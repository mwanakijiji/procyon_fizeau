"""
Detector helpers for the background stages: masking of unusable detector areas, the
channel-pedestal basis, and PCA bases.

The readout channel layout (``N_CHANNELS`` channels, each ``CHANNEL_WIDTH`` columns wide;
for LMIRcam 32 x 64 = 2048, NOUTPUTS = 32 in the headers) is set in ``[detector]`` of the
dataset config. The Procyon frames are the Center_512 subarray (512 x 2048), so the channel
basis is built for whatever frame shape is passed in.

The area masked by ``DetectorMask`` (edge width and boxes) is set in ``[detector_mask]``
of the dataset config.
"""

import logging

import numpy as np

from modules.shared.dataset import _parse_slice


def channel_layout(config):
    '''
    Readout channel layout from ``[detector]`` of the merged config.

    INPUTS:
    - config (ConfigParser): Merged config

    OUTPUTS:
    - n_channels (int): Number of readout channels (``N_CHANNELS``)
    - channel_width (int): Columns per channel (``CHANNEL_WIDTH``)
    '''
    sec = config["detector"]
    return sec.getint("N_CHANNELS"), sec.getint("CHANNEL_WIDTH")


class DetectorMask:
    '''
    Sets to NaN the detector edges and any boxes listed in the config, and sets stray
    32-bit-float pixels (|value| > 1e5) to 0. Used when making and fitting the PCA backgrounds.
    '''

    def __init__(self, edge_width=0, boxes=()):
        '''
        INPUTS:
        - edge_width (int): Number of rows/columns masked at each edge of the frame
        - boxes (list of tuple): ``(y_slice, x_slice)`` boxes to mask

        OUTPUTS:
        - None
        '''
        self.edge_width = int(edge_width)
        self.boxes = list(boxes)

    @classmethod
    def from_config(cls, config):
        '''
        Read ``[detector_mask]`` from the merged config: ``MASK_EDGE_WIDTH`` (int) and
        ``MASK_BOXES`` (``y_start:y_stop, x_start:x_stop`` boxes separated by ``;``,
        Python slice convention; may be blank).

        INPUTS:
        - config (ConfigParser): Merged config

        OUTPUTS:
        - mask (DetectorMask): The mask
        '''
        sec = config["detector_mask"]
        boxes = []
        for box in sec.get("MASK_BOXES", "").split(";"):
            if not box.strip():
                continue
            ys, xs = box.split(",")
            boxes.append((_parse_slice(ys), _parse_slice(xs)))
        return cls(edge_width=sec.getint("MASK_EDGE_WIDTH"), boxes=boxes)

    def __call__(self, image):
        '''
        Apply the mask in place.

        INPUTS:
        - image (numpy array): 2D image or 3D cube (frames along the first axis); modified in place

        OUTPUTS:
        - image (numpy array): The masked image (same object as the input)
        '''
        e = self.edge_width
        if e > 0:
            image[..., :e, :] = np.nan
            image[..., -e:, :] = np.nan
            image[..., :, :e] = np.nan
            image[..., :, -e:] = np.nan
        for ys, xs in self.boxes:
            image[..., ys, xs] = np.nan

        # deal with a bug associated with 32-bit floats,
        # where stray pixels can be assigned value +-2.1474842e+09
        # (left un-fixed, it screws up the whole PCA decomposition)
        image[np.abs(image) > 1e+05] = 0

        return image


def channels_PCA_cube(shape, n_channels, channel_width):
    '''
    Basis representing the bias changes of the detector readout channels.

    INPUTS:
    - shape (tuple of int): Frame shape ``(ny, nx)``; ``nx`` must be ``n_channels * channel_width``
    - n_channels (int): Number of readout channels
    - channel_width (int): Columns per channel

    OUTPUTS:
    - channel_vars_PCA (numpy array): Cube (n_channels, ny, nx); slice k is 1 in channel k,
      0 elsewhere
    '''
    ny, nx = shape
    if nx != n_channels * channel_width:
        raise ValueError(f"Expected {n_channels} x {channel_width} = {n_channels * channel_width} "
                         f"columns ([detector] in the config), got {nx}")
    channel_vars_PCA = np.zeros((n_channels, ny, nx))
    for chNum in range(n_channels):
        channel_vars_PCA[chNum, :, chNum * channel_width:(chNum + 1) * channel_width] = 1.
    return channel_vars_PCA


def PCA_basis(training_cube_masked_weird, n_PCA):
    '''
    PCA basis of a training cube, using only the pixels finite in its first frame (randomized
    SVD); pixels not used are NaN in the basis.

    INPUTS:
    - training_cube_masked_weird (numpy array): Training images (num_images, y, x), with the
      bad regions already masked (NaN)
    - n_PCA (int): Number of PCA components

    OUTPUTS:
    - pca_comp_cube (numpy array): PCA basis (n_PCA, y, x), float32
    '''
    from sklearn.decomposition import PCA

    shape_img = np.shape(training_cube_masked_weird[0, :, :])

    # flatten each individual frame into a 1D array
    logging.info("Flattening the training cube...")
    test_cube_1_1ds = np.reshape(training_cube_masked_weird,
                                 (np.shape(training_cube_masked_weird)[0], shape_img[0] * shape_img[1]))

    # keep only the pixels finite in the first frame
    idx = np.isfinite(test_cube_1_1ds[0, :])
    training_set_1ds_noNaN = test_cube_1_1ds[:, idx]

    logging.info("Doing PCA to make PCA basis cube...")
    pca = PCA(n_components=n_PCA, svd_solver="randomized")
    pca.fit(training_set_1ds_noNaN)
    del training_set_1ds_noNaN

    # reinsert the NaN values into each slice of the PCA basis set
    logging.info('Putting PCA components into cube...')
    pca_comp_cube = np.full((n_PCA, shape_img[0], shape_img[1]), np.nan, dtype=np.float32)
    for slicenum in range(n_PCA):
        slice_1d = np.full(test_cube_1_1ds.shape[1], np.nan)
        slice_1d[idx] = pca.components_[slicenum]
        pca_comp_cube[slicenum, :, :] = slice_1d.reshape(shape_img).astype(np.float32)

    return pca_comp_cube
