"""
Optional bkg_nod stage: subtract a median of nearby opposite-nod frames.

For each frame, take the NOD_N_FRAMES frames nearest in frame number (i.e. time) that
have the same target and filter but the star in a different region (the other nod),
mask the star in each of them, and subtract their median. This removes thermal and sky
structure before the channel and region PCA passes.

Opposite-nod frames whose block has no star on the detector (blank ``quadrant``, e.g. the
off-source nod) are used unmasked. Where the star masks of all opposite-nod frames overlap
(around the other nod's star position), the median is NaN, and so is the subtracted frame.
"""

import logging
import os
import warnings
from collections import OrderedDict

import numpy as np

from modules.shared.dataset import block_for_frame, is_sci
from modules.preprocessing.frame_io import read_frame, write_frame
from modules.preprocessing.star import locate_star, block_hint, block_has_star, star_nan_mask


class NodSubtractSingle:
    '''
    Callable for one frame; keeps a small cache of star-masked frames so that neighbouring
    frames (which share most of their opposite-nod frames) are read only once per worker.
    '''

    def __init__(self, write_dir, files_by_frame, blocks, n_frames, psf_radius, halfbox=50,
                 prefix=""):
        '''
        INPUTS:
        - write_dir (str): Output directory
        - files_by_frame (dict): Map of frame number to input file, for all available frames
        - blocks (DataFrame): Blocks CSV (``load_blocks``)
        - n_frames (int): Number of opposite-nod frames in each median
        - psf_radius (float): Radius (pix) of the star mask in the opposite-nod frames
        - halfbox (int): Half-width of the star search box around the blocks-CSV position
        - prefix (str): Filename prefix, to parse frame numbers

        OUTPUTS:
        - None
        '''
        self.write_dir = write_dir
        self.files_by_frame = files_by_frame
        self.blocks = blocks
        self.n_frames = n_frames
        self.psf_radius = psf_radius
        self.halfbox = halfbox
        self.prefix = prefix
        self._cache = OrderedDict()
        self._cache_size = 3 * n_frames

        # usable frames of each (target, filter, region), sorted by frame number
        self._groups = {}
        for n in sorted(files_by_frame):
            b = block_for_frame(blocks, n)
            if not is_sci(b):
                continue
            self._groups.setdefault((b["target"], b["filter"], b["quadrant"]), []).append(n)

    def opposite_nod_frames(self, frame_num):
        '''
        The ``n_frames`` opposite-nod frame numbers nearest to ``frame_num``.

        INPUTS:
        - frame_num (int): Frame number

        OUTPUTS:
        - nums (list of int): Sorted frame numbers (empty if there are none)
        '''
        b = block_for_frame(self.blocks, frame_num)
        if not is_sci(b):
            # darks, calibrations and frames in no block have no opposite nod
            return []
        others = [n for (t, f, q), nums in self._groups.items()
                  if t == b["target"] and f == b["filter"] and q != b["quadrant"]
                  for n in nums]
        others.sort(key=lambda n: (abs(n - frame_num), n))
        return sorted(others[:self.n_frames])

    def _masked(self, frame_num):
        '''
        One frame with its star masked (NaN within ``psf_radius``; unmasked if its block has
        no star on the detector), from the per-worker cache.

        INPUTS:
        - frame_num (int): Frame number

        OUTPUTS:
        - masked (numpy array): Star-masked frame
        '''
        if frame_num in self._cache:
            self._cache.move_to_end(frame_num)
            return self._cache[frame_num]
        frame, _ = read_frame(self.files_by_frame[frame_num])
        b = block_for_frame(self.blocks, frame_num)
        if block_has_star(b):
            star = locate_star(frame, block_hint(b), self.halfbox)
            masked = frame * star_nan_mask(frame.shape, star, self.psf_radius)
        else:
            masked = frame
        self._cache[frame_num] = masked
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return masked

    def __call__(self, abs_sci_name):
        '''
        Subtract the median of the nearest opposite-nod frames (stars masked) from one frame and
        write it to ``write_dir`` (with NODN, NODFRST and NODLAST in the header). Frames with no
        opposite-nod frames are skipped.

        INPUTS:
        - abs_sci_name (str): Frame path

        OUTPUTS:
        - None
        '''
        base = os.path.basename(abs_sci_name)
        frame_num = int(base.split(self.prefix, 1)[1].split(".fits")[0])
        nums = self.opposite_nod_frames(frame_num)
        if not nums:
            logging.info(f"No opposite-nod frames for {base}; skipping")
            return
        sci, header = read_frame(abs_sci_name)
        stack = np.stack([self._masked(n) for n in nums])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN pixels at the other nod's star
            nod_median = np.nanmedian(stack, axis=0)
        write_frame(os.path.join(self.write_dir, base), sci - nod_median, header, "nod_subtracted",
                    extra={"NODN": (len(nums), "opposite-nod frames in subtracted median"),
                           "NODFRST": (nums[0], "first opposite-nod frame"),
                           "NODLAST": (nums[-1], "last opposite-nod frame")})
        logging.info(f"Wrote nod-subtracted frame {base} (median of {len(nums)} frames {nums[0]}-{nums[-1]})")
