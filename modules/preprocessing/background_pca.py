"""
PCA background subtraction (bkg_channel and bkg_regions stages) and fusing of the
region-subtracted frames (fuse stage).

Adapted from modules/preprocessing/basic_red_fcns.py of the AB Aur pipeline; the detector
mask is passed in (``DetectorMask``) instead of being hard-coded for a 1024 x 2048 frame.
"""

import logging
import os
import time

import matplotlib
matplotlib.use('agg') # avoids some crashes when multiprocessing
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from modules.preprocessing.detector import channel_layout, channels_PCA_cube
from modules.preprocessing.frame_io import read_frame, write_frame
from modules.preprocessing.star import find_airy_psf, star_nan_mask


class FuseRegions:
    '''
    Fuse the background-subtracted frames of each detector region into one frame.

    Each input frame is NaN outside its region (its PCA basis is NaN there). Where regions
    overlap, the frames are averaged; pixels in no region stay NaN.
    '''

    def __init__(self, write_dir, region_dirs):
        '''
        INPUTS:
        - write_dir (str): Directory for the fused frames
        - region_dirs (list of str): One directory per region, holding frames with the same basenames

        OUTPUTS:
        - None
        '''
        self.write_dir = write_dir
        self.region_dirs = list(region_dirs)

    def __call__(self, abs_sci_name):
        '''
        Average the region frames with this basename where they overlap (pixels in no region
        stay NaN) and write the fused frame to ``write_dir``.

        INPUTS:
        - abs_sci_name (str): Any frame path whose basename names the frame to fuse

        OUTPUTS:
        - None
        '''
        basename = os.path.basename(abs_sci_name)
        frames = []
        header_sci = None
        for region_dir in self.region_dirs:
            frame, header = read_frame(os.path.join(region_dir, basename))
            frames.append(frame)
            if header_sci is None:
                header_sci = header
        stack = np.stack(frames)
        n_finite = np.sum(np.isfinite(stack), axis=0)
        fused = np.nansum(stack, axis=0) / np.maximum(n_finite, 1)
        fused[n_finite == 0] = np.nan

        write_frame(os.path.join(self.write_dir, basename), fused, header_sci, "regions_fused")
        logging.info("Writing out region-fused frame " + basename)


def pca_backg_maker_channels_only(abs_pca_cube_file_name, shape, n_channels, channel_width):
    '''
    Write a simple PCA basis consisting of the readout-channel variations alone.

    INPUTS:
    - abs_pca_cube_file_name (str): Output FITS path
    - shape (tuple of int): Frame shape ``(ny, nx)``
    - n_channels (int): Number of readout channels
    - channel_width (int): Columns per channel

    OUTPUTS:
    - None
    '''
    fits.writeto(filename=abs_pca_cube_file_name, data=channels_PCA_cube(shape, n_channels, channel_width),
                 header=None, overwrite=True)
    logging.info("Wrote out background PCA cube, consisting of channel variations only, as " +
                 abs_pca_cube_file_name)


class BackgroundPCASubtSingle:
    '''
    Does a PCA decomposition of the background of a science frame and subtracts it
    (N.b. remaining pedestal should be photons alone).

    Steps are
    1. reads in PCA component cube
    2. masks and subtracts the median (just a constant) from each science frame
    3. decomposes each science frame into its PCA components (with a mask over the PSF)
    4. subtracts the reconstructed background
    5. saves the background-subtracted images
    '''

    def __init__(self, n_PCA, pca_cube_file, write_dir, mask, config_data,
                 psf_mask_radius=150., write_diagnostics=False):
        '''
        Read the PCA component cube and mask the bad detector regions in it.

        INPUTS:
        - n_PCA (int): Number of components to reconstruct the background with
        - pca_cube_file (str): PCA cube
        - write_dir (str): Directory for the background-subtracted frames (with trailing slash)
        - mask (DetectorMask): Mask of the unusable detector areas
        - config_data (ConfigParser): Configuration data
        - psf_mask_radius (float): Radius (pix) around the star excluded from the background fit
        - write_diagnostics (bool): For every frame, write the reconstructed background and the
          masked frames before/after subtraction to DIR_OTHER_FITS, and a stats PNG to DIR_FYI_INFO

        OUTPUTS:
        - None
        '''
        self.config_data = config_data
        self.write_dir = write_dir
        self.n_PCA = n_PCA
        self.mask = mask
        self.psf_mask_radius = psf_mask_radius
        self.write_diagnostics = write_diagnostics
        self.cube_tag = os.path.splitext(os.path.basename(pca_cube_file))[0]

        self.pca_cube = fits.getdata(pca_cube_file, 0, header=False)

        # apply mask over unusable detector regions to PCA cube
        # (N.b. otherwise the PCA won't converge!)
        self.pca_cube = self.mask(self.pca_cube.astype(np.float32))

    def __call__(self, abs_sci_name):
        '''
        PCA background subtraction of one frame (one frame per call, so the job can be
        parallelized). The PSF is masked within ``psf_mask_radius`` for the fit; the reconstructed
        background is subtracted from the whole frame, which is written to ``write_dir``.

        INPUTS:
        - abs_sci_name (str): Science frame path

        OUTPUTS:
        - None
        '''
        start_time = time.time()
        sciImg_subtracted_name = os.path.join(self.write_dir, os.path.basename(abs_sci_name))

        # read in the science frame
        sciImg, header_sci = read_frame(abs_sci_name)
        sciImg = sciImg.astype(np.float64)

        # apply mask over unusable detector regions to science image
        sciImg = self.mask(sciImg)

        ## mask the PSF out to psf_mask_radius, so that the halo does not enter the fit
        # (find_airy_psf turns NaNs into zeros in place, so give it a copy)
        psf_loc = find_airy_psf(np.copy(sciImg)) # center of science PSF
        logging.info("PSF location in " + os.path.basename(abs_sci_name) + \
              ": [y,x]=[" + str(psf_loc[0]) + ", " + str(psf_loc[1]) + "]")
        psf_mask = star_nan_mask(np.shape(sciImg), psf_loc, self.psf_mask_radius)

        sciImg_masked = np.multiply(sciImg,psf_mask) # this is now the masked science frame

        # subtract the median (just a constant) from the remaining science image
        sciImg_psf_masked = np.subtract(sciImg_masked,np.nanmedian(sciImg_masked)) # where PSF is masked
        sciImg_psf_not_masked = np.subtract(sciImg,np.nanmedian(sciImg_masked)) # where PSF is not masked

        ## PCA-decompose, using only the pixels that are finite in both the PCA cube and the
        ## masked science frame (equivalent to masking the PCA cube, without copying it)
        pca_1ds = np.reshape(self.pca_cube[0:self.n_PCA,:,:], (self.n_PCA, -1))
        sci_masked_1d = np.ravel(sciImg_psf_masked)
        idx = np.logical_and(np.isfinite(pca_1ds[0,:]), np.isfinite(sci_masked_1d))

        # the vector of component amplitudes
        soln_vector = np.linalg.lstsq(pca_1ds[:,idx].astype(np.float64).T, sci_masked_1d[idx], rcond=None)

        # reconstruct the background based on that vector
        # note that the PCA components WITHOUT masking of the PSF location is being
        # used to reconstruct the background
        recon_backgrnd_2d = np.reshape(np.dot(soln_vector[0], pca_1ds), np.shape(sciImg))

        # do the actual subtraction
        sciImg_subtracted = np.subtract(sciImg_psf_not_masked,recon_backgrnd_2d)

        write_frame(sciImg_subtracted_name, sciImg_subtracted, header_sci, "pca_background_subtracted",
                    extra={"PSFMASKR": (float(self.psf_mask_radius), "radius (pix) masked in bkgd fit"),
                           "PCABASIS": (self.cube_tag, "PCA basis of the background fit")})
        logging.info('Background-subtracted frame '+
              os.path.basename(abs_sci_name)+
              ' written out. PCA = '+str(self.n_PCA))

        if self.write_diagnostics:
            other = self.config_data["data_dirs"]["DIR_OTHER_FITS"]
            base = os.path.basename(abs_sci_name)

            # reconstructed background, and the masked science frame before and after subtraction
            background_subtracted_masked = np.multiply(self.mask(np.copy(sciImg_subtracted)), psf_mask)
            for label, data in (("recon_bkgd", recon_backgrnd_2d),
                                ("masked_sci_before_bkd_subt", sciImg_psf_masked),
                                ("masked_sci_after_bkd_subt", background_subtracted_masked)):
                fits.writeto(os.path.join(other, f"{label}_{self.cube_tag}_{base}"),
                             data.astype(np.float32), overwrite=True)
            logging.info("Wrote background-subtraction diagnostics for " + base)

            self.vital_stats(file_base_name=base,
                             sci_img_pre=sciImg_masked,
                             sci_img_post_all_subt=background_subtracted_masked,
                             pca_spec=soln_vector[0])

        logging.info('Elapsed time: %.2f s', time.time() - start_time)

    def vital_stats(self, file_base_name, sci_img_pre, sci_img_post_all_subt, pca_spec):
        '''
        Save an FYI plot of the PCA spectrum and of the counts before and after background
        subtraction (to DIR_FYI_INFO), and log the noise. The PSF and the masked detector areas
        are NaN in both images, so the statistics are over the rest of the frame.

        INPUTS:
        - file_base_name (str): Frame basename
        - sci_img_pre (numpy array): Masked science frame before background subtraction
        - sci_img_post_all_subt (numpy array): Masked science frame after subtraction of the whole
          reconstructed background
        - pca_spec (numpy array): PCA amplitudes; in the channel pass these are the channel pedestals

        OUTPUTS:
        - None
        '''
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15,5))

        pre = sci_img_pre[np.isfinite(sci_img_pre)]
        post = sci_img_post_all_subt[np.isfinite(sci_img_post_all_subt)]

        axes[0].set_title(f'Pre-backgd subtraction\nMedian: {np.median(pre)}\nStdev: {np.std(pre)}')
        axes[0].hist(pre, bins=200)

        axes[1].set_title('PCA Spec')
        axes[1].plot(pca_spec)
        n_channels = channel_layout(self.config_data)[0]
        if len(pca_spec) > n_channels:
            axes[1].axvline(n_channels, color='k', linestyle=':')

        axes[2].set_title(f'Post-backgd subtraction\nMedian: {np.median(post)}\nStdev: {np.std(post)}')
        axes[2].hist(post, bins=200)

        logging.info('before any subt, %s: std=%s, med=%s', file_base_name, np.std(pre), np.median(pre))
        logging.info('after all, %s: std=%s, med=%s', file_base_name, np.std(post), np.median(post))

        plt.suptitle("frame " + file_base_name)
        plt.tight_layout()
        abs_vital_stats_png = os.path.join(self.config_data["data_dirs"]["DIR_FYI_INFO"],
                                           'bkgd_subt_stats_' + file_base_name.split(".")[0] + '.png')
        plt.savefig(abs_vital_stats_png)
        plt.close(fig)
        logging.info("Wrote out vital stats PNG for " + file_base_name)
