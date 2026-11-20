"""
Bad-pixel stage: interpolate over the pixels flagged in the master bad-pixel mask.

Each bad pixel is replaced by the mean of the good pixels in the 3x3 patch around it
(astropy ``interpolate_replace_nans``); bad pixels whose whole 3x3 patch is bad fall back
to a 5x5 patch. The mask file is named by ``BADPIX_MASTER`` in
``[file_names]`` of the dataset config (nonzero = bad, same shape as the frames).

Ported from modules/preprocessing/fix_bad_pix.py of the AB Aur pipeline.
"""

import logging
import os
import warnings

import numpy as np
from astropy.convolution import interpolate_replace_nans
from astropy.io import fits
from astropy.utils.exceptions import AstropyUserWarning

from modules.preprocessing.frame_io import read_frame, write_frame

# patch around each bad pixel whose good pixels are averaged (normalized by astropy)
DEFAULT_KERNEL = np.ones((3, 3))
# larger patch for bad pixels whose whole DEFAULT_KERNEL patch is bad
FALLBACK_KERNEL = np.ones((5, 5))


def fix_bad_pixels(frame, bad_mask, kernel=None):
    '''
    Replace the bad pixels of a frame by interpolation over their good neighbors.

    INPUTS:
    - frame (numpy array): 2D frame
    - bad_mask (numpy array): Boolean (or 0/nonzero) array of the frame's shape; True = bad
    - kernel (numpy array or None): Interpolation kernel; None uses ``DEFAULT_KERNEL`` (3x3)

    OUTPUTS:
    - fixed (numpy array): Copy of ``frame`` (same dtype) with the bad pixels replaced. Good
      pixels are unchanged. A bad pixel with no good pixel in its ``kernel`` patch is retried
      with ``FALLBACK_KERNEL`` (5x5); only if that patch is all bad too does it stay NaN.
    '''
    bad_mask = np.asarray(bad_mask).astype(bool)
    if bad_mask.shape != np.shape(frame):
        raise ValueError(f"Bad-pixel mask shape {bad_mask.shape} does not match frame shape {np.shape(frame)}")
    kernel = DEFAULT_KERNEL if kernel is None else kernel

    sci_badnan = np.array(frame, dtype=np.float64)
    sci_badnan[bad_mask] = np.nan
    with warnings.catch_warnings():
        # astropy warns when NaNs survive; those are handled by the fallback below
        warnings.simplefilter("ignore", AstropyUserWarning)
        fixed = interpolate_replace_nans(array=sci_badnan, kernel=kernel)

        # fallback: bad pixels whose whole patch was bad (still NaN) are retried with a
        # larger patch, using only the originally good pixels (first-pass values are not reused)
        left = np.isnan(fixed) & bad_mask
        if left.any():
            fallback = interpolate_replace_nans(array=sci_badnan, kernel=FALLBACK_KERNEL)
            fixed[left] = fallback[left]
            n_still = int(np.isnan(fixed[bad_mask]).sum())
            logging.debug('Bad-pixel fallback (%dx%d) applied to %d pixels; %d still NaN',
                          *FALLBACK_KERNEL.shape, int(left.sum()), n_still)

    return fixed.astype(np.asarray(frame).dtype, copy=False)


class FixPixSingle:
    '''
    Interpolates over the bad pixels of a single frame.
    '''

    def __init__(self, write_dir, master_bad_pix_file, config):
        '''
        Read the bad-pixel mask.

        INPUTS:
        - write_dir (str): Directory for the fixed frames (with trailing slash)
        - master_bad_pix_file (str): Bad-pixel mask FITS path (0 = good, nonzero = bad)
        - config (ConfigParser): Configuration data

        OUTPUTS:
        - None
        '''
        self.write_dir = write_dir
        self.config = config

        with fits.open(master_bad_pix_file) as hdul:
            self.bad_mask = np.asarray(hdul[0].data) != 0
        logging.info('Read bad-pixel mask %s (%d bad pixels)', master_bad_pix_file, int(self.bad_mask.sum()))

    def __call__(self, file_name):
        '''
        Interpolate over the bad pixels of one frame and write it to ``write_dir`` (failures are
        logged and skipped; one frame per call, so the job can be parallelized).

        INPUTS:
        - file_name (str): Frame path

        OUTPUTS:
        - None
        '''
        frame_data, header = read_frame(file_name)

        try:
            fixed = fix_bad_pixels(frame_data, self.bad_mask)

            file_name_write = self.write_dir + os.path.basename(file_name)
            write_frame(file_name_write, fixed, header, "bad-pixel-fixed")
            logging.info('Wrote %s', file_name_write)

        except Exception as e:
            logging.warning('Failed to fix bad pixels in %s: %s', os.path.basename(file_name), e)
