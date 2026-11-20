import logging
import os

import numpy as np

from modules.preprocessing.frame_io import read_frame, write_frame
from modules.preprocessing.star import locate_star, star_nan_mask

class RemoveStrayRamp:
    '''
    Removes an additive electronic artifact illumination ramp in y (which appears to be linear) at the top of the
    LMIRcam readouts. (The ramp has something to do with resetting of the detector after each block of frames.
    See emails from J. Leisenring and J. Stone, Sept. 5/6 2018, about the non-linear ramp effect in the pre-2018 detector.)

    Even if a stray gradient is not present, this function acts to remove the residual backround that remains 
    following dark-subtraction.

    JL, referring to a particular dataset:
    > It appears to also be affecting the side column reference pixels (which aren't light sensitive), so it's not an illumination problem. 
    There is a phenomenon that occurs in these electronics known as a "reset anomaly." The basic premise is that when you reset the detector, 
    a reference voltage gets pulled because of the increased current flowing through the reset switch. After the reset frame is finished, it 
    takes some time for that reference voltage to settle back to its original location. This is usually imprinted on the raw frame as a 
    gradient in the y-direction. However, that behavior is generally pretty consistent, so it should subtract away, unlike in your bg-subtracted data. 
    Also, your raw plots show a very strange profile, so I'm not really sure what to make of this other than to chalk it up to weirdness of the ISDEC.
    Given that you have a large background region surrounding your PSF, you can just take a section of the frame to create an average column value 
    and then subtract that away from all of the data to remove the gradient and flatten things out. Hopefully there will be no residuals.
    '''

    def __init__(self, write_dir, config, psf_mask_radius=150.):

        '''
        INPUTS:
        - write_dir (str): Directory for the ramp-removed frames (with trailing slash)
        - config (ConfigParser): Configuration data
        - psf_mask_radius (float): Radius (pix) around the star excluded from the ramp fit

        OUTPUTS:
        - None
        '''
        self.write_dir = write_dir
        self.config_data = config
        self.psf_mask_radius = psf_mask_radius

    def __call__(self, abs_sci_name):
        '''
        Fit a line to the row medians of one frame (star masked, 5 rows trimmed at each end),
        subtract it from every column, and write the frame to ``write_dir`` (with RAMPMED, the
        median of the row medians, in the header).

        INPUTS:
        - abs_sci_name (str): Frame path

        OUTPUTS:
        - None
        '''

        # read in the science frame from data directory
        sci, header_sci = read_frame(abs_sci_name)

        # mask the star (and its halo out to about the AO control radius), so that it
        # does not enter the ramp estimate, wherever on the detector it is
        psf_loc = locate_star(sci)
        sci_star_masked = sci * star_nan_mask(sci.shape, psf_loc, self.psf_mask_radius)

        # find the median in x across the whole array as a function of y
        # (N.b. offsets in the channels shouldn't be a problem, since both
        # the ramp and the channel offsets should be additive)
        stray_ramp = np.nanmedian(sci_star_masked, axis=1)
        # if the median of the median has a value between 20 and 50, then the
        # ramp is there and needs to be removed
        median_ramp_val = np.nanmedian(stray_ramp)

        # make linear fit and subtract it
        x_dummy = np.arange(len(stray_ramp))
        slope, intercept = np.polyfit(x_dummy[5:-5], stray_ramp[5:-5], 1) # avoid edges in fit
        linear_fit = slope * x_dummy + intercept

        # note this effect is subtracted, based on JL's advice
        image_ramp_subted = np.subtract(sci, linear_fit[:, np.newaxis]).astype(np.float32)

        # indicate last reduction step
        red_step = "ramp_removed"

        # write file out
        abs_image_ramp_subted_name = str(self.write_dir + os.path.basename(abs_sci_name))
        extra = {}
        if np.isfinite(median_ramp_val):
            extra["RAMPMED"] = (float(median_ramp_val), "median of row medians, star masked")
        write_frame(abs_image_ramp_subted_name, image_ramp_subted, header_sci, red_step, extra=extra)
        logging.info("Writing out ramp-removed-fixed frame " + os.path.basename(abs_image_ramp_subted_name))
