# dark-subtracts frames

import numpy as np
from astropy.io import fits
import os
import logging

from modules.preprocessing.frame_io import read_frame, write_frame


class DarkSubtSingle:

    '''
    Dark subtraction of a single slope frame.
    '''
    def __init__(self, write_dir, master_dark, config):

        '''
        Read the master dark.

        INPUTS:
        - write_dir (str): Directory for the dark-subtracted frames (with trailing slash)
        - master_dark (str): Master dark FITS path
        - config (ConfigParser): Configuration data

        OUTPUTS:
        - None
        '''
        self.config_data = config
        self.write_dir = write_dir

        with fits.open(master_dark) as hdul_dark:
            self.master_dark = hdul_dark[0].data.astype(np.float64)

    def __call__(self, file_name):

        '''
        Subtract the master dark from one frame and write it to ``write_dir`` (failures are logged
        and skipped).

        INPUTS:
        - file_name (str): Frame path

        OUTPUTS:
        - None
        '''
        self.file_name = file_name

        frame_data, header = read_frame(self.file_name)

        try:

            dark_subted = np.subtract(frame_data.astype(np.float64), self.master_dark)

            # Write the array to a new FITS file, keeping the header (needed downstream for LBT_PARA)
            file_name_write = self.write_dir + os.path.basename(file_name)
            write_frame(file_name_write, dark_subted, header, "dark-subtraction")
            logging.info('Wrote %s', file_name_write)

            del dark_subted

        except Exception as e:

            logging.warning('Failed to dark-subtract %s: %s', os.path.basename(file_name), e)

        del frame_data
