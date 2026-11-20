import numpy as np
from astropy.io import fits
import os
import logging

from modules.preprocessing.frame_io import write_frame

class MakeSlopeFrameSingle:
    '''
    Make 2D frames from the raw cubes. A 2-read cube gives the slope frame (second read minus
    first); a 1-read cube (as in the Procyon data, NGROUPS = 1) has no slope to take, and its
    single read is passed through.
    '''

    def __init__(self, write_dir, config):

        '''
        INPUTS:
        - write_dir (str): Directory for the slope frames (with trailing slash)
        - config (ConfigParser): Configuration data

        OUTPUTS:
        - None
        '''
        self.config_data = config
        self.write_dir = write_dir

    def __call__(self, file_name):

        '''
        Make the 2D frame of one raw cube and write it to ``write_dir`` (failures are logged
        and skipped).

        INPUTS:
        - file_name (str): Raw cube path (1 or 2 reads)

        OUTPUTS:
        - None
        '''
        self.file_name = file_name

        with fits.open(self.file_name) as hdul:

            try:

                # (reading the data inside the try, so a truncated file is skipped, not fatal)
                frame_data = hdul[0].data
                header = hdul[0].header.copy()
                n_reads = frame_data.shape[0] if frame_data.ndim == 3 else 1

                if n_reads == 2:
                    # subtract in signed 32-bit integers, so negative results don't wrap around
                    frame_slope = np.subtract(frame_data[1,:,:].astype(np.int32),frame_data[0,:,:].astype(np.int32))
                    red_step = "slope"
                elif n_reads == 1:
                    frame_slope = np.reshape(frame_data, frame_data.shape[-2:]).astype(np.float32)
                    red_step = "single_read"
                else:
                    raise ValueError(f"expected 1 or 2 reads, got {n_reads}")

                # Write the array to a new FITS file, keeping the raw header
                file_name_write = self.write_dir + os.path.basename(file_name)
                write_frame(file_name_write, frame_slope, header, red_step)
                logging.info('Wrote %s frame %s', red_step, file_name_write)

                del frame_slope

            except Exception as e:

                logging.warning('Failed to make frame from %s: %s', os.path.basename(file_name), e)
