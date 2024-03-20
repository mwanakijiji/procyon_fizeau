# Dark subtracts all frames in the dataset

# Created 2023 Dec. 3

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
#import sys
import glob, os

stem_read = '/Volumes/One Touch/procyon_fizeau_2023B/raw_data/'
stem_write = '/Volumes/One Touch/procyon_fizeau_2023B/01_dark_subtracted/'

master_dark_file = '/Volumes/One Touch/procyon_fizeau_2023B/calib_files/dark_median.fits'

file_list = np.sort(glob.glob(stem_read + '*fits'))

with fits.open(master_dark_file) as hdul_dark:

    master_dark = hdul_dark[0].data
    print(np.shape(master_dark))

for t in range(100, len(file_list)):
    file_name = file_list[t]
    print(file_name)
    with fits.open(file_name) as hdul:
        frame_data = hdul[0].data
        print(np.shape(frame_data))
        try:
            dark_subted = np.subtract(frame_data, master_dark)
            # Write the array to a new FITS file
            fits.writeto(stem_write + os.path.basename(file_name), dark_subted, overwrite=False)
            '''
            plt.hist(np.ndarray.flatten(frame_slope), bins=50000)
            plt.xlim([200,1000])
            plt.show()
            '''
            del dark_subted
            del frame_data
            hdul.close()  # Close the FITS file
        except:
            print('Failed to dark-subtract ', os.path.basename(file_name))

