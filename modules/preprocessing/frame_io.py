"""
Reading and writing single detector frames between preprocessing stages.

Every stage keeps the original FITS header (so ``LBT_PARA`` etc. survive to the end),
records the stage in ``RED_STEP``, and stores the data as float32.
"""

import numpy as np
from astropy.io import fits

# keywords that describe the raw (integer, 3D) data and must not be copied onto float frames
_RAW_ONLY_KEYS = ("BZERO", "BSCALE", "BLANK", "NAXIS3")


def read_frame(path):
    '''
    Read a 2D frame as float32 with its header.

    INPUTS:
    - path (str): FITS file

    OUTPUTS:
    - data (numpy array): float32 frame
    - header (Header): Primary header
    '''
    with fits.open(path) as hdul:
        data = np.asarray(hdul[0].data, dtype=np.float32)
        header = hdul[0].header.copy()
    return data, header


def clean_header(header):
    '''
    Copy of ``header`` without the keywords that only apply to the raw integer cube.

    INPUTS:
    - header (Header): FITS header

    OUTPUTS:
    - header (Header): Cleaned copy
    '''
    header = header.copy()
    for key in _RAW_ONLY_KEYS:
        header.remove(key, ignore_missing=True, remove_all=True)
    return header


def write_frame(path, data, header, red_step, extra=None):
    '''
    Write a frame as float32, with the cleaned header and ``RED_STEP`` set.

    INPUTS:
    - path (str): Output FITS file
    - data (numpy array): Frame
    - header (Header): Header to carry along
    - red_step (str): Value for ``RED_STEP``
    - extra (dict): Further keywords, ``{key: value}`` or ``{key: (value, comment)}``
    '''
    header = clean_header(header)
    header["RED_STEP"] = red_step
    for key, value in (extra or {}).items():
        header[key] = value
    fits.writeto(path, np.asarray(data, dtype=np.float32), header=header, overwrite=True)
