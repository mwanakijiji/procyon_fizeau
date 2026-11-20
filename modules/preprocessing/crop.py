"""Crop odd-sized frames about their center (crop stage).

The cutout and register stages leave the star at ((N-1)/2, (N-1)/2) of odd N x N frames;
cropping about the center keeps it there.

Adapted from modules/adi/pad_frames.py of the AB Aur pipeline."""

import glob
import logging
import os

from astropy.io import fits

logger = logging.getLogger(__name__)


def crop_center(frame, size, header=None):
    '''
    Crop the central ``size`` x ``size`` pixels of an odd-sized square frame.

    With the star at ((N-1)/2, (N-1)/2) in the input, it is at ((size-1)/2, (size-1)/2) in
    the output.

    INPUTS:
    - frame (numpy array): Square frame, odd side
    - size (int): Output side (odd, <= the input side)
    - header (Header): If given, a copy with CUTSIZE, STARX, STARY, CUTX0 and CUTY0 updated
      is returned too

    OUTPUTS:
    - cropped (numpy array): Central crop (a view)
    - header (Header): Updated header copy (only if ``header`` is given)
    '''
    n_y, n_x = frame.shape
    if n_y != n_x or n_y % 2 != 1 or size % 2 != 1 or size > n_y:
        raise ValueError(f"Need an odd square frame and an odd size <= its side; got {frame.shape}, {size}")
    start = (n_y - size) // 2
    cropped = frame[start:start + size, start:start + size]
    if header is None:
        return cropped
    header = header.copy()
    c = (size - 1) / 2.
    header["CUTSIZE"] = (size, "cutout side (pix)")
    header["STARX"] = (c, "star x in this frame (0-indexed)")
    header["STARY"] = (c, "star y in this frame (0-indexed)")
    for key in ("CUTX0", "CUTY0"):
        if key in header:
            header[key] = header[key] + start
    return cropped, header


def crop_fits_dir(input_dir, output_dir, size, skip_existing=False, only=None):
    '''
    Write ``crop_center`` crops of every FITS file in ``input_dir`` to ``output_dir``
    (or only of the basenames in ``only``, if given).

    INPUTS:
    - input_dir (str): Directory of odd-sized square frames
    - output_dir (str): Output directory (created if needed)
    - size (int): Output side (odd)
    - skip_existing (bool): Skip files already in ``output_dir``
    - only (set of str): Basenames to crop; None crops all

    OUTPUTS:
    - written (list of str): Paths written
    '''
    os.makedirs(output_dir, exist_ok=True)
    written = []
    for fits_file in sorted(glob.glob(os.path.join(str(input_dir), "*.fits"))):
        if only is not None and os.path.basename(fits_file) not in only:
            continue
        output_file = os.path.join(str(output_dir), os.path.basename(fits_file))
        if skip_existing and os.path.exists(output_file):
            continue
        with fits.open(fits_file) as hdul:
            cropped, header = crop_center(hdul[0].data, size, hdul[0].header)
        fits.PrimaryHDU(data=cropped, header=header).writeto(output_file, overwrite=True)
        written.append(output_file)
    logger.info("Cropped %d frames to %d pix into %s", len(written), size, output_dir)
    return written
