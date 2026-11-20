"""
Finding the star on a detector frame and masking it.
"""

import numpy as np
from scipy import ndimage


def find_airy_psf(image):
    '''
    Find the star as the maximum of the Gaussian-smoothed (sigma 6 pix) image. NaNs in the
    input are set to 0 in place.

    INPUTS:
    - image (numpy array): 2D image; its NaNs are replaced by 0

    OUTPUTS:
    - center (list of int): ``[cy, cx]`` of the PSF
    '''

    # replace NaNs with zeros to get the Gaussian filter to work
    image[~np.isfinite(image)] = 0

    # Gaussian filter for further removing effect of bad pixels
    imageG = ndimage.gaussian_filter(image, 6)
    loc = np.argwhere(imageG == np.max(imageG))
    cx = loc[0, 1]
    cy = loc[0, 0]

    return [cy, cx]


def locate_star(frame, hint=None, halfbox=50):
    '''
    Integer pixel position of the star, from the peak of the Gaussian-smoothed frame.

    If ``hint`` is given, only a box of half-width ``halfbox`` around it is searched;
    otherwise the whole frame is. The input frame is not modified.

    INPUTS:
    - frame (numpy array): 2D frame (NaNs allowed)
    - hint (tuple or None): Approximate ``(y, x)`` of the star
    - halfbox (int): Half-width of the search box around ``hint``

    OUTPUTS:
    - (cy, cx) (tuple of int): Star position in frame coordinates
    '''
    img = np.array(frame, dtype=np.float64, copy=True)
    if hint is None:
        cy, cx = find_airy_psf(img)
        return int(cy), int(cx)
    hy, hx = int(hint[0]), int(hint[1])
    y0, x0 = max(hy - halfbox, 0), max(hx - halfbox, 0)
    y1, x1 = min(hy + halfbox + 1, img.shape[0]), min(hx + halfbox + 1, img.shape[1])
    cy, cx = find_airy_psf(img[y0:y1, x0:x1])
    return int(cy) + y0, int(cx) + x0


def block_hint(block):
    '''
    Star position from a blocks-CSV row.

    INPUTS:
    - block (Series or None): Row from ``block_for_frame``

    OUTPUTS:
    - hint (tuple of int or None): ``(star_y, star_x)``, or None if the row is missing or blank
    '''
    if block is None or block.isna()["star_y"] or block.isna()["star_x"]:
        return None
    return int(block["star_y"]), int(block["star_x"])


def block_has_star(block):
    '''
    Whether the star is on the detector in this block (its ``quadrant`` is not blank;
    blank means e.g. an off-source nod).

    INPUTS:
    - block (Series or None): Row from ``block_for_frame``

    OUTPUTS:
    - has_star (bool): True if the block names the region holding the star
    '''
    return block is not None and block["quadrant"] != ""


def star_nan_mask(shape, center, radius):
    '''
    Multiplicative mask: 1 everywhere, NaN within ``radius`` of ``center``.

    INPUTS:
    - shape (tuple): Frame shape
    - center (tuple): ``(y, x)`` of the star
    - radius (float): Mask radius in pixels

    OUTPUTS:
    - mask (numpy array): float32 array of ones and NaNs
    '''
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    inside = (yy - center[0]) ** 2 + (xx - center[1]) ** 2 <= radius ** 2
    mask = np.ones(shape, dtype=np.float32)
    mask[inside] = np.nan
    return mask
