"""
Cutout stage: odd-sized square cutouts centered on the star, NaN-padded off the detector.

Center convention used from here on: in an N x N frame (N odd), the star is at pixel
((N-1)/2, (N-1)/2), 0-indexed. This is written to the header as STARX, STARY.
"""

import logging
import os

import numpy as np

from modules.shared.dataset import block_for_frame, frame_number
from modules.preprocessing.frame_io import read_frame, write_frame
from modules.preprocessing.star import locate_star, block_hint


def make_cutout(frame, center_yx, size):
    '''
    Cut a ``size`` x ``size`` square centered on pixel ``center_yx``.

    Returns ``frame[cy-h : cy+h+1, cx-h : cx+h+1]`` with ``h = (size-1)//2``; rows and
    columns beyond the frame edges are NaN.

    INPUTS:
    - frame (numpy array): 2D frame
    - center_yx (tuple of int): Center pixel ``(cy, cx)``
    - size (int): Cutout side (odd)

    OUTPUTS:
    - cutout (numpy array): float32 cutout; the center pixel is at ``(h, h)``
    - origin (tuple of int): ``(y0, x0)``, frame coordinates of cutout pixel (0, 0)
    '''
    if size % 2 != 1:
        raise ValueError(f"Cutout size must be odd, got {size}")
    h = (size - 1) // 2
    cy, cx = int(center_yx[0]), int(center_yx[1])
    y0, x0 = cy - h, cx - h
    ny, nx = frame.shape
    cutout = np.full((size, size), np.nan, dtype=np.float32)
    fy0, fy1 = max(y0, 0), min(y0 + size, ny)
    fx0, fx1 = max(x0, 0), min(x0 + size, nx)
    if fy1 > fy0 and fx1 > fx0:
        cutout[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0] = frame[fy0:fy1, fx0:fx1]
    return cutout, (y0, x0)


class CutoutSingle:
    '''
    Cut out one frame around its star.

    The star is found with ``find_airy_psf`` in a box around the block's ``star_y, star_x``
    from the blocks CSV, or on the whole frame if those are blank.
    '''

    def __init__(self, write_dir, blocks, prefix, size, halfbox=50):
        '''
        INPUTS:
        - write_dir (str): Directory for the cutouts
        - blocks (DataFrame): Blocks table (``load_blocks``), for the star-position hints
        - prefix (str): Filename prefix before the frame number (e.g. ``lm_190108_``)
        - size (int): Cutout side (odd)
        - halfbox (int): Half-width of the star search box around the block's hint

        OUTPUTS:
        - None
        '''
        self.write_dir = write_dir
        self.blocks = blocks
        self.prefix = prefix
        self.size = size
        self.halfbox = halfbox

    def __call__(self, abs_sci_name):
        '''
        Locate the star in one frame, cut a ``size`` x ``size`` cutout centered on it (NaN off the
        detector), and write it with CUTSIZE, STARX, STARY ((N-1)/2), CUTX0 and CUTY0 in the header.

        INPUTS:
        - abs_sci_name (str): Frame path

        OUTPUTS:
        - None
        '''
        base = os.path.basename(abs_sci_name)
        hint = block_hint(block_for_frame(self.blocks, frame_number(base, self.prefix)))
        frame, header = read_frame(abs_sci_name)
        cy, cx = locate_star(frame, hint, self.halfbox)
        cutout, (y0, x0) = make_cutout(frame, (cy, cx), self.size)
        c = (self.size - 1) / 2.
        write_frame(os.path.join(self.write_dir, base), cutout, header, "cutout", extra={
            "CUTSIZE": (self.size, "cutout side (pix)"),
            "STARX": (c, "star x in this frame (0-indexed)"),
            "STARY": (c, "star y in this frame (0-indexed)"),
            "CUTX0": (x0, "detector x of cutout pixel 0 (0-indexed)"),
            "CUTY0": (y0, "detector y of cutout pixel 0 (0-indexed)"),
        })
        logging.info(f"Wrote {self.size}x{self.size} cutout {base}, star at detector (y, x) = ({cy}, {cx})")
