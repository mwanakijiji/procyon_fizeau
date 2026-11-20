"""
Registration stage: put the star at ((N-1)/2, (N-1)/2) in every cutout, block by block.

For each block of the blocks CSV:
1. fit a 2D Gaussian near the center of each frame and compute the shift that puts its
   centroid at ((N-1)/2, (N-1)/2);
2. take the median of (up to REG_REF_MAX_FRAMES of) the shifted frames as the block's reference;
3. measure the residual shift of each shifted frame against the reference with
   ``skimage.registration.phase_cross_correlation`` (upsampled, unnormalized) on a central
   box, and add it to the frame's shift; repeat 2-3 REG_N_ITER times, then shift the whole
   block so the Gaussian center of its median is at ((N-1)/2, (N-1)/2);
4. shift each original frame once, by its total shift (cubic spline);
5. restore NaNs by shifting the NaN mask by the same amount.

No hand-picked reference frames, and no empirical 1-pixel correction.
"""

import logging
import os
import warnings

import numpy as np
from astropy.modeling import models, fitting
from scipy import ndimage
from skimage.registration import phase_cross_correlation

from modules.preprocessing.frame_io import read_frame, write_frame


def gaussian_centroid(image, guess_yx, halfbox=15):
    '''
    Centroid of the star from a 2D Gaussian (plus constant) fit in a box around ``guess_yx``.

    INPUTS:
    - image (numpy array): 2D frame (NaNs allowed; they are left out of the fit)
    - guess_yx (tuple): Starting ``(y, x)``
    - halfbox (int): Half-width of the fitting box

    OUTPUTS:
    - (y, x) (tuple of float): Centroid in image coordinates, or ``None`` if the fit fails
      or ends outside the box
    '''
    gy, gx = int(round(guess_yx[0])), int(round(guess_yx[1]))
    y0, x0 = max(gy - halfbox, 0), max(gx - halfbox, 0)
    y1, x1 = min(gy + halfbox + 1, image.shape[0]), min(gx + halfbox + 1, image.shape[1])
    box = image[y0:y1, x0:x1]
    yy, xx = np.mgrid[y0:y1, x0:x1]
    good = np.isfinite(box)
    if good.sum() < 10:
        return None
    z = box[good]
    floor = np.median(z)
    peak = np.unravel_index(np.argmax(np.where(good, box, -np.inf)), box.shape)
    model = models.Gaussian2D(amplitude=z.max() - floor, x_mean=x0 + peak[1], y_mean=y0 + peak[0],
                              x_stddev=4., y_stddev=4.) + models.Const2D(amplitude=floor)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            fit = fitting.LevMarLSQFitter()(model, xx[good], yy[good], z)
        except Exception:
            return None
    cy, cx = fit.y_mean_0.value, fit.x_mean_0.value
    if not np.isfinite(cy + cx) or not (y0 <= cy < y1 and x0 <= cx < x1):
        return None
    return float(cy), float(cx)


def shift_frame(frame, shift):
    '''
    Shift a frame by ``(dy, dx)`` with a cubic spline, keeping NaNs.

    NaNs are set to zero for the interpolation, and every output pixel that draws on a NaN
    pixel or on a pixel beyond the edge is NaN again (the NaN mask is shifted along both
    axes with linear interpolation, so a NaN region grows by up to one pixel).

    INPUTS:
    - frame (numpy array): 2D frame
    - shift (tuple): ``(dy, dx)`` in pixels

    OUTPUTS:
    - shifted (numpy array): float32 shifted frame
    '''
    nan = ~np.isfinite(frame)
    filled = np.where(nan, 0., frame).astype(np.float64)
    shifted = ndimage.shift(filled, shift, order=3, mode="constant", cval=0.)
    nan_shifted = ndimage.shift(nan.astype(np.float64), shift, order=1, mode="constant", cval=1.) > 1e-3
    shifted[nan_shifted] = np.nan
    return shifted.astype(np.float32)


def central_box(frame, side):
    '''
    Central ``side`` x ``side`` box of an odd-sized frame.

    INPUTS:
    - frame (numpy array): Square frame, odd side
    - side (int): Box side (odd)

    OUTPUTS:
    - box (numpy array): The central box (a view)
    '''
    c = (frame.shape[0] - 1) // 2
    h = (side - 1) // 2
    return frame[c - h:c + h + 1, c - h:c + h + 1]


def residual_shift(reference_box, moving_box, upsample=20):
    '''
    Shift ``(dy, dx)`` that registers ``moving_box`` onto ``reference_box``
    (``ndimage.shift(moving, (dy, dx))`` lines up with the reference). NaNs count as zero.

    Uses plain (unnormalized) upsampled cross-correlation: the default phase normalization
    whitens the spectrum, so for a smooth PSF it is dominated by noise at high spatial
    frequencies (on synthetic noisy stars it was off by up to ~18 pix, vs. < 0.05 pix here).

    INPUTS:
    - reference_box (numpy array): Reference image
    - moving_box (numpy array): Image to register (same shape)
    - upsample (int): Upsampling factor (shift precision 1/upsample pix)

    OUTPUTS:
    - shift (numpy array): ``(dy, dx)`` (pix)
    '''
    shift, _, _ = phase_cross_correlation(np.nan_to_num(reference_box), np.nan_to_num(moving_box),
                                          upsample_factor=upsample, normalization=None)
    return np.asarray(shift, dtype=float)


def _median(stack):
    '''
    NaN-ignoring median of a list of images (all-NaN warnings suppressed).

    INPUTS:
    - stack (list of numpy array): Images of the same shape

    OUTPUTS:
    - median (numpy array): Pixel-wise median
    '''
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(np.stack(stack), axis=0)


def register_block(files, write_dir, box=101, upsample=20, centroid_halfbox=15, ref_max_frames=300,
                   n_iter=3):
    '''
    Register all cutouts of one block (see the module docstring).

    Steps 2-3 are iterated ``n_iter`` times: each time the reference is rebuilt from the
    frames as currently aligned, so it sharpens and the alignment converges. Then the whole
    block is shifted so that the Gaussian center of the final median is at ((N-1)/2, (N-1)/2).
    The iterations only shift a central crop (box + 40 pix); each full frame is interpolated
    once, when it is written.

    INPUTS:
    - files (list of str): Cutouts of the block (all N x N, N odd)
    - write_dir (str): Output directory
    - box (int): Side of the central box for phase cross-correlation
    - upsample (int): Upsampling factor for phase cross-correlation
    - centroid_halfbox (int): Half-width of the Gaussian-fit box
    - ref_max_frames (int): Maximum number of frames in the median reference
    - n_iter (int): Number of reference/alignment iterations

    OUTPUTS:
    - summary (dict): Numbers of frames written and failed, the median shift correction (pix)
      in the last iteration, and the block-wide recentering shift (pix)
    '''
    if not files:
        return {"n_written": 0, "n_failed": 0}
    n_side = read_frame(files[0])[0].shape[0]
    c = (n_side - 1) / 2.
    crop_side = min(box + 40, n_side)

    # 1. centroid shifts; keep the central crops in memory for the iterations
    shifts, crops = {}, {}
    n_failed = 0
    for f in files:
        frame, _ = read_frame(f)
        cen = gaussian_centroid(frame, (c, c), centroid_halfbox)
        if cen is None:
            logging.warning("Gaussian fit failed; skipping %s", f)
            n_failed += 1
            continue
        shifts[f] = np.array([c - cen[0], c - cen[1]])
        crops[f] = central_box(frame, crop_side)
    good = [f for f in files if f in shifts]
    if not good:
        return {"n_written": 0, "n_failed": n_failed}
    pick = [good[i] for i in np.unique(np.linspace(0, len(good) - 1, min(ref_max_frames, len(good))).astype(int))]

    # 2.-3. iterate: median reference of the aligned frames, then align each frame to it
    corrections = [0.]
    for _ in range(n_iter):
        ref_box = central_box(_median([shift_frame(crops[f], shifts[f]) for f in pick]), box)
        corrections = []
        for f in good:
            d = residual_shift(ref_box, central_box(shift_frame(crops[f], shifts[f]), box), upsample)
            shifts[f] = shifts[f] + d
            corrections.append(np.hypot(*d))

    # recenter the block: Gaussian center of the final median at (c, c)
    final = _median([shift_frame(crops[f], shifts[f]) for f in pick])
    cc = (crop_side - 1) / 2.
    cen = gaussian_centroid(final, (cc, cc), centroid_halfbox)
    recenter = np.zeros(2) if cen is None else np.array([cc - cen[0], cc - cen[1]])
    del crops

    # 4.-5. one combined shift per frame, write
    for f in good:
        frame, header = read_frame(f)
        total = shifts[f] + recenter
        write_frame(os.path.join(write_dir, os.path.basename(f)), shift_frame(frame, total), header,
                    "registered", extra={
                        "REGDX": (float(total[1]), "x shift applied in registration (pix)"),
                        "REGDY": (float(total[0]), "y shift applied in registration (pix)"),
                        "STARX": (c, "star x in this frame (0-indexed)"),
                        "STARY": (c, "star y in this frame (0-indexed)"),
                    })
    return {"n_written": len(good), "n_failed": n_failed,
            "median_last_correction_pix": float(np.median(corrections)),
            "recenter_pix": float(np.hypot(*recenter))}
