"""
PCA bases of the read-noise background, one per detector region (train_regions stage).

For a region R, the training frames are channel-background-subtracted frames whose star
is in another region and at least PSF_MASK_RADIUS (+ margin) away from R, or frames whose
block has no star on the detector (blank ``quadrant``, e.g. the off-source nod), so R holds
background only. The regions and the star positions come from the dataset config and the
blocks CSV.

By default only science frames (block ``type = sci``) are candidates. With
``restrict_to_sci=False`` (used when the frames are listed explicitly in TRAIN_FRAMES, e.g.
darks) every frame passed in is a candidate, whatever its type or block.

Adapted from modules/preprocessing/region_pca.py of the AB Aur pipeline.
"""

import logging
import os

import numpy as np
from astropy.io import fits

from modules.shared.dataset import block_for_frame, is_sci
from modules.preprocessing.detector import PCA_basis
from modules.preprocessing.star import locate_star, block_hint, block_has_star

# extra distance (pix) between the star and a region, on top of the PSF mask radius,
# to allow for the error of the approximate star position
STAR_MARGIN = 20


def distance_to_region(point, region, shape):
    '''
    Distance (pix) from ``point`` to the nearest pixel of ``region``.

    INPUTS:
    - point (tuple): ``(y, x)``
    - region (tuple of slice): ``(y_slice, x_slice)``
    - shape (tuple of int): Frame shape, to resolve open slice ends

    OUTPUTS:
    - distance (float): 0 if the point is inside the region
    '''
    dists = []
    for p, sl, n in zip(point, region, shape):
        lo, hi, _ = sl.indices(n)
        dists.append(max(lo - p, 0, p - (hi - 1)))
    return float(np.hypot(*dists))


def train_region_pca(files_by_frame, blocks, region_name, region, n_frames, n_pca,
                     psf_radius, out_path, mask, halfbox=50, seed=0, restrict_to_sci=True):
    '''
    Build and write the PCA basis for one region.

    INPUTS:
    - files_by_frame (dict): Map of frame number to channel-background-subtracted frame
    - blocks (DataFrame): Blocks CSV (``load_blocks``)
    - region_name (str): Region name (as in the ``quadrant`` column)
    - region (tuple): ``(y_slice, x_slice)`` of the region on the detector
    - n_frames (int): Number of training frames
    - n_pca (int): Number of PCA modes to keep
    - psf_radius (float): PSF mask radius (pix)
    - out_path (str): Output FITS file (n_pca, ny, nx), NaN outside the region
    - mask (DetectorMask): Mask of the unusable detector areas
    - halfbox (int): Half-width of the star search box around the blocks-CSV position
    - seed (int): Random seed for choosing the training frames
    - restrict_to_sci (bool): True: only frames in ``type = sci`` blocks whose star is not in
      this region. False: every frame in ``files_by_frame`` (frames in blocks with a star are
      still skipped if the star is too close to the region)

    OUTPUTS:
    - used (list of str): Training frames used
    '''
    candidates = []
    for n, path in sorted(files_by_frame.items()):
        b = block_for_frame(blocks, n)
        if restrict_to_sci and not is_sci(b):
            continue
        if b is not None and b["quadrant"] == region_name:
            continue  # the star is in this region
        candidates.append((path, block_hint(b), block_has_star(b)))
    if not candidates:
        raise RuntimeError(f"No training frames for region {region_name}")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(candidates))

    shape = None
    cube, used, finite_ref = [], [], None
    n_skipped_star = n_skipped_nan = 0
    for i in order:
        if len(cube) >= n_frames:
            break
        path, hint, has_star = candidates[i]
        if has_star and hint is not None and shape is not None and \
                distance_to_region(hint, region, shape) <= psf_radius + STAR_MARGIN:
            # skip without reading the frame
            n_skipped_star += 1
            continue
        with fits.open(path) as hdul:
            frame = np.asarray(hdul[0].data, dtype=np.float64)
        shape = frame.shape
        if has_star:
            star = locate_star(frame, hint, halfbox)
            if distance_to_region(star, region, shape) <= psf_radius + STAR_MARGIN:
                n_skipped_star += 1
                continue
        crop = mask(frame)[region]
        finite = np.isfinite(crop)
        if finite_ref is None:
            finite_ref = finite
        elif not np.array_equal(finite, finite_ref):
            # PCA_basis needs the same finite pixels in every frame
            n_skipped_nan += 1
            continue
        cube.append(crop.astype(np.float32))
        used.append(path)
        if len(cube) % 100 == 0:
            logging.info(f"Region {region_name}: {len(cube)} training frames read")

    logging.info(f"Region {region_name}: {len(cube)} training frames "
          f"({n_skipped_star} skipped for star too close, {n_skipped_nan} for a different NaN pattern)")
    if len(cube) < n_pca:
        raise RuntimeError(f"Region {region_name}: only {len(cube)} training frames for {n_pca} PCA modes")

    basis_crop = PCA_basis(np.array(cube), n_PCA=n_pca)
    del cube

    basis = np.full((n_pca,) + shape, np.nan, dtype=np.float32)
    basis[(slice(None),) + tuple(region)] = basis_crop

    header = fits.Header()
    header["REGION"] = (region_name, "detector region of this basis")
    ys, xs = region
    header["REGY0"], header["REGY1"], _ = ys.indices(shape[0])
    header["REGX0"], header["REGX1"], _ = xs.indices(shape[1])
    header["NTRAIN"] = (len(used), "number of training frames")
    header["PSFMASKR"] = (float(psf_radius), "PSF mask radius (pix)")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fits.writeto(out_path, basis, header=header, overwrite=True)
    with open(out_path + ".frames.txt", "w") as f:
        f.write("\n".join(os.path.basename(p) for p in used) + "\n")
    logging.info(f"Wrote region PCA basis {out_path} (list of training frames in {out_path}.frames.txt)")
    return used
