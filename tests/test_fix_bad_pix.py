"""Tests of the bad-pixel stage (modules/preprocessing/fix_bad_pix.py); synthetic data only."""
import os

import numpy as np
import pytest
from astropy.io import fits

from modules.preprocessing.fix_bad_pix import FixPixSingle, fix_bad_pixels
from modules.preprocessing.frame_io import read_frame
from modules.preprocessing.pipeline import Pipeline
from modules.shared.dataset import load_dataset_config

SHAPE = (64, 96)


@pytest.fixture
def gradient():
    '''Linear ramp in y and x: reproduced exactly by the mean of a symmetric 3x3 patch.'''
    y, x = np.mgrid[:SHAPE[0], :SHAPE[1]]
    return (100. + 2.5 * y - 1.25 * x).astype(np.float32)


@pytest.fixture
def bad_mask():
    '''Isolated interior bad pixels (no two adjacent), plus one on the frame corner.'''
    rng = np.random.default_rng(1)
    mask = np.zeros(SHAPE, dtype=bool)
    ys = rng.choice(np.arange(2, SHAPE[0] - 2, 3), size=15, replace=False)
    xs = rng.choice(np.arange(2, SHAPE[1] - 2, 3), size=15, replace=False)
    mask[ys, xs] = True
    return mask


def corrupt(frame, mask):
    '''Put junk (hot, cold, NaN) values into the bad pixels.'''
    out = frame.copy()
    junk = np.resize(np.array([1e6, -1e6, np.nan], dtype=np.float32), mask.sum())
    out[mask] = junk
    return out


def test_bad_pixels_interpolated(gradient, bad_mask):
    fixed = fix_bad_pixels(corrupt(gradient, bad_mask), bad_mask)
    np.testing.assert_allclose(fixed[bad_mask], gradient[bad_mask], rtol=0, atol=1e-3)


def test_good_pixels_unchanged(gradient, bad_mask):
    sci = corrupt(gradient, bad_mask)
    fixed = fix_bad_pixels(sci, bad_mask)
    np.testing.assert_array_equal(fixed[~bad_mask], sci[~bad_mask])


def test_no_nans_shape_dtype(gradient, bad_mask):
    sci = corrupt(gradient, bad_mask)
    fixed = fix_bad_pixels(sci, bad_mask)
    assert np.all(np.isfinite(fixed))
    assert fixed.shape == sci.shape
    assert fixed.dtype == sci.dtype
    # input not modified in place
    assert np.isnan(sci[bad_mask]).any()


def test_integer_mask_accepted(gradient, bad_mask):
    '''Masks on disk are 0/1 integers, not booleans.'''
    sci = corrupt(gradient, bad_mask)
    np.testing.assert_array_equal(fix_bad_pixels(sci, bad_mask.astype(np.int16)),
                                  fix_bad_pixels(sci, bad_mask))


def test_fallback_fills_3x3_cluster(gradient):
    '''The center of a 3x3 block of bad pixels has no good 3x3 neighbor; the 5x5 fallback
    fills it (exactly, for a linear field, since the ring of good pixels is symmetric).'''
    mask = np.zeros(SHAPE, dtype=bool)
    mask[20:23, 30:33] = True
    fixed = fix_bad_pixels(corrupt(gradient, mask), mask)
    assert np.all(np.isfinite(fixed))
    np.testing.assert_allclose(fixed[21, 31], gradient[21, 31], rtol=0, atol=1e-3)
    np.testing.assert_array_equal(fixed[~mask], gradient[~mask])


def test_fallback_limit_5x5_cluster_center_stays_nan(gradient):
    '''A 5x5 block of bad pixels: the center has no good pixel even in the 5x5 patch.'''
    mask = np.zeros(SHAPE, dtype=bool)
    mask[20:25, 30:35] = True
    fixed = fix_bad_pixels(corrupt(gradient, mask), mask)
    assert np.isnan(fixed[22, 32])
    assert np.isnan(fixed).sum() == 1


def test_shape_mismatch_raises(gradient):
    with pytest.raises(ValueError):
        fix_bad_pixels(gradient, np.zeros((SHAPE[0] + 1, SHAPE[1]), dtype=bool))


def test_fixpixsingle_end_to_end(tmp_path, gradient, bad_mask):
    mask_file = tmp_path / "bad_pix.fits"
    fits.writeto(mask_file, bad_mask.astype(np.int16))

    read_dir = tmp_path / "in"
    write_dir = tmp_path / "out"
    read_dir.mkdir()
    write_dir.mkdir()
    header = fits.Header()
    header["LBT_PARA"] = 12.5
    sci_file = read_dir / "lm_240131_000123.fits"
    fits.writeto(sci_file, corrupt(gradient, bad_mask), header=header)

    fn = FixPixSingle(write_dir=str(write_dir) + "/", master_bad_pix_file=str(mask_file), config=None)
    fn(str(sci_file))

    fixed, header_out = read_frame(write_dir / sci_file.name)
    np.testing.assert_allclose(fixed[bad_mask], gradient[bad_mask], rtol=0, atol=1e-3)
    np.testing.assert_array_equal(fixed[~bad_mask], gradient[~bad_mask])
    assert header_out["RED_STEP"] == "bad-pixel-fixed"
    assert header_out["LBT_PARA"] == 12.5


@pytest.mark.parametrize("badpix_on", [True, False])
def test_pipeline_ramp_reads_badpix_output(badpix_on):
    '''With the real dataset config, ramp reads badpix's output when badpix is enabled, and
    the dark-subtracted frames when it is not.'''
    config = load_dataset_config("procyon_240131")
    # full default chain, independent of the flags currently set in the .ini
    for s in ("slope", "dark", "ramp", "bkg_channel", "train_regions", "bkg_regions", "fuse",
              "cutout", "register", "crop"):
        config["pipeline"][s] = "True"
    config["pipeline"]["bkg_nod"] = "False"
    config["pipeline"]["START_STAGE"] = "slope"
    config["pipeline"]["START_DIR"] = ""
    config["pipeline"]["badpix"] = str(badpix_on)
    plan = Pipeline(config).plan
    write = config["data_dirs"]["DIR_WRITE"]
    expected = "02_bad_pix_fixed" if badpix_on else "01_dark_subtracted"
    assert plan.in_dir("ramp") == os.path.join(write, expected, "")
