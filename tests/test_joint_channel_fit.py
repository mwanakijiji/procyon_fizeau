"""Joint fit of channel pedestals and region PCA modes (bkg_regions, JOINT_CHANNEL_FIT).
Synthetic data only."""
import numpy as np
from astropy.io import fits

from modules.preprocessing.background_pca import BackgroundPCASubtSingle
from modules.preprocessing.detector import DetectorMask, region_channel_modes
from modules.preprocessing.frame_io import read_frame
from modules.preprocessing.star import star_nan_mask

SHAPE = (48, 96)
N_CH, CH_W = 6, 16
STAR = (24, 40)
PSF_R = 10


def test_region_channel_modes_keeps_overlapping_channels_only():
    region = (slice(0, 48), slice(20, 50))       # channels 1, 2, 3
    modes = region_channel_modes(SHAPE, region, N_CH, CH_W)
    assert modes.shape == (3, *SHAPE)
    assert np.isnan(modes[:, :, :20]).all() and np.isnan(modes[:, :, 50:]).all()
    np.testing.assert_array_equal(modes[0, :, 20:32], 1)
    np.testing.assert_array_equal(modes[0, :, 32:50], 0)


def zero_mean_per_channel(img):
    out = img.copy()
    for k in range(N_CH):
        out[:, k * CH_W:(k + 1) * CH_W] -= out[:, k * CH_W:(k + 1) * CH_W].mean()
    return out


def channel_offsets(img):
    '''Median of each channel outside the PSF mask.'''
    m = img * star_nan_mask(SHAPE, STAR, PSF_R)
    return np.array([np.nanmedian(m[:, k * CH_W:(k + 1) * CH_W]) for k in range(N_CH)])


def run(tmp_path, joint):
    rng = np.random.default_rng(3)
    # read-noise-like modes, zero mean in every channel (as if trained on channel-subtracted darks)
    modes = np.stack([zero_mean_per_channel(rng.normal(size=SHAPE)) for _ in range(3)]).astype(np.float32)
    cube = tmp_path / "region_cube.fits"
    fits.writeto(cube, modes, overwrite=True)

    y, x = np.mgrid[:SHAPE[0], :SHAPE[1]]
    star = 1e4 * np.exp(-((y - STAR[0]) ** 2 + (x - STAR[1]) ** 2) / (2 * 2.0 ** 2))
    offsets = np.repeat(rng.normal(scale=20, size=N_CH), CH_W)[None, :]
    sci = star + 5 * modes[0] - 3 * modes[1] + offsets + rng.normal(scale=0.1, size=SHAPE)
    sci_path = tmp_path / "lm_240131_000001.fits"
    fits.writeto(sci_path, sci.astype(np.float32), overwrite=True)

    out_dir = tmp_path / ("joint" if joint else "alone")
    out_dir.mkdir()
    region = (slice(None), slice(None))
    extra = region_channel_modes(SHAPE, region, N_CH, CH_W) if joint else None
    fn = BackgroundPCASubtSingle(3, pca_cube_file=str(cube), write_dir=str(out_dir) + "/",
                                 mask=DetectorMask(0), config_data=None, psf_mask_radius=PSF_R,
                                 extra_modes=extra)
    fn(str(sci_path))
    return read_frame(out_dir / sci_path.name)


def test_joint_fit_removes_channel_offsets(tmp_path):
    out_joint, hdr = run(tmp_path, joint=True)
    out_alone, _ = run(tmp_path, joint=False)
    assert hdr["PCABASIS"] == f"region_cube+{N_CH}extra"
    resid_joint = np.abs(channel_offsets(out_joint)).max()
    resid_alone = np.abs(channel_offsets(out_alone)).max()
    assert resid_joint < 0.1
    assert resid_alone > 10 * resid_joint
