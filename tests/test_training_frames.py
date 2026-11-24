"""Tests of the blocks ``type`` column, frame selection by type, and region-PCA training on
explicitly listed frames (TRAIN_FRAMES, e.g. darks). Synthetic data only."""
import os

import numpy as np
import pytest
from astropy.io import fits

from modules.preprocessing.detector import DetectorMask
from modules.preprocessing.pipeline import Pipeline
from modules.preprocessing.region_pca import train_region_pca
from modules.preprocessing.stage_plan import STAGES
from modules.shared.dataset import (
    blocks_csv_path, load_blocks, load_dataset_config, sci_frame_numbers,
)

PREFIX = "lm_240131_"
HEADER = "frame_start,frame_stop,target,filter,quadrant,star_y,star_x,type\n"


def write_csv(tmp_path, rows):
    path = tmp_path / "blocks.csv"
    path.write_text(HEADER + "\n".join(rows) + "\n")
    return path


def touch(directory, nums):
    os.makedirs(directory, exist_ok=True)
    for n in nums:
        open(os.path.join(directory, f"{PREFIX}{n:06d}.fits"), "w").close()


def config_with(tmp_path, train_frames=""):
    '''Real dataset config, full default chain, DIR_WRITE in tmp_path.'''
    config = load_dataset_config("procyon_240131")
    for s in STAGES:
        config["pipeline"][s] = str(s != "bkg_nod")
    config["pipeline"]["START_STAGE"] = "slope"
    config["pipeline"]["START_DIR"] = ""
    config["pipeline"]["CLOBBER"] = "False"
    config["data_dirs"]["DIR_WRITE"] = str(tmp_path)
    config["preprocess"]["TRAIN_FRAMES"] = train_frames
    return config


# ---------- blocks CSV ----------

def test_load_blocks_type_column(tmp_path):
    csv = write_csv(tmp_path, ["1,10,Star,F1,A,5,5,sci",
                               "11,20,dark,Blank,,,,Dark",
                               "21,25,flat,F1,,,, cal "])
    blocks = load_blocks(csv)
    assert list(blocks["type"]) == ["sci", "dark", "cal"]
    assert sci_frame_numbers(blocks) == set(range(1, 11))


def test_load_blocks_rejects_unknown_type(tmp_path):
    csv = write_csv(tmp_path, ["1,10,Star,F1,A,5,5,True"])
    with pytest.raises(ValueError, match="unknown block type"):
        load_blocks(csv)


def test_sci_block_needs_target_and_filter(tmp_path):
    csv = write_csv(tmp_path, ["1,10,,F1,A,5,5,sci"])
    with pytest.raises(ValueError, match="type=sci"):
        load_blocks(csv)
    # a dark block may have no target
    load_blocks(write_csv(tmp_path, ["1,10,,Blank,,,,dark"]))


def test_procyon_blocks_darks_are_not_sci():
    blocks = load_blocks(blocks_csv_path(load_dataset_config("procyon_240131")))
    sci = sci_frame_numbers(blocks)
    assert 11800 in sci and 11801 not in sci and 12800 not in sci
    assert set(blocks.loc[blocks["frame_start"] == 11801, "type"]) == {"dark"}


# ---------- frame selection ----------

def test_files_include_every_type_but_star_only_drops_darks(tmp_path):
    pipe = Pipeline(config_with(tmp_path), ncpu=1)
    d = tmp_path / "in"
    touch(d, [600, 10600, 11900, 20000])   # sci+star, sci off-source, dark, in no block
    names = [os.path.basename(p) for p in pipe.files(str(d))]
    assert names == [f"{PREFIX}{n:06d}.fits" for n in (600, 10600, 11900, 20000)]
    starry = [os.path.basename(p) for p in pipe.files(str(d), star_only=True)]
    assert starry == [f"{PREFIX}000600.fits"]


def test_training_files_are_exactly_train_frames(tmp_path):
    pipe = Pipeline(config_with(tmp_path, "11801-11805, 11810"), ncpu=1)
    read_dir = pipe.plan.in_dir("train_regions")
    assert read_dir == pipe.plan.out_dir("bkg_channel")
    touch(read_dir, [600, 10600, 11801, 11802, 11803, 11810, 11811])
    assert sorted(pipe.training_files()) == [11801, 11802, 11803, 11810]


def test_training_files_without_train_frames_lists_everything(tmp_path):
    pipe = Pipeline(config_with(tmp_path), ncpu=1)
    read_dir = pipe.plan.in_dir("train_regions")
    touch(read_dir, [600, 10600, 11900])
    assert sorted(pipe.training_files()) == [600, 10600, 11900]


# ---------- train_region_pca on non-science frames ----------

@pytest.fixture
def dark_set(tmp_path):
    '''8 small noise frames in a dark block, plus 8 starless science frames.'''
    blocks = load_blocks(write_csv(tmp_path, ["1,8,Star,F1,,,,sci", "101,108,dark,Blank,,,,dark"]))
    rng = np.random.default_rng(0)
    files = {}
    for n in list(range(1, 9)) + list(range(101, 109)):
        path = tmp_path / f"{PREFIX}{n:06d}.fits"
        fits.writeto(path, rng.normal(size=(12, 16)).astype(np.float32))
        files[n] = str(path)
    return files, blocks


@pytest.mark.parametrize("restrict, expected", [(False, range(101, 109)), (True, range(1, 9))])
def test_train_region_pca_restrict_to_sci(tmp_path, dark_set, restrict, expected):
    files, blocks = dark_set
    darks = {n: p for n, p in files.items() if n > 100}
    region = (slice(0, 12), slice(0, 16))
    out = tmp_path / "calib" / "cube_A.fits"
    pool = files if restrict else darks
    used = train_region_pca(pool, blocks, "A", region, n_frames=50, n_pca=3, psf_radius=5,
                            out_path=str(out), mask=DetectorMask(0), restrict_to_sci=restrict)
    assert sorted(os.path.basename(p) for p in used) == [f"{PREFIX}{n:06d}.fits" for n in expected]
    basis = fits.getdata(out)
    assert basis.shape == (3, 12, 16)
    assert fits.getheader(out)["NTRAIN"] == 8


def test_train_region_pca_default_skips_darks(tmp_path, dark_set):
    files, blocks = dark_set
    darks = {n: p for n, p in files.items() if n > 100}
    with pytest.raises(RuntimeError, match="No training frames"):
        train_region_pca(darks, blocks, "A", (slice(None), slice(None)), n_frames=50, n_pca=3,
                         psf_radius=5, out_path=str(tmp_path / "x.fits"), mask=DetectorMask(0))


# ---------- FRAME_SUBSET ----------

@pytest.mark.parametrize("value, expected", [
    ("all", None), (" ALL ", None), ("", None),
    ("11801-12800", [(11801, 12800)]), ("5, 10-12", [(5, 5), (10, 12)]),
])
def test_frame_subset(value, expected):
    from modules.shared.dataset import frame_subset
    config = load_dataset_config("procyon_240131")
    config["dataset"]["FRAME_SUBSET"] = value
    assert frame_subset(config) == expected
