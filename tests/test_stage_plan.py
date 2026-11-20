"""Tests of the stage plan (modules/preprocessing/stage_plan.py): which stages run, the
dynamic directory numbering, START_STAGE / START_DIR, and CLOBBER. No data needed."""
import os

import pytest

from modules.preprocessing.pipeline import Pipeline
from modules.preprocessing.stage_plan import STAGES, StagePlan
from modules.shared.dataset import load_dataset_config

STEMS = {
    "slope": "slope_frames", "dark": "dark_subtracted", "badpix": "bad_pix_fixed",
    "ramp": "ramp_removed", "bkg_nod": "nod_subted", "bkg_channel": "pcab_subted",
    "bkg_regions": "pcab_subted_higher_order_region_{region}", "fuse": "regions_recombined",
    "cutout": "cutouts_{size}", "register": "registered_{size}",
}
DEFAULT_FLAGS = {s: s != "bkg_nod" for s in STAGES}
WRITE = "/w"
RAW = "/raw/"


def make_plan(start_stage="slope", start_dir=None, clobber=False, dir_write=WRITE, **flag_overrides):
    flags = dict(DEFAULT_FLAGS, **flag_overrides)
    return StagePlan(flags, STEMS, dir_write=dir_write, dir_raw=RAW, start_stage=start_stage,
                     start_dir=start_dir, clobber=clobber, regions=["A", "B"], sizes=[257, 129])


def w(name):
    return os.path.join(WRITE, name, "")


def dataset_config():
    '''The real dataset config, with [pipeline] reset to the default full chain (so the tests do
    not depend on the flags currently set in the .ini).'''
    config = load_dataset_config("procyon_240131")
    for s in STAGES:
        config["pipeline"][s] = str(DEFAULT_FLAGS[s])
    config["pipeline"]["START_STAGE"] = "slope"
    config["pipeline"]["START_DIR"] = ""
    config["pipeline"]["CLOBBER"] = "False"
    return config


# ---------- numbering and wiring ----------

def test_default_numbering_and_inputs():
    p = make_plan()
    assert p.out_dir("slope") == w("00_slope_frames")
    assert p.out_dir("dark") == w("01_dark_subtracted")
    assert p.out_dir("badpix") == w("02_bad_pix_fixed")
    assert p.out_dir("ramp") == w("03_ramp_removed")
    assert p.out_dir("bkg_channel") == w("04_pcab_subted")
    assert p.out_dir("bkg_regions", region="A") == w("05_pcab_subted_higher_order_region_A")
    assert p.out_dir("fuse") == w("06_regions_recombined")
    assert p.out_dir("cutout", size=257) == w("07_cutouts_257")
    assert p.out_dir("register", size=257) == w("08_registered_257")
    assert p.out_dir("crop", size=129) == w("08_registered_129")

    assert p.in_dir("slope") == RAW
    assert p.in_dir("dark") == w("00_slope_frames")
    assert p.in_dir("ramp") == w("02_bad_pix_fixed")
    # train_regions and bkg_regions both read bkg_channel's output
    assert p.in_dir("train_regions") == w("04_pcab_subted")
    assert p.in_dir("bkg_regions") == w("04_pcab_subted")
    assert p.in_dir("fuse", region="B") == w("05_pcab_subted_higher_order_region_B")
    assert p.in_dir("crop", size=257) == w("08_registered_257")
    assert p.to_run == [s for s in STAGES if DEFAULT_FLAGS[s]]


def test_disabled_stage_renumbers_and_rewires():
    p = make_plan(dark=False)
    assert p.out_dir("badpix") == w("01_bad_pix_fixed")
    assert p.in_dir("badpix") == w("00_slope_frames")
    assert p.out_dir("ramp") == w("02_ramp_removed")
    assert "dark" not in p.to_run
    with pytest.raises(ValueError):
        p.out_dir("dark")


def test_nod_stage_inserted():
    p = make_plan(bkg_nod=True)
    assert p.out_dir("bkg_nod") == w("04_nod_subted")
    assert p.in_dir("bkg_channel") == w("04_nod_subted")
    assert p.out_dir("bkg_channel") == w("05_pcab_subted")


def test_without_region_stages_cutout_reads_bkg_channel():
    p = make_plan(train_regions=False, bkg_regions=False, fuse=False)
    assert p.in_dir("cutout", size=257) == w("04_pcab_subted")
    assert p.out_dir("cutout", size=257) == w("05_cutouts_257")


# ---------- START_STAGE / START_DIR ----------

def test_start_stage_reads_previous_output():
    p = make_plan(start_stage="bkg_channel")
    assert p.to_run[0] == "bkg_channel"
    assert "ramp" not in p.to_run
    assert p.in_dir("bkg_channel") == w("03_ramp_removed")
    # numbering is the same as for a full run, so a previous full run's output is found
    assert p.out_dir("bkg_channel") == w("04_pcab_subted")


def test_start_dir_overrides_only_the_start_input(tmp_path):
    p = make_plan(start_stage="dark", start_dir=str(tmp_path))
    assert p.in_dir("dark") == str(tmp_path)
    assert p.in_dir("badpix") == w("01_dark_subtracted")


def test_start_dir_feeds_bkg_regions_when_starting_at_train_regions(tmp_path):
    '''bkg_regions reads the same (pre-START_STAGE) frames as train_regions.'''
    p = make_plan(start_stage="train_regions", start_dir=str(tmp_path))
    assert p.in_dir("train_regions") == str(tmp_path)
    assert p.in_dir("bkg_regions") == str(tmp_path)


def test_start_dir_with_region_template(tmp_path):
    for r in ("A", "B"):
        (tmp_path / f"reg_{r}").mkdir()
    p = make_plan(start_stage="fuse", start_dir=str(tmp_path / "reg_{region}"))
    assert p.in_dir("fuse", region="B") == str(tmp_path / "reg_B")


def test_first_stage_without_slope_needs_start_dir(tmp_path):
    with pytest.raises(ValueError, match="START_DIR"):
        make_plan(start_stage="dark", slope=False)
    p = make_plan(start_stage="dark", slope=False, start_dir=str(tmp_path))
    assert p.out_dir("dark") == w("00_dark_subtracted")


# ---------- validation ----------

@pytest.mark.parametrize("kwargs, match", [
    (dict(start_stage="nonsense"), "not a stage"),
    (dict(start_stage="bkg_nod"), "not enabled"),
    (dict(start_dir="/does/not/exist"), "does not exist"),
    (dict(cutout=False), "needs stage 'cutout'"),
    (dict(register=False), "needs stage 'register'"),
    (dict(bkg_regions=False), "needs stage 'bkg_regions'"),
    (dict(fuse=False), "must be followed by 'fuse'"),
])
def test_validation_errors(kwargs, match):
    with pytest.raises(ValueError, match=match):
        make_plan(**kwargs)


def test_flags_must_cover_every_stage():
    flags = dict(DEFAULT_FLAGS)
    del flags["dark"]
    with pytest.raises(ValueError, match="missing"):
        StagePlan(flags, STEMS, WRITE, RAW, "slope")
    with pytest.raises(ValueError, match="unknown"):
        StagePlan(dict(DEFAULT_FLAGS, darkk=True), STEMS, WRITE, RAW, "slope")


def test_missing_stem():
    stems = dict(STEMS)
    del stems["ramp"]
    with pytest.raises(ValueError, match="stage_dirs"):
        StagePlan(DEFAULT_FLAGS, stems, WRITE, RAW, "slope")


# ---------- CLOBBER ----------

def test_clobber_dirs_cover_exactly_the_stages_that_run():
    p = make_plan(start_stage="bkg_channel")
    assert p.clobber_dirs() == [
        w("04_pcab_subted"),
        w("05_pcab_subted_higher_order_region_A"), w("05_pcab_subted_higher_order_region_B"),
        w("06_regions_recombined"), w("07_cutouts_257"), w("08_registered_257"), w("08_registered_129"),
    ]


def test_clobber_refuses_to_delete_start_dir(tmp_path):
    '''START_DIR pointing at the output directory of a stage that runs.'''
    out = tmp_path / "04_pcab_subted"
    out.mkdir()
    p = make_plan(start_stage="bkg_channel", start_dir=str(out), dir_write=str(tmp_path), clobber=True)
    with pytest.raises(ValueError, match="input of this run"):
        p.clobber_dirs()


def test_pipeline_clobber_deletes_only_later_outputs(tmp_path):
    '''End to end with the real dataset config, DIR_WRITE in tmp_path.'''
    config = dataset_config()
    config["data_dirs"]["DIR_WRITE"] = str(tmp_path)
    config["pipeline"]["START_STAGE"] = "ramp"
    config["pipeline"]["CLOBBER"] = "True"
    pipe = Pipeline(config, ncpu=1)
    plan = pipe.plan

    keep = plan.out_dir("badpix")        # input of the run (before START_STAGE)
    gone = [plan.out_dir("ramp"), plan.out_dir("fuse"), plan.out_dir("register", size=pipe.sizes[0])]
    for d in [keep] + gone:
        os.makedirs(d)
        open(os.path.join(d, "lm_240131_000001.fits"), "w").close()
    calib_like = tmp_path / "unrelated"
    calib_like.mkdir()

    pipe.clobber()

    assert os.path.exists(os.path.join(keep, "lm_240131_000001.fits"))
    assert calib_like.is_dir()
    for d in gone:
        assert not os.path.exists(d)


def test_from_config_start_dir_interpolates_data_dirs(tmp_path):
    '''START_DIR may use %(DIR_HOME)s etc. from [data_dirs], as the .ini comment suggests.'''
    (tmp_path / "00_slope_frames").mkdir()
    config = dataset_config()
    config["data_dirs"]["DIR_HOME"] = str(tmp_path)
    config["pipeline"]["START_STAGE"] = "dark"
    config["pipeline"]["START_DIR"] = "%(DIR_HOME)s/00_slope_frames/"
    p = StagePlan.from_config(config, regions=["A"])
    assert p.in_dir("dark") == f"{tmp_path}/00_slope_frames/"


def test_from_config_reads_pipeline_section():
    config = dataset_config()
    config["pipeline"]["dark"] = "False"
    config["pipeline"]["START_STAGE"] = "badpix"
    p = StagePlan.from_config(config, regions=["A"])
    assert p.to_run[0] == "badpix"
    assert p.in_dir("badpix") == os.path.join(config["data_dirs"]["DIR_WRITE"], "00_slope_frames", "")
    assert not p.clobber
