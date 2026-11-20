"""
Dataset descriptors: everything that differs from one observing dataset to the next.

A dataset ``<name>`` is described by two files in ``scripts/config_files/datasets/``:

1. ``<name>.ini``: directories, filename prefix, calibration file names, detector
   regions and cutout sizes. It is read after ``scripts/config_files/preprocess.ini``, so its
   values override the base config.

2. The *blocks CSV* (``<name>_blocks.csv``, or whatever ``[dataset] BLOCKS_CSV``
   names). This is the observing log of the night, reduced to what the pipeline
   needs. A night's frames are numbered consecutively (``lm_240131_000001.fits``,
   ``..._000002.fits``, ...), and they come in runs where nothing changes: same
   target, same filter, same nod position. Each such run of consecutive frame
   numbers is one *block*, and each block is one row of the CSV:

   ============  ==========================================================
   column        meaning
   ============  ==========================================================
   frame_start   first frame number of the block (inclusive)
   frame_stop    last frame number of the block (inclusive)
   target        target name, e.g. ``Procyon``
   filter        filter name, e.g. ``Br-Alpha-On``
   quadrant      name of the detector region (see ``[regions]`` in the
                 ``.ini``) that holds the star in this block; this is
                 also the nod position; blank if the star is
                 not on the detector (e.g. an off-source nod)
   star_y        approximate star row on the detector (may be blank)
   star_x        approximate star column on the detector (may be blank)
   use           ``True`` for good frames; ``False`` for bad frames
   ============  ==========================================================

   For example, the row ``501,10500,Procyon,Br-Alpha-On,A,138,915,True`` says that
   frames 501 to 10500 are good frames of Procyon in Br-Alpha-On, with the star near
   detector pixel (y, x) = (138, 915) in region A.

   Blocks must not overlap. A range of bad frames inside a block gets its own
   row with ``use = False``, and the good block is split around it. Frames that
   are in no row at all (e.g. calibration frames) are ignored by every stage that
   uses the blocks.

   Everything that depends on the frame-number range (cutout centers, which region
   holds the PSF, bad frames) is read from here.
"""

import configparser
import os
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "scripts" / "config_files"
BASE_CONFIG = CONFIG_DIR / "preprocess.ini"
DATASETS_DIR = CONFIG_DIR / "datasets"

BLOCK_COLUMNS = ["frame_start", "frame_stop", "target", "filter", "quadrant",
                 "star_y", "star_x", "use"]


def dataset_ini_path(dataset):
    '''
    Resolve a dataset name or a path to its ``.ini`` file.

    INPUTS:
    - dataset (str): Name in ``scripts/config_files/datasets/`` (e.g. ``procyon_240131``) or a path to an ``.ini``

    OUTPUTS:
    - path (Path): Absolute path to the dataset ``.ini``
    '''
    path = Path(dataset).expanduser()
    if path.suffix != ".ini":
        path = DATASETS_DIR / f"{dataset}.ini"
    if not path.is_file():
        raise FileNotFoundError(f"Dataset config not found: {path}")
    return path.resolve()


def load_dataset_config(dataset, base_ini=BASE_CONFIG):
    '''
    Read the base preprocessing config, then the dataset config on top of it.

    INPUTS:
    - dataset (str): Dataset name or path to its ``.ini``
    - base_ini (str or Path): Base config read first

    OUTPUTS:
    - config (ConfigParser): Merged config; ``config['dataset']['INI_PATH']`` holds the dataset file
    '''
    ini = dataset_ini_path(dataset)
    config = configparser.ConfigParser()
    read = config.read([str(base_ini), str(ini)])
    if str(ini) not in read:
        raise IOError(f"Could not read {ini}")
    config["dataset"]["INI_PATH"] = str(ini)
    return config


def blocks_csv_path(config):
    '''
    Path of the blocks CSV named by ``[dataset] BLOCKS_CSV`` (relative to the dataset ``.ini``).

    INPUTS:
    - config (ConfigParser): Merged config (``[dataset] BLOCKS_CSV`` and ``INI_PATH``)

    OUTPUTS:
    - csv (Path): Blocks CSV path
    '''
    csv = Path(config["dataset"]["BLOCKS_CSV"]).expanduser()
    if not csv.is_absolute():
        csv = Path(config["dataset"]["INI_PATH"]).parent / csv
    return csv


def _parse_bool(value):
    '''
    Parse a boolean from a CSV cell.

    INPUTS:
    - value (object): bool, or a string such as ``true``/``false``, ``1``/``0``, ``yes``/``no``

    OUTPUTS:
    - flag (bool): Parsed value (raises ValueError if it does not parse)
    '''
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    s = str(value).strip().lower()
    if s in ("true", "1", "yes", "y"):
        return True
    if s in ("false", "0", "no", "n"):
        return False
    raise ValueError(f"Cannot parse boolean from {value!r}")


def load_blocks(csv):
    '''
    Read and validate a blocks CSV (see the module docstring for what a block is).

    INPUTS:
    - csv (str or Path): Blocks CSV

    OUTPUTS:
    - blocks (DataFrame): One row per block, sorted by ``frame_start``; ``star_y``/``star_x``
      are nullable integers, ``use`` is boolean
    '''
    blocks = pd.read_csv(csv, comment="#", skipinitialspace=True,
                         dtype={"target": str, "filter": str, "quadrant": str})
    missing = [c for c in BLOCK_COLUMNS if c not in blocks.columns]
    if missing:
        raise ValueError(f"{csv}: missing columns {missing}")

    blocks = blocks[BLOCK_COLUMNS].copy()
    blocks["frame_start"] = blocks["frame_start"].astype(int)
    blocks["frame_stop"] = blocks["frame_stop"].astype(int)
    blocks["star_y"] = pd.to_numeric(blocks["star_y"]).astype("Int64")
    blocks["star_x"] = pd.to_numeric(blocks["star_x"]).astype("Int64")
    blocks["use"] = blocks["use"].map(_parse_bool)
    for col in ("target", "filter", "quadrant"):
        blocks[col] = blocks[col].fillna("").str.strip()
    blocks = blocks.sort_values("frame_start").reset_index(drop=True)

    if (blocks["frame_stop"] < blocks["frame_start"]).any():
        raise ValueError(f"{csv}: a block has frame_stop < frame_start")
    overlap = blocks["frame_start"].values[1:] <= blocks["frame_stop"].values[:-1]
    if overlap.any():
        i = int(np.argmax(overlap))
        raise ValueError(f"{csv}: blocks overlap at rows {i} and {i + 1}")
    used = blocks[blocks["use"]]
    if (used["target"] == "").any() or (used["filter"] == "").any():
        raise ValueError(f"{csv}: every block with use=True needs a target and filter")
    return blocks


def block_for_frame(blocks, frame_num):
    '''
    Return the block containing ``frame_num``, or ``None`` if no block contains it.

    INPUTS:
    - blocks (DataFrame): Output of ``load_blocks``
    - frame_num (int): Frame number

    OUTPUTS:
    - block (Series or None): The matching row (its index ``.name`` is the block id)
    '''
    i = np.searchsorted(blocks["frame_start"].values, frame_num, side="right") - 1
    if i < 0 or frame_num > blocks["frame_stop"].iat[i]:
        return None
    return blocks.iloc[i]


def frame_number(file_name, prefix):
    '''
    Frame number from a filename like ``<prefix><NNNNNN>.fits``.

    INPUTS:
    - file_name (str): File name or path
    - prefix (str): Text just before the frame number (e.g. ``lm_240131_``)

    OUTPUTS:
    - frame_num (int or None): Frame number, or None if it does not parse
    '''
    base = os.path.basename(file_name)
    if prefix not in base:
        return None
    try:
        return int(base.split(prefix, 1)[1].split(".fits")[0])
    except ValueError:
        return None


def used_frame_numbers(blocks):
    '''
    Set of all frame numbers in blocks with ``use = True``.

    INPUTS:
    - blocks (DataFrame): Output of ``load_blocks``

    OUTPUTS:
    - nums (set of int): Frame numbers
    '''
    nums = set()
    for row in blocks[blocks["use"]].itertuples():
        nums.update(range(int(row.frame_start), int(row.frame_stop) + 1))
    return nums


def _parse_slice(text):
    '''
    Parse one region axis ``start:stop`` (either end may be blank) into a slice.

    INPUTS:
    - text (str): E.g. ``"0:512"`` or ``"512:"``

    OUTPUTS:
    - axis (slice): The slice (raises ValueError if malformed)
    '''
    parts = text.strip().split(":")
    if len(parts) != 2:
        raise ValueError(f"Region axis must look like 'start:stop', got {text!r}")
    lo, hi = (int(p) if p.strip() else None for p in parts)
    return slice(lo, hi)


def parse_regions(config):
    '''
    Detector regions from the ``[regions]`` config section.

    Each entry reads ``<name>: y_start:y_stop, x_start:x_stop`` in detector pixels,
    Python slice convention (stop is exclusive; either end may be blank). The names
    are the values used in the ``quadrant`` column of the blocks CSV.

    INPUTS:
    - config (ConfigParser): Merged config

    OUTPUTS:
    - regions (dict): Map of region name (as written, e.g. ``I``) to ``(y_slice, x_slice)``
    '''
    if not config.has_section("regions"):
        return {}
    # ConfigParser lower-cases keys; re-read with case preserved to keep the names as written
    raw = configparser.ConfigParser()
    raw.optionxform = str
    raw.read([str(BASE_CONFIG), config["dataset"]["INI_PATH"]])
    regions = {}
    for name, value in raw.items("regions"):
        ys, xs = value.split(",")
        regions[name] = (_parse_slice(ys), _parse_slice(xs))
    return regions


def parse_frame_ranges(text):
    '''
    Parse ``"501-600,10501-10600,7"`` into a list of inclusive ``(lo, hi)`` tuples.

    INPUTS:
    - text (str): Comma-separated frame numbers and ``lo-hi`` ranges

    OUTPUTS:
    - ranges (list of tuple): Inclusive ``(lo, hi)`` ranges; a single number gives ``(n, n)``
    '''
    ranges = []
    for tok in text.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            lo, hi = tok.split("-")
            ranges.append((int(lo), int(hi)))
        else:
            ranges.append((int(tok), int(tok)))
    return ranges
