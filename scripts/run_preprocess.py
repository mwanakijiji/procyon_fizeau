"""Preprocessing pipeline entrypoint.

Runs the preprocessing stages (see modules/preprocessing/pipeline.py) for one dataset,
described by scripts/config_files/datasets/<name>.ini and its blocks CSV.

Examples:
    # everything, for the Procyon dataset
    python scripts/run_preprocess.py --dataset procyon_240131

    # background stages only, on one block from each nod
    python scripts/run_preprocess.py --dataset procyon_240131 \\
        --stages ramp,bkg_channel,bkg_regions,fuse --frames 501-600,10501-10600

    # build the region PCA bases (needed once per dataset, after bkg_channel)
    python scripts/run_preprocess.py --dataset procyon_240131 --stages train_regions

Stages (in order): slope, dark, badpix, ramp, bkg_nod, bkg_channel, train_regions, bkg_regions,
fuse, cutout, register, crop. By default all run except the optional bkg_nod and
train_regions.
"""
import argparse
import datetime
import logging
import multiprocessing
import os
import sys
from pathlib import Path

# Allow running as ``python scripts/run_preprocess.py`` from the repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

STAGES = ["slope", "dark", "badpix", "ramp", "bkg_nod", "bkg_channel", "train_regions",
          "bkg_regions", "fuse", "cutout", "register", "crop"]
DEFAULT_STAGES = [s for s in STAGES if s not in ("bkg_nod", "train_regions")]


def _limit_blas_threads(ncpu):
    '''
    Give each of the ``ncpu`` workers an equal share of the cores for BLAS (numpy/scipy),
    instead of every worker using all cores. Must run before numpy is imported.

    INPUTS:
    - ncpu (int): Number of worker processes

    OUTPUTS:
    - None
    '''
    n = str(max(1, multiprocessing.cpu_count() // ncpu))
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, n)


def main():
    '''
    Parse the command line and run the selected preprocessing stages for one dataset.

    INPUTS:
    - None (command-line arguments; see the module docstring)

    OUTPUTS:
    - None
    '''
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("\n\n", 1)[1])
    parser.add_argument("--dataset", required=True,
                        help="dataset name in scripts/config_files/datasets/ (e.g. procyon_240131) or path to its .ini")
    parser.add_argument("--stages", default=",".join(DEFAULT_STAGES),
                        help=f"comma-separated stages from {','.join(STAGES)} (default: %(default)s)")
    parser.add_argument("--frames", default=None,
                        help="restrict to frame ranges, e.g. 501-600,10501-10600 (default: all)")
    parser.add_argument("--sample", type=int, default=None,
                        help="process only N random good frames from within --frames (same frames "
                             "for every stage; for quick tests)")
    parser.add_argument("--seed", type=int, default=0, help="random seed for --sample (default: %(default)s)")
    parser.add_argument("--ncpu", type=int, default=4, help="worker processes (default: %(default)s)")
    parser.add_argument("--overwrite", action="store_true",
                        help="reprocess frames whose output already exists (default: skip them)")
    args = parser.parse_args()
    ncpu = max(1, min(args.ncpu, multiprocessing.cpu_count()))
    _limit_blas_threads(ncpu)

    # load these inside main(), after BLAS linalg thread limits are set
    # (so each worker process does not grab every core for itself)
    from modules.shared.dataset import load_dataset_config, parse_frame_ranges
    from modules.preprocessing import pipeline as pipeline_module
    assert pipeline_module.STAGES == STAGES, "keep STAGES in run_preprocess.py and pipeline.py in sync"

    config = load_dataset_config(args.dataset)
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    frame_ranges = parse_frame_ranges(args.frames) if args.frames else None

    log_dir = Path(config["data_dirs"]["DIR_HOME"]) / "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = log_dir / f"preprocess_{datetime.datetime.now():%Y%m%d_%H%M%S}.log"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)])
    logging.info("Dataset %s (%s); stages %s; frames %s; %d CPUs; skip existing: %s",
                 config["dataset"]["NAME"], config["dataset"]["INI_PATH"], stages,
                 args.frames or "all", ncpu, not args.overwrite)

    pipeline = pipeline_module.Pipeline(config, ncpu=ncpu, skip_existing=not args.overwrite,
                                        frame_ranges=frame_ranges, sample=args.sample, seed=args.seed)
    pipeline.run(stages)
    logging.info("Done; log in %s", log_file)


# entry point
if __name__ == '__main__':
    main()
