"""Preprocessing pipeline entrypoint.

Runs the preprocessing stages (see modules/preprocessing/pipeline.py) for one dataset,
described by scripts/config_files/datasets/<name>.ini and its blocks CSV.

Which stages run is set in the config, not on the command line: the ``[pipeline]`` section
(True/False per stage, START_STAGE, START_DIR, CLOBBER) of the dataset .ini or of
scripts/config_files/preprocess.ini. See modules/preprocessing/stage_plan.py.

Examples:
    # the stages enabled in the config, from START_STAGE on
    python scripts/run_preprocess.py --dataset procyon_240131

    # (which frames are processed: [dataset] FRAME_SUBSET of the .ini, 'all' or frame ranges)

    # the same, on one block from each nod (overriding FRAME_SUBSET)
    python scripts/run_preprocess.py --dataset procyon_240131 --frames 501-600,10501-10600

    # the same, on 5 random good frames
    python scripts/run_preprocess.py --dataset procyon_240131 --sample 5
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
    parser.add_argument("--frames", default=None,
                        help="restrict to frame ranges, e.g. 501-600,10501-10600 (overrides "
                             "[dataset] FRAME_SUBSET of the dataset .ini; default: FRAME_SUBSET)")
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
    from modules.shared.dataset import load_dataset_config, parse_frame_ranges, frame_subset
    from modules.preprocessing import pipeline as pipeline_module

    config = load_dataset_config(args.dataset)
    # --frames overrides [dataset] FRAME_SUBSET
    if args.frames:
        frames_text = args.frames
        frame_ranges = parse_frame_ranges(args.frames)
    else:
        frames_text = config["dataset"].get("FRAME_SUBSET", fallback="").strip() or "all"
        frame_ranges = frame_subset(config)

    log_dir = Path(config["data_dirs"]["DIR_HOME"]) / "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = log_dir / f"preprocess_{datetime.datetime.now():%Y%m%d_%H%M%S}.log"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)])
    logging.info("Dataset %s (%s); frames %s; %d CPUs; skip existing: %s",
                 config["dataset"]["NAME"], config["dataset"]["INI_PATH"],
                 frames_text, ncpu, not args.overwrite)

    pipeline = pipeline_module.Pipeline(config, ncpu=ncpu, skip_existing=not args.overwrite,
                                        frame_ranges=frame_ranges, sample=args.sample, seed=args.seed)
    pipeline.run()
    logging.info("Done; log in %s", log_file)


# entry point
if __name__ == '__main__':
    main()
