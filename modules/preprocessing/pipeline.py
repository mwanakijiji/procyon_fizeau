"""
Preprocessing stages, in order, driven by scripts/run_preprocess.py.

Stage          reads                        writes
-------------  ---------------------------  ---------------------------------------
slope          DIR_RAW_DATA                 DIR_SLOPE_FRAMES
dark           DIR_SLOPE_FRAMES             DIR_DARK_SUBTED         (skipped if DARK_MASTER = None)
badpix         DIR_DARK_SUBTED (or _SLOPE_) DIR_PIXL_CORRTD         (skipped if BADPIX_MASTER = None)
ramp           newest of the above          DIR_RAMP_REMOVED
bkg_nod        DIR_RAMP_REMOVED             DIR_NOD_SUBTED          (optional)
bkg_channel    DIR_RAMP_REMOVED or _NOD_    DIR_PCAB_SUBTED
train_regions  DIR_PCAB_SUBTED              REGION_PCA_CUBE files   (when the cubes are missing)
bkg_regions    DIR_PCAB_SUBTED              DIR_PCAB_SUBTED_REGION (one per region)
fuse           DIR_PCAB_SUBTED_REGION       DIR_ALL_QUADS_RECOMBINED
cutout         DIR_ALL_QUADS_RECOMBINED     DIR_CUTOUT_FRAMES       (first of CUTOUT_SIZES)
register       DIR_CUTOUT_FRAMES            DIR_REGISTERED_FRAMES   (first size)
crop           DIR_REGISTERED_FRAMES        DIR_REGISTERED_FRAMES   (other sizes)

slope, dark, badpix and ramp process every frame in their input directory. From bkg_nod
on, only frames in blocks with use = True are processed; cutout, register and crop further
skip blocks with no star on the detector (blank ``quadrant`` in the blocks CSV).

ramp reads DIR_PIXL_CORRTD if the badpix stage is enabled, else DIR_DARK_SUBTED if the dark
stage is, else DIR_SLOPE_FRAMES.

Adapted from modules/preprocessing/pipeline.py of the AB Aur pipeline.
"""

import glob
import logging
import os
from multiprocessing import get_context

import numpy as np

from modules.shared.dataset import (
    blocks_csv_path, load_blocks, block_for_frame, frame_number, parse_regions,
    used_frame_numbers,
)
from modules.preprocessing import make_slope_frames, dark_subtract, fix_bad_pix, ramp_removal
from modules.preprocessing.background_pca import (
    BackgroundPCASubtSingle, FuseRegions, pca_backg_maker_channels_only,
)
from modules.preprocessing.detector import DetectorMask
from modules.preprocessing.frame_io import read_frame
from modules.preprocessing.nod_subtract import NodSubtractSingle
from modules.preprocessing.region_pca import train_region_pca
from modules.preprocessing.cutouts import CutoutSingle
from modules.preprocessing.crop import crop_fits_dir
from modules.preprocessing.star import block_has_star

STAGES = ["slope", "dark", "badpix", "ramp", "bkg_nod", "bkg_channel", "train_regions",
          "bkg_regions", "fuse", "cutout", "register", "crop"]
DEFAULT_STAGES = [s for s in STAGES if s not in ("bkg_nod", "train_regions")]

# the callable run by pool workers; set before the pool is forked, so workers inherit it
# instead of having it (and e.g. a large PCA cube) pickled with every task
_WORKER = None


def _call_worker(arg):
    '''
    Run the module-level ``_WORKER`` callable on one argument (the target of pool workers).

    INPUTS:
    - arg (object): One work item (e.g. a file path, or a ``(block_id, files)`` job)

    OUTPUTS:
    - result (object): Whatever ``_WORKER`` returns
    '''
    if isinstance(arg, str):
        logging.info("Processing %s", os.path.basename(arg))
    return _WORKER(arg)


class Pipeline:
    '''
    Runs preprocessing stages for one dataset (see the module docstring for the stages,
    and the directories they read and write).
    '''

    def __init__(self, config, ncpu=4, skip_existing=True, frame_ranges=None, sample=None, seed=0):
        '''
        Set up the pipeline from the merged base + dataset config.

        INPUTS:
        - config (ConfigParser): Merged base + dataset config (``load_dataset_config``)
        - ncpu (int): Number of worker processes
        - skip_existing (bool): Skip frames whose output file already exists
        - frame_ranges (list of tuple or None): Inclusive ``(lo, hi)`` frame ranges to process;
          ``None`` processes all frames
        - sample (int or None): Process only this many randomly chosen good frames (use=True)
          from within ``frame_ranges``, e.g. for quick tests. The choice depends only on the
          blocks CSV and ``seed``, so every stage gets the same frames. (train_regions and the
          opposite-nod frames of bkg_nod still use every available frame.)
        - seed (int): Random seed for ``sample``

        OUTPUTS:
        - None
        '''
        self.config = config
        self.ncpu = ncpu
        self.skip_existing = skip_existing
        self.frame_ranges = frame_ranges
        self.dirs = config["data_dirs"]
        self.pp = config["preprocess"]
        self.prefix = config["dataset"]["FILE_PREFIX"]
        self.blocks = load_blocks(blocks_csv_path(config))
        self.sample_frames = None
        if sample:
            candidates = sorted(n for n in used_frame_numbers(self.blocks) if self._in_ranges(n))
            rng = np.random.default_rng(seed)
            chosen = rng.choice(candidates, size=min(sample, len(candidates)), replace=False)
            self.sample_frames = set(int(n) for n in chosen)
            logging.info(f"Random sample of {len(self.sample_frames)} frames (seed {seed}): {sorted(self.sample_frames)}")
        self.regions = parse_regions(config)
        self.mask = DetectorMask.from_config(config)
        self.sizes = [int(s) for s in config["dataset"]["CUTOUT_SIZES"].split(",")]
        self.psf_radius = self.pp.getfloat("PSF_MASK_RADIUS")
        self.halfbox = self.pp.getint("STAR_SEARCH_HALFBOX")
        self.use_nod = self.pp.getboolean("USE_NOD_SUBTRACTION")
        # DARK_MASTER = None skips dark subtraction; ramp then reads the slope frames
        self.use_dark = config["file_names"]["DARK_MASTER"].strip() != "None"
        # BADPIX_MASTER = None (or absent) skips bad-pixel fixing
        self.use_badpix = config["file_names"].get("BADPIX_MASTER", "None").strip() != "None"

    # ---------- helpers ----------

    def _pre_badpix_dir(self):
        '''
        Input directory of the badpix stage.

        INPUTS:
        - None

        OUTPUTS:
        - path (str): DIR_DARK_SUBTED if the dark stage is enabled, else DIR_SLOPE_FRAMES
        '''
        return self.dirs["DIR_DARK_SUBTED"] if self.use_dark else self.dirs["DIR_SLOPE_FRAMES"]

    def _pre_ramp_dir(self):
        '''
        Input directory of the ramp stage (output of the last enabled stage before it).

        INPUTS:
        - None

        OUTPUTS:
        - path (str): DIR_PIXL_CORRTD if the badpix stage is enabled, else ``_pre_badpix_dir()``
        '''
        return self.dirs["DIR_PIXL_CORRTD"] if self.use_badpix else self._pre_badpix_dir()

    def _in_ranges(self, n):
        '''
        Whether frame number ``n`` is inside ``frame_ranges`` (always True if none were given).

        INPUTS:
        - n (int): Frame number

        OUTPUTS:
        - inside (bool): True if ``n`` is in one of the inclusive ``frame_ranges``
        '''
        return self.frame_ranges is None or any(lo <= n <= hi for lo, hi in self.frame_ranges)

    def _selected(self, n):
        '''
        Whether frame number ``n`` is selected by ``frame_ranges`` and the random ``sample``.

        INPUTS:
        - n (int): Frame number

        OUTPUTS:
        - selected (bool): True if ``n`` is in range and in the sample (if any)
        '''
        return self._in_ranges(n) and (self.sample_frames is None or n in self.sample_frames)

    def files(self, directory, used_only=False, all_frames=False, star_only=False):
        '''
        Sorted ``<prefix>*.fits`` files in ``directory``, restricted to ``--frames`` and
        ``--sample`` (unless ``all_frames``), if ``used_only`` to frames in blocks with
        ``use = True``, and if ``star_only`` to frames in blocks with the star on the detector.

        INPUTS:
        - directory (str): Directory to list
        - used_only (bool): Keep only frames in blocks with ``use = True``
        - all_frames (bool): Ignore ``frame_ranges`` and ``sample``
        - star_only (bool): Keep only frames in blocks with a non-blank ``quadrant``

        OUTPUTS:
        - files (list of str): Sorted FITS paths
        '''
        out = []
        for path in sorted(glob.glob(os.path.join(directory, f"{self.prefix}*.fits"))):
            n = frame_number(path, self.prefix)
            if n is None or (not all_frames and not self._selected(n)):
                continue
            if used_only or star_only:
                b = block_for_frame(self.blocks, n)
                if b is None or (used_only and not b["use"]) or (star_only and not block_has_star(b)):
                    continue
            out.append(path)
        return out

    def files_by_frame(self, directory):
        '''
        All used frames in ``directory`` (ignoring ``frame_ranges`` and ``sample``), by frame number.

        INPUTS:
        - directory (str): Directory to list

        OUTPUTS:
        - files (dict): Map of frame number (int) to FITS path (str)
        '''
        return {frame_number(p, self.prefix): p for p in self.files(directory, used_only=True, all_frames=True)}

    def _todo(self, files, write_dir):
        '''
        The input files whose output is not yet in ``write_dir`` (all of them unless ``skip_existing``).

        INPUTS:
        - files (list of str): Input FITS paths
        - write_dir (str): Output directory (outputs keep the input basenames)

        OUTPUTS:
        - todo (list of str): Input paths still to process
        '''
        if not self.skip_existing:
            return files
        todo = [f for f in files if not os.path.exists(os.path.join(write_dir, os.path.basename(f)))]
        if len(todo) < len(files):
            logging.info(f"  skipping {len(files) - len(todo)} frames already in {write_dir}")
        return todo

    def map(self, fn, files, write_dir, chunksize=None):
        '''
        Run ``fn`` on each input file not yet in ``write_dir``, in ``ncpu`` forked workers
        (or serially if ``ncpu`` <= 1). ``fn`` writes its own output.

        INPUTS:
        - fn (callable): Per-file worker, called as ``fn(path)``
        - files (list of str): Input FITS paths
        - write_dir (str): Output directory (created if needed)
        - chunksize (int or None): Pool chunk size; default ``len(todo) / (4 ncpu)``, between 1 and 32

        OUTPUTS:
        - None
        '''
        global _WORKER
        os.makedirs(write_dir, exist_ok=True)
        todo = self._todo(files, write_dir)
        logging.info(f"  {len(todo)} frames to process -> {write_dir}")
        if not todo:
            return
        _WORKER = fn
        if self.ncpu <= 1:
            for f in todo:
                _call_worker(f)
            return
        if chunksize is None:
            chunksize = max(1, min(32, len(todo) // (4 * self.ncpu)))
        with get_context("fork").Pool(self.ncpu) as pool:
            pool.map(_call_worker, todo, chunksize=chunksize)

    def region_dir(self, region):
        '''
        Output directory of the region-PCA background subtraction for one region.

        INPUTS:
        - region (str): Region name (from the dataset config)

        OUTPUTS:
        - path (str): ``DIR_PCAB_SUBTED_REGION`` with ``{region}`` filled in
        '''
        return self.dirs["DIR_PCAB_SUBTED_REGION"].format(region=region)

    def region_cube_path(self, region):
        '''
        Path of the PCA basis cube for one region (``_nodsub`` variant if nod subtraction is used).

        INPUTS:
        - region (str): Region name (from the dataset config)

        OUTPUTS:
        - path (str): Path under ``DIR_CALIB_FRAMES``
        '''
        name = self.config["file_names"]["REGION_PCA_CUBE"].format(region=region)
        if self.use_nod:
            name = name.replace(".fits", "_nodsub.fits")
        return os.path.join(self.dirs["DIR_CALIB_FRAMES"], name)

    def cutout_dir(self):
        '''
        Output directory of the cutout stage (first size in ``CUTOUT_SIZES``).

        INPUTS:
        - None

        OUTPUTS:
        - path (str): ``DIR_CUTOUT_FRAMES`` with ``{size}`` filled in
        '''
        return self.dirs["DIR_CUTOUT_FRAMES"].format(size=self.sizes[0])

    def registered_dir(self, size):
        '''
        Directory of the registered frames of one cutout size.

        INPUTS:
        - size (int): Cutout side (pix)

        OUTPUTS:
        - path (str): ``DIR_REGISTERED_FRAMES`` with ``{size}`` filled in
        '''
        return self.dirs["DIR_REGISTERED_FRAMES"].format(size=size)

    def calib(self, key):
        '''
        Path of a calibration file named in the ``[file_names]`` config section.

        INPUTS:
        - key (str): Key in ``[file_names]`` (e.g. ``DARK_MASTER``)

        OUTPUTS:
        - path (str): Path under ``DIR_CALIB_FRAMES``; raises FileNotFoundError if missing
        '''
        path = os.path.join(self.dirs["DIR_CALIB_FRAMES"], self.config["file_names"][key])
        if not os.path.exists(path):
            raise FileNotFoundError(f"Calibration file {key} not found: {path}")
        return path

    def run(self, stages):
        '''
        Run the given stages in order (each must be in ``STAGES``).

        INPUTS:
        - stages (list of str): Stage names

        OUTPUTS:
        - None
        '''
        for stage in stages:
            if stage not in STAGES:
                raise ValueError(f"Unknown stage {stage!r}; choose from {STAGES}")
        for stage in stages:
            logging.info(f"=== stage {stage} ===")
            getattr(self, f"stage_{stage}")()

    # ---------- stages ----------

    def stage_slope(self):
        '''
        Make 2D frames from the raw readouts: slope frames from 2-read cubes, or the single read
        of 1-read cubes (DIR_RAW_DATA -> DIR_SLOPE_FRAMES).

        INPUTS:
        - None

        OUTPUTS:
        - None
        '''
        logging.info('Making 2D frames from the raw readouts...')
        fn = make_slope_frames.MakeSlopeFrameSingle(write_dir=self.dirs["DIR_SLOPE_FRAMES"], config=self.config)
        self.map(fn, self.files(self.dirs["DIR_RAW_DATA"]), self.dirs["DIR_SLOPE_FRAMES"])

    def stage_dark(self):
        '''
        Subtract the master dark (DIR_SLOPE_FRAMES -> DIR_DARK_SUBTED); skipped if DARK_MASTER
        is None.

        INPUTS:
        - None

        OUTPUTS:
        - None
        '''
        if not self.use_dark:
            logging.info('DARK_MASTER is None; skipping dark subtraction')
            return
        logging.info('Dark-subtracting frames...')
        fn = dark_subtract.DarkSubtSingle(write_dir=self.dirs["DIR_DARK_SUBTED"],
                                          master_dark=self.calib("DARK_MASTER"), config=self.config)
        self.map(fn, self.files(self.dirs["DIR_SLOPE_FRAMES"]), self.dirs["DIR_DARK_SUBTED"])

    def stage_badpix(self):
        '''
        Interpolate over the pixels in the master bad-pixel mask (DIR_DARK_SUBTED, or
        DIR_SLOPE_FRAMES if DARK_MASTER is None -> DIR_PIXL_CORRTD); skipped if BADPIX_MASTER
        is None.

        INPUTS:
        - None

        OUTPUTS:
        - None
        '''
        if not self.use_badpix:
            logging.info('BADPIX_MASTER is None; skipping bad-pixel fixing')
            return
        logging.info('Fixing bad pixels...')
        fn = fix_bad_pix.FixPixSingle(write_dir=self.dirs["DIR_PIXL_CORRTD"],
                                      master_bad_pix_file=self.calib("BADPIX_MASTER"), config=self.config)
        self.map(fn, self.files(self._pre_badpix_dir()), self.dirs["DIR_PIXL_CORRTD"])

    def stage_ramp(self):
        '''
        Remove stray-illumination ramps (``_pre_ramp_dir()`` -> DIR_RAMP_REMOVED).

        INPUTS:
        - None

        OUTPUTS:
        - None
        '''
        logging.info('Removing stray illumination ramps...')
        read_dir = self._pre_ramp_dir()
        fn = ramp_removal.RemoveStrayRamp(write_dir=self.dirs["DIR_RAMP_REMOVED"], config=self.config,
                                          psf_mask_radius=self.psf_radius)
        self.map(fn, self.files(read_dir), self.dirs["DIR_RAMP_REMOVED"])

    def stage_bkg_nod(self):
        '''
        Subtract the median of nearby opposite-nod frames (DIR_RAMP_REMOVED -> DIR_NOD_SUBTED;
        used frames only).

        INPUTS:
        - None

        OUTPUTS:
        - None
        '''
        logging.info('Subtracting medians of opposite-nod frames...')
        fn = NodSubtractSingle(write_dir=self.dirs["DIR_NOD_SUBTED"],
                               files_by_frame=self.files_by_frame(self.dirs["DIR_RAMP_REMOVED"]),
                               blocks=self.blocks, n_frames=self.pp.getint("NOD_N_FRAMES"),
                               psf_radius=self.psf_radius, halfbox=self.halfbox, prefix=self.prefix)
        # contiguous chunks, so each worker's cache of opposite-nod frames is reused
        self.map(fn, self.files(self.dirs["DIR_RAMP_REMOVED"], used_only=True), self.dirs["DIR_NOD_SUBTED"],
                 chunksize=64)

    def stage_bkg_channel(self):
        '''
        Subtract channel-pedestal PCA backgrounds (DIR_RAMP_REMOVED, or DIR_NOD_SUBTED with nod
        subtraction -> DIR_PCAB_SUBTED; used frames only). Builds the channel PCA cube, at the
        shape of the input frames, if missing.

        INPUTS:
        - None

        OUTPUTS:
        - None
        '''
        logging.info('Subtracting channel-pedestal PCA backgrounds...')
        read_dir = self.dirs["DIR_NOD_SUBTED"] if self.use_nod else self.dirs["DIR_RAMP_REMOVED"]
        files = self.files(read_dir, used_only=True)
        if not files:
            logging.info(f"  no frames in {read_dir}")
            return
        cube = os.path.join(self.dirs["DIR_OTHER_FITS"], "background_PCA_vector_channel_vars_only.fits")
        if not os.path.exists(cube):
            os.makedirs(os.path.dirname(cube), exist_ok=True)
            pca_backg_maker_channels_only(cube, read_frame(files[0])[0].shape)
        fn = BackgroundPCASubtSingle(
            self.pp.getint("N_PCA_CHANNEL"), pca_cube_file=cube, write_dir=self.dirs["DIR_PCAB_SUBTED"],
            mask=self.mask, config_data=self.config, psf_mask_radius=self.psf_radius,
            write_diagnostics=self.pp.getboolean("WRITE_BKG_DIAGNOSTICS"))
        self.map(fn, files, self.dirs["DIR_PCAB_SUBTED"])

    def stage_train_regions(self):
        '''
        Train the PCA background basis of each detector region from DIR_PCAB_SUBTED frames
        (skips regions whose basis exists, if ``skip_existing``).

        INPUTS:
        - None

        OUTPUTS:
        - None
        '''
        logging.info('Training PCA background bases for each detector region...')
        files = self.files_by_frame(self.dirs["DIR_PCAB_SUBTED"])
        for name, region in self.regions.items():
            out = self.region_cube_path(name)
            if self.skip_existing and os.path.exists(out):
                logging.info(f"  region {name}: basis exists, skipping ({out})")
                continue
            train_region_pca(files, self.blocks, name, region,
                             n_frames=self.pp.getint("N_TRAIN_FRAMES"),
                             n_pca=self.pp.getint("N_PCA_REGION"),
                             psf_radius=self.psf_radius, out_path=out, mask=self.mask,
                             halfbox=self.halfbox)

    def stage_bkg_regions(self):
        '''
        Subtract each region's PCA background (DIR_PCAB_SUBTED -> one DIR_PCAB_SUBTED_REGION per
        region; used frames only). The region bases must exist (train_regions).

        INPUTS:
        - None

        OUTPUTS:
        - None
        '''
        logging.info('Subtracting region PCA backgrounds...')
        files = self.files(self.dirs["DIR_PCAB_SUBTED"], used_only=True)
        for name in self.regions:
            cube = self.region_cube_path(name)
            if not os.path.exists(cube):
                raise FileNotFoundError(f"PCA basis for region {name} not found: {cube}. "
                                        "Run the train_regions stage first.")
            logging.info(f"  region {name}")
            fn = BackgroundPCASubtSingle(
                self.pp.getint("N_PCA_REGION"), pca_cube_file=cube, write_dir=self.region_dir(name),
                mask=self.mask, config_data=self.config, psf_mask_radius=self.psf_radius,
                write_diagnostics=self.pp.getboolean("WRITE_BKG_DIAGNOSTICS"))
            self.map(fn, files, self.region_dir(name))
            del fn

    def stage_fuse(self):
        '''
        Fuse the background-subtracted regions into whole frames (DIR_PCAB_SUBTED_REGION ->
        DIR_ALL_QUADS_RECOMBINED), for frames present in every region directory.

        INPUTS:
        - None

        OUTPUTS:
        - None
        '''
        logging.info('Fusing background-subtracted regions into whole frames...')
        region_dirs = [self.region_dir(name) for name in self.regions]
        # frames present in every region directory
        names = set.intersection(*[set(map(os.path.basename, self.files(d, used_only=True))) for d in region_dirs])
        files = [os.path.join(region_dirs[0], n) for n in sorted(names)]
        fn = FuseRegions(write_dir=self.dirs["DIR_ALL_QUADS_RECOMBINED"], region_dirs=region_dirs)
        self.map(fn, files, self.dirs["DIR_ALL_QUADS_RECOMBINED"])

    def stage_cutout(self):
        '''
        Cut odd-sized cutouts centered on the star (DIR_ALL_QUADS_RECOMBINED ->
        DIR_CUTOUT_FRAMES, first size in CUTOUT_SIZES; used frames with the star on the detector).

        INPUTS:
        - None

        OUTPUTS:
        - None
        '''
        logging.info('Making initial cutouts centered on the star...')
        fn = CutoutSingle(write_dir=self.cutout_dir(), blocks=self.blocks, prefix=self.prefix,
                          size=self.sizes[0], halfbox=self.halfbox)
        self.map(fn, self.files(self.dirs["DIR_ALL_QUADS_RECOMBINED"], used_only=True, star_only=True),
                 self.cutout_dir())

    def stage_register(self):
        '''
        Register the cutouts within each block (DIR_CUTOUT_FRAMES -> DIR_REGISTERED_FRAMES of the
        first size), one block per worker; blocks already fully registered are skipped if
        ``skip_existing``. Logs a summary per block.

        INPUTS:
        - None

        OUTPUTS:
        - None
        '''
        # scikit-image is needed only here
        from modules.preprocessing.registration import register_block

        logging.info('Registering cutouts within each block...')
        write_dir = self.registered_dir(self.sizes[0])
        os.makedirs(write_dir, exist_ok=True)
        by_block = {}
        for path in self.files(self.cutout_dir(), used_only=True, star_only=True):
            b = block_for_frame(self.blocks, frame_number(path, self.prefix))
            by_block.setdefault(b.name, []).append(path)
        jobs = []
        for block_id, files in sorted(by_block.items()):
            if self.skip_existing and all(os.path.exists(os.path.join(write_dir, os.path.basename(f)))
                                          for f in files):
                logging.info(f"  block {block_id}: all {len(files)} frames registered, skipping")
                continue
            jobs.append((block_id, files))
        logging.info(f"  {len(jobs)} blocks to register -> {write_dir}")

        kwargs = dict(box=self.pp.getint("REG_BOX"), upsample=self.pp.getint("REG_UPSAMPLE"),
                      centroid_halfbox=self.pp.getint("REG_CENTROID_HALFBOX"),
                      ref_max_frames=self.pp.getint("REG_REF_MAX_FRAMES"),
                      n_iter=self.pp.getint("REG_N_ITER"))

        def one_block(job):
            '''
            Register one block (the pool worker of ``stage_register``).

            INPUTS:
            - job (tuple): ``(block_id, files)``, the block id and its cutout paths

            OUTPUTS:
            - result (tuple): ``(block_id, frame_start, frame_stop, summary)``, with the
              ``register_block`` summary dict
            '''
            block_id, files = job
            summary = register_block(files, write_dir, **kwargs)
            b = self.blocks.loc[block_id]
            return block_id, b["frame_start"], b["frame_stop"], summary

        global _WORKER
        _WORKER = one_block
        if self.ncpu <= 1 or len(jobs) <= 1:
            results = [one_block(j) for j in jobs]
        else:
            with get_context("fork").Pool(min(self.ncpu, len(jobs))) as pool:
                results = pool.map(_call_worker, jobs, chunksize=1)
        for block_id, lo, hi, s in results:
            msg = f"  block {block_id} ({lo}-{hi}): {s['n_written']} registered, {s['n_failed']} failed"
            if "median_last_correction_pix" in s:
                msg += (f", median shift correction in last iteration {s['median_last_correction_pix']:.3f} pix"
                        f", block recentered by {s['recenter_pix']:.3f} pix")
            logging.info(msg)

    def stage_crop(self):
        '''
        Crop the registered frames to the other sizes in CUTOUT_SIZES (DIR_REGISTERED_FRAMES of the
        first size -> DIR_REGISTERED_FRAMES of each other size), keeping the star at ((N-1)/2, (N-1)/2).

        INPUTS:
        - None

        OUTPUTS:
        - None
        '''
        logging.info('Cropping registered frames to the user-desired cutout sizes...')
        src = self.registered_dir(self.sizes[0])
        for size in self.sizes[1:]:
            dst = self.registered_dir(size)
            names = {os.path.basename(p) for p in self.files(src, used_only=True, star_only=True)}
            written = crop_fits_dir(src, dst, size, skip_existing=self.skip_existing, only=names)
            logging.info(f"  cropped {len(written)} frames to {size} pix -> {dst}")
