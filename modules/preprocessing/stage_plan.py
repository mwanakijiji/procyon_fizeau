"""
Which preprocessing stages run, and the directory each one reads and writes.

The ``[pipeline]`` config section has a True/False flag for every stage in ``STAGES``; the
enabled stages form the processing chain, in ``STAGES`` order. Each enabled stage that
writes frames (``FRAME_STAGES``) gets the output directory ``DIR_WRITE/<NN>_<stem>/``, where
``<stem>`` comes from ``[stage_dirs]`` and ``NN`` counts up from 00 over the enabled frame
stages. A stage reads the output of the previous enabled frame stage (``slope`` reads
``DIR_RAW_DATA``), so turning a stage on or off renumbers and rewires the later ones.

``START_STAGE`` is the first stage to run; enabled stages before it are taken as done, and
their outputs are read from disk. ``START_DIR``, if set, replaces the output of the stages
before ``START_STAGE`` (e.g. frames from a run with different flags). ``CLOBBER = True``
deletes the output directories of the stages that run before running them.

Two stages write no numbered directory of their own: ``train_regions`` writes PCA basis cubes
to ``DIR_CALIB_FRAMES``, and ``crop`` writes the other ``CUTOUT_SIZES`` next to ``register``'s
output (same number, other ``{size}``).

No numpy here, so scripts can import this before setting BLAS thread limits.
"""

import os

STAGES = ["slope", "dark", "badpix", "ramp", "bkg_nod", "bkg_channel", "train_regions",
          "bkg_regions", "fuse", "cutout", "register", "crop"]
# stages that write frames to their own numbered directory
FRAME_STAGES = [s for s in STAGES if s not in ("train_regions", "crop")]

# stage -> stage that must also be enabled
_REQUIRES = {
    "fuse": "bkg_regions",    # fuses bkg_regions' per-region directories
    "register": "cutout",     # registers cutouts, not whole frames
    "crop": "register",       # crops the registered frames, written next to them
}


class StagePlan:
    '''
    The enabled stages, the ones to run now, and their input/output directories.
    '''

    def __init__(self, flags, stems, dir_write, dir_raw, start_stage, start_dir=None,
                 clobber=False, regions=(), sizes=()):
        '''
        Validate the chain and number its frame stages.

        INPUTS:
        - flags (dict): Stage name -> bool, for every stage in ``STAGES``
        - stems (dict): Frame stage name -> directory stem (may contain ``{region}``/``{size}``)
        - dir_write (str): Root of the numbered output directories
        - dir_raw (str): Raw data directory (input of ``slope``)
        - start_stage (str): First stage to run (must be enabled)
        - start_dir (str or None): Replaces the output of the stages before ``start_stage``
        - clobber (bool): Delete the output directories of the stages that run, before running
        - regions (sequence of str): Region names (for ``{region}`` directories)
        - sizes (sequence of int): Cutout sizes; the first is registered, the others cropped

        OUTPUTS:
        - None (raises ValueError if the configuration is inconsistent)
        '''
        unknown = set(flags) - set(STAGES)
        missing = set(STAGES) - set(flags)
        if unknown or missing:
            raise ValueError(f"[pipeline] flags: unknown {sorted(unknown)}, missing {sorted(missing)}; "
                             f"need one True/False flag per stage in {STAGES}")
        self.flags = {s: bool(flags[s]) for s in STAGES}
        self.enabled_stages = [s for s in STAGES if self.flags[s]]
        missing_stems = [s for s in FRAME_STAGES if self.flags[s] and not stems.get(s)]
        if missing_stems:
            raise ValueError(f"[stage_dirs] has no directory name for {missing_stems}")
        self.stems = dict(stems)
        self.dir_write = dir_write
        self.dir_raw = dir_raw
        self.regions = list(regions)
        self.sizes = [int(s) for s in sizes]
        self.clobber = bool(clobber)

        if start_stage not in STAGES:
            raise ValueError(f"START_STAGE {start_stage!r} is not a stage; choose from {STAGES}")
        if not self.flags[start_stage]:
            raise ValueError(f"START_STAGE {start_stage!r} is not enabled in [pipeline]")
        self.start_stage = start_stage
        self.start_dir = start_dir or None
        if self.start_dir is not None and "{" not in self.start_dir and not os.path.isdir(self.start_dir):
            raise ValueError(f"START_DIR does not exist: {self.start_dir}")

        self.numbers = {s: i for i, s in enumerate(s for s in FRAME_STAGES if self.flags[s])}
        self.to_run = [s for s in self.enabled_stages if STAGES.index(s) >= STAGES.index(start_stage)]

        for stage, needed in _REQUIRES.items():
            if self.flags[stage] and not self.flags[needed]:
                raise ValueError(f"stage {stage!r} needs stage {needed!r} to be enabled too")
        # bkg_regions writes one directory per region, which only fuse can read
        if self.flags["bkg_regions"]:
            later = [s for s in FRAME_STAGES[FRAME_STAGES.index("bkg_regions") + 1:] if self.flags[s]]
            if later and later[0] != "fuse":
                raise ValueError("stage 'bkg_regions' must be followed by 'fuse' (stage "
                                 f"{later[0]!r} cannot read per-region directories)")
        for stage in self.to_run:
            self._producer(stage)  # raises if the stage has no input

    @classmethod
    def from_config(cls, config, regions=None):
        '''
        Build the plan from the merged config (``[pipeline]``, ``[stage_dirs]``, ``[data_dirs]``,
        ``[dataset] CUTOUT_SIZES``).

        INPUTS:
        - config (ConfigParser): Merged base + dataset config
        - regions (sequence of str or None): Region names; None reads them from the config
          (``modules.shared.dataset.parse_regions``)

        OUTPUTS:
        - plan (StagePlan): The plan
        '''
        sec = config["pipeline"]
        reserved = {"start_stage", "start_dir", "clobber"}
        flags = {k: sec.getboolean(k) for k in sec if k not in reserved}
        stems = {k: v.strip() for k, v in config["stage_dirs"].items()} if config.has_section("stage_dirs") else {}
        if regions is None:
            from modules.shared.dataset import parse_regions
            regions = list(parse_regions(config))
        sizes = [int(s) for s in config["dataset"]["CUTOUT_SIZES"].split(",")]
        return cls(flags, stems,
                   dir_write=config["data_dirs"]["DIR_WRITE"],
                   dir_raw=config["data_dirs"]["DIR_RAW_DATA"],
                   start_stage=sec.get("START_STAGE", STAGES[0]).strip(),
                   # %(DIR_HOME)s etc. in START_DIR resolve against [data_dirs]
                   start_dir=sec.get("START_DIR", "", vars=dict(config["data_dirs"])).strip() or None,
                   clobber=sec.getboolean("CLOBBER", fallback=False),
                   regions=regions, sizes=sizes)

    def enabled(self, stage):
        '''
        Whether ``stage`` is in the chain.

        INPUTS:
        - stage (str): Stage name

        OUTPUTS:
        - enabled (bool): Its ``[pipeline]`` flag
        '''
        return self.flags[stage]

    def _producer(self, stage):
        '''
        Where ``stage`` reads its frames from.

        INPUTS:
        - stage (str): Stage name

        OUTPUTS:
        - producer (str): The previous enabled frame stage, ``"START_DIR"`` if that stage is
          before ``START_STAGE`` and ``START_DIR`` is set (or there is no previous stage), or
          ``"RAW"`` for ``slope`` without ``START_DIR``. Raises ValueError if there is no input.
        '''
        i = STAGES.index(stage)
        prev = [s for s in FRAME_STAGES if self.flags[s] and STAGES.index(s) < i]
        prev = prev[-1] if prev else None
        if self.start_dir is not None and (prev is None or STAGES.index(prev) < STAGES.index(self.start_stage)):
            return "START_DIR"
        if prev is not None:
            return prev
        if stage == "slope":
            return "RAW"
        raise ValueError(f"stage {stage!r} has no enabled stage before it to read from; "
                         "enable one, or set START_DIR")

    def out_dir(self, stage, **fmt):
        '''
        Output directory of a stage.

        INPUTS:
        - stage (str): Frame stage name, or ``crop`` (written next to ``register``'s output)
        - **fmt: ``region=`` / ``size=`` for stems with ``{region}`` / ``{size}``

        OUTPUTS:
        - path (str): ``DIR_WRITE/<NN>_<stem>/`` (with trailing slash)
        '''
        if stage == "crop":
            stage = "register"
        if stage not in self.numbers:
            raise ValueError(f"stage {stage!r} is not an enabled frame stage")
        name = f"{self.numbers[stage]:02d}_" + self.stems[stage].format(**fmt)
        return os.path.join(self.dir_write, name, "")

    def in_dir(self, stage, **fmt):
        '''
        Input directory of a stage.

        INPUTS:
        - stage (str): Stage name
        - **fmt: ``region=`` / ``size=`` for directories with ``{region}`` / ``{size}``
          (e.g. ``fuse`` reads one directory per region; ``crop`` reads the first size)

        OUTPUTS:
        - path (str): The directory
        '''
        producer = self._producer(stage)
        if producer == "START_DIR":
            return self.start_dir.format(**fmt)
        if producer == "RAW":
            return self.dir_raw
        return self.out_dir(producer, **fmt)

    def stage_out_dirs(self, stage):
        '''
        Every output directory of a stage (all regions / sizes); none for ``train_regions``.

        INPUTS:
        - stage (str): Stage name

        OUTPUTS:
        - dirs (list of str): The directories
        '''
        if stage == "train_regions":
            return []
        if stage == "bkg_regions":
            return [self.out_dir(stage, region=r) for r in self.regions]
        if stage == "register":
            return [self.out_dir(stage, size=self.sizes[0])]
        if stage == "crop":
            return [self.out_dir(stage, size=s) for s in self.sizes[1:]]
        if stage == "cutout":
            return [self.out_dir(stage, size=self.sizes[0])]
        return [self.out_dir(stage)]

    def clobber_dirs(self):
        '''
        The output directories of the stages that run (what ``CLOBBER`` deletes). Never
        includes the calibration or other-FITS directories (PCA cubes are kept).

        INPUTS:
        - None

        OUTPUTS:
        - dirs (list of str): The directories (whether or not they exist). Raises ValueError if
          one is outside ``DIR_WRITE`` or is an input of a stage that runs.
        '''
        dirs = [d for s in self.to_run for d in self.stage_out_dirs(s)]

        # inputs that this run reads but does not produce itself (START_DIR, raw data, or
        # outputs of enabled stages before START_STAGE): these must survive the clobber
        external = set()
        for s in self.to_run:
            if self._producer(s) in self.to_run:
                continue
            if s == "fuse":
                external.update(os.path.realpath(self.in_dir(s, region=r)) for r in self.regions)
            else:
                external.add(os.path.realpath(self.in_dir(s, size=self.sizes[0] if self.sizes else 0)))

        root = os.path.realpath(self.dir_write)
        for d in dirs:
            real = os.path.realpath(d)
            if real == root or os.path.commonpath([real, root]) != root:
                raise ValueError(f"refusing to clobber {d}: not inside DIR_WRITE ({self.dir_write})")
            if real in external:
                raise ValueError(f"refusing to clobber {d}: it is an input of this run (START_DIR?)")
        return dirs

    def describe(self):
        '''
        A table of the enabled stages, their input and output directories, and whether they run.

        INPUTS:
        - None

        OUTPUTS:
        - text (str): Multi-line table
        '''
        lines = [f"{'stage':<14}{'run':<6}reads  ->  writes"]
        for s in self.enabled_stages:
            fmt = dict(region="{region}", size=self.sizes[0] if self.sizes else "{size}")
            if s not in self.to_run:
                src = "(done earlier)"
            else:
                src = self.in_dir(s, **fmt)
            if s == "train_regions":
                dst = "region PCA cubes in DIR_CALIB_FRAMES"
            elif s == "crop":
                dst = ", ".join(self.stage_out_dirs(s)) or "(no other sizes)"
            else:
                dst = self.out_dir(s, **fmt)
            lines.append(f"{s:<14}{'yes' if s in self.to_run else 'done':<6}{src}  ->  {dst}")
        if self.clobber:
            lines.append("CLOBBER = True: the output directories of the 'yes' stages are deleted first")
        return "\n".join(lines)
