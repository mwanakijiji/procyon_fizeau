"""
Preprocessing stage: slope (single-read) frames, dark subtraction, bad-pixel fixing, ramp removal,
PCA background subtraction, cutouts and registration.

Import explicitly, e.g.::

    from modules.preprocessing import make_slope_frames
    from modules.preprocessing import dark_subtract
    from modules.preprocessing.background_pca import FuseRegions
"""

__all__ = [
    'make_slope_frames',
    'dark_subtract',
    'fix_bad_pix',
    'ramp_removal',
    'detector',
    'background_pca',
    'frame_io',
    'star',
    'nod_subtract',
    'region_pca',
    'cutouts',
    'registration',
    'crop',
    'pipeline',
]
