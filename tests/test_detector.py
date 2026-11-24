"""Tests of the readout-channel layout from the config (modules/preprocessing/detector.py)."""
import numpy as np
import pytest

from modules.preprocessing.detector import channel_layout, channels_PCA_cube
from modules.shared.dataset import load_dataset_config


def test_channel_layout_from_dataset_config():
    assert channel_layout(load_dataset_config("procyon_240131")) == (32, 64)


def test_channels_pca_cube():
    cube = channels_PCA_cube((5, 12), n_channels=3, channel_width=4)
    assert cube.shape == (3, 5, 12)
    # each column belongs to exactly one channel; channel k covers columns 4k..4k+3
    np.testing.assert_array_equal(cube.sum(axis=0), 1.)
    np.testing.assert_array_equal(cube[1, :, 4:8], 1.)
    assert cube[1, :, :4].sum() == 0 and cube[1, :, 8:].sum() == 0


def test_channels_pca_cube_width_mismatch():
    with pytest.raises(ValueError, match="detector"):
        channels_PCA_cube((5, 12), n_channels=3, channel_width=5)
