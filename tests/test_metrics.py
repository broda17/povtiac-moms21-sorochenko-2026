import math

import numpy as np
import pytest

from bioprep.metrics import psnr


def test_psnr_identical_images_is_infinite():
    img = np.random.default_rng(0).random((8, 8))
    assert psnr(img, img) == math.inf


def test_psnr_known_value():
    ref = np.zeros((4, 4))
    res = np.full((4, 4), 0.1)  # MSE = 0.01 -> PSNR = 20 дБ
    assert psnr(ref, res) == pytest.approx(20.0)


def test_psnr_shape_mismatch():
    with pytest.raises(ValueError):
        psnr(np.zeros((2, 2)), np.zeros((3, 3)))
