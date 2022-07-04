import numpy as np

from megdiffusion.utils.vision import make_grid

def test_make_grid():
    image = np.random.random((32, 3, 5, 5))
    grid = make_grid(image)
    assert grid.shape == (3, 30, 58)
