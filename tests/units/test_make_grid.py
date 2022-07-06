import numpy as np

from megdiffusion.utils.vision import make_grid

def test_make_grid_input_convert():
    image = np.random.random((5, 5))
    grid = make_grid(image)
    

def test_make_grid_shape():
    image = np.random.random((32, 3, 5, 5))
    grid = make_grid(image)
    assert grid.shape == (3, 30, 58)
