import math

import cv2
import numpy as np

def make_grid(inp: np.ndarray, nrow: int = 8, padding: int = 2, pad_value: float = 0.0):
    """Make a grid of images. Port from torchvision.utils.make_grid implementation."""
    # convert to (N, C, H, W) with 3 channels
    if inp.ndim == 2:  # single image H x W
        inp = np.expand_dims(inp, axis=0)
    if inp.ndim == 3:  # single image C x H x W
        if inp.shape[0] == 1:  # if single-channel, convert to 3-channel
            inp = np.concatenate((inp, inp, inp), axis=0)
        inp = np.expand_dims(inp, axis=0)
    if inp.ndim == 4 and inp.shape[1] == 1:  # single-channel images
        inp = np.concatenate((inp, inp, inp), axis=1)

    assert inp.ndim == 4

    # make the mini-batch of images into a grid
    N, C, H, W = inp.shape
    nmaps = N
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(H + padding), int(W + padding)
    num_channels = C
    grid = np.full((num_channels, height * ymaps + padding, width * xmaps + padding), pad_value)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            grid[:,y * height + padding: (y+1) * height,  x * width + padding: (x+1) * width] = inp[k]
            k = k + 1
    return grid.astype("uint8")

def save_image(inp: np.ndarray, path, order: str = "bgr"):
    order = order.lower()
    assert order in ["rgb", "bgr"]
    
    inp = inp.transpose(1, 2, 0) # CHW to HWC
    if order == "rgb":
        inp = cv2.cvtColor(inp, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, inp) 


