from megengine import Tensor

def batch_broadcast(inp, shape):
    """Convert 1d (t,) Tensor to shape (t, 1, 1, ...)."""
    assert len(inp) == shape[0]
    return inp.reshape(-1, *( (1,) * (len(shape) - 1) ))

def mean_flat(tensor: Tensor):
    """Take the mean over all non-batch dimensions."""
    return tensor.mean(axis=list(range(1, len(tensor.shape))))