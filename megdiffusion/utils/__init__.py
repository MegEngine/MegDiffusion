def batch_broadcast(inp, shape):
    """Convert 1d (t,) Tensor to shape (t, 1, 1, ...)."""
    assert len(inp) == shape[0]
    return inp.reshape(-1, *( (1,) * (len(shape) - 1) ))