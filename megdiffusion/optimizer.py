import megengine.optimizer as optim

def build_optimizer(type: str, params, **kwargs):

    mapping = {
        "sgd": optim.SGD,
        "adam": optim.Adam,
    }

    optimizer: optim.Optimizer = mapping[type.lower()]

    return optimizer(params, **kwargs)