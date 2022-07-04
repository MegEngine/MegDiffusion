import numpy as np

def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in target_dict.keys():
        target_dict[key] = np.array(target_dict[key] * decay + source_dict[key] * (1 - decay), dtype="float32")
    target.load_state_dict(target_dict)