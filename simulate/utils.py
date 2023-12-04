
import numpy as np
import torch

def np2tensor(*args):
    return tuple(map(lambda x: torch.from_numpy(x) if type(x) == np.ndarray else x, args))

