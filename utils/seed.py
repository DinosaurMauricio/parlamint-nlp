import torch
import random
import numpy as np


def set_seed(seed):
    # Set seed for PyTorch
    torch.manual_seed(seed)

    # Set seed for CUDA
    torch.cuda.manual_seed(seed)

    # Set seed for Python's random module
    random.seed(seed)

    # Set seed for NumPy
    np.random.seed(seed)

    # Ensure deterministic behavior for PyTorch operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
