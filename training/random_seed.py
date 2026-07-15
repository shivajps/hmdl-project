"""
Seed control for reproducibility (manuscript Section 4.3.4 / 5.3).
Reported runs use seeds {7, 42, 123, 2024, 31415}, two shuffles per seed = 10 runs.
"""
import random
import numpy as np
import torch

SEEDS = [7, 42, 123, 2024, 31415]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False