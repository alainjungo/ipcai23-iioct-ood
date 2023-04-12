import random

import torch
import numpy as np


def do_seed(seed: int, with_cudnn: bool):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed) -> no longer required since 1.3.1

    # makes it perfectly deterministic but slower (without is already very good)
    if with_cudnn:
        torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    # This function is to be used as 'worker_init_fn' argument to the pytorch dataloader
    # Needed because the numpy seed is shared among the workers
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)






