import torch


def print_rank_0(*args, **kwargs):
    if torch.distributed.get_rank() == 0:
        print(*args, **kwargs)
