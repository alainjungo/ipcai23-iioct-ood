import os
import glob
import re


def get_checkpoint_dir(parent_dir):
    return os.path.join(parent_dir, 'checkpoints')


def get_checkpoint_path(checkpoint_dir: str, epoch: int, best_in_metric: str = None):
    if best_in_metric is not None:
        return os.path.join(checkpoint_dir, f'checkpoint_best-{best_in_metric}_ep{epoch:04}.pth')
    return os.path.join(checkpoint_dir, f'checkpoint_ep{epoch:04}.pth')


def get_checkpoints(checkpoint_dir: str):
    exp = re.compile(r'checkpoint_ep(\d*).pth')

    checkpoint_paths = glob.glob(checkpoint_dir + '/checkpoint_ep*.pth')
    epoch_dict = {}
    for checkpoint_path in checkpoint_paths:
        epoch = int(exp.match(os.path.basename(checkpoint_path)).groups()[0])
        epoch_dict[epoch] = checkpoint_path

    return epoch_dict


def get_last_checkpoint(checkpoint_dir: str):
    epoch_dict = get_checkpoints(checkpoint_dir)
    max_epoch = max(k for k in epoch_dict)
    return epoch_dict[max_epoch]


def get_best_checkpoints(checkpoint_dir: str):
    exp = re.compile(r'checkpoint_best-(.*)_ep(\d*).pth')

    best_paths = glob.glob(checkpoint_dir + '/checkpoint_best-*ep*.pth')
    best_dict = {}
    for best_path in best_paths:
        metric, epoch_str = exp.match(os.path.basename(best_path)).groups()
        best_dict[metric] = best_path, int(epoch_str)

    return best_dict
