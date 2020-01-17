import os
import shutil
import time

import slugify
import torch

__all__ = ['generate_root',
           'save_checkpoint',
           'WorkingDirectory',
           'report_loss',
           'RunningAverage']


def generate_root(name):
    """Generate a root path.

    Args:
        name (str): Name of the experiment.

    Returns:

    """
    now = time.strftime('%Y-%m-%d_%H-%M-%S')
    return os.path.join('_experiments', f'{now}_{slugify.slugify(name)}')


def save_checkpoint(wd, state, is_best):
    """Save a checkpoint.

    Args:
        wd (:class:`.experiment.WorkingDirectory`): Working directory.
        state (dict): State to save.
        is_best (bool): This model is the best so far.
    """
    fn = wd.file('checkpoint.pth.tar')
    torch.save(state, fn)
    if is_best:
        fn_best = wd.file('model_best.pth.tar')
        shutil.copyfile(fn, fn_best)


class WorkingDirectory:
    """Working directory.

    Args:
        root (str): Root of working directory.
        override (bool, optional): Delete working directory if it already
            exists. Defaults to `False`.
    """

    def __init__(self, root, override=False):
        self.root = root

        # Delete if the root already exists.
        if os.path.exists(self.root) and override:
            print('Experiment directory already exists. Overwriting.')
            shutil.rmtree(self.root)

        print('Root:', self.root)

        # Create root directory.
        os.makedirs(self.root, exist_ok=True)

    def file(self, *name, exists=False):
        """Get the path of a file.

        Args:
            *name (str): Path to file, relative to the root directory. Use
                different arguments for directories.
            exists (bool): Assert that the file already exists. Defaults to
                `False`.

        Returns:
            str: Path to file.
        """
        path = os.path.join(self.root, *name)

        # Ensure that path exists.
        if exists and not os.path.exists(path):
            raise AssertionError('File "{}" does not exist.'.format(path))
        elif not exists:
            path_dir = os.path.join(self.root, *name[:-1])
            os.makedirs(path_dir, exist_ok=True)

        return path


def report_loss(name, loss, step, freq=1):
    """Print loss.

    Args:
        name (str): Name of loss.
        loss (float): Loss value.
        step (int or str): Step or name of step.
        freq (int, optional): If `step` is an integer, this specifies the
            frequency at which the loss should be printed. If `step` is a
            string, the loss is always printed.
    """
    if isinstance(step, int):
        if step == 0 or (step + 1) % freq == 0:
            print('{name:15s} {step:5d}: {loss:.3e}'
                  ''.format(name=name, step=step + 1, loss=loss))
    else:
        print('{name:15s} {step:>5s}: {loss:.3e}'
              ''.format(name=name, step=step, loss=loss))


class RunningAverage:
    """Maintain a running average."""

    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def reset(self):
        """Reset the running average."""
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        """Update the running average.

        Args:
            val (float): Value to update with.
            n (int): Number elements used to compute `val`.
        """
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt
