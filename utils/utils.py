import random
import subprocess
import os
import torch
from torch import nn
import numpy as np
from scipy.io.wavfile import read
import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt
import numpy as np


def save_figure_to_numpy(fig):
    # save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data

def plot_spectrogram_to_numpy(spectrogram):
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data

def num_params(model, print_out=True):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    if print_out:
        print('Trainable Parameters: %.3fM' % parameters)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_commit_hash():
    message = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
    return message.strip().decode('utf-8')


def read_wav_np(path):
    sr, wav = read(path)

    if len(wav.shape) == 2:
        wav = wav[:, 0]

    if wav.dtype == np.int16:
        wav = wav / 32768.0
    elif wav.dtype == np.int32:
        wav = wav / 2147483648.0
    elif wav.dtype == np.uint8:
        wav = (wav - 128) / 128.0

    wav = wav.astype(np.float32)

    return sr, wav


def remove_weight_norms(m):
    if hasattr(m, 'weight_g'):
        nn.utils.remove_weight_norm(m)


def add_weight_norms(m):
    if hasattr(m, 'weight'):
        nn.utils.weight_norm(m)


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def set_deterministic_pytorch():
    """Ensures pytorch produces deterministic results depending on the program arguments
    :param Namespace args: The program arguments
    """

    # debug mode setting
    # 0 would be fastest, but 1 seems to be reasonable
    # considering reproducibility
    # remove type check
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = (
        False  # https://github.com/pytorch/pytorch/issues/6351
    )