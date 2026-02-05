import h5py
import mat73
from matplotlib.colors import hsv_to_rgb
import numpy as np
import scipy.io
import torch


def _keep_keys(dictionary: dict, keys_to_keep: list[str]) -> dict:
    if keys_to_keep is None:
        return dictionary
    else:
        return {key: value for key, value in dictionary.items() if key in keys_to_keep}


def load_matfile(filepath: str, keys: list[str] = None) -> dict:
    try:
        loaded_data = scipy.io.loadmat(filepath)
    except NotImplementedError:
        loaded_data = mat73.loadmat(filepath)
    loaded_data = _keep_keys(loaded_data.copy(), keys)
    return loaded_data


def load_h5file(filepath: str, keys: list[str] = None) -> dict:
    with h5py.File(filepath, 'r') as hf:
        loaded_data = {}
        for key in hf.keys():
            loaded_data[key] = np.array(hf[key])
    return _keep_keys(loaded_data, keys)


def real_gaussian_noise(tensor, sigma: float, mu: float = 0.0):
    if isinstance(tensor, torch.Tensor):
        return torch.randn_like(tensor) * sigma + mu
    elif isinstance(tensor, np.ndarray):
        return np.random.randn(*tensor.shape) * sigma + mu
    else:
        raise TypeError("Input must be a torch.Tensor or np.ndarray")


def complex_to_rgb(z, max=None, exponent: float = 1):
    """Map complex value to RGB using phase as hue, power as brightness."""
    phase = np.angle(z)  # [-π, π]
    ampl = np.power(np.abs(z), exponent)

    if max is None:
        max_ampl = np.max(ampl)

    ampl = np.clip(ampl, 0, max_ampl)

    hue = (phase + np.pi) / (2 * np.pi)  # [0, 1]
    sat = np.ones_like(hue)  # full saturation
    val = ampl / max_ampl

    return hsv_to_rgb(np.stack([hue, sat, val], axis=-1))