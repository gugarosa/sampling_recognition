import numpy as np


def load_signal(path, delimiter):
    print(f'Loading signal from: {path} ...')

    # Loading a single signal file
    signal = np.loadtxt(path, delimiter=delimiter)

    return signal
