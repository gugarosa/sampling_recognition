import numpy as np


def load_signal(path, delimiter):
    """Loads a signal from a specified file.

    Args:
        path (str): The signal's path to be loaded.
        delimiter (str): A delimiter character, e.g., ' ' or '\t'.

    Returns:
        The loaded signal in a numpy array.

    """

    print(f'Loading signal from: {path} ...')

    # Loading a single signal file
    signal = np.loadtxt(path, delimiter=delimiter)

    return signal
