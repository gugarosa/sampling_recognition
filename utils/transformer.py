import numpy as np
import pywt


def cwt(signal, transform_size, n_channels, wavelet='morl'):
    """Transforms a signal using Continuous Wavelet Transform.

    Args:
        signal (np.array): Pre-loaded signal.
        transform_size (int): Size of the transform.
        n_channels (int): Number of signal's channels.
        wavelet (str): Name of the wavelet.

    Returns:
        A numpy array holding the transformed signal.

    """

    print(
        f'Transforming signal using CWT with {wavelet} and size {transform_size} ...')

    # Creates an empty array to hold the transformed signal
    transformed_signal = np.zeros((transform_size, transform_size, n_channels))

    # Creates a range of scales
    scales = range(1, transform_size+1)

    # For every channel
    for i in range(n_channels):
        # Apply the corresponding CWT
        c, _ = pywt.cwt(signal[:, i], scales, wavelet, 1)

        # Adds the resulting coefficients to the array
        transformed_signal[:, :, i] = c[:, :transform_size]

    return transformed_signal
