import pywt
import numpy as np

WAVELET = 'morl'
SCALES = range(1, 128)

def cwt(signal, n_channels):
    """
    """

    transformed_signal = np.zeros((127, 127, n_channels))

    for i in range(6):
        c, _ = pywt.cwt(signal[:,i], SCALES, WAVELET, 1)
        transformed_signal[:,:,i] = c[:,:127]


    return transformed_signal