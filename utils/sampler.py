import numpy as np


def sample_signal(signal, n_samples, n_channels):
    print(f'Sampling signal with {n_samples} samples ...')

    # Gathering the signal shape
    size = signal.shape[0]

    # Calculating its step
    step = int(size / n_samples)

    # Creating an empty array to the sampled signal
    sampled_signal = np.zeros((n_samples, n_channels))

    # Initializing the counter
    c = 0

    # For every step
    for i in range(0, size, step):
        # Gathers the mean of neighbours
        sampled_signal[c, :] = np.mean(signal[i:i+step-1], axis=0)

        # Increments the counter
        c += 1

        # If the counter reaches the number of samples
        if c == n_samples:
            # Breaks the loop
            break

    return sampled_signal
