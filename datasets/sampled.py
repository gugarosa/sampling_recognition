import numpy as np

import utils.loader as l
import utils.sampler as s

DATASETS = ['signrec', 'handpd']


class SampledDataset():
    """A SampledDataset class to initialize the desired signal's datasets.

    Properties:
        x (np.array): SampledDataset's data.
        y (np.array): SampledDataset's labels.

    """

    def __init__(self, name='', n_persons=1, id_tests=[1], n_samples=128, n_channels=6):
        """Initalization method.

        Args:
            name (str): The dataset's name.
            n_persons (int): Number of dataset's persons.
            id_tests (list): List of the tests identifiers.
            n_samples (int): Number of samples per signal.
            n_channels (int): Number of channels per signal.

        """

        # Checks if name belongs to a known dataset
        if name not in DATASETS:
            # If not, raises an error
            raise NameError(f'Dataset name should be in {DATASETS}.')

        # If it is supposed to load the signec dataset
        if name == 'signrec':
            # Loads using the space delimiter
            self.x, self.y = _load_dataset(
                'signrec', n_persons, id_tests, n_samples, n_channels, delimiter=' ')

        # If it is supposed to load the handpd dataset
        elif name == 'handpd':
            # Loads using the tab delimiter
            self.x, self.y = _load_dataset(
                'handpd', n_persons, id_tests, n_samples, n_channels, delimiter='\t')


def _load_dataset(name, n_persons, id_tests, n_samples, n_channels, delimiter):
    """Loads a specific dataset.

    Args:
        name (str): The dataset's name.
        n_persons (int): Number of dataset's persons.
        id_tests (list): List of the tests identifiers.
        n_samples (int): Number of samples per signal.
        n_channels (int): Number of channels per signal.
        delimiter (str): A delimiter character, e.g., ' ' or '\t'.

    Returns:
        The dataset's data and labels.

    """

    print(f'Loading dataset: {name} ...\n')

    # Creating an empty array to hold the data
    x = np.zeros((n_persons*len(id_tests), n_samples, n_channels))

    # Creating an empty list to hold the labels
    y = []

    # For every person
    for i in range(n_persons):
        # For every test
        for j, test in enumerate(id_tests):
            # Loads the signal
            signal = l.load_signal(f'data/{name}/{i+1}/{test}.txt', delimiter)

            # Samples the signal
            sampled_signal = s.sample_signal(signal, n_samples, n_channels)

            # Saves to array
            x[i*len(id_tests)+j, :, :] = sampled_signal

            # Also, creates the label
            y.append(i)

    # Transforms the list into array
    y = np.asarray(y)

    print(f'\nDataset loaded.')
    print(f'X: {x.shape} | Y: {y.shape}')

    return x, y
