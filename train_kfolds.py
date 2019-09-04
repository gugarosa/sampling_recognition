import keras
import numpy as np
from sklearn.model_selection import StratifiedKFold

from models.lenet import Lenet
from utils import plotter as p
from utils.dataset import Dataset

# Number of persons to load the data
N_PERSONS = 66

# Identifier of tests to be loaded
ID_TESTS = [9, 10, 11, 12]

# Number of samples for further signal sampling
N_SAMPLES = 256

# Number of signals' channels
N_CHANNELS = 6

# Number of classes
N_CLASSES = 2

# Loads the HandPD dataset
d = Dataset(name='handpd', n_persons=N_PERSONS, id_tests=ID_TESTS,
            n_samples=N_SAMPLES, n_channels=N_CHANNELS)

# Re-shapes data
d.x = np.reshape(d.x, (N_PERSONS*len(ID_TESTS),
                       int(np.sqrt(N_SAMPLES)), int(np.sqrt(N_SAMPLES)), N_CHANNELS))

# Re-define labels for Parkinson's identification
d.y[:len(ID_TESTS)*35] = 0
d.y[len(ID_TESTS)*35:] = 1

# Creates the input shape
input_shape = (d.x.shape[1], d.x.shape[2], d.x.shape[3])

# Creating a K-Folds cross-validation
k_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=5)

# Creating a history and a score list
history = []
loss = []
accuracy = []

# Iterating through every possible fold
for train, test in k_fold.split(d.x, d.y):
    # Pre-processing the training data
    X_train = d.x[train].astype('float32')
    Y_train = keras.utils.to_categorical(d.y[train], N_CLASSES)

    # Pre-processing the testing data
    X_test = d.x[test].astype('float32')
    Y_test = keras.utils.to_categorical(d.y[test], N_CLASSES)

    # Initializes the corresponding model
    model = Lenet(input_shape=input_shape, n_classes=N_CLASSES, lr=0.0001)

    # Fits the model
    history.append(model.fit(X_train, Y_train,
                             batch_size=16, epochs=300, verbose=1))

    # Evaluates the model
    score = model.evaluate(X_test, Y_test)

    # Appending metrics
    loss.append(score[0])
    accuracy.append(score[1])

# Plotting last iteration results
# p.plot_accuracy(history[-1], validation=False)
# p.plot_loss(history[-1], validation=False)

# Printing output
print(f'Loss: {np.mean(loss)} +- {np.std(loss)} | Accuracy: {np.mean(accuracy)} +- {np.std(accuracy)}')
