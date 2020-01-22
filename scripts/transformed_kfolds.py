import keras
import numpy as np
from sklearn.model_selection import StratifiedKFold

from datasets.transformed import TransformedDataset
from models.lenet import Lenet

# Number of persons to load the data
N_PERSONS = 26

# Identifier of tests to be loaded
ID_TESTS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Size of the transform
N_TRANSFORM = 128

# Number of signals' channels
N_CHANNELS = 6

# Number of classes
N_CLASSES = 26

# Loads the SignRec dataset
d = TransformedDataset(name='signrec', n_persons=N_PERSONS, id_tests=ID_TESTS,
                       transform_size=N_TRANSFORM, n_channels=N_CHANNELS)

# Re-define labels for Parkinson's identification
# d.y[:len(ID_TESTS)*35] = 0
# d.y[len(ID_TESTS)*35:] = 1

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

# Printing output
print(f'Loss: {np.mean(loss)} +- {np.std(loss)} | Accuracy: {np.mean(accuracy)} +- {np.std(accuracy)}')
