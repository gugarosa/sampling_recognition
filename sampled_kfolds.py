import pickle
import time

import keras
import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import StratifiedKFold

from datasets.sampled import SampledDataset
from models.lenet import Lenet

# Number of persons to load the data
N_PERSONS = 66

# Identifier of tests to be loaded
ID_TESTS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

# Number of samples for further signal sampling
N_SAMPLES = 64

# Number of signals' channels
N_CHANNELS = 6

# Number of classes
N_CLASSES = 66

# Defining the output file
OUTPUT_FILE = 'lenet_' + str(N_SAMPLES) + '.pkl'

# Loads the HandPD dataset
d = SampledDataset(name='handpd', n_persons=N_PERSONS, id_tests=ID_TESTS,
            n_samples=N_SAMPLES, n_channels=N_CHANNELS)

# Re-shapes data
d.x = np.reshape(d.x, (N_PERSONS*len(ID_TESTS),
                       int(np.sqrt(N_SAMPLES)), int(np.sqrt(N_SAMPLES)), N_CHANNELS))

# Creates the input shape
input_shape = (d.x.shape[1], d.x.shape[2], d.x.shape[3])

# Creating a K-Folds cross-validation
k_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

# Creating metrics lists
train_loss = []
test_loss = []
test_accuracy = []
test_precision = []
test_recall = []
test_f1 = []
train_time = []

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

    # Starting the timer
    start = time.time()

    # Fits the model
    history = model.fit(X_train, Y_train, batch_size=16, epochs=300, verbose=1)

    # Ending the timer
    end = time.time()

    # Evaluates the model
    score = model.evaluate(X_test, Y_test)

    # Predicts with the model
    preds = model.predict(X_test)

    # Transform the predictions into categorical labels
    preds = keras.utils.to_categorical(np.argmax(preds, axis=1), N_CLASSES)

    # Calculating metrics
    t_accuracy = accuracy_score(preds, Y_test)
    t_precision = precision_score(preds, Y_test, average='macro')
    t_recall = recall_score(preds, Y_test, average='macro')
    t_f1 = f1_score(preds, Y_test, average='macro')

    # Appending metrics
    train_loss.append(history.history['loss'])
    train_time.append(end - start)
    test_loss.append(score[0])
    test_accuracy.append(t_accuracy)
    test_precision.append(t_precision)
    test_recall.append(t_recall)
    test_f1.append(t_f1)

# Opening file
with open(OUTPUT_FILE, 'wb') as f:
    # Saving output to pickle
    pickle.dump({
        'train_loss': train_loss,
        'train_time': train_time,
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1
    }, f)
