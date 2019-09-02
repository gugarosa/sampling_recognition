import keras
import numpy as np
from sklearn import model_selection

from models.lenet import Lenet
from utils.dataset import Dataset

# Number of persons to load the data
N_PERSONS = 2

# Identifier of tests to be loaded
ID_TESTS = [9, 10, 11, 12]

# Number of samples for further signal sampling
N_SAMPLES = 256

# Number of signals' channels
N_CHANNELS = 6

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

#
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(
    d.x, d.y, test_size=0.5, random_state=5, stratify=d.y)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

Y_train = keras.utils.to_categorical(Y_train, 2)
Y_test = keras.utils.to_categorical(Y_test, 2)


# Initializes the corresponding model
model = Lenet(input_shape=input_shape, n_classes=2, lr=0.0001)

# Fits the model
model.fit(X_train, Y_train, batch_size=16, epochs=300, verbose=1,
          validation_data=(X_test, Y_test), callbacks=[model.history])
