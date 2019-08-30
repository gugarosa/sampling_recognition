import keras
import numpy as np
from keras.callbacks import History
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
from sklearn import model_selection

from utils.datasets import Dataset

N_PERSONS = 66
ID_TESTS = [9, 10, 11, 12]

# Loads the HandPD dataset
d = Dataset(name='handpd', n_persons=N_PERSONS, id_tests=ID_TESTS, n_samples=256, n_channels=6)

# Re-shaping data
d.x = np.reshape(d.x, (N_PERSONS*len(ID_TESTS), 16, 16, 6))

# Re-define labels for Parkinson's identification
d.y[:len(ID_TESTS)*35] = 0
d.y[len(ID_TESTS)*35:] = 1

#
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(d.x, d.y, test_size=0.5, random_state=5, stratify=d.y)

history = History()

img_x = 16
img_y = 16
img_z = 6
input_shape = (img_x, img_y, img_z)

batch_size = 16
num_classes = 2
epochs = 300
 
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
 
Y_train = keras.utils.to_categorical(Y_train, num_classes)
Y_test = keras.utils.to_categorical(Y_test, num_classes)
 
 
model = Sequential()
model.add(Conv2D(20, kernel_size=(5, 5), strides=(1, 1),
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(50, (5, 5)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
 
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=0.0001),
              metrics=['accuracy'])
 
model.fit(X_train, Y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test, Y_test),
          callbacks=[history])