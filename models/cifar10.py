import keras
from keras.callbacks import History
from keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential


class Cifar10(Sequential):
    """A simple architecture copying the default's CIFAR10.

    """

    def __init__(self, input_shape=(), n_classes=2, lr=0.0001):
        """Initialization methods.

        Args:
            input_shape (tuple): The input shape of the network.
            n_classes (int): Number of classes.
            lr (float): Learning rate.

        """

        # Overriding its parent class
        super(Cifar10, self).__init__()

        # Defining the model itself
        # Conv + Max Pool block
        self.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
        self.add(Activation('relu'))
        self.add(Conv2D(32, (3, 3)))
        self.add(Activation('relu'))
        self.add(MaxPooling2D(pool_size=(2, 2)))
        self.add(Dropout(0.25))

        # Conv + Max Pool block
        self.add(Conv2D(64, (3, 3), padding='same'))
        self.add(Activation('relu'))
        self.add(Conv2D(64, (3, 3)))
        self.add(Activation('relu'))
        self.add(MaxPooling2D(pool_size=(2, 2)))
        self.add(Dropout(0.25))
        self.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # Flattenning the arrays
        self.add(Flatten())

        # Performing the last fully connections
        self.add(Dense(512, activation='relu'))
        self.add(Dropout(0.5))
        self.add(Dense(n_classes, activation='softmax'))

        # Compiling the model
        self.compile(loss=keras.losses.categorical_crossentropy,
                     optimizer=keras.optimizers.Adam(lr), metrics=['accuracy'])
