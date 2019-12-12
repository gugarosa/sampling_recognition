import keras
from keras.callbacks import History
from keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.models import Sequential


class Alexnet(Sequential):
    """A simple architecture copying the default's AlexNet.

    """

    def __init__(self, input_shape=(), n_classes=2, lr=0.0001):
        """Initialization methods.

        Args:
            input_shape (tuple): The input shape of the network.
            n_classes (int): Number of classes.
            lr (float): Learning rate.

        """

        # Overriding its parent class
        super(Alexnet, self).__init__()

        self.add(Conv2D(96, (11, 11), input_shape=input_shape, padding='same'))
        self.add(BatchNormalization())
        self.add(Activation('relu'))
        self.add(MaxPooling2D(pool_size=(2, 2)))

        # Layer 2
        self.add(Conv2D(256, (5, 5), padding='same'))
        self.add(BatchNormalization())
        self.add(Activation('relu'))
        self.add(MaxPooling2D(pool_size=(2, 2)))

        # Layer 3
        self.add(ZeroPadding2D((1, 1)))
        self.add(Conv2D(512, (3, 3), padding='same'))
        self.add(BatchNormalization())
        self.add(Activation('relu'))
        self.add(MaxPooling2D(pool_size=(2, 2)))

        # Layer 4
        self.add(ZeroPadding2D((1, 1)))
        self.add(Conv2D(1024, (3, 3), padding='same'))
        self.add(BatchNormalization())
        self.add(Activation('relu'))

        # Layer 5
        self.add(ZeroPadding2D((1, 1)))
        self.add(Conv2D(1024, (3, 3), padding='same'))
        self.add(BatchNormalization())
        self.add(Activation('relu'))
        self.add(MaxPooling2D(pool_size=(2, 2)))

        # Layer 6
        self.add(Flatten())
        self.add(Dense(3072))
        self.add(BatchNormalization())
        self.add(Activation('relu'))
        self.add(Dropout(0.5))

        # Layer 7
        self.add(Dense(4096))
        self.add(BatchNormalization())
        self.add(Activation('relu'))
        self.add(Dropout(0.5))

        # Layer 8
        self.add(Dense(n_classes))
        self.add(BatchNormalization())
        self.add(Activation('softmax'))

        # Compiling the model
        self.compile(loss=keras.losses.categorical_crossentropy,
                     optimizer=keras.optimizers.Adam(lr), metrics=['accuracy'])
