import keras
from keras.callbacks import History
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.models import Sequential


class Lenet(Sequential):
    """
    """

    def __init__(self, input_shape=(), n_classes=2, lr=0.0001):
        """
        """

        # Overriding its parent class
        super(Lenet, self).__init__()

        # Creating a history property
        self.history = History()

        # Defining the model itself
        # Conv + Max Pool block
        self.add(Conv2D(20, kernel_size=(5, 5),
                        strides=(1, 1), input_shape=input_shape))
        self.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # Conv + Max Pool block
        self.add(Conv2D(50, (5, 5)))
        self.add(MaxPooling2D(pool_size=(2, 2)))

        # Flattenning the arrays
        self.add(Flatten())

        # Performing the last fully connections
        self.add(Dense(500, activation='relu'))
        self.add(Dense(n_classes, activation='softmax'))

        # Compiling the model
        self.compile(loss=keras.losses.categorical_crossentropy,
                     optimizer=keras.optimizers.Adam(lr), metrics=['accuracy'])
