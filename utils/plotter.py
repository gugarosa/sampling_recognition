import matplotlib.pyplot as plt


def plot_accuracy(history, validation=False):
    """Plots an accuracy x epochs chart.

    Args:
        history (History): A Keras history object.
        validation (bool): A boolean indicating if validation is used or not.

    """

    # Plots the accuracy
    plt.plot(history.history['acc'])

    # Plots the legend
    plt.legend(['Training'], loc='upper left')

    # Check if there is a validation
    if validation:
        # Plots the validation accuracy
        plt.plot(history.history['val_acc'])

        # Adjusts the legend
        plt.legend(['Training', 'Validation'], loc='upper left')

    # Plots the labels
    plt.title('Accuracy x Epochs')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')

    # Showing plot
    plt.show()


def plot_loss(history, validation=False):
    """Plots a loss x epochs chart.

    Args:
        history (History): A Keras history object.
        validation (bool): A boolean indicating if validation is used or not.

    """

    # Plots the loss
    plt.plot(history.history['loss'])

    # Plots the legend
    plt.legend(['Training'], loc='upper left')

    # Check if there is a validation
    if validation:
        # Plots the validation loss
        plt.plot(history.history['val_loss'])

        # Adjusts the legend
        plt.legend(['Training', 'Validation'], loc='upper left')

    # Plots the labels
    plt.title('Loss x Epochs')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')

    # Showing plot
    plt.show()
