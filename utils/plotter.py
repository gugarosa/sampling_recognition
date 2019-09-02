import matplotlib.pyplot as plt

def plot_accuracy(history):
    """
    """

    plt.plot(history.history['acc'])
    plt.title('Model Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['Training'], loc='upper left')
    plt.show()

def plot_loss(history):
    """
    """

    plt.plot(history.history['loss'])
    plt.title('Model Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Training'], loc='upper right')
    plt.show()
