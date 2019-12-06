import pickle

import matplotlib.pyplot as plt
import numpy as np

# Defining an input file
input_file = 'lenet_256.pkl'

# Opening input file
with open(input_file, 'rb') as f:
    # Loading pickle object
    metrics = pickle.load(f)

# Calculates the mean and standard deviation
mean_loss = np.mean(metrics['train_loss'], axis=0)
std_loss = np.std(metrics['train_loss'], axis=0)

# Plots the mean loss
plt.plot(mean_loss)

# Fills the mean with standard deviation
plt.fill_between(range(300), mean_loss - std_loss, mean_loss + std_loss, alpha=0.35)

# Plots the legend
plt.legend(['$L_{64}$'], loc='upper left')

# Plots the labels
plt.title('Training convergence over all architectures')
plt.ylabel('Loss')
plt.xlabel('Epoch')

# Showing plot
plt.show()