import pickle

import matplotlib.pyplot as plt
import numpy as np

# Defining input files
input_file = ['lenet_256.pkl']

# Defining input models
input_model = ['$L_{256}$']

# Creating an empty plot
fig, ax = plt.subplots()

# Iterate over all possible models
for i in input_file:
    # Opening input file
    with open(i, 'rb') as f:
        # Loading pickle object
        metrics = pickle.load(f)

    # Calculates the mean and standard deviation
    mean_loss = np.mean(metrics['train_loss'], axis=0)
    std_loss = np.std(metrics['train_loss'], axis=0)

    # Plots the mean loss
    ax.plot(mean_loss)

    # Fills the mean with standard deviation
    ax.fill_between(range(300), mean_loss - std_loss, mean_loss + std_loss, alpha=0.35)

# Setting limits
ax.set_xlim([0, 300])

# Plots the legend
ax.grid()
ax.legend(input_model, loc='upper right')

# Plots the labels
plt.title('Mean training convergence comparison over NewHandPD')
plt.ylabel('loss')
plt.xlabel('epoch')

# Showing plot
plt.show()