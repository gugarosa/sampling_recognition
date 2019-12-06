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

    # Plots the mean time
    ax.plot(range(1, 11), metrics['train_time'])

# Setting limits
ax.set_xlim([1, 10])

# Plots the legend
ax.grid()
ax.legend(input_model, loc='upper right')

# Plots the labels
plt.title('Mean computational load comparison over NewHandPD')
plt.ylabel('time (s)')
plt.xlabel('$k$-fold index')

# Showing plot
plt.show()