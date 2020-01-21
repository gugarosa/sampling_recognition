import pickle

import matplotlib.pyplot as plt
import numpy as np

# Defining input files
input_file = ['output/all_lenet_256.pkl', 'output/all_lenet_576.pkl', 'output/all_lenet_1024.pkl', 'output/all_cifar10_256.pkl', 'output/all_cifar10_576.pkl', 'output/all_cifar10_1024.pkl', 'output/all_alexnet_256.pkl', 'output/all_alexnet_576.pkl', 'output/all_alexnet_1024.pkl']

# Defining input models
input_model = ['$L_{256}$', '$L_{576}$', '$L_{1024}$', '$C_{256}$', '$C_{576}$', '$C_{1024}$', '$A_{256}$', '$A_{576}$', '$A_{1024}$']

# Creating an empty plot
fig, ax = plt.subplots()

# Iterate over all possible models
for i in input_file:
    # Opening input file
    with open(i, 'rb') as f:
        # Loading pickle object
        metrics = pickle.load(f)

    # Reshaping the variable
    metrics['train_time'] = np.reshape(metrics['train_time'], (15, 12))

    # Gathering its mean value
    metrics['train_time'] = np.mean(metrics['train_time'], axis=0)

    # Plots the mean time
    ax.plot(range(1, 13), metrics['train_time'])

# Setting limits
ax.set_xlim([1, 12])
ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

# Plots the legend
ax.grid()
ax.legend(input_model, loc='upper right')

# Plots the labels
plt.title('Mean computational load comparison over NewHandPD (all)')
plt.ylabel('time (s)')
plt.xlabel('$k$-fold index')

# Showing plot
plt.show()