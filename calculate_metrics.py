import pickle

import numpy as np

# Defining an input file
input_file = 'lenet_256.pkl'

# Opening input file
with open(input_file, 'rb') as f:
    # Loading pickle object
    metrics = pickle.load(f)

# Calculating metrics
print(f"Test Loss: {np.mean(metrics['test_loss'])} +- {np.std(metrics['test_loss'])}")
print(f"Test Accuracy: {np.mean(metrics['test_accuracy'])} +- {np.std(metrics['test_accuracy'])}")
print(f"Test Precision: {np.mean(metrics['test_precision'])} +- {np.std(metrics['test_precision'])}")
print(f"Test Recall: {np.mean(metrics['test_recall'])} +- {np.std(metrics['test_recall'])}")
print(f"Test F1-Score: {np.mean(metrics['test_f1'])} +- {np.std(metrics['test_f1'])}")
