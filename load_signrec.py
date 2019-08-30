from utils.datasets import Dataset

# Loads the SignRec dataset
d = Dataset(name='signrec', n_persons=26, n_tests=10, n_samples=128, n_channels=6)

# Access its data and labels
print(d.x, d.y)