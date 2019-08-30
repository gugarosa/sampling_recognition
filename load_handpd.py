from utils.datasets import Dataset

# Loads the HandPD dataset
d = Dataset(name='handpd', n_persons=3, n_tests=10, n_samples=128, n_channels=6)

# Re-define labels for Parkinson's identification
pass

# Access its data and labels
print(d.x, d.y)