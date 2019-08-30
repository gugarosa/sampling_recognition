from utils.datasets import Dataset

ID_TESTS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

# Loads the HandPD dataset
d = Dataset(name='handpd', n_persons=66, id_tests=ID_TESTS, n_samples=128, n_channels=6)

# Re-define labels for Parkinson's identification
#d.y[:len(ID_TESTS)*35] = 0
#d.y[len(ID_TESTS)*35:] = 1

# Access its data and labels
print(d.x, d.y)