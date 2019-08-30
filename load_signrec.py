from utils.datasets import Dataset

ID_TESTS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Loads the SignRec dataset
d = Dataset(name='signrec', n_persons=26, id_tests=ID_TESTS, n_samples=128, n_channels=6)

# Access its data and labels
print(d.x, d.y)