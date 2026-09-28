import numpy as np

# This will raise an error if the file contains pickled objects
try:
    X = np.load("cleandataset/email_clean_y.npy", allow_pickle=False)
    print("Safe: plain numeric array")
except ValueError:
    print("Caution: file contains pickled Python objects")