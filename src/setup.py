from helpers import read_corpus
from sklearn.model_selection import train_test_split
import numpy as np

# Read in the data
X, y = read_corpus("data/reviews.txt", False)

# Split the dataset in a stratified manner
X_cross_idxs, X_test_idxs, _, _ = train_test_split(list(range(len(X))), y, test_size=0.2, random_state=42, stratify=y)

# Load in the whole dataset
with open("data/reviews.txt", "r", encoding="utf-8") as f:
    file_contents = np.array(f.read().split("\n")[:-1])

# Write the cross validation portion
with open("data/cross.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(file_contents[X_cross_idxs]))

# Write the inference portion
with open("data/inference.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(file_contents[X_test_idxs]))