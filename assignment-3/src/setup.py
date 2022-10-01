"""
This file generates the cross validation and inference split in a stratified manner.
"""

import numpy as np
from helpers import read_corpus
from sklearn.model_selection import train_test_split

# Read in the data
X, y = read_corpus("data/reviews.txt", False)

print(len(X))
print(len(y))

train_size = 0.6
dev_size = 0.2
test_size = 0.2

# Split the dataset into a train, development, and test set.
train_dev_idxs, test_idxs = train_test_split(list(range(len(X))), test_size=test_size, random_state=42)
train_idxs, dev_idxs = train_test_split(train_dev_idxs, test_size=dev_size/(train_size+test_size), random_state=42)

print(len(train_idxs))
print(len(dev_idxs))
print(len(test_idxs))

# # Split the dataset in a stratified manner
# X_cross_idxs, X_test_idxs, _, _ = train_test_split(list(range(len(X))), y, test_size=0.2, random_state=42, stratify=y)

# Load in the whole dataset
with open("data/reviews.txt", "r", encoding="utf-8") as f:
    file_contents = np.array(f.read().split("\n")[:-1])

# Write the train set
with open("data/train.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(file_contents[train_idxs]))

# Write the dev set
with open("data/dev.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(file_contents[dev_idxs]))

# Write the test set
with open("data/test.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(file_contents[test_idxs]))
