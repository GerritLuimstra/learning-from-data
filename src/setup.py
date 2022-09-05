from helpers import read_corpus
from sklearn.model_selection import train_test_split
import numpy as np

SENTIMENT = True

# Read in the data
X, y = read_corpus("data/reviews.txt", SENTIMENT)

X_train_idxs, X_test_idxs, _, _ = train_test_split(list(range(len(X))), y, test_size=0.2, random_state=42, stratify=y)

with open("data/reviews.txt", "r", encoding="utf-8") as f:
    file_contents = np.array(f.read().split("\n")[:-1])

with open("data/train.txt", "w", encoding="utf-8") as f:
    train_content = "\n".join(file_contents[X_train_idxs])
    f.write(train_content)

with open("data/test.txt", "w", encoding="utf-8") as f:
    test_content = "\n".join(file_contents[X_test_idxs])
    f.write(test_content)