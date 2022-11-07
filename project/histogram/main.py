import argparse
import json
import random as python_random
from tkinter import Y
from typing import NamedTuple
import matplotlib.pyplot as plt
import numpy as np
from preprocess_twitter import hashtag, tokenize
from collections import Counter
from matplotlib.patches import Rectangle
from sklearn.metrics import accuracy_score, f1_score

def read_lexicon(path):
    with open(path) as f:
        lines = f.read().split("\n")
        lexicon_mapping = {}
        for line in lines:
            splits = line.split("\t")
            lexicon_mapping[splits[0]] = float(splits[1])
        return lexicon_mapping

def read_tweets(corpus_file):
    """Read in tweets dataset and return tweets and toxicity labels."""
    tweets = []
    labels = []
    with open(corpus_file, encoding="utf-8") as corpus:
        for line in corpus:
            tweet = line[:-4].strip()
            label = line.strip()[-3:]
            tweets.append(tweet)
            labels.append(label)
    return tweets, labels

# Read in the data
X_train, Y_train = read_tweets("../data/train.tsv")
X_dev, Y_dev = read_tweets("../data/dev.tsv")

to_beat = np.unique(Y_train, return_counts=True)[1][0]/len(Y_train)
lexicon = read_lexicon("unigrams-pmilexicon.txt")

# Normalize the lexicon
a = -1
b = 1
min_ = np.min(list(lexicon.values()), axis=0)
max_ = np.max(list(lexicon.values()), axis=0)
normalized = {}
for word, value in lexicon.items():
    normalized[word] = (b - a) * (value - min_) / (max_ - min_) + a

# Preprocess the text
X_train = list(map(lambda x: tokenize(x), X_train))

# Obtain the sentiment
func = lambda tweet: [normalized.get(word, np.nan) for word in tweet.split(" ")]
X_train_sent = list(map(func, X_train))

# for s, v in zip(X_train[:10], X_train_sent[:10]):
#     print("===")
#     print(s)
#     print(v)
# exit(0)

X_train_scores = []
X_train_idxs = []
for idx, doc in enumerate(X_train_sent):
    if sum(np.isnan(doc)) == len(doc):
        continue
    X_train_idxs.append(idx)
    # X_train_scores.append(doc[np.nanargmax(np.abs(doc))])
    X_train_scores.append(np.nanmean(doc))

Y_train = np.array(Y_train)[X_train_idxs]

B = 100
n, bins, patches = plt.hist(X_train_scores, bins=B)
X_train_scores_ = np.array(X_train_scores)
for bin, patch in zip(bins, patches):
    scores_ = Y_train[(X_train_scores_ >= bin) & (X_train_scores_ < bin + 2 / B)]
    if len(scores_) == 0:
        patch.set_facecolor("white")
        continue
    
    most_common = Counter(scores_).most_common()
    if len(most_common) != 1:
        alpha = max(most_common[0][1], most_common[1][1]) / len(scores_)
    else:
        alpha = 1
    most_common = most_common[0][0]
    patch.set_facecolor({"OFF": "red", "NOT": "green"}[most_common])
    patch.set_alpha(alpha)

plt.title("Actual sentiment distribution (maximum absolute value)")
plt.xlabel("Sentence Sentiment")
plt.ylabel("Frequency")
plt.yticks([], [])
handles = [Rectangle((0,0),1,1,color=c,ec="k") for c in ["red", "green"]]
labels= ["Offensive", "Not offensive"]
plt.legend(handles, labels)
plt.show()