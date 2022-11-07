import matplotlib.pyplot as plt
import numpy as np
from preprocess_twitter import tokenize
from utilities import read_tweets, load_lexicon, obtain_sentiment

# Read in the data
X_train, Y_train = read_tweets("../data/train.tsv")

# Read in the lexicon
lexicon = load_lexicon()

# Preprocess the text
X_train = list(map(lambda x: tokenize(x), X_train))

# Obtain the sentiment
X_train_sent = list(map(lambda tweet: [lexicon.get(word, np.nan) for word in tweet.split(" ")], X_train))

# Obtain the tweet level sentiment scores
X_train_scores = []
X_train_idxs = []
for idx, doc in enumerate(X_train_sent):
    if sum(np.isnan(doc)) == len(doc):
        continue
    X_train_idxs.append(idx)
    X_train_scores.append(obtain_sentiment(doc, reduction='mean'))
Y_train = np.array(Y_train)[X_train_idxs]

# Display the histogram
B = 100
plt.hist(np.array(X_train_scores)[Y_train == "NOT"], bins=B, alpha=0.5, label='Not offensive', color='green')
plt.hist(np.array(X_train_scores)[Y_train == "OFF"], bins=B, alpha=0.5, label='Offensive', color='red')
plt.title("Actual sentiment distribution (mean)")
plt.xlabel("Sentence Sentiment")
plt.ylabel("Frequency")
plt.yticks([], [])
plt.legend()
plt.show()