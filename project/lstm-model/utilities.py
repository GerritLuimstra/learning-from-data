import logging

import numpy as np

def read_tweets(corpus_file):
    """Read the tweets dataset and return tweets and toxicity labels."""

    logging.info(f"Reading tweets from {corpus_file}")
    tweets = []
    labels = []

    with open(corpus_file, encoding="utf-8") as corpus:
        for line in corpus:
            # Data of the form <tweet>\t<label> with <label> = NOT | OFF.
            split_line = [chunk.strip() for chunk in line.split("\t")]
            tweets.append(split_line[0])
            labels.append(0 if split_line[1] == "NOT" else 1)

    return np.asarray(tweets), np.asarray(labels)
