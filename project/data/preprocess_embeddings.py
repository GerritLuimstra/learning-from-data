#!/usr/bin/env python

"""
Python script for preprocessing GloVe embeddings text files. Words not
occurring in the vocabulary are removed and the resulting embedding dictionary
is pickled to a compact binary file. Run using:

python preprocess_embeddings.py glove_embeddings_file pickle_file

The embeddings files can be obtained from:

https://nlp.stanford.edu/projects/glove/
"""

import pickle
import sys

import numpy as np


def read_embeddings(embeddings_file):
    """Read in word embeddings from a text file and return as a dictionary ."""

    print(f"Reading embeddings from {embeddings_file}")

    embeddings = {}

    with open(embeddings_file, "r", encoding="utf-8") as handle:
        for line in handle:
            split_line = line.split()
            word = split_line[0]
            embedding = np.array(split_line[1:], dtype=np.float64)
            embeddings[word] = embedding

    return embeddings


def read_vocabulary(corpus_files):
    """Read the datasets and return the set of unique tokens."""

    vocabulary = set()

    for corpus_file in corpus_files:
        print(f"Reading vocabulary from {corpus_file}")
        with open(corpus_file, encoding="utf-8") as corpus:
            for line in corpus:
                words = line.split("\t")[0].strip().split()
                for word in words:
                    vocabulary.add(word)

    return vocabulary


def filter_embeddings(embeddings, vocabulary):
    """Remove words from the embeddings that don't occur in the vocabulary."""

    count = 0
    print("Filtering embeddings")

    for word in list(embeddings.keys()):
        if word not in vocabulary:
            del embeddings[word]
            count += 1

    print(f"Removed {count} embedding vectors not occurring in vocabulary")
    return embeddings


def main():
    # Read the embeddings text file.
    embeddings_file = sys.argv[1]
    embeddings = read_embeddings(embeddings_file)

    # Restrict the embeddings to the train + dev vocabulary.
    vocabulary = read_vocabulary(["train_glove.tsv", "dev_glove.tsv"])
    embeddings = filter_embeddings(embeddings, vocabulary)

    # Dump the filtered embeddings to a pickle file.
    with open(f"{sys.argv[2]}.pickle", "wb") as output:
        pickle.dump(embeddings, output)


if __name__ == '__main__':
    main()
