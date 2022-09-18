import argparse
import numpy as np

def read_corpus(corpus_file, use_sentiment):
    '''TODO: add function description'''
    documents = []
    labels = []
    with open(corpus_file, encoding='utf-8') as in_file:
        for line in in_file:
            tokens = line.strip().split()
            documents.append(tokens[3:])
            if use_sentiment:
                # 2-class problem: positive vs negative
                labels.append(tokens[1])
            else:
                # 6-class problem: books, camera, dvd, health, music, software
                labels.append(tokens[0])
    return documents, labels

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--cross_file", type=str,
                        help="Txt file that will be used to perform cross-validation.")
    parser.add_argument("-if", "--inference_file", default=None, type=str,
                        help="Optional test set to run inferences on. (default None).")
    parser.add_argument("-t", "--tfidf", action="store_true",
                        help="Use the TF-IDF vectorizer instead of CountVectorizer")
    parser.add_argument("-l", "--lemmatization", action="store_true",
                        help="Whether to use Lemmatization (default False)")
    parser.add_argument("-s", "--stemming", action="store_true",
                        help="Whether to use Stemming (default False).")
    parser.add_argument("-n", "--ngram_range", type=int, default=1, 
                        help="The upper n-gram range. This includes n-grams in the range (1, n). (default 1)")
    parser.add_argument("-m", "--model_name", type=str, default='nb', help="The model to use. Can be one of ['nb', 'dt', 'rf', 'knn', 'svm']")
    parser.add_argument("-f", "--folds", type=int, default=10, help="The amount of folds to use for the cross validation")
    parser.add_argument("-a", "--args", default=[], nargs='+', help="The arguments passed to the ML model")
    args = parser.parse_args()
    return args

def parse_values(values):
    values_ = []
    for value in values:
        if "'" in value:
            values_.append(value.replace("'", ""))
        elif "." in value:
            values_.append(float(value))
        elif value == "None":
            values_.append(None)
        else:
            values_.append(int(value))
    return values_

def create_vocabulary(X, stemmer=None, lemmatizer=None):

    # Obtain all words in the dataset
    flattened = [word for sample in X for word in sample]

    # Obtain the unique words and their frequencies
    words, frequency = np.unique(flattened, return_counts=True)

    # Remove all words that have a frequency of less than 5
    words = words[frequency >= 5]

    # Remove qoutes from words
    words = list(map(lambda word: word.replace("'", ""), words))
    words = list(map(lambda word: word.replace('"', ""), words))

    # Remove words that contain numbers
    words = list(filter(lambda word: not any(char.isdigit() for char in word), words))

    # Load in all the words from the english dictionary
    # from https://github.com/dwyl/english-words
    with open("src/english_wordlist.txt") as f:
        english_words = set(f.read().split("\n"))
    
    # Remove words that are not in the english language dictionary
    words = list(filter(lambda word: word in english_words, words))

    # Load in the stop words
    # from https://github.com/dwyl/english-words
    with open("src/stopwords.txt") as f:
        stopwords = set(f.read().split("\n"))

    # Remove stopwords
    words = list(filter(lambda word: word not in stopwords, words))

    # Perform stemming or lemmatization
    if lemmatizer is not None:
        words = list(map(lambda word: lemmatizer.lemmatize(word), words))
    if stemmer is not None:
        words = list(map(lambda word: stemmer.stem(word), words))

    return set(words)