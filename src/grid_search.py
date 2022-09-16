'''TODO: add high-level description of this Python script'''

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score, recall_score, precision_score, accuracy_score
from helpers import create_vocabulary, read_corpus
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import numpy as np
import tqdm

import mlflow
import mlflow.sklearn
import random

# Ensure reproducability
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

if __name__ == "__main__":

    # Setup the connection to ML flow (for tracking)
    mlflow.set_tracking_uri("http://localhost:5050")
    _ = mlflow.set_experiment("Learning From Data Assignment 1")

    # Read in the data from the train file
    X, y = read_corpus("data/reviews.txt", False)

    classifier_options = {
        'dt': {
            'base': DecisionTreeClassifier,
            'grid': {
                'max_depth': [5, 10, 20, 30, 40, 50, None],
                'min_samples_leaf': [1, 5, 10, 20, 30],
                'ccp_alpha': [0.0005, 0.0001, 0.001, 0, 0]
            }
        },
        'rf': {
            'base': RandomForestClassifier,
            'grid': {
                'max_depth': [5, 10, 20, 30, 40, 50, None],
                'min_samples_leaf': [1, 5, 10, 20, 30],
                'ccp_alpha': [0, 0.0005, 0.0001, 0.001, 0, 0],
                'n_estimators': [100, 200, 300],
                'n_jobs': [-1]
            }
        },
        'knn': {
            'base': KNeighborsClassifier,
            'grid': {
                'n_neighbors': [1, 3, 5, 7, 11]
            }
        },
        'nb': {'base': MultinomialNB, 'grid': {}},
        'svm': {
            'base': SVC,
            'grid': {'kernel': ('linear', 'rbf'), 'C':[1, 0.5, 2]}
        },
        'linearsvc': {
            'base': LinearSVC,
            'grid': {'C':[1, 0.5, 2]}
        }
    }

    # Setup the metrics to track
    metrics = {
        'test_f1_macro': make_scorer(f1_score, average='macro'),
        'test_accuracy': make_scorer(accuracy_score)
    }
    for c in np.unique(y):
        metrics |= {
            'f1_' + str(c): make_scorer(f1_score, average=None, labels=[c]),
            'recall_' + str(c): make_scorer(recall_score, average=None, labels=[c]),
            'precision_' + str(c): make_scorer(precision_score, average=None, labels=[c])
        }

    
    done = set([])
    for _ in tqdm.tqdm(range(5000)):

        use_stemming = random.random() > 0.5
        use_lemmatization = random.random() > 0.5 if not use_stemming else False
        use_tfidf = random.random() > 0.5
        ngram_range = (1, random.choice([1, 2, 3]))

        # Compute the weights
        weights = [np.product([len(l) for l in classifier_options[algo]['grid'].values()]) for algo in classifier_options.keys()]

        # Sample a random algorithm
        algo = random.choices(list(classifier_options.keys()), weights=weights, k=1)[0]

        # Randomly sample from the grid
        options = {}
        for key in classifier_options[algo]["grid"]:
            options[key] = random.choice(classifier_options[algo]["grid"][key])

        # Create the classifier with the given parameters
        classifier = classifier_options[algo]['base'](**options)

        # Hash the experiment settings
        run_string = str(algo) + str(use_stemming) + str(use_lemmatization) + str(use_tfidf) + str(ngram_range) + "".join([str(value) for value in options.values()])

        # We only run unique experiments
        if run_string in done:
            continue
        done.add(run_string)

        # Setup the stemmer and lemmatizer
        stemmer = PorterStemmer()
        lemmatizer = WordNetLemmatizer()

        # Create the vocabulary
        if use_stemming:
            vocabulary = create_vocabulary(X, stemmer=stemmer)
        elif use_lemmatization:
            vocabulary = create_vocabulary(X, lemmatizer=lemmatizer)
        else:
            vocabulary = create_vocabulary(X)

        # Convert the texts to vectors
        if use_tfidf:
            vec = TfidfVectorizer(vocabulary=vocabulary, preprocessor=lambda x: x, tokenizer=lambda x: x, ngram_range=ngram_range)
        else:
            # Bag of Words vectorizer
            vec = CountVectorizer(vocabulary=vocabulary, preprocessor=lambda x: x, tokenizer=lambda x: x, ngram_range=ngram_range)

        # Transform the input data to the new vocabulary
        X_ = vec.fit_transform(X)

        # Start the experiment
        with mlflow.start_run():

            # Log the experiment in ML flow
            mlflow.log_param("SEED", SEED)
            mlflow.log_param("TFIDF", use_tfidf)
            mlflow.log_param("MODEL NAME", classifier.__class__.__name__)
            mlflow.log_param("FOLDS", 10)
            mlflow.log_param("LEMMATIZATION", use_lemmatization)
            mlflow.log_param("STEMMING", use_stemming)
            mlflow.log_param("NGRAM_RANGE", ngram_range)
            mlflow.log_param("VOCAB SIZE", len(vec.vocabulary_))
            mlflow.log_params(classifier.get_params())

            # Setup stratified cross validation
            # Stratification ensures that each fold has the 
            # same class proportion as the main dataset
            # https://en.wikipedia.org/wiki/Stratified_sampling
            skf = StratifiedKFold(n_splits=10, shuffle=True)

            # Obtain the scores
            scores = cross_validate(classifier, X_, y, cv=skf, scoring=metrics)
            for metric in scores:
                if not "test_" in metric:
                    continue
                mlflow.log_metric(metric.replace("test_", "") + " std", np.std(scores[metric]))
                mlflow.log_metric(metric.replace("test_", "") + " mean", np.mean(scores[metric]))