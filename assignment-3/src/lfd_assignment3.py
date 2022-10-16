#!/usr/bin/env python

"""TODO: add high-level description of this Python script"""

import argparse
import json
import random as python_random
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (classification_report, ConfusionMatrixDisplay,
                             confusion_matrix)
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import (Bidirectional, Dense, Embedding, LSTM,
                                     TextVectorization)
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

# Make results reproducible.
np.random.seed(1234)
tf.random.set_seed(1234)
python_random.seed(1234)


class ModelSettings(NamedTuple):
    """Settings used to construct the model."""

    embeddings: str
    embedding_dimension: int
    trainable: bool
    learning_rate: float
    use_lr_decay: bool
    optimizer: str
    layers: int
    dropout: float
    recurrent_dropout: float
    bidirectional: bool


class TrainSettings(NamedTuple):
    """Settings used to train the model."""

    epochs: int
    batch_size: int
    patience: int
    verbose: int
    loss_plot: str
    confusion_matrix: bool


def create_arg_parser():
    """Create an argument parser and return the parsed command line input."""

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--train_file", default="data/train.txt",
                        type=str, help="Input file to learn from")
    parser.add_argument("-d", "--dev_file", type=str, default="data/dev.txt",
                        help="Separate dev set to read in")
    parser.add_argument("-t", "--test_file", type=str,
                        help="If added, use trained model to predict on test \
                        set")
    parser.add_argument("-lm", "--language_model", type=str,
                        help="If added, the pre-trained language model to use")
    parser.add_argument("-s", "--sequence_length", type=int, default=50,
                        help="Number of tokens before the input is cut off")
    parser.add_argument("-e", "--embeddings", type=str,
                        help="If added, pre-trained embedding file to use")
    parser.add_argument("-ed", "--embedding_dimension", type=int, default=300,
                        help="Dimension of the embeddings, ignored if a \
                        embedding file is specified")
    parser.add_argument("-tr", "--trainable", default=False,
                        action="store_true", help="Whether the embedding \
                        layer is updated during training")
    parser.add_argument("-lr", "--learning_rate", type=float, default=5e-3,
                        help="Learning rate used to train the model")
    parser.add_argument("-ld", "--use_lr_decay", default=False,
                        action="store_true", help="Whether to use Exponential \
                        LR decay")
    parser.add_argument("-o", "--optimizer", choices=["sgd", "adam"],
                        default="sgd", help="Optimizer used to train the \
                        model")
    parser.add_argument("-ly", "--layers", type=int, default=1,
                        help="Number of LSTM layers in the model")
    parser.add_argument("-do", "--dropout", type=float, default=0,
                        help="Dropout fraction to use for linear \
                        transformation of inputs in LSTM")
    parser.add_argument("-rdo", "--recurrent_dropout", type=float, default=0,
                        help="Dropout fraction to use for linear \
                        transformation of recurrent states in LSTM")
    parser.add_argument("-bi", "--bidirectional", default=False,
                        action="store_true", help="If added, use \
                        bidirectional LSTM layers in the model")
    parser.add_argument("-ep", "--epochs", type=int, default=50,
                        help="Maximum number of epochs to train the model for")
    parser.add_argument("-p", "--patience", type=int, default=5,
                        help="Number of epochs with no improvement before \
                        training is stopped early")
    parser.add_argument("-b", "--batch_size", type=int, default=16,
                        help="Batch size used to train the model")
    parser.add_argument("-v", "--verbose", type=int, choices=[0, 1, 2],
                        default=1, help="Verbosity level of the model \
                        training process")
    parser.add_argument("-lp", "--loss_plot", type=str,
                        help="If added, file name of loss curve plot")
    parser.add_argument("-cm", "--confusion_matrix", default=False,
                        action="store_true", help="If added, plot and save a \
                        confusion matrix")
    args = parser.parse_args()
    return args


def read_args():
    """Read the parsed command line arguments into the ModelSettings and
    TrainSettings classes."""

    args = create_arg_parser()
    model_settings = ModelSettings(args.embeddings, args.embedding_dimension,
                                   args.trainable, args.learning_rate,
                                   args.use_lr_decay, args.optimizer,
                                   args.layers, args.dropout,
                                   args.recurrent_dropout, args.bidirectional)
    train_settings = TrainSettings(args.epochs, args.batch_size, args.patience,
                                   args.verbose, args.loss_plot,
                                   args.confusion_matrix)
    return args, model_settings, train_settings


def read_corpus(corpus_file):
    """Read in review data set and return docs and labels."""

    documents = []
    labels = []
    with open(corpus_file, encoding="utf-8") as corpus:
        for line in corpus:
            tokens = line.strip()
            documents.append(" ".join(tokens.split()[3:]).strip())
            # 6-class problem: books, camera, dvd, health, music, software.
            labels.append(tokens.split()[0])
    return documents, labels


def read_embeddings(embeddings_file):
    """Read in word embeddings from a JSON file and save as a NumPy array."""

    embeddings = json.load(open(embeddings_file, "r"))
    return {word: np.array(embeddings[word]) for word in embeddings}


def get_embedding_matrix(vocabulary, embeddings):
    """Get embedding matrix given the vocabulary and embeddings."""

    # Construct an empty embedding matrix of the correct size.
    num_tokens = len(vocabulary) + 2
    embedding_dim = len(embeddings["the"])  # Bit of a hack.
    embedding_matrix = np.zeros((num_tokens, embedding_dim))

    # Get an embedding vector for each word in the vocabulary.
    for index, word in enumerate(vocabulary):
        embedding_vector = embeddings.get(word)
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            embedding_matrix[index] = embedding_vector

    return embedding_matrix


def create_optimizer(name, learning_rate):
    """Create an optimizer object based on its name."""
    optimizer_dict = {"sgd": SGD, "adam": Adam}
    return optimizer_dict[name](learning_rate=learning_rate)


def create_lstm_layers(num_layers, units, dropout, recurrent_dropout):
    """Create a list of LSTM layers with the specified settings."""

    layers = []

    # Add n-1 LSTM layers that return sequences and one that does not.
    for _ in range(num_layers - 1):
        layers.append(LSTM(units, dropout=dropout,
                           recurrent_dropout=recurrent_dropout,
                           return_sequences=True))
    layers.append(LSTM(units, dropout=dropout,
                       recurrent_dropout=recurrent_dropout))

    return layers


def create_lstm_model(Y_train, embedding_matrix, settings):
    """Create the LSTM model to use."""

    # Get the embedding dimension and size from the embedding matrix.
    embedding_dim = len(embedding_matrix[0])
    num_tokens = len(embedding_matrix)
    num_labels = len(set(Y_train))

    # Start building the model.
    model = Sequential()

    # Add the embedding layer. Initialize it with uniform random values if no
    # embeddings file was provided.
    if settings.embeddings:
        initializer = Constant(embedding_matrix)
    else:
        initializer = "uniform"
    model.add(Embedding(num_tokens, embedding_dim,
                        embeddings_initializer=initializer,
                        trainable=settings.trainable))

    # Create the specified number of LSTM layers.
    lstm_layers = create_lstm_layers(settings.layers, embedding_dim,
                                     settings.dropout,
                                     settings.recurrent_dropout)

    # Make the LSTM layers bidirectional if specified.
    for layer in lstm_layers:
        if settings.bidirectional:
            model.add(Bidirectional(layer))
        else:
            model.add(layer)

    # Ultimately, end with a dense layer with softmax.
    model.add(Dense(input_dim=embedding_dim, units=num_labels,
                    activation="softmax"))

    # Setup the optimizer
    if settings.use_lr_decay:
        lr_schedule = ExponentialDecay(
            initial_learning_rate=settings.learning_rate,
            decay_steps=1000,
            decay_rate=0.9
        )
        optimizer = create_optimizer(settings.optimizer, lr_schedule)
    else:
        optimizer = create_optimizer(settings.optimizer,
                                     settings.learning_rate)

    # Compile the model using our settings, check for accuracy.
    model.compile(loss="categorical_crossentropy", optimizer=optimizer,
                  metrics=["accuracy"])

    return model


def create_language_model(model_name, settings):
    """Create the pre-trained language model to use."""

    # Load the pre-trained language model.
    model = TFAutoModelForSequenceClassification.from_pretrained(model_name,
                                                                 num_labels=6)

    # Decide whether we want to fine-tune the language model.
    model.layers[0].trainable = settings.trainable

    # Compile the model using our settings, check for accuracy.
    loss_function = CategoricalCrossentropy(from_logits=True)

    if settings.use_lr_decay:
        lr_schedule = ExponentialDecay(
            initial_learning_rate=settings.learning_rate,
            decay_steps=1_000,
            decay_rate=0.9
        )
        optimizer = create_optimizer(settings.optimizer, lr_schedule)
    else:
        optimizer = create_optimizer(settings.optimizer,
                                     settings.learning_rate)

    model.compile(loss=loss_function, optimizer=optimizer,
                  metrics=["accuracy"])

    print(model.summary())

    return model


def plot_loss(history, file_name):
    """Plot a loss curve from the given model history."""

    train_loss = history["loss"]
    val_loss = history["val_loss"]
    plt.title("Train and Dev Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(range(1, len(train_loss) + 1), train_loss, label="train")
    plt.plot(range(1, len(train_loss) + 1), val_loss, label="dev")
    plt.legend()
    plt.savefig(file_name, bbox_inches="tight", format="pdf")


def train_model(model, X_train, Y_train, X_dev, Y_dev, settings, labels):
    """Trains the model using the specified training and validation sets."""

    # Stop training when there are some number (3 by default) consecutive
    # epochs without improving.
    callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                patience=settings.patience)

    # Fit the model to our data.
    history = model.fit(X_train, Y_train, verbose=settings.verbose,
                        epochs=settings.epochs, callbacks=[callback],
                        batch_size=settings.batch_size,
                        validation_data=(X_dev, Y_dev))

    # Plot a loss curve if specified.
    if settings.loss_plot:
        plot_loss(history.history, settings.loss_plot)

    # Print the final accuracy for the model.
    test_set_predict(model, X_dev, Y_dev, "dev", labels,
                     settings.confusion_matrix)
    return model


def test_set_predict(model, X_test, Y_test, ident, labels, plot_cm):
    """Do predictions and measure accuracy on the given test set."""

    # Get predictions using the trained model.
    Y_pred = model.predict(X_test)
    if isinstance(Y_pred, dict):
        Y_pred = Y_pred["logits"]  # Get logits in the case of language model.

    # Convert to numerical labels to get scores with sklearn.
    Y_pred = np.argmax(Y_pred, axis=1)
    Y_test = np.argmax(Y_test, axis=1)

    # Print a classification report.
    print(f"Classification results on {ident} set:")
    print(classification_report(Y_test, Y_pred, target_names=labels))

    # Save a confusion matrix if specified.
    if plot_cm:
        cm = confusion_matrix(Y_pred, Y_test)
        ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=labels).plot()
        plt.savefig(f"confusion_matrix_{ident}.pdf", bbox_inches="tight",
                    format="pdf")


def run_lstm_model(args, model_settings, train_settings):
    """Constructs, trains, and runs the specified LSTM model."""

    # Read in the data.
    X_train, Y_train = read_corpus(args.train_file)
    X_dev, Y_dev = read_corpus(args.dev_file)

    # Read in the embeddings if specified, otherwise create dummy embeddings.
    if args.embeddings:
        embeddings = read_embeddings(args.embeddings)
    else:
        embeddings = {"the": np.zeros(args.embedding_dimension)}

    # Transform words to indices using a vectorizer.
    vectorizer = TextVectorization(standardize=None,
                                   output_sequence_length=args.sequence_length)

    # Use train and dev to create the vocabulary.
    text_ds = tf.data.Dataset.from_tensor_slices(X_train + X_dev)
    vectorizer.adapt(text_ds)
    vocabulary = vectorizer.get_vocabulary()

    # Construct the embedding matrix.
    emb_matrix = get_embedding_matrix(vocabulary, embeddings)

    # Transform string labels to one-hot encodings.
    encoder = LabelBinarizer()
    Y_train_bin = encoder.fit_transform(Y_train)
    Y_dev_bin = encoder.fit_transform(Y_dev)

    # Create the model.
    model = create_lstm_model(Y_train, emb_matrix, model_settings)

    # Transform input to vectorized input.
    X_train_vect = vectorizer(np.array([[s] for s in X_train])).numpy()
    X_dev_vect = vectorizer(np.array([[s] for s in X_dev])).numpy()

    # Train the model.
    model = train_model(model, X_train_vect, Y_train_bin, X_dev_vect,
                        Y_dev_bin, train_settings, encoder.classes_)

    # If specified, do predictions on the test set.
    if args.test_file:
        # Read in test set and vectorize.
        X_test, Y_test = read_corpus(args.test_file)
        Y_test_bin = encoder.fit_transform(Y_test)
        X_test_vect = vectorizer(np.array([[s] for s in X_test])).numpy()

        # Do the predictions.
        test_set_predict(model, X_test_vect, Y_test_bin, "test",
                         encoder.classes_, train_settings.confusion_matrix)


def run_language_model(args, model_settings, train_settings):
    """Constructs, trains, and runs the specified pre-trained language
    model."""

    # Read in the data.
    X_train, Y_train = read_corpus(args.train_file)
    X_dev, Y_dev = read_corpus(args.dev_file)

    # Transform string labels to one-hot encodings.
    encoder = LabelBinarizer()
    Y_train_bin = encoder.fit_transform(Y_train)
    Y_dev_bin = encoder.fit_transform(Y_dev)

    # Create the model.
    model = create_language_model(args.language_model, model_settings)

    # Tokenize the input.
    tokenizer = AutoTokenizer.from_pretrained(args.language_model)
    tokens_train = tokenizer(X_train, padding=True,
                             max_length=args.sequence_length, truncation=True,
                             return_tensors="np").data
    tokens_dev = tokenizer(X_dev, padding=True,
                           max_length=args.sequence_length,
                           truncation=True, return_tensors="np").data

    # Train the model.
    model = train_model(model, tokens_train, Y_train_bin, tokens_dev,
                        Y_dev_bin, train_settings, encoder.classes_)

    # If specified, do predictions on the test set.
    if args.test_file:
        # Read in test set and tokenize.
        X_test, Y_test = read_corpus(args.test_file)
        Y_test_bin = encoder.fit_transform(Y_test)
        tokens_test = tokenizer(X_test, padding=True,
                                max_length=args.sequence_length,
                                truncation=True, return_tensors="np").data

        # Do the predictions.
        test_set_predict(model, tokens_test, Y_test_bin, "test",
                         encoder.classes_, train_settings.confusion_matrix)


def main():
    """Main function to train and test neural network given command line
    arguments."""

    # Read the command line arguments.
    args, model_settings, train_settings = read_args()

    # Run a pre-trained language model if specified.
    if args.language_model:
        run_language_model(args, model_settings, train_settings)
    # Run an LSTM model otherwise.
    else:
        run_lstm_model(args, model_settings, train_settings)


if __name__ == "__main__":
    main()
