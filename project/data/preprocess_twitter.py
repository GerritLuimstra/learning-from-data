#!/usr/bin/env python

"""
Python script for preprocessing Twitter text for use with GloVe embeddings.
Adapted for the OLID dataset. Run using:

python preprocess_twitter.py -cf input_file > output_file

Original script for preprocessing tweets by Romain Paulus with small
modifications by Jeffrey Pennington with translation to Python by Motoki Wu.

The original Ruby script:

http://nlp.stanford.edu/projects/glove/preprocess-twitter.rb
"""

import argparse
import regex as re

import emoji
import wordsegment

FLAGS = re.MULTILINE | re.DOTALL


def create_arg_parser():
    """Create an argument parser and return the parsed command line input."""

    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--corpus_file", type=str,
                        help="The name of the corpus file to preprocess.")
    parser.add_argument("-re", "--remove_emoji", default=False,
                        action="store_true", help="Whether emoji are removed.")
    parser.add_argument("-rh", "--remove_hashtags", default=False,
                        action="store_true",
                        help="Whether hashtags are removed.")
    parser.add_argument("-rp", "--remove_punctuation", default=False,
                        action="store_true",
                        help="Whether punctuation is removed.")
    args = parser.parse_args()
    return args


def allcaps(text):
    """Lower case input text and append an <allcaps> token."""

    text = text.group()
    return text.lower() + " <allcaps>"


def remove_emoji(text):
    """Remove all emoji from text."""

    # Replace emoji with space for case word<emoji>word.
    return emoji.replace_emoji(text, " ")


def substitute_emoji(text):
    """Replace emoji with space separated words describing the emoji."""

    def replace_underscore(text):
        return text.group()[1:-1].replace("_", " ")

    text = emoji.demojize(text, delimiters=(" :", ": "))
    text = re.sub(r":\w+:", replace_underscore, text, flags=FLAGS)
    return text


def remove_hashtags(text):
    """Remove all hashtags from text."""

    return re.sub(r"#\S+", " ", text, flags=FLAGS)


def substitute_hashtags(text):
    """Replace hashtags with space separated words and add <hashtag> token."""

    def hashtag(text):
        text = text.group()
        hashtag_body = text[1:]

        # Segment the hashtag into lower cased component words.
        words = wordsegment.segment(hashtag_body)

        # Add an <allcaps> token behind each word if the tweet is capitalized.
        if hashtag_body.isupper():
            words = [word + " <allcaps>" for word in words]

        result = " ".join(["<hashtag>"] + words)
        return result

    return re.sub(r"#\S+", hashtag, text, flags=FLAGS)


def remove_punctuation(text):
    """Remove punctuation from tokenized text."""

    old = ""
    punctuation = r"@!\"#\$%&'\(\)\*\+,-\./:;<=>\?@\[\\\]\^_`\{\|\}~"
    while old != text:
        old = text
        text = re.sub(f"^[{punctuation}]\\s(?!(\\s)*<repeat>)", " ",
                      text, flags=FLAGS)  # Skip punctuation before <repeat>.
        text = re.sub(f"\\s+[{punctuation}]\\s(?!(\\s)*<repeat>)", " ",
                      text, flags=FLAGS)  # Skip punctuation before <repeat>.
        text = re.sub(f"\\s+[{punctuation}]$", " ", text, flags=FLAGS)
    return text


def glove_tokenize(text):
    """Tokenize text to match GloVe embeddings."""

    # Different regex parts for smiley faces.
    eyes = r"[8:=;]"
    nose = r"['`\-]?"

    # Utility function to avoid repetition.
    def re_sub(pattern, repl):
        return re.sub(pattern, repl, text, flags=FLAGS)

    # Tokenization in line with GloVe's preprocessing Ruby script.
    text = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "<url>")
    text = re_sub(r"URL", "<url>")  # Our dataset has its own url token.
    text = re_sub(r"@\w+", "<user>")
    text = re_sub(f"{eyes}{nose}[)dD]+|[)dD]+{nose}{eyes}", "<smile>")
    text = re_sub(f"{eyes}{nose}p+", "<lolface>")
    text = re_sub(f"{eyes}{nose}\\(+|\\)+{nose}{eyes}", "<sadface>")
    text = re_sub(f"{eyes}{nose}[\\/|l*]", "<neutralface>")
    text = re_sub(r"<3", "<heart>")
    text = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "<number>")
    text = re_sub(r"([!?.]){2,}", r"\1 <repeat>")
    text = re_sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2 <elong>")
    text = re_sub(r"([A-Z]){2,}", allcaps)

    # Replace special ... character which has no embedding.
    text = re_sub(r"…", r" . <repeat> ")

    # Add white space around <tokens>.
    text = re_sub(r"(<[^\s>]+>)", r" \1 ")

    # Insert white space around any non-token < and > characters.
    text = re_sub(r"<([^\s>]+)(\s)", r"< \1\2")
    text = re_sub(r"(\s)([^\s<]+)>", r"\1\2 >")

    # Repeatedly insert white space around punctuation characters.
    old = ""
    punctuation = r"@!\"#\$%&'\(\)\*\+,-\./:;<=>\?@\[\\\]\^_`\{\|\}~"
    while old != text:
        old = text
        text = re_sub(f"([{punctuation.replace('<', '')}])(\\S+)", r" \1 \2")
        text = re_sub(f"(\\S+)([{punctuation.replace('>', '')}])", r"\1 \2 ")

    return text


def pre_cleanup(text):
    """Perform some initial preprocessing steps to clean up text."""

    # Resolve some HTML codes to their ASCII characters.
    text = re.sub(r"&amp;", r"&", text, flags=FLAGS)
    text = re.sub(r"&gt;", r">", text, flags=FLAGS)
    text = re.sub(r"&lt;", r"<", text, flags=FLAGS)

    # Rewrite unusual hyphen character.
    text = re.sub(r"—", r"-", text, flags=FLAGS)

    # Rewrite unusual quote characters.
    text = re.sub(r"`", r"'", text, flags=FLAGS)
    text = re.sub(r"‘", r"'", text, flags=FLAGS)
    text = re.sub(r"’", r"'", text, flags=FLAGS)
    text = re.sub(r"“", r'"', text, flags=FLAGS)
    text = re.sub(r"”", r'"', text, flags=FLAGS)

    return text


def post_cleanup(text):
    """Perform some final preprocessing steps to clean up text."""

    # Remove duplicate or alternate white space characters.
    text = re.sub(r"\s+", r" ", text, flags=FLAGS)
    text = text.strip()

    return text


def expand_contraction(text):
    """Expand contractions such as I've to I have."""

    text = re.sub(r"\swon\'t", " will not", text, flags=FLAGS)
    text = re.sub(r"\scan\'t", " can not", text, flags=FLAGS)
    text = re.sub(r"n\'t\s", " not ", text, flags=FLAGS)
    text = re.sub(r"\'re\s", " are ", text, flags=FLAGS)
    text = re.sub(r"\'s\s", " is ", text, flags=FLAGS)
    text = re.sub(r"\'d\s", " would ", text, flags=FLAGS)
    text = re.sub(r"\'ll\s", " will ", text, flags=FLAGS)
    text = re.sub(r"\'t\s", " not ", text, flags=FLAGS)
    text = re.sub(r"\'ve\s", " have ", text, flags=FLAGS)
    text = re.sub(r"\'m\s", " am ", text, flags=FLAGS)

    return text


def preprocess(text, args):
    """Preprocess tweets."""

    text = pre_cleanup(text)

    if args.remove_emoji:
        text = remove_emoji(text)
    else:
        text = substitute_emoji(text)

    if args.remove_hashtags:
        text = remove_hashtags(text)
    else:
        text = substitute_hashtags(text)

    text = expand_contraction(text)
    text = glove_tokenize(text)

    if args.remove_punctuation:
        text = remove_punctuation(text)

    text = post_cleanup(text)
    return text.lower()


if __name__ == '__main__':
    # Read the command line arguments.
    args = create_arg_parser()

    # Load in data from disk for word segmentation.
    wordsegment.load()

    # Process the corpus line by line.
    corpus_file = args.corpus_file
    with open(corpus_file, encoding="utf-8") as corpus:
        for line in corpus:
            # Split and only tokenize the first part (second part is label).
            split_line = line.split('\t')
            tokens = preprocess(split_line[0], args)
            print(tokens + '\t' + split_line[1], end='')
