#!/usr/bin/env python

"""
Python script for preprocessing Twitter text for use with GloVe embeddings.
Adapted for the OLID dataset. Run using:

python preprocess_twitter.py input_file > output_file

Original script for preprocessing tweets by Romain Paulus with small
modifications by Jeffrey Pennington with translation to Python by Motoki Wu.

The original Ruby script:

http://nlp.stanford.edu/projects/glove/preprocess-twitter.rb
"""

import sys
import regex as re

FLAGS = re.MULTILINE | re.DOTALL


def hashtag(text):
    """Break input text into words and add <hashtag> token."""

    text = text.group()
    hashtag_body = text[1:]

    # All-caps hashtags are one word with <allcaps> token.
    if hashtag_body.isupper():
        result = f"<hashtag> {hashtag_body.lower()} <allcaps>"
    # Other hashtags are broken up using CamelCase rules.
    else:
        words = re.split(r"(?=[A-Z])", hashtag_body, flags=FLAGS)
        words = list(filter(None, words))
        result = " ".join(["<hashtag>"] + words)
    return result


def allcaps(text):
    """Lower case input text and append an <allcaps> token."""

    text = text.group()
    return text.lower() + " <allcaps>"

def tokenize(text, remove_emojis=True):
    """Tokenize input text."""

    # Different regex parts for smiley faces.
    eyes = r"[8:=;]"
    nose = r"['`\-]?"

    # Utility function to avoid repetition.
    def re_sub(pattern, repl):
        return re.sub(pattern, repl, text, flags=FLAGS)

    # Remove duplicate or alternate white space characters.
    text = re_sub(r"\s+", r" ")
    text = text.strip()

    # Resolve some HTML codes to their ASCII characters.
    text = re_sub(r"&amp;", r"&")
    text = re_sub(r"&gt;", r">")
    text = re_sub(r"&lt;", r"<")

    # Tokenization in line with GloVe's preprocessing Ruby script.
    text = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "<url>")
    text = re_sub(r"URL", "<url>")  # Our dataset has its own url token.
    text = re_sub(r"@\w+", "<user>")
    text = re_sub(f"{eyes}{nose}[)dD]+|[)dD]+{nose}{eyes}", "<smile>")
    text = re_sub(f"{eyes}{nose}p+", "<lolface>")
    text = re_sub(f"{eyes}{nose}\\(+|\\)+{nose}{eyes}", "<sadface>")
    text = re_sub(f"{eyes}{nose}[\\/|l*]", "<neutralface>")
    text = re_sub(r"❤|<3", "<heart>")
    text = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "<number>")
    text = re_sub(r"#\S+", hashtag)
    text = re_sub(r"([!?.]){2,}", r"\1 <repeat>")
    text = re_sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2 <elong>")
    text = re_sub(r"([A-Z]){2,}", allcaps)


    if remove_emojis:
        # Remove emoij since they have no embedding.
        # https://gist.github.com/Alex-Just/e86110836f3f93fe7932290526529cd1
        emoji_pattern = re.compile(
            "(["
            "\U0001F1E0-\U0001F1FF"  # Flags (iOS).
            "\U0001F300-\U0001F5FF"  # Symbols & pictographs.
            "\U0001F600-\U0001F64F"  # Emoticons.
            "\U0001F680-\U0001F6FF"  # Transport & map symbols.
            "\U0001F700-\U0001F77F"  # Alchemical symbols.
            "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended.
            "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C.
            "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs.
            "\U0001FA00-\U0001FA6F"  # Chess Symbols.
            "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A.
            "\U00002702-\U000027B0"  # Dingbats.
            "]+)"
        )
        text = re.sub(emoji_pattern, r" ", text)  # Space for case word<emoji>word.

    # Rewrite unusual hyphen character.
    text = re_sub(r"—", r"-")

    # Rewrite unusual quote characters.
    text = re_sub(r"`", r"'")
    text = re_sub(r"‘", r"'")
    text = re_sub(r"’", r"'")
    text = re_sub(r"“", r'"')
    text = re_sub(r"”", r'"')

    # Clean up lines that are wrapped in quotes (quote tweets?).
    text = re_sub(r"\"", r"")

    # Expand common contractions.
    text = re_sub(r"\swon\'t", " will not")
    text = re_sub(r"\scan\'t", " can not")
    text = re_sub(r"n\'t\s", " not ")
    text = re_sub(r"\'re\s", " are ")
    text = re_sub(r"\'s\s", " is ")
    text = re_sub(r"\'d\s", " would ")
    text = re_sub(r"\'ll\s", " will ")
    text = re_sub(r"\'t\s", " not ")
    text = re_sub(r"\'ve\s", " have ")
    text = re_sub(r"\'m\s", " am ")

    # Add white space around <tokens>.
    text = re_sub(r"(<[^\s>]+>)", r" \1 ")

    # Insert white space around any non-token < and > characters.
    text = re_sub(r"<([^\s>]+)(\s)", r"< \1\2")
    text = re_sub(r"(\s)([^\s<]+)>", r"\1\2 >")

    # Replace special ... character which has no embedding.
    text = re_sub(r"…", r" . <repeat> ")

    # Repeatedly insert white space around punctuation characters.
    old = ""
    punctuation = r"!\"#\$%&'\(\)\*\+,-\./:;<=>\?@\[\\\]\^_`\{\|\}~"
    while old != text:
        old = text
        text = re_sub(f"([{punctuation.replace('<', '')}])(\\S+)", r" \1 \2")
        text = re_sub(f"(\\S+)([{punctuation.replace('>', '')}])", r"\1 \2 ")

    # Remove duplicate or alternate white space characters.
    text = re_sub(r"\s+", r" ")
    text = text.strip()

    return text.lower()


if __name__ == '__main__':
    corpus_file = sys.argv[1]
    with open(corpus_file, encoding="utf-8") as corpus:
        for line in corpus:
            # Split and only tokenize the first part (second part is label).
            split_line = line.split('\t')
            tokens = tokenize(split_line[0])
            print(tokens + '\t' + split_line[1], end='')
