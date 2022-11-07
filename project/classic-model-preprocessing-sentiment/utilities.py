import argparse
import numpy as np
import random
import regex as re
import emoji

from nltk import pos_tag

FLAGS = re.MULTILINE | re.DOTALL

def read_tweets(corpus_file):
    """Read the tweets dataset and return tweets and toxicity labels."""
    documents = []
    labels = []
    with open(corpus_file, encoding='utf-8') as in_file:
        for line in in_file:
            tokens = line.split('\t')
            documents.append(tokens[0])
            labels.append(tokens[1][:-1])
    return documents, labels

def read_args():
    """Read the parsed command line arguments"""
    args = create_arg_parser()

    # Check if the classifier arguments are properly given
    if not len(args.args) % 2 == 0:
        print("Invalid arguments specified. Should be in the form: param1 value1 param2 value2")
        exit(0)

    # Parse the arguments
    params = args.args[0::2]
    values = parse_values(args.args[1::2])
    param_dict = dict(zip(params, values))

    return args, param_dict

def create_arg_parser():
    """Create an argument parser and return the parsed command line input."""

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train_file", type=str,
                        help="Txt file that will be used to perform cross-validation.")
    parser.add_argument("-d", "--dev_file", type=str,
                        help="File that will be used to perform testing.")
    parser.add_argument("-if", "--inference_file", default=None, type=str,
                        help="Optional test set to run inferences on. (default None).")
    parser.add_argument("-tf", "--tfidf", action="store_true",
                        help="Use the TF-IDF vectorizer instead of CountVectorizer")
    parser.add_argument("-l", "--lemmatization", action="store_true",
                        help="Whether to use Lemmatization (default False)")
    parser.add_argument("-s", "--stemming", action="store_true",
                        help="Whether to use Stemming (default False).")
    parser.add_argument("-n", "--ngram_range", type=int, default=1, 
                        help="The upper n-gram range. This includes n-grams in the range (1, n). (default 1)")
    parser.add_argument("-g", "--glove", action="store_true", default=False, 
                        help="Preprocess the data using glove embedding.")
    parser.add_argument("-pt", "--pos_tags", action="store_true", default=False, 
                        help="Use pos tagging to filter words.")
    parser.add_argument("-r", "--remove_emojis", action="store_true", default=False, 
                        help="Remove emojis.")
    parser.add_argument("-m", "--model_name", type=str, default='nb', help="The model to use. Can be one of ['nb', 'dt', 'rf', 'knn', 'svm']")
    parser.add_argument("-a", "--args", default=[], nargs='+', help="The arguments passed to the ML model")
    args = parser.parse_args()
    return args
    

def parse_values(values):
    """Parses the values of the classifier"""
    values_ = []
    for value in values:
        if ":" in value:
            values_.append(value.replace(":", ""))
        elif "." in value:
            values_.append(float(value))
        elif value == "None":
            values_.append(None)
        else:
            values_.append(int(value))
    return values_

def set_seed(seed):
    """Set the seed to obtain reproducible results."""

    np.random.seed(seed)
    random.seed(seed)

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

def rm_emojis(text):
    """ Remove emoijs
        https://gist.github.com/Alex-Just/e86110836f3f93fe7932290526529cd1
    """
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
    return text

def substitute_emoji(text):
    """Replace emoji with space separated words describing the emoji."""

    def replace_underscore(text):
        return text.group()[1:-1].replace("_", " ")

    text = emoji.demojize(text, delimiters=(" :", ": "))
    text = re.sub(r":\w+:", replace_underscore, text, flags=FLAGS)
    return text

def read_lexicon(path):
    with open(path) as f:
        lines = f.read().split("\n")
        lexicon_mapping = {}
        for line in lines:
            splits = line.split("\t")
            lexicon_mapping[splits[0]] = float(splits[1])
        return lexicon_mapping

def load_lexicon():
    # Load in the lexicon
    lexicon = read_lexicon("../data/unigrams-pmilexicon.txt")

    # Normalize the lexicon
    a = -1
    b = 1
    min_ = np.min(list(lexicon.values()), axis=0)
    max_ = np.max(list(lexicon.values()), axis=0)
    normalized = {}
    for word, value in lexicon.items():
        normalized[word] = (b - a) * (value - min_) / (max_ - min_) + a

    return normalized

def obtain_sentiment(tweet, reduction='mean'):
    if sum(np.isnan(tweet)) == len(tweet):
        return -999
    if reduction == 'mean':
        return np.nanmean(tweet)
    if reduction == 'median':
        return np.nanmedian(tweet)
    return tweet[np.nanargmax(np.abs(tweet))]

def glove_tokenize(text):
    """
    Based on script for preprocessing tweets by Romain Paulus with small
    modifications by Jeffrey Pennington with translation to Python by Motoki Wu.

    The original Ruby script:

    http://nlp.stanford.edu/projects/glove/preprocess-twitter.rb
    """
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

    return text

def pos_tagging(text):
    """Pos tagging by only considering nouns, proper nouns and adjectives"""
    pos_tags = pos_tag(text.split())
    text = [tag[0] for tag in pos_tags if tag[-1] in ["NN", "NNP", "JJ", "JJR", "JJS", "NNS", "NNPS"]]
    text = " ".join(text)
    return text

def preprocessor(text, glove=False, remove_emojis=False, pos_tags=False, stemmer=None, lemmatizer=None):
    """Preprocesses tweets."""

    # Emoji removal or substitution
    if remove_emojis:
        text = rm_emojis(text)
    else: 
        text = substitute_emoji(text)

    # Glove preprocessing
    if glove:
        text = glove_tokenize(text)

    # POS-tagging
    if pos_tags:
        text = pos_tagging(text)

    # Perform stemming or lemmatization
    text = text.split()
    if lemmatizer is not None:
        text = list(map(lambda word: lemmatizer.lemmatize(word), text))
    if stemmer is not None:
        text = list(map(lambda word: stemmer.stem(word), text))
    text = " ".join(text)

    return text.lower()
