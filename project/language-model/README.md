### Used language models
The exact version of the used language models can be found in the following table:
| Algorithm  | Model Name  |
|---|---|
| BERT | bert-base-uncased |
| DistilBERT | distilbert-base-uncased |
| RoBERTa | roberta-base |
| AlBERT | albert-base-v2 |
| Electra | google/electra-base-discriminator |
| DeBERTa | microsoft/deberta-base |
| BertTweet | vinai/bertweet-base |

### Running a model

All model parameters we have used in our experiments can be specified as command line arguments. The program will display a description of each available command line argument by specifying the `--help` flag. For example:
```{bash}
python main.py --help
```

The following experiments were ran and can be found in the report:
```{python}
# Models
python main.py -lm roberta-base -o adam -ep 10 -lr 0.000005 -b 64 -s 100 --trainable                       # W: 0.80, AUC: 0.795, M: 0.79
python main.py -lm bert-base-uncased -o adam -ep 10 -lr 0.000005 -b 64 -s 100 --trainable                  # W: 0.80, AUC: 0.784, M: 0.80
python main.py -lm google/electra-base-discriminator -o adam -ep 10 -lr 0.000005 -b 64 -s 100 --trainable  # W: 0.80, AUC: 0.786, M: 0.78
python main.py -lm distilbert-base-uncased -o adam -ep 10 -lr 0.000005 -b 64 -s 100 --trainable            # W: 0.81, AUC: 0.782, M: 0.79
python main.py -lm albert-base-v2 -o adam -ep 10 -lr 0.000005 -b 64 -s 100 --trainable                     # W: 0.80, AUC: 0.776, M: 0.78
python main.py -lm microsoft/deberta-base -o adam -ep 10 -lr 0.000005 -b 32 -s 100 --trainable             # W: 0.80, AUC: 0.769, M: 0.78
python main.py -lm vinai/bertweet-base -o adam -ep 10 -lr 0.000005 -b 64 -s 100 --trainable                # W: 0.80, AUC: 0.790, M: 0.79

# Sequence length
python main.py -lm bert-base-uncased -o adam -ep 10 -lr 0.000005 -b 32 -s 150 --trainable                  # W: 0.80, AUC: 0.781, M: 0.78
python main.py -lm bert-base-uncased -o adam -ep 10 -lr 0.000005 -b 32 -s 200 --trainable                  # W: 0.80, AUC: 0.787, M: 0.78
python main.py -lm bert-base-uncased -o adam -ep 10 -lr 0.000005 -b 32 -s 50  --trainable                  # W: 0.80, AUC: 0.779, M: 0.78

# Learning rate decay
python main.py -lm roberta-base -o adam -ep 10 -lr 0.000005 -b 64 -s 100 --trainable -ld                   # W: 0.80, AUC: 0.795, M: 0.79
python main.py -lm bert-base-uncased -o adam -ep 10 -lr 0.000005 -b 64 -s 100 --trainable -ld              # W: 0.79, AUC: 0.776, M: 0.77

# Just final layer
python main.py -lm roberta-base -o adam -ep 10 -lr 0.005 -b 64 -s 100                                     # W: 0.69, AUC: 0.636, M: 0.64
python main.py -lm vinai/bertweet-base -o adam -ep 10 -lr 0.005 -b 64 -s 100                              # W: 0.71, AUC: 0.659, M: 0.71
python main.py -lm bert-base-uncased -o adam -ep 10 -lr 0.005 -b 64 -s 100                                # W: 0.66, AUC: 0.608, M: 0.60

# Extra
python main.py -lm distilbert-base-uncased -o adam -ep 10 -lr 0.000005 -b 128 -s 100 --trainable          # W: 0.81, AUC: 0.782, M: 0.79
```

These commands will print out a classification report. A confusion matrix will be plotted and saved if the `--confusion_matrix` flag is provided. To run these models on the test set the argument `-t=../data/test.txt` should be provided.

**NOTE**: All these examples assume that the code is run from the same directory as this README file. If you want to run the code from another directory the paths shown above should be updated to reflect this. Additionally, the correct paths to the train and developments files will need to be specified manually.
