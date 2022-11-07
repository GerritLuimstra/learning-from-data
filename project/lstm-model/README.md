# Setting up the environment

In order to set up the environment that we used to train and test our models, the first thing to do would be to set up the conda environment. We have created a conda `environment.yml` file, so setting up the environment should be as simple as:
```{bash}
conda env create -f environment.yml
```
After the environment is created, you can activate it using:
```{bash}
conda activate lstm_model
```

### Running a model

Running a model is done in three stages using the `train.py`, `predict.py`, and `evaluate.py` scripts. Training a model is done using the `train.py` script with a command such as:
```{bash}
python train.py -tf ../data/train_glove.tsv -vf ../data/dev_glove.tsv -mf model-name -ef ../data/glove_twitter_200d.pickle
```
Here the `-mf` argument specifies the name of the directory in which the trained model will be stored. This can be read by `predict.py` to make predictions on new data. For example, to use the trained model on the development set one can run
```{bash}
python predict.py -if ../data/dev_glove.tsv -of output.out -mf model-name
```
Here the `-of` argument specifies the file where the predictions will be stored. These predictions can then be read by `evaluate.py` to obtain the macro F1 score. This is done by running:
```{bash}
python evaluate.py -tf ../data/dev_glove.tsv -of score.txt -pf output.out
```
Here the `-of` argument specifies the file where the macro F1 score will be written, but it should also be printed on the command line directly.

Because running these files separately is rather tedious we use the `pipeline.py` script to run the entire pipeline at once. This script stores all results in a user specified directory and can train the model multiple times in a row with different seeds to obtain a more accurate estimate of the model's performance. For example, one can run:

```{bash}
python pipeline.py -n experiment-name -r 10 -ef ../data/glove_twitter_200d.pickle
```
This does 10 runs of our baseline LSTM model which is also one of our best models. By providing the `-t` or `--test` flag to `pipeline.py` the trained models are also used to predict on `../data/test_glove.tsv`. Note that `pipeline.py` can not be used to test a model on new data since the train, dev, and test files are hard-coded to be those we use for our experiments. Predicting on new data should be done using the individual `train.py`, `predict.py`, and `evaluate.py` scripts.
