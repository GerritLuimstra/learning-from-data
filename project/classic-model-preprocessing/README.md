# Running a model

Running a model is done in three stages using the `train.py`, `predict.py`, and `evaluate.py` scripts. Training a model is done using the `train.py` script with a command such as:
```{bash}
python train.py -tf ../data/train.tsv -mf model-name -ff feature_stats -m 'et' -g -st -rm -a ccp_alpha 0.0001 class_weight :balanced: min_samples_split 5 n_jobs -1
```
Here the `-mf` argument specifies the location in which the trained model will be stored. The `-ff` argument specifies the location of the file that contains the vocabulary size and the top 10 best features (for the ExtraTrees classifier). The stored model can be read by `predict.py` to make predictions on new data. For example, to use the trained model on the development set one can run
```{bash}
python predict.py -if ../data/test.tsv -of output.out -mf model-name
```
Here the `-of` argument specifies the file where the predictions will be stored. These predictions can then be read by `evaluate.py` to obtain the macro F1 score. This is done by running:
```{bash}
python evaluate.py -tf ../data/test.tsv -of score.txt -pf output.out
```
Here the `-of` argument specifies the file where the macro F1 score will be written, but it should also be printed on the command line directly.

Because running these files separately is rather tedious we use the `pipeline.py` script to run the entire pipeline at once. This script stores all results in a user specified directory and can train the model multiple times in a row with different seeds to obtain a more accurate estimate of the model's performance. For example, one can run:

```{bash}
python pipeline.py -n experiment-name -r 5 -m 'et' -t -g -st -rm -a ccp_alpha 0.0001 class_weight :balanced: min_samples_split 5 n_jobs -1
```
This does 5 runs with the ExtraTreeClassifier which is also our best model. By providing the `-t` or `--test` flag to `pipeline.py` the trained models are also used to predict on `../data/test.tsv`. The results are stored in results/experiment-name. Note that `pipeline.py` can not be used to test a model on new data since the train, dev, and test files are hard-coded to be those we use for our experiments. Predicting on new data should be done using the individual `train.py`, `predict.py`, and `evaluate.py` scripts.
