# Setting up the environment

In order to set up the environment that we used to train and test our models, the first thing to do would be to set up the conda environment. We have created a conda `environment.yml` file, so setting up the environment should be as simple as:
```{bash}
conda env create -f environment.yml
```
After the environment is created, you can activate it using:
```{bash}
conda activate learningfromdata_group20
```
Our train, development, and test files should already be present in the `data` directory. If not, they can be regenerated from a source `data/reviews.txt` file by running:
```{bash}
python src/setup.py
```

### Running a model

All model parameters we have used in our experiments can be specified as command line arguments. The program will display a description of each available command line argument by specifying the `--help` flag. For example:
```{bash}
python src/lfd_assignment3.py --help
```

Our baseline LSTM model can be run using:
```{bash}
python src/lfd_assignment3.py -e=data/glove_reviews.json
```
Our best LSTM model can be run using:
```{bash}
python src/lfd_assignment3.py -e=data/glove_reviews.json -o=adam -lr=1e-3 -s=100 -do=0.75
```
Our best pretrained language model can be run using:
```{bash}
python src/lfd_assignment3.py -lm=bert-base-uncased -tr -b=32 -s=100 -o=adam -lr=5e-5 -ld -p=3 -ep=10
```

These commands will print out a classification report. A confusion matrix will be plotted and saved if the `--confusion_matrix` flag is provided. To run these models on the test set the argument `-t=data/test.txt` should be provided.

**NOTE**: All these examples assume that the code is run from the same directory as this README file. If you want to run the code from another directory the paths shown above should be updated to reflect this. Additionally, the correct paths to the train and developments files will need to be specified manually.
