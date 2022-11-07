# Project Description

This directory contains the files relating to our experiments on the [OLID dataset](https://scholar.harvard.edu/malmasi/olid). This is a dataset of tweets that are labelled as offensive or inoffensive. The data can be found in the `data` directory.

We have performed experiments using classical models, which can be found in the `classical-model` directory. A version of the same experiments with additional preprocessing can be found in the `classical-model-preprocessing` directory.

We have also performed experiments using LSTMs and pretrained language models. The files and instructions for running these experiments can be found in the `lstm-model` and `language-model` respectively.

Finally, we have investigated the relation between tweet sentiment and tweet offensiveness. The files relating to this can be found in the `sentiment-enhanced-model` directory.

# Setting up the environment

In order to set up the environment that we used to train and test our models, the first thing to do would be to set up the conda environment. We have created a conda `environment.yml` file, so setting up the environment should be as simple as:
```{bash}
conda env create -f environment.yml
```
After the environment is created, you can activate it using:
```{bash}
conda activate learningfromdata_group20
```
