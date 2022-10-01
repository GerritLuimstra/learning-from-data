# Setting up the environment

In order to setup the environment that we used to train and test our models, the first thing to do would be to setup the conda environment.
We have created a conda environment yml file, so setting up the environent should be as simple as:
```{bash}
conda env create -f environment.yml
```
After the environment is created, you can activate it using:
```{bash}
conda activate learningfromdata_group20
```
Since we are using cross validation (cross.txt) while also having an addition test set (inference.txt), we first need to generate those seperate files.
This can be done by running:
```{bash}
python src/setup.py
```
however this should **not** be necessary as we already generated the cross.txt and inference.txt files.

### Running a model
**TODO**
