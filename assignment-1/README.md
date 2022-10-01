# Setting up the environment

In order to setup the environment that we used to train and test our models, the first thing to do would be to setup the conda environment.
We have created a conda environment yml file, so setting up the environent should be as simple as:
```{bash}
conda env create -f environment.yml
```
After the environment is created, you can activate it and run MLFlow, which we used to track our experiment results.
This can be done by issuing the following commands in the root folder of this project:
```{bash}
conda activate learningfromdata_group20
mlflow server --host localhost --port 5050 --backend-store-uri file:./mlflow/tracking --serve-artifacts --default-artifact-root ./mlflow/artifacts
```
Since we are using cross validation (cross.txt) while also having an addition test set (inference.txt), we first need to generate those seperate files.
This can be done by running:
```{bash}
python src/setup.py
```
however this should **not** be necessary as we already generated the cross.txt and inference.txt files.

### Running a model
The argument parser that we use is quite generic, allowing us to specify an arbitratry amount of parameters. For more information, you should check out the ```helper.py``` file and look at the documentation of the  ```create_arg_parser()``` function.

As an example, to run a random forest (rf) with a max depth of 10 and 50 estimators you can type from the root of the project folder:
```{bash}
python src/lfd_assignment1.py -cf data/cross.txt -m rf -a max_depth 10 n_estimators 50 n_jobs -1
```

To run our best models (as found on the test set), you can run either one of (first one recommended):
```{bash}
# Recommended
python src/lfd_assignment1.py -cf data/cross.txt -if data/inference.txt --tfidf -nr -m linearsvc -a C 0.05

# Also possible
python src/lfd_assignment1.py -cf data/cross.txt -if data/inference.txt --lemmatization -nr -m nb
python src/lfd_assignment1.py -cf data/cross.txt -if data/inference.txt --tfidf -nr -m svm -a C 0.5 kernel 'rbf'
python src/lfd_assignment1.py -cf data/cross.txt -if data/inference.txt -nr -m rf -a ccp_alpha 0.0005

# These have poor performance
python src/lfd_assignment1.py -cf data/cross.txt -if data/inference.txt -nr -m dt -a ccp_alpha 0.0005
python src/lfd_assignment1.py -cf data/cross.txt -if data/inference.txt --lemmatization -m knn -a n_neighbors 15
```

This will print out all the required metrics and will plot a confusion matrix using matplotlib.
To run our best model on a custom inference set (test set), simply replace ```data/inference.txt``` with the location of your test file.

**NOTE**: It is required to have the MLFlow server running in order to run the code. Otherwise a connection timed out error will be printed.

## MLFlow
Furthermore, all experiments that we ran can be found at localhost:5050 (or whatever port you used for MLFlow above).
To access the data, go to the Learning From Data Assignment 1 Experiment and click Refresh (this can take a minute to load).
