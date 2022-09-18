# Setup the environment

## Create an anaconda environment
```{bash}
conda env create -f environment.yml
```

## Start MLFlow

Run the following from the project root
```{bash}
conda activate learningfromdata_group20
mlflow server --host localhost --port 5050 --backend-store-uri file:./mlflow/tracking --serve-artifacts --default-artifact-root ./mlflow/artifacts
```

## Setup the cross validation set and inference set

## Running a test

As an example, here is a run with a random forest (rf) with a max depth of 10 and 5 estimators
```{bash}
python src/lfd_assignment1.py -cd data/cross.txt -m rf -a max_depth 10 n_estimators 5
```
As another example, here is a run with a SVM model (svm) with a linear kernel
```{bash}
python src/lfd_assignment1.py -tf data/cross.txt -m svm -a kernel 'linear'
```
The result of these runs can be found in localhost:5050 (or whatever port you used for MLFlow above)