# Setup the environment

## Create an anaconda environment
```{bash}
conda create --name learningfromdata
```
## Setup the environment
```{bash}
conda activate learningfromdata
conda install --file requirements.txt
``` 

## Start MLFlow

Run the following from the project root
```{bash}
conda activate learningfromdata
mlflow server --host localhost --port 5000 --backend-store-uri file:./mlflow/tracking --serve-artifacts --default-artifact-root ./mlflow/artifacts
```

## Running a test

As an example, here is a run with a random forest (rf) with a max depth of 10 and 5 estimators
```{bash}
python src/lfd_assignment1.py -tf data/train.txt -df data/test.txt -m rf -a max_depth 10 n_estimators 5
```
As another example, here is a run with a SVM model (svm) with a linear kernel
```{bash}
python src/lfd_assignment1.py -tf data/train.txt -df data/test.txt -m svm -a kernel 'linear'
```
The result of these runs can be found in localhost:5000 (or whatever port you used for MLFlow above)