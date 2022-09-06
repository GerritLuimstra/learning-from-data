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
