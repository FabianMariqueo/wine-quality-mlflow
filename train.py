import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier

import mlflow
from mlflow import sklearn


data = pd.read_csv("https://raw.githubusercontent.com/cvidalse/modelo-prediccion/master/data_exp.csv",parse_dates=['Fecha'])

def modelo_prediccion(data):
  X=data.drop(columns=['Fecha', 'PM2.5'])
  y=data['PM2.5']
  X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=4)
  knn = KNeighborsClassifier(n_neighbors=5)#parametro n_neighbors cantidad de vecinos a utilizar
  knn.fit(X_train, y_train)#fit entrena el modelo con los valores X_train y y_train
  return knn


modelo = modelo_prediccion(data)

##mlflow.set_experiment('Prediccion-contaminacion')
with mlflow.start_run():
    
    conda = {
        'name': 'mlflow-env',
        'channels': ['defaults'],
        'dependencies': [
            'python=3.7.0',
            'scikit-learn=0.19.2'
        ]
    }

    mlflow.sklearn.log_model(modelo, 'model', serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE, conda_env = conda)
    artifact_path = mlflow.get_artifact_uri()
    artifact_path = artifact_path.replace("file://", "")

    os.environ["MLFLOW-ARTIFACT-PATH"] = artifact_path