import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier


data = pd.read_csv("https://raw.githubusercontent.com/cvidalse/modelo-prediccion/master/data_exp.csv",parse_dates=['Fecha'])

def modelo_prediccion(data):
  X=data.drop(columns=['Fecha', 'PM2.5'])
  y=data['PM2.5']
  X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=4)
  knn = KNeighborsClassifier(n_neighbors=5)#parametro n_neighbors cantidad de vecinos a utilizar
  knn.fit(X_train, y_train)#fit entrena el modelo con los valores X_train y y_train
  return knn


def get_model():
    return modelo_prediccion(data)