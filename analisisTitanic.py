import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

#Carga de datos
train = pd.read_csv('train.csv')
#print (train.head())

#exploración de los datos
nameColumn = train.columns
tamData = train.shape
#infoData = train.info()
variableData = train.describe()
variableCatData = train.describe(include=['O'])
print(variableCatData)
