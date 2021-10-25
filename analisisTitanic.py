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
#print(variableCatData)

#variable target
survived = train.groupby(['Survived']).count()['PassengerId']
#target vs sex
survivedSex = train.groupby(['Survived','Sex']).count()['PassengerId']
(survivedSex.unstack(level=0).plot.bar())
plt.show()

print (survivedSex)
