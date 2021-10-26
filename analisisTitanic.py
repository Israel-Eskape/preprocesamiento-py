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

survivedAge = train.groupby(['Sex','Age']).count()['PassengerId']
(survivedAge.unstack(level=0).plot.bar())

classEmbarked = train.groupby(['Pclass','Embarked']).count()['PassengerId']
(classEmbarked.unstack(level=0).plot.bar())

#plt.show()
#print (survivedAge)


#procesamiento de datos
#variables Survived Sex Age Pclass

procesData = train[['Survived', 'Sex', 'Age', 'Pclass']].head(3)
#procesDataInfo = train[['Survived', 'Sex', 'Age', 'Pclass']].info() #estudiar variables nulos
train[['Survived', 'Sex', 'Age', 'Pclass']].info()
datosNulosEdad = (train[train['Age'].isna()].groupby(['Sex', 'Pclass']).count()['PassengerId'].unstack(level = 0))

datosAge = (train[train['Age'].isna()].groupby(['SibSp','Parch']).count()['PassengerId'].unstack(level=0))
#calcular la media de la edad
mediaAge = (train['Age'].median())
train['Age'] = train['Age'].fillna(mediaAge)
train[['Survived', 'Sex', 'Age', 'Pclass']].info()


print(mediaAge)
