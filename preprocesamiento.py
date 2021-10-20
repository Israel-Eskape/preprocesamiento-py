from os import replace
import pandas as pd 
import numpy as np

df = pd.read_csv('train.csv')
alldata = df.head() #muestra todos los datos
rows = df.tail()  #muestra las últimas n filas
datatypes = df.dtypes #muestra los tipos de datos de las columnas
describedata = df.describe(include="all") #Analisis estadístico de los datos

rowRemove = df.dropna(axis=0)    #dropna elimina datos faltantes
                            # axis = 0 elimina fila and axis = 1 elimina columnas 
#Elimina datos en una columna especifica
removeRowName = df.dropna(subset=["Cabin"], axis = 0)

promedio = df["Age"].mean()
promedio = round(promedio, 0)

replaceAge = df["Age"].replace(np.nan,promedio)
replaceAge = replaceAge.head(50)
print(replaceAge)
