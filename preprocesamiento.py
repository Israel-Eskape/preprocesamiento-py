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

df = pd.get_dummies(df, columns=["Sex"], drop_first=True) #Cambia el valor de las categorias por variables numéricas 
#Agrupar datos
# 1 = 0 a 5 años
# 2 = 6 a 12 años
# 3 = 13 a 18 años 
# 4 = 19 a 35 años
# 5 = 36 a 60 años
# 6 = 60 a 100 años
# #

bins = [0,5,12,18,35,60,100]
namesAge = ["1","2","3","4","5","6"]
df["Age"] = pd.cut(df["Age"],bins, labels=namesAge)

print(df.head(10))
