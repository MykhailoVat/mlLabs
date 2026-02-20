import pandas as pd
pd.set_option('display.precision', 2)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

#1) Отримати дані
titanic = pd.read_csv('https://vincentarelbundock.github.io/Rdatasets/csv/carData/TitanicSurvival.csv')

#2) Переглянути перші й останні 5 рядків
#print(titanic.head(),"\n")
#print(titanic.tail(),"\n")

#3) Перейменувати стовпці
titanic.columns = ["name","survived","sex","age","class"]
#print(titanic.head(),"\n")
#print(titanic.tail(),"\n")