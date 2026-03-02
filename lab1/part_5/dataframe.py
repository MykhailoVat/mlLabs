import pandas as pd

#робота з наявними даними:

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

data = pd.read_csv("../data_zdo/zdo_data.csv")

# Налаштування індексів DataFrame з використанням атрибута index
data_c1 = pd.DataFrame(data)
data_c1.index = [f"row{row}" for row in range(1, 36)]
print(data_c1, "\n")

# Звернення до стовпців DataFrame
print("######"*20)
print(data.Year, "\n")

# Вибір рядків з використанням атрибутів loc і iloc
print("######"*20)
print(data_c1.loc['row5'], "\n")
print(data.iloc[4], "\n")
print(data_c1.loc['row5':'row10'], "\n")
print(data.iloc[4:9],"\n", "\n")

# Вибір підмножин рядків і стовпців
print("######"*20)
print(data_c1.loc['row5':'row10',['Year']], "\n")
print(data_c1.iloc[[1,10],0:3],"\n")

# Логічне індексування
print("######"*20)
print(data[data != 2005], "\n")

# Звернення до конкретного осередку DataFrame по рядку і стовпцю
print("######"*20)
print(data_c1.at['row5','Year'], "\n") #= 1990
print(data_c1.iat[2,0],"\n") #= 1990

# Описова статистика
print("######"*20)
print(data.describe(), "\n")
print(data.median(), "\n")

# Транспонування DataFrame з використанням атрибута T
print("######"*20)
print(data.T, "\n")

#Сортування рядків за індексами
print("######"*20)
print(data.sort_index(ascending= False), "\n")
print(data_c1.sort_index(ascending= False), "\n")

# Сортування за індексами стовпців
print("######"*20)
print(data.sort_index(axis=1), "\n")

# Сортування за значеннями стовпців
print("######"*20)
print(data.sort_values(1,axis=1,ascending=False), "\n")

# робота з вигаданими даними

#Створення DataFrame на базі словника
print("######"*20)
dict = {"1990":[26,23.4],"1991":[24,22.4],"1992":[22.3,23.454]}
data_2 = pd.DataFrame(dict)
print(data_2)