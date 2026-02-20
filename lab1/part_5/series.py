import pandas as pd
from lab1.tools.get_data_column import get_data_column

#робота з наявними даними:

data_col = get_data_column(4)
data_col_str = list(map(lambda x: str(x), data_col))

# Створення Series з індексами за замовчуванням + Виведення колекції Series
coverage = pd.Series(data_col)
print(coverage, "\n")

# Звернення до елементів Series
print("######"*20)
print(coverage[1], "\n")

# Обчислення описових статистик для Series
print("######"*20)
print(coverage.count())
print(coverage.mean())
print(coverage.min())
print(coverage.max())
print(coverage.std())
print(coverage.describe(), "\n")

# Створення колекції Series з нестандартними індексами
print("######"*20)
coverage_unusual_i = pd.Series(data_col, index=range(1990, 2025))
print(coverage_unusual_i, "\n")

# Звернення до елементів Series з використанням нестандартних індексів
print("######"*20)
print(coverage_unusual_i[1990], "\n")

# Створення колекції Series із строковими елементами
print("######"*20)
coverage_str = pd.Series(data_col_str)
print(coverage_str, "\n")
print(coverage_str.str.contains("5"),"\n")

#робота з вигаданими даними:

# Створення колекції Series з однаковими значеннями
print("######"*20)
print(pd.Series(5,range(5)),"\n")

# Словники як ініціалізатор
print("######"*20)
dictionary = {"1990":50,"1991":55,"1992":60}
coverage_from_dict = pd.Series(dictionary)
print(coverage_from_dict, "\n")





