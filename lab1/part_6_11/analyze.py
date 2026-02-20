from titanic import titanic

# наймалолодший пасажир:
print("\n","Наймолодший пасажир:")
youngest_name = titanic.sort_values(by=["age"]).head(1)["name"]
print(youngest_name)

# найстарший пасажир:
print("\n","Найстарший пасажир:")
oldest_name = titanic.sort_values(by=["age"],na_position='first').tail(1)["name"]
print(oldest_name)

# Середній вік
print("\n","Середній вік:")
avg_age = titanic["age"].mean()
print(avg_age)

# Статистика по тим, що вижили
print("\n","Ті, що вижили:")
print(titanic[titanic["survived"] == "yes"])

# знайти жінок першого класу, знайти наймолодшу й найстаршу, кількість вцілілих
print("\n","Жінки першого класу:")
fst_class_women = titanic[(titanic["class"] == "1st") & (titanic["sex"] == "female")]
print(fst_class_women)

print("\n","Наймолодша серед них:")
f_c_women_sorted = fst_class_women[fst_class_women["age"] >= 0].sort_values(by=["age"])
print(f_c_women_sorted.head(1))

print("\n","Найстарша серед них:")
print(f_c_women_sorted.tail(1))

print("\n","Кількість вцілілих серед них:")
print(fst_class_women["survived"][fst_class_women["survived"] == "yes"].count())
