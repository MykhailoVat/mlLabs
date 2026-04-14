from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

#--- splitting

nyc = pd.read_csv('data/data_1895-2020.csv')

X_train, X_test, y_train, y_test = train_test_split(
    nyc.Date.values.reshape(-1, 1), nyc.Temperature.values,
    random_state=42
)

#print(X_test)
#print(y_train)

print(X_train.shape)
print(X_test.shape)

#--- learning

linreg = LinearRegression()
linreg.fit(X_train, y_train)

#y=kx+b
k=linreg.coef_[0]
b=linreg.intercept_

#print(k)
#print(b)

#--- testing

predicted = linreg.predict(X_test)
expected = y_test

for p,e in zip(predicted[::5],expected[::5]):
    print(f'predicted:{p:.2f},expected:{e:.2f}')
print('\n')

#--- prediction

predict = lambda x: k*x + b

for year in range(2021, 2027):
    print(f'prediction on {year}: {predict(year)}')

for year in range(1894, 1888, -1):
    print(f'prediction on {year}: {predict(year)}')

#--- visualizing
#scatter
axes = sns.scatterplot(nyc,x='Date',y='Temperature',
                       hue='Temperature', palette='muted', legend=False)
axes.set_ylim(10, 70)

#line
x=np.array([min(nyc.Date.values), max(nyc.Date.values)])
y = predict(x)

line = plt.plot(x,y)

plt.savefig('./data/plots/linreg.png')
plt.close()







