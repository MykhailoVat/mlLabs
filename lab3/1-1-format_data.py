import pandas as pd

rnyc = pd.read_csv('data/data.csv')

rnyc.columns = ['Date', 'Temperature', 'Anomaly']
rnyc.Date = rnyc.Date.floordiv(100)

print(' Formatted data '.center(20, '#'))
print(rnyc.head())
print(rnyc.tail())

nyc_1895_2020 = rnyc[rnyc['Date'] <= 2020]

nyc_1895_2020.to_csv('data/data_1895-2020.csv', index=False)

#print(nyc.Date.dtype)
