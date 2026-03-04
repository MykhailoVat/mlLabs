import pandas as pd

rnyc = pd.read_csv('data/data.csv')

print(' Raw data '.center(20, '#'))
print(rnyc.head())
print(rnyc.tail())

rnyc.columns = ['Date', 'Temperature', 'Anomaly']
#print(nyc.Date.dtype)
rnyc.Date = rnyc.Date.floordiv(100)

print(' Formatted data '.center(20, '#'))
print(rnyc.head())
print(rnyc.tail())

rnyc.to_csv('data/formatted_data.csv', index=False)

