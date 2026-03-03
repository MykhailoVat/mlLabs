import pandas as pd

nyc = pd.read_csv('data/data.csv')

print(' Raw data '.center(20, '#'))
print(nyc.head())
print(nyc.tail())

nyc.columns = ['Date','Temperature','Anomaly']
#print(nyc.Date.dtype)
nyc.Date = nyc.Date.floordiv(100)

print(' Formatted data '.center(20, '#'))
print(nyc.head())
print(nyc.tail())

nyc.to_csv('data/formatted_data.csv', index=False)

