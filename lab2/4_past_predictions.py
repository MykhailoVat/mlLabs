import pandas as pd
import scipy.stats as stats

nyc = pd.read_csv('./data/formatted_data.csv')

linear_regression = stats.linregress(nyc['Date'], nyc['Temperature'])

#print(linear_regression.slope)
#print(linear_regression.intercept)

pred_1894 = linear_regression.slope * 1894 + linear_regression.intercept
pred_1893 = linear_regression.slope * 1893 + linear_regression.intercept
pred_1892 = linear_regression.slope * 1892 + linear_regression.intercept
pred_1891 = linear_regression.slope * 1891 + linear_regression.intercept
pred_1890 = linear_regression.slope * 1890 + linear_regression.intercept
pred_1889 = linear_regression.slope * 1889 + linear_regression.intercept

print('Predictions')
print('1894:', pred_1894)
print('1893:', pred_1893)
print('1892:', pred_1892)
print('1891:', pred_1891)
print('1890:', pred_1890)
print('1889:', pred_1889)