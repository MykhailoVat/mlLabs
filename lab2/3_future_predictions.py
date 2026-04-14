import pandas as pd
import scipy.stats as stats

nyc = pd.read_csv('./data/formatted_data.csv')
linear_regression = stats.linregress(nyc['Date'], nyc['Temperature'])

#print(linear_regression.slope)
#print(linear_regression.intercept)

pred_2021 = linear_regression.slope * 2021 + linear_regression.intercept
pred_2022 = linear_regression.slope * 2022 + linear_regression.intercept
pred_2023 = linear_regression.slope * 2023 + linear_regression.intercept
pred_2024 = linear_regression.slope * 2024 + linear_regression.intercept
pred_2025 = linear_regression.slope * 2025 + linear_regression.intercept
pred_2026 = linear_regression.slope * 2026 + linear_regression.intercept

print('Predictions')
print('2021:', pred_2021)
print('2022:', pred_2022)
print('2023:', pred_2023)
print('2024:', pred_2024)
print('2025:', pred_2025)
print('2026:', pred_2026)
