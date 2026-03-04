import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

fnyc = pd.read_csv('data/formatted_data.csv')

years = fnyc['Date'][fnyc['Date'] <= 2020]
#print(years)
temp = fnyc['Temperature'][fnyc['Date'] <= 2020]
#print(temp)

sns.regplot(x=years, y=temp,ci=None,scatter = False)
plt.savefig('./data/plots/linear_regression.png')
plt.close()