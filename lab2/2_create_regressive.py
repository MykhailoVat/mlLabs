import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

nyc = pd.read_csv('data/formatted_data.csv')

years = nyc['Date'][nyc['Date'] <= 2020]
#print(years)
temp = nyc['Temperature'][nyc['Date'] <= 2020]
#print(temp)

sns.regplot(x=years, y=temp, ci=None,scatter = False)
plt.savefig('./data/plots/linregr_partial.png')
plt.close()