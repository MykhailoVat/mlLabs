import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

nyc = pd.read_csv('data/formatted_data.csv')

sns.set_style('whitegrid')
axes = sns.regplot(x=nyc['Date'], y=nyc['Temperature'],ci=None)
plt.savefig('./data/plots/linregr_full.png')
#plt.close()

#formatting

axes.set_ylim(10,70)

plt.savefig('./data/plots/linregr_formatted.png')
plt.close()