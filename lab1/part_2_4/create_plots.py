import matplotlib.pyplot as plt
from lab1.tools.get_data_column import get_data_column

#number of rows
BINS = 20

hists = [
    (1,"institutes","institutes(thsd)"),
    (2,"places","places(thsd)"),
    (3,"persons","persons(thsd)"),
    (4,"coverage","coverage(%)")
]

for col,name,xlb in hists:
    plt.hist(get_data_column(col), bins=BINS)
    plt.grid(True)
    plt.xlabel(xlb)
    plt.ylabel("Frequency")
    plt.savefig(f"histograms/{name}.png")
    plt.clf()


