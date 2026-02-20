import matplotlib.pyplot as plt
from titanic import titanic

titanic.hist(column="age")
plt.savefig("histograms_t/titanic_age.png")
plt.close()

