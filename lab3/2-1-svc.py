from sklearn.svm import SVC
from matplotlib import pyplot as plt
import numpy as np
from lab3.plot_decision_regions import pdr

#--- data

np.random.seed(1)
X_xor = np.random.randn(200,2)
y_xor = np.logical_xor(X_xor[:,0] > 0, X_xor[:,1] > 0)
y_xor = np.where(y_xor, 1, -1)
plt.scatter(X_xor[y_xor == 1, 0], X_xor[y_xor == 1, 1],
            c='b', marker='x', label='1')
plt.scatter(X_xor[y_xor == -1, 0], X_xor[y_xor == -1, 1],
            c='r', marker='s', label='-1')

plt.xlim([-3,4])
plt.ylim([-3,3])
plt.legend(loc='best')
plt.savefig('./data/plots/xor_data.png')
plt.close()

#--- SVC

svm = SVC(kernel='rbf', random_state=1, gamma=0.1, C=10)
svm.fit(X_xor, y_xor)
pdr(X_xor, y_xor, svm)
plt.legend(loc='best')
plt.savefig('./data/plots/svc.png')
plt.close()
