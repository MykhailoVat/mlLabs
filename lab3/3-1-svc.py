from sklearn.svm import SVC
from learn_procs import learn_wine

svm = SVC(kernel='rbf',probability=True, random_state=1, gamma=0.1)

learn_wine(svm,'wine_svc')

