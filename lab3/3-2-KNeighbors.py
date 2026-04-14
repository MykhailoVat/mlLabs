from sklearn.neighbors import KNeighborsClassifier
from learn_procs import learn_wine

knm = KNeighborsClassifier(n_neighbors=5, weights='uniform')

learn_wine(knm,'wine_KNeighbors')

