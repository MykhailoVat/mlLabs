from sklearn.naive_bayes import GaussianNB
from learn_procs import learn_wine

gnb = GaussianNB()

learn_wine(gnb,'wine_GaussianNB')

