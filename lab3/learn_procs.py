from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
from lab3.plot_decision_regions import pdr

def learn_wine(classifier, res_file_name):
    wine = load_wine()

    X = wine.data[:,:2]
    y = wine.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    classifier.fit(X_train, y_train)

    X_comb = np.vstack((X_train, X_test))
    y_comb = np.hstack((y_train, y_test))
    t_ind = range(len(y_train),len(y_comb))

    pdr(X_comb, y_comb, classifier, test_idx=t_ind)
    plt.legend(loc='best')
    plt.savefig(f'./data/plots/{res_file_name}.png')
    plt.close()

    #accuracy = classifier.score(X_test, y_test)
    #print(f"\nOverall accuracy: {accuracy:.2%}")

    y_pred = classifier.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=wine.target_names, digits=4))


