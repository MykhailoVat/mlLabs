from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from lab4.tools import report
import numpy as np

data = np.load("shared/s_datasets.npz")

X_train = data["X_train"]
Y_train = data["Y_train"]
x_test = data["x_test"]
y_test = data["y_test"]

X = np.concatenate((data["X_train"], data["x_test"]))
y = np.concatenate((data["Y_train"], data["y_test"]))

for k in range(1, 10):
    model = KNeighborsClassifier(n_neighbors=k)

    #cross validation
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=kf)
    print(f"KNN (k={k}): {scores.mean():.2%} (+/- {scores.std():.2%})")

    #models themselves
    #print('\nModels themselves:')
    #model.fit(X_train, Y_train)
    #report(f"KNN (k={k})", model.predict(x_test), y_test)

