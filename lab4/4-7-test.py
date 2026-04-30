from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import joblib

data = np.load("shared/s_datasets.npz")
x_test = data["x_test"]
y_test = data["y_test"]

knn = joblib.load("shared/knn_model.pkl")

predicted = knn.predict(x_test)
expected = y_test

#print(f"Predicted:{predicted[:36]}")
#print(f"Expected: {expected[:36]}")

#print(f"{knn.score(x_test, y_test):.2%}")

#print(f"{confusion_matrix(y_true=expected, y_pred=predicted)}")

#names = [str(dig) for dig in range(0,10)]
#print(classification_report(expected, predicted, target_names=names))

print("Pred vs Exp (first 36):")
print(" ".join(f"{p}/{e}" for p, e in zip(predicted[:36], expected[:36])))

print(f"\nAccuracy: {knn.score(x_test, y_test):.2%}")

print("\nConfusion matrix:")
cm = confusion_matrix(expected, predicted)
for row in cm:
    print(" ".join(f"{num:3d}" for num in row))

print("\nReport:")
names = [str(i) for i in range(10)]
print(classification_report(expected, predicted, target_names=names, digits=4))