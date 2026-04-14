from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

def report(mname, predicted, expected):
    print(f"\n=== {mname} ===")

    names = [str(i) for i in range(10)]
    print(classification_report(expected, predicted, target_names=names, digits=4))

    cm = confusion_matrix(expected, predicted)
    for row in cm:
        print(" ".join(f"{num:3d}" for num in row))
