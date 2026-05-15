import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["KERAS_BACKEND"] = "torch"

from keras.models import load_model
from sklearn.metrics import classification_report
import numpy as np

def test_proc(num):
     model = load_model(f"models/model_{num}.keras")

     data = np.load("data/dataset_1.npz")

     X_test = data["X_test"]
     y_test = data["y_test"]

     y_pred = model.predict(X_test)
     y_pred = (y_pred > 0.5).astype(int)

     print(classification_report(y_test,y_pred, digits=4))

test_proc(8)
