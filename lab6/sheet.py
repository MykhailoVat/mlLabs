import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["KERAS_BACKEND"] = "torch"

import pandas as pd
import joblib
from keras.src.saving import load_model
from generators import generate_dataset

dataset , _ = generate_dataset(1)

scaler = joblib.load("models/scaler.pkl")
dataset_n = scaler.transform(dataset)

print(dataset_n.shape)

model = load_model(f"models/model_1.keras")
prediction = model.predict(dataset_n)
prediction = (prediction > 0.5).astype(int)

prediction = prediction.flatten()

df = pd.DataFrame(dataset)

df["prediction"] = prediction

df.to_excel("data/results.xlsx", index=False)