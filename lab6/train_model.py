import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["KERAS_BACKEND"] = "torch"

from keras import models
from keras import layers
import numpy as np

def train_model_1(x,y):
    network = models.Sequential()

    network.add(layers.Dense(64, activation='relu'))
    network.add(layers.Dense(1, activation='sigmoid'))

    network.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    network.fit(x, y, epochs=5, batch_size=32)

    network.save("models/model_1.keras")

    return network

def train_model_2(x,y):
    network = models.Sequential()

    network.add(layers.Dense(64, activation='relu'))
    network.add(layers.Dense(32, activation='relu'))
    network.add(layers.Dense(1, activation='sigmoid'))

    network.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    network.fit(x, y, epochs=5, batch_size=32)

    network.save("models/model_2.keras")

    return network

def train_model_3(x,y):
    network = models.Sequential()

    network.add(layers.Dense(64, activation='silu'))
    network.add(layers.BatchNormalization())
    network.add(layers.Dense(32, activation='silu'))
    network.add(layers.BatchNormalization())
    network.add(layers.Dense(1, activation='sigmoid'))

    network.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    network.fit(x, y, epochs=5, batch_size=64)

    network.save("models/model_3.keras")

    return network

def train_model_5(x,y):
    network = models.Sequential()

    network.add(layers.Dense(64, activation='relu'))
    network.add(layers.Dense(1, activation='sigmoid'))

    network.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    network.fit(x, y, epochs=5, batch_size=16)

    network.save("models/model_5.keras")

    return network

def train_model_6(x,y):
    network = models.Sequential()

    network.add(layers.Dense(64, activation='gelu'))
    network.add(layers.Dense(1, activation='sigmoid'))

    network.compile(optimizer='adam', loss='binary_focal_crossentropy', metrics=['accuracy'])

    network.fit(x, y, epochs=5, batch_size=32)

    network.save("models/model_6.keras")

    return network

def train_model_7(x,y):
    network = models.Sequential()

    network.add(layers.Dense(32, activation='relu'))
    network.add(layers.Dense(1, activation='sigmoid'))

    network.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    network.fit(x, y, epochs=5, batch_size=32)

    network.save("models/model_7.keras")

    return network

def train_model_8(x,y):
    network = models.Sequential()

    network.add(layers.Dense(64, activation='relu'))
    network.add(layers.Dense(32, activation='relu'))
    network.add(layers.Dense(1, activation='sigmoid'))

    network.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    network.fit(x, y, epochs=5, batch_size=32)

    network.save("models/model_8.keras")

    return network


data = np.load("data/dataset_1.npz")

X_train = data["X_train"]
y_train = data["y_train"]

print(X_train.shape)
print(y_train.shape)

train_model_8(X_train,y_train)
