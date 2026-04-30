import os
os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
from keras import models
from keras import layers

data = np.load("data/datasets.npz")

train_images = data["train_images"]
train_labels = data["train_labels"]

network = models.Sequential()

network.add(layers.Dense(512, activation='relu'))
network.add(layers.Dense(10, activation='softmax'))

network.compile(optimizer='adam',loss = 'categorical_crossentropy',metrics=['accuracy'])

network.fit(train_images,train_labels,epochs=10,batch_size=128)

network.summary()

network.save("model/fmnist_model.keras")