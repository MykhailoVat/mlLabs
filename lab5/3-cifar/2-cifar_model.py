import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
from keras import models
from keras import layers

data = np.load("data/datasets.npz")

train_images = data["train_images"]
train_labels = data["train_labels"]

network = models.Sequential()

#l1
network.add(layers.Conv2D(32, (3, 3), activation='relu'))
network.add(layers.BatchNormalization())
network.add(layers.MaxPooling2D((2, 2)))
network.add(layers.Dropout(0.3))

#l2
network.add(layers.Conv2D(64, (3, 3), activation='relu'))
network.add(layers.BatchNormalization())
network.add(layers.MaxPooling2D((2, 2)))
network.add(layers.Dropout(0.3))

#l3
network.add(layers.Conv2D(128, (3, 3), activation='relu'))
network.add(layers.BatchNormalization())
network.add(layers.MaxPooling2D((2, 2)))
network.add(layers.Dropout(0.3))

#l5
network.add(layers.Conv2D(128, (3, 3), activation='relu'))
network.add(layers.BatchNormalization())
network.add(layers.MaxPooling2D((2, 2)))
network.add(layers.Dropout(0.3))

#l6
network.add(layers.Flatten())
network.add(layers.Dense(128, activation='relu'))
network.add(layers.BatchNormalization())
network.add(layers.Dropout(0.4))
network.add(layers.Dense(10, activation='softmax'))

network.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

network.fit(train_images, train_labels, epochs=20, batch_size=64)

network.save("model/cifar_model.keras")
