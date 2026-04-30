import os
os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
from keras import models
from keras import layers

data = np.load("data/datasets.npz")

train_images = data["train_images"]
train_labels = data["train_labels"]

network = models.Sequential()

# Перший блок згорток: шукаємо прості форми
network.add(layers.Conv2D(32, (3, 3), activation='relu'))
network.add(layers.MaxPooling2D((2, 2)))

# Другий блок: шукаємо складніші структури
network.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Третій блок: глибоке вивчення ознак
network.add(layers.Conv2D(128, (3, 3), activation='relu'))

# Перетворення 2D-карт у 1D-вектор для фінальної класифікації
network.add(layers.Flatten())
network.add(layers.Dense(32, activation='relu'))
# Вихідний шар: 10 класів об'єктів
network.add(layers.Dense(10, activation='softmax'))

network.compile(optimizer='adam',loss = 'categorical_crossentropy',metrics=['accuracy'])

network.fit(train_images,train_labels,epochs=10,batch_size=128)

network.summary()

network.save("model/cifar_model.keras")