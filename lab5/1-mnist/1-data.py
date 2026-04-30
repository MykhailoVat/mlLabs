import os
os.environ["KERAS_BACKEND"] = "torch"

from keras.datasets import mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

#load data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#check data
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap='gray')
    plt.xlabel(train_labels[i])
plt.savefig('./plots/check_data.png')
plt.close()

#format data
train_images = train_images.reshape((60000, 784))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 784))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

np.savez("data/datasets.npz",
         train_images=train_images,
         test_images=test_images,
         train_labels=train_labels,
         test_labels=test_labels)