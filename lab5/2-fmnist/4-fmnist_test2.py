import os
os.environ["KERAS_BACKEND"] = "torch"

from keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt

mnist_model = load_model("model/fmnist_model.keras")

file_names = ['tshort.png', 'pants.png', 'shorts.png']
real = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
items = ["Футболка", "Штани", "Шорти"]
processed = []

for name in file_names:
    full_path = os.path.join('data/handw/', name)

    tst = 255 - cv2.imread(full_path, 0)
    tst = cv2.resize(tst, (28, 28))
    tst = tst.reshape((1,784))
    tst = tst.astype('float32') / 255

    processed.append(tst)

for tst, item in zip(processed, items):
    predictions = mnist_model.predict(tst, verbose=0)
    prediction = np.argmax(predictions)
    confidence = np.max(predictions) * 100

    print(f"Predicted: {real[prediction]} | Real: {item} | Confidence: {confidence:.2f}%")

#check data
plt.figure(figsize=(10, 5))
for i in range(3):
    plt.subplot(1, 4, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)

    display_img = processed[i].reshape(28, 28)

    plt.imshow(display_img, cmap='gray')
    plt.xlabel(f"Real: {items[i]}")
plt.tight_layout()
plt.savefig('./plots/formatted_hndw.png')
plt.close()
