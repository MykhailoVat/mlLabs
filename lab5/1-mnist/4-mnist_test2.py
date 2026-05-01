import os
os.environ["KERAS_BACKEND"] = "torch"

from keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt

mnist_model = load_model("model/mnist_model.keras")

file_names = ['entry2.png', 'entry5.png', 'entry3.png', 'entry7.png']
nums = ["2", "5", "3", "7"]
processed = []

for name in file_names:
    full_path = os.path.join('data/handw/', name)

    tst = 255 - cv2.imread(full_path, 0)
    tst = cv2.resize(tst, (28, 28))
    tst = tst.reshape((1,784))
    tst = tst.astype('float32') / 255

    processed.append(tst)

for tst, num in zip(processed, nums):
    predictions = mnist_model.predict(tst, verbose=0)
    prediction = np.argmax(predictions)
    confidence = np.max(predictions) * 100

    print(f"Predicted: {prediction} | Real: {num} | Confidence: {confidence:.2f}%")

#check data
plt.figure(figsize=(10, 5))
for i in range(4):
    plt.subplot(1, 4, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)

    display_img = processed[i].reshape(28, 28)

    plt.imshow(display_img, cmap='gray')
    plt.xlabel(f"Real: {nums[i]}")
plt.tight_layout()
plt.savefig('./plots/formatted_hndw.png')
plt.close()
