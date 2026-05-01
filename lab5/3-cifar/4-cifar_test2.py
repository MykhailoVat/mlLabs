import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["KERAS_BACKEND"] = "torch"

from keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt

cifar_model = load_model("model/cifar_model.keras")

file_names = ['frog.png', 'plane.png', 'car.png']
items = ['frog', 'airplane', 'automobile']
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck']
processed = []

for name in file_names:
    full_path = os.path.join('data/handw/', name)

    tst = cv2.imread(full_path)
    tst = cv2.cvtColor(tst, cv2.COLOR_BGR2RGB)
    tst = cv2.resize(tst, (32, 32))
    tst = tst.astype('float32') / 255
    tst = np.expand_dims(tst, axis=0)

    processed.append(tst)

for tst, item in zip(processed, items):
    predictions = cifar_model.predict(tst, verbose=0)
    prediction = np.argmax(predictions)
    confidence = np.max(predictions) * 100

    print(f"Predicted: {labels[prediction]} | Real: {item} | Confidence: {confidence:.2f}%")

#check data
plt.figure(figsize=(10, 5))
for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)

    display_img = processed[i][0]

    plt.imshow(display_img, cmap='gray')
    plt.xlabel(f"Real: {labels[i]}")
plt.tight_layout()
plt.savefig('./plots/formatted_hndw.png')
plt.close()
