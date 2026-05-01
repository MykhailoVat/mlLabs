import os
os.environ["KERAS_BACKEND"] = "torch"

from keras.models import load_model
import numpy as np

data = np.load("data/datasets.npz")

test_images = data["test_images"]
test_labels = data["test_labels"]

cifar_model = load_model("model/cifar_model.keras")

test_loss, test_acc = cifar_model.evaluate(test_images, test_labels)

print("\nTest accuracy:", test_acc)
print("Test loss:", test_loss)