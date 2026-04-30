import os
os.environ["KERAS_BACKEND"] = "torch"

from keras.models import load_model
import numpy as np

data = np.load("data/datasets.npz")

test_images = data["test_images"]
test_labels = data["test_labels"]

mnist_model = load_model("model/mnist_model.keras")

test_loss, test_acc = mnist_model.evaluate(test_images,test_labels)

print("Test accuracy:", test_acc)
print("\nTest loss:", test_loss)