import os
os.environ["KERAS_BACKEND"] = "torch"

from keras.models import load_model
import numpy as np

data = np.load("data/datasets.npz")

test_images = data["test_images"]
test_labels = data["test_labels"]

fmnist_model = load_model("model/fmnist_model.keras")

test_loss, test_acc = fmnist_model.evaluate(test_images, test_labels)

print("Test accuracy:", test_acc)
print("\nTest loss:", test_loss)