from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

digits = load_digits()

def show_digits(axes, fname):
    for item in zip(axes.ravel(), digits.images, digits.target):
        axs, image, target = item
        axs.imshow(image, cmap='gray_r')

        axs.set_xticks([])
        axs.set_yticks([])

        axs.set_title(target)

    plt.tight_layout()
    plt.savefig(f'plots/{fname}.png')
    plt.close()

figure1, axes1 = plt.subplots(nrows=4, ncols=6, figsize=(6, 4))
show_digits(axes1, 'digits24')
figure2, axes2 = plt.subplots(nrows=4, ncols=8, figsize=(8, 4))
show_digits(axes2, 'digits36')







