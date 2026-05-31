import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.datasets import fashion_mnist

sns.set_style('whitegrid')

print('Loading data...')

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

def analyse_product_images(images, labels, target_label, product_name):
    boolean_mask =(labels == target_label)
    print(images)

    subset_images = images[boolean_mask]
    avg_image = np.mean(subset_images, axis=0)

    plt.figure(figsize=(8, 6))
    plt.imshow(avg_image, cmap='viridis')
    plt.title(f'Average Image of {product_name}s (Label {target_label})', fontsize=16)
    plt.axis('off')
    plt.show()

#x_train is the label eg. 0 for t-shirt/top, 1 for trouser, etc.
#y_train is the image data, a 28x28 pixel grayscale image

#analysing the images of a specific product, e.g. T-shirt/top (label 0)
analyse_product_images(x_train, y_train, target_label=0, product_name='T-shirt')
analyse_product_images(x_train, y_train, target_label=1, product_name='Trouser')