#%%
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Flatten



if tf.config.list_physical_devices('GPU') != list():
    print("Graphic card supported")
else:
    raise EnvironmentError('Graphic card not initialized')

# get mnsit
mnist_dataset = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist_dataset.load_data()

#scale 
scaled_train_images, scaled_test_images = train_images/255.0, test_images/255.0
scaled_train_images, scaled_test_images = scaled_train_images[..., np.newaxis], scaled_test_images[..., np.newaxis]

