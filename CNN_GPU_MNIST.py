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

# build model
input_shape = scaled_train_images[0].shape

model = tf.keras.Sequential([
    Conv2D(8, (3,3), input_shape=input_shape, activation='relu', padding='same'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
# %%
history  = model.fit(scaled_train_images, train_labels, epochs=5)
# %%
frame = pd.DataFrame(history.history)
acc_plot = frame.plot(y="acc", title="Accuracy vs Epochs", legend=False)
acc_plot.set(xlabel="Epochs", ylabel="Accuracy")

acc_plot = frame.plot(y="loss", title = "Loss vs Epochs",legend=False)
acc_plot.set(xlabel="Epochs", ylabel="Loss")

# %%

test = model.evaluate(scaled_test_images, test_labels)
test_loss, test_accuracy = test
print(f"Test loss: {test_loss}")
print(f"Test accuracy: {test_accuracy}")
# %%
