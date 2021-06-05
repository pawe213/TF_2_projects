# %%
import sklearn
import tensorflow as tf
from tensorflow.python.keras.backend import dropout



if tf.config.list_physical_devices('GPU') != list():
    print("Graphic card supported")
else:
    raise EnvironmentError('Graphic card not initialized')
# %%
from numpy.random import seed
seed(8)
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, model_selection 

# %%


def read_in_and_split_data(sk_dataset):
    images, targets = sk_dataset['data'], sk_dataset['target']
    train_data, test_data, train_targets, test_targets = model_selection.train_test_split(images, targets, test_size=0.1)
    return (train_data, test_data, train_targets, test_targets)


# %%
dataset_images = datasets.load_iris()
train_data, test_data, train_targets, test_targets = read_in_and_split_data(dataset_images)

# %%
train_targets = tf.keras.utils.to_categorical(np.array(train_targets))
test_targets = tf.keras.utils.to_categorical(np.array(test_targets))
print(train_data.shape)
# %%
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, input_shape=train_data[0].shape,
     kernel_initializer=tf.keras.initializers.he_uniform(),
     bias_initializer='one', activation='relu'),
     tf.keras.layers.Dense(128, activation='relu'),
     tf.keras.layers.Dense(128, activation='relu'),
     tf.keras.layers.Dense(128, activation='relu'),
     tf.keras.layers.Dense(128, activation='relu'),
     tf.keras.layers.Dense(64, activation='relu'),
     tf.keras.layers.Dense(64, activation='relu'),
     tf.keras.layers.Dense(64, activation='relu'),
     tf.keras.layers.Dense(64, activation='relu'),
     tf.keras.layers.Dense(3, activation='softmax')
])
# %%
opt = tf.keras.optimizers.Adam(learning_rate=(0.0001))
loss = tf.keras.losses.CategoricalCrossentropy()
model.compile(optimizer=opt, loss=loss, metrics=['acc'])
# %%
epochs = 800
history = model.fit(train_data, train_targets, batch_size=40, validation_split=0.15, epochs=epochs)
# %%
try:
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
except KeyError:
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
plt.title('Accuracy vs. epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='lower right')
plt.show() 
# %%
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss vs. epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper right')
plt.show() 
