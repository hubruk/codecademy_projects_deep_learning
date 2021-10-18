import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split
from utils import load_galaxy_data
from tensorflow.keras import activations
from tensorflow.keras import layers

import app

input_data, labels = load_galaxy_data()
print(input_data.shape, labels.shape)

input_data_train, input_data_test, labels_train, labels_test = train_test_split(input_data, labels, test_size = 0.2, shuffle=True, random_state = 222, stratify=labels)

img_gen = ImageDataGenerator(rescale=1./255)
input_data_train_flow = img_gen.flow(input_data_train, labels_train, batch_size=5)
input_data_validation_flow = img_gen.flow(input_data_train, labels_train, batch_size=5)

model = tf.keras.Sequential()
model.add(tf.keras.Input(shape = (128, 128, 3)))
model.add(tf.keras.layers.Dense(4))
model.add(layers.Activation(activations.softmax))

model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001), loss = tf.keras.losses.CategoricalCrossentropy(), metrics = [tf.keras.metrics.CategoricalAccuracy(),tf.keras.metrics.AUC()])

