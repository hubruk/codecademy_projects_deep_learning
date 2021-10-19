import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from visualize import visualize_activations

from sklearn.model_selection import train_test_split
from utils import load_galaxy_data
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
model.add(tf.keras.layers.Conv2D(8, 3 ,strides = 2, activation = "relu"))
model.add(tf.keras.layers.MaxPooling2D(pool_size =(2, 2), strides =(2,2)))
model.add(tf.keras.layers.Conv2D(8, 3 ,strides = 2, activation = "relu"))
model.add(tf.keras.layers.MaxPooling2D(pool_size =(2, 2), strides =(2,2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(16,activation = 'relu'))
model.add(tf.keras.layers.Dense(4,activation = 'softmax'))

model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001), loss = tf.keras.losses.CategoricalCrossentropy(), metrics = [tf.keras.metrics.CategoricalAccuracy(),tf.keras.metrics.AUC()])
model.summary()

hist = model.fit(input_data_train_flow, steps_per_epoch = len(input_data_train)/5,epochs = 12, validation_data = input_data_validation_flow, validation_steps = len(input_data_validation_flow)/5)

visualize_activations(model, input_data_validation_flow)

print(hist.history)
