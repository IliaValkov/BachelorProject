from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import time

print(f"TensorFlow version: {tf.__version__}")

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(f"Training images shape: {train_images.shape}")
print(f"Amount of training labels: {len(train_labels)}")


# normalize data 
train_images = train_images / 255.0

test_images = test_images / 255.0


fmnist_train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
fmnist_train_ds = fmnist_train_ds.shuffle(5000).batch(32)


#declare the network layers
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

optimizer = tf.keras.optimizers.Adam()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

def loss(model, x, y):
  y_ = model(x)

  return loss_object(y_true=y, y_pred=y_)


def training_step(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets)

  grads = tape.gradient(loss_value, model.trainable_variables)
 
  # apply gradients to model
  optimizer.apply_gradients(zip(grads, model.trainable_variables))

  return loss_value  

num_epochs = 2
train_loss_results = []
train_accuracy_results = []

start = time.perf_counter()
for epoch in range(num_epochs):
  
  # COMPUTES THE (WEIGHTED) MEAN OF THE GIVEN VALUES
  epoch_loss_avg = tf.keras.metrics.Mean()
  
  # CALCULATES HOW OFTEN PREDICTIONS MATCHES INTEGER LABELS
  epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

  # TRAINING LOOP
  for x, y in fmnist_train_ds:
    # Optimize the model
   
    # Compute loss value and gradients
    loss_value = training_step(model, x, y)
    
    # Track progress
    epoch_loss_avg(loss_value)  # Add current batch loss
    # Compare predicted label to actual label
    epoch_accuracy(y, model(x))

  # End epoch
  train_loss_results.append(epoch_loss_avg.result())
  train_accuracy_results.append(epoch_accuracy.result())
  print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,epoch_loss_avg.result(),epoch_accuracy.result()))

end = time.perf_counter()
print(f"Finished training loop in {round(end-start,2)} second(s).")

#calculate accuracy
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

