from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import time
from distribute_framework import Distribute

dist = Distribute()
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

dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
fmnist_train_ds = dist.distribute_dataset(dataset)

fmnist_train_ds = fmnist_train_ds.shuffle(5000).batch(32)


#declare the network layers
model = dist.replicate_model(keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
]))

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
  grads = dist.simple_all_reduce(grads)
  # grads = dist.ring_all_reduce_large_lists(grads)
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
    
    # apply gradients to model
    
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

#make predictions
predictions = model.predict(test_images)

#graph
def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array, true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array, true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

# num_rows = 5
# num_cols = 3
# num_images = num_rows*num_cols
# plt.figure(figsize=(2*2*num_cols, 2*num_rows))
# for i in range(num_images):
#   plt.subplot(num_rows, 2*num_cols, 2*i+1)
#   plot_image(i, predictions[i], test_labels, test_images)
#   plt.subplot(num_rows, 2*num_cols, 2*i+2)
#   plot_value_array(i, predictions[i], test_labels)
# plt.tight_layout()
# plt.show()

