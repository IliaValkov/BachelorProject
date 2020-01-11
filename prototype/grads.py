from __future__ import absolute_import, division, print_function, unicode_literals 
import os 
import matplotlib.pyplot as plt 

import tensorflow as tf 
import time

from my_framework_prototype import Dist



print(f"Tensorflow vesion: {tf.__version__}") 
print(f"Eager execition: {tf.executing_eagerly()}")

# GET THE DATA
train_dataset_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"
train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(train_dataset_url),
                                           origin=train_dataset_url)
print("Local copy of the dataset file: {}".format(train_dataset_fp))


# PREPARE THE FEATURES AND LABELS NAMES
column_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
features_names = column_names[:-1]
label_name = column_names[-1]
print(f"Features names: {features_names}")
print(f"Label: {label_name}")


# DECLARE CLASS NAMES
class_names = ["Iris_setosa", "Iris_versicolor", "Iris_virginica"]

# SPECIFY BATCH SIZE AND FORMAT THE DATA USING DATASET
batch_size = 10
print(f"batch_size: {batch_size}")
train_dataset = tf.data.experimental.make_csv_dataset(
    train_dataset_fp,
    batch_size,
    column_names = column_names,
    label_name = label_name,
    num_epochs = 10)

def pack_features_vector(features, labels):
  """Pack the features into a single array."""
  features = tf.stack(list(features.values()), axis=1)
  return features, labels

# CREATE SUITABLE FEATURES-LABEL PAIRS
train_dataset = train_dataset.map(pack_features_vector)

# DECLARE THE MODEL
model = tf.keras.Sequential([
  tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,)),  # input shape required
  tf.keras.layers.Dense(10, activation=tf.nn.relu),
  tf.keras.layers.Dense(3)
])

# DECALARE A LOSS FUNCTION
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


# FUNCTION TO CALCULATE THE LOSS
def loss(model, x, y):
  y_ = model(x)

  return loss_object(y_true=y, y_pred=y_)

# CALCULATE LOSS PRE-TRAINING
features, labels = next(iter(train_dataset))

l = loss(model, features, labels)
print("Loss test: {}".format(l))

with tf.GradientTape() as tape:
    loss_value = loss(model, features, labels)

grads = tape.gradient(loss_value, model.trainable_variables)

ds = Dist()
import numpy as np
deconstructed = ds.deconstruct(grads)

# print(deconstructed)

array_form_tensor = deconstructed.numpy()
        
new_list = np.array_split(array_form_tensor,2)
new_list = [tf.convert_to_tensor(arr) for arr in new_list]
print(new_list)


# reconstructed = ds.reconstruct(deconstructed)

# print(reconstructed)

# print(grads[2])

# tensor_shapes = []
# tensor_sizes = []

# tensor_shapes_n = tf.shape_n(grads)

# for i, t in enumerate(grads): 
#     tensor_sizes.append(tf.size(t))
#     if tf.rank(t) > 1: 
#         print(f"t size is {tf.size(t)}")
#         grads[i] = tf.reshape(t, [tf.size(t)])

# # print(f"tensor_sizes {tensor_sizes}")
# # print(f"tensor_shapes {tensor_shapes}")


# new_t = tf.concat(grads,0)    
# print(new_t)
# # here im ready to all reduce
# offset = 0
# # after all reduce i have rebuild the shape of the tensor
# reconstructed_grads = []
# for size, shape in zip(tensor_sizes, tensor_shapes_n): 
#     # print(f"size {size}")
#     print(f"shape {shape}")
#     t = tf.slice(new_t, [offset], [size])
#     t = tf.reshape( t , shape)
#     offset = offset + size
#     reconstructed_grads.append(t)

# print("reconstructed_grads".upper())
# print(reconstructed_grads[2])