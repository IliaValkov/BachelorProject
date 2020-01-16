from __future__ import absolute_import, division, print_function, unicode_literals 
import os 
import matplotlib.pyplot as plt 

import tensorflow as tf 
import time

from my_framework_prototype import Dist
from tensorflow.python.util import nest


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

gradients_sizes = [tf.size(g) for g in grads]
gradients_shapes = tf.shape_n(grads) 
        
simple_arr = []
rt = 0
return_grads = []
begin = 0
import math
def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i+n]

dt = grads[0].dtype
for i in range(1):
  return_grads = []
  s = time.perf_counter()
  # DECONSTRUCTION  
  for g in grads: 
    simple_arr +=[v for v in g.numpy().flatten()]

    # TODO: Find a good splitting function
  n = math.ceil(len(simple_arr)/4)
  div_list = list(chunks(simple_arr,n))
  
  # RECONSTRUCTION
  new_arr = []
  for a in div_list:
    new_arr += a

  i = 0
  for size, shape in zip(gradients_sizes, gradients_shapes):
    g = new_arr[begin:size+begin]
    print(grads[i].numpy().tolist())
    g = nest.pack_sequence_as(grads[i].numpy().tolist(), g)  
    begin = begin + size
    i += 1
    return_grads.append(tf.constant(g))
  
  print(return_grads)

  e = time.perf_counter()
  rt = rt + (e - s)

print("time",round((e - s), 2))



# d = Dist()

# for i in range(1): 
#   e = time.perf_counter()
#   print(grads[0].dtype)
#   g = d.deconstruct_faster(grads)  
#   g = d.reconstruct_faster(g)

#   e = time.perf_counter()
#   rt = rt + (e - s)



# for i in range(201): 
#   e = time.perf_counter()
  
#   g = d.deconstruct(grads)  
#   g = d.reconstruct(g)

#   e = time.perf_counter()
#   rt = rt + (e - s)

# print("time",round((e - s), 2))
