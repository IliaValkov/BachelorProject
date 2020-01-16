from __future__ import absolute_import, division, print_function, unicode_literals 
import os 
import matplotlib.pyplot as plt 

import tensorflow as tf 
import time
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

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
    num_epochs = 1)

def pack_features_vector(features, labels):
  """Pack the features into a single array."""
  features = tf.stack(list(features.values()), axis=1)
  return features, labels

# CREATE SUITABLE FEATURES-LABEL PAIRS
train_dataset = train_dataset.map(pack_features_vector)

# DECLARE THE MODEL
layers = [
  tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,)),  # input shape required
  tf.keras.layers.Dense(10, activation=tf.nn.relu),
  tf.keras.layers.Dense(3)
]

model = tf.keras.Sequential(layers)

if rank == 0: 
  own_layers = layers = [tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,))]  # input shape required
elif rank == 1: 
  own_layers = [tf.keras.layers.Dense(10, activation=tf.nn.relu),tf.keras.layers.Dense(3)]  

print(f"Process {rank} has {len(own_layers)} layers")

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# GET A BATCH OF FEATURES AND LABELS
features, labels = next(iter(train_dataset))

# FUNCTION TO CALCULATE THE LOSS

some_loss_value = loss_object(y_true = labels, y_pred = model(features))

print("Loss: ",loss_object(y_true = labels, y_pred = model(features)))

own_model = tf.keras.Sequential(own_layers)

def loss(model, x, y):
  y_ = model(x)
  return loss_object(y_true = y, y_pred = y_ )



# FORWARD PASS
if rank == 0:
  with tf.GradientTape() as tape:
    loss_value = loss(own_model, features, labels)
  
  tape.gradient(loss_value, own_model.trainable_variables)



#   comm.send(outputs, dest = 1)

# elif rank == 1:
#   inputs = comm.recv(source = 0)

#   for l in own_layers: 
#     outputs = l(inputs)
#     inputs = outputs
    
  

