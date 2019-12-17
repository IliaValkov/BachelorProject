from __future__ import absolute_import, division, print_function, unicode_literals 
import os 
import matplotlib.pyplot as plt 

import tensorflow as tf 
import time

print(f"Tensorflow vesion: {tf.__version__}") 
print(f"Eager execition: {tf.executing_eagerly()}")

strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())

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
batch_size = 5
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

train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)

# DECLARE THE MODEL
def create_model():
  model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,)),  # input shape required
    tf.keras.layers.Dense(10, activation=tf.nn.relu),
    tf.keras.layers.Dense(3)
  ])
  return model

# DECALARE A LOSS FUNCTION
with strategy.scope():
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
  # FUNCTION TO CALCULATE THE LOSS
  def loss(model, x, y):
    y_ = model(x)

    return loss_object(y_true=y, y_pred=y_)


  # FUNCTION TO CALCULATE THE GRADIENTS
  def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
      loss_value = loss(model, inputs, targets)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)

  # APPLIES THE COMPUTED GRADIENTS TO THE MODEL'S VARIABLES TO MINIMIZE THE LOSS FUNCTION
  optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
  model = create_model()
  
  # Keep results for plotting
  train_loss_results = []
  train_accuracy_results = []

  num_epochs = 201
  def train_step(model, x, y): 
    # Compute loss value and gradients
      loss_value, grads = grad(model, x, y)
      
      # apply gradients to model
      optimizer.apply_gradients(zip(grads, model.trainable_variables))
      return loss_value

  @tf.function
  def distribute_train_step(model, x, y):
    per_replica_losses = strategy.experimental_run_v2(train_step,
                                                      args=(model, x, y))
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                           axis=None)

  start = time.perf_counter()
  # EPOCH LOOP
  for epoch in range(num_epochs):
    
    # COMPUTES THE (WEIGHTED) MEAN OF THE GIVEN VALUES
    epoch_loss_avg = tf.keras.metrics.Mean()
    
    # CALCULATES HOW OFTEN PREDICTIONS MATCHES INTEGER LABELS
    epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    # TRAINING LOOP - using batches of 32
    for x, y in train_dataset:
      # Optimize the model
     
      loss_value = distribute_train_step(model, x, y)
      # Track progress
      epoch_loss_avg(loss_value)  # Add current batch loss
      # Compare predicted label to actual label
      epoch_accuracy(y, model(x))

    # End epoch
    train_loss_results.append(epoch_loss_avg.result())
    train_accuracy_results.append(epoch_accuracy.result())

    if epoch % 50 == 0:
      print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,epoch_loss_avg.result(),epoch_accuracy.result()))

  finish = time.perf_counter()
  