from __future__ import absolute_import, division, print_function, unicode_literals 
import os 
import matplotlib.pyplot as plt 

import tensorflow as tf 
import numpy as np
# tf.debugging.set_log_device_placement(True)
print(f"Tensorflow vesion: {tf.__version__}") 
print(f"Eager execition: {tf.executing_eagerly()}")

strategy = tf.distribute.MirroredStrategy()

print ('Number of devices: {}'.format(strategy.num_replicas_in_sync))


# GET THE DATA
train_dataset_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"
train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(train_dataset_url),
                                           origin=train_dataset_url)
print("Local copy of the dataset file: {}".format(train_dataset_fp))

test_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv"

test_fp = tf.keras.utils.get_file(fname=os.path.basename(test_url),
                                  origin=test_url)



# PREPARE THE FEATURES AND LABELS NAMES
column_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
features_names = column_names[:-1]
label_name = column_names[-1]
print(f"Features names: {features_names}")
print(f"Label: {label_name}")

BATCH_SIZE_PER_REPLICA = 32
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

EPOCHS = 201




# DECLARE CLASS NAMES
class_names = ["Iris_setosa", "Iris_versicolor", "Iris_virginica"]

# SPECIFY BATCH SIZE AND FORMAT THE DATA USING DATASET
batch_size = 10
train_dataset = tf.data.experimental.make_csv_dataset(
    train_dataset_fp,
    BATCH_SIZE_PER_REPLICA,
    column_names = column_names,
    label_name = label_name,
    num_epochs = 1)

test_dataset = tf.data.experimental.make_csv_dataset(
    test_fp,
    batch_size,
    column_names=column_names,
    label_name='species',
    num_epochs=1,
    shuffle=False)

def pack_features_vector(features, labels):
  """Pack the features into a single array."""
  features = tf.stack(list(features.values()), axis=1)
  return features, labels

# CREATE SUITABLE FEATURES-LABEL PAIRS

train_dataset = train_dataset.map(pack_features_vector)

test_dataset = test_dataset.map(pack_features_vector)

# CREATE A DISTRIBUTED DATASET

train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
test_dist_dataset = strategy.experimental_distribute_dataset(test_dataset)



# DECLARE THE MODEL
def create_model():

  model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,)),  # input shape required
    tf.keras.layers.Dense(10, activation=tf.nn.relu),
    tf.keras.layers.Dense(3)
  ])
  return model

# model and optimizer must be created under `strategy.scope`.
with strategy.scope():
  model = create_model()

  optimizer = tf.keras.optimizers.SGD(learning_rate=0.03)

  #checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

with strategy.scope():
  # DEFINE THE METRICS TO TRACK LOSS AND ACCURACY
  test_loss = tf.keras.metrics.Mean(name='test_loss')

  train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
      name='train_accuracy')
  test_accuracy = tf.keras.metrics.Accuracy(
      name='test_accuracy')

with strategy.scope():
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
      reduction=tf.keras.losses.Reduction.NONE)
  # or loss_fn = tf.keras.losses.sparse_categorical_crossentropy
  def compute_loss(x, y):
    # print(f"inputs.shape {x.shape}")
    # print(f"predictions.shape {y.shape}")
    per_example_loss = loss_object(x, y)
    return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)

  def loss(model, x, y):
    y_ = model(x)
    return loss_object(y_true=y, y_pred=y_)

  def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
      predictions = model(inputs, training=True)
      # print(f"predictions.shape {predictions.shape}")
      # print(f"inputs.shape {inputs.shape}")
      
      loss_value = compute_loss(targets, predictions)

    return loss_value, tape.gradient(loss_value, model.trainable_variables), predictions
  
  def train_step(x, y):
    loss_value, grads, predictions = grad(model, x, y)
    
    # apply gradients to model
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    train_accuracy.update_state(y, predictions)

    return loss_value
    
  def test_step(x,y):
    logits = model(x, training=False)
    prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
    t_loss = loss_object(y, logits)
    print(f"y and prdition: {y , prediction}")
    test_loss.update_state(t_loss)
    test_accuracy.update_state(y, prediction)

with strategy.scope():
  # `experimental_run_v2` replicates the provided computation and runs it
  # with the distributed input.
  @tf.function
  def distributed_train_step(x, y):
    per_replica_losses = strategy.experimental_run_v2(train_step, args=(x, y))
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,axis=None)

  @tf.function
  def distributed_test_step(x, y):
    return strategy.experimental_run_v2(test_step, args=(x,y))


  for epoch in range(EPOCHS):
    # TRAIN LOOP
    total_loss = 0.0
    num_batches = 0
    for x, y in train_dist_dataset:
      total_loss += distributed_train_step(x, y)
      num_batches += 1
    train_loss = total_loss / num_batches

    # TEST LOOP
    for x, y in test_dist_dataset:
      distributed_test_step(x, y)

    if epoch%50 == 0:
      template = ("Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, "
                  "Test Accuracy: {}")
      print (template.format(epoch+1, train_loss,
                             train_accuracy.result()*100, test_loss.result(),
                             test_accuracy.result()*100))

    test_loss.reset_states()
    train_accuracy.reset_states()
    test_accuracy.reset_states()
