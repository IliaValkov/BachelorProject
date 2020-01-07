from __future__ import absolute_import, division, print_function, unicode_literals 
import os 
import tensorflow as tf 
from my_framework_prototype import Dist

dist = Dist()
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
    shuffle = False,
    column_names = column_names,
    label_name = label_name,
    num_epochs = 1).unbatch()

dist_train_dataset = train_dataset.shard(dist.size, dist.rank).batch(batch_size)

def pack_features_vector(features, labels):
  """Pack the features into a single array."""
  features = tf.stack(list(features.values()), axis=1)
  return features, labels

# CREATE SUITABLE FEATURES-LABEL PAIRS
dist_train_dataset = dist_train_dataset.map(pack_features_vector)

# DECLARE THE MODEL
model = dist.replicate_model(model=tf.keras.Sequential([
  tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,)),  # input shape required
  tf.keras.layers.Dense(10, activation=tf.nn.relu),
  tf.keras.layers.Dense(3)
]))

#DECALARE A LOSS FUNCTION
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# GET A BATCH OF FEATURES AND LABELS
features, labels = next(iter(dist_train_dataset))

# FUNCTION TO CALCULATE THE LOSS
def loss(model, x, y):
  y_ = model(x)

  return loss_object(y_true=y, y_pred=y_)

# CALCULATE LOSS PRE-TRAINING
l = loss(model, features, labels)
print("Loss test: {}".format(l))

# FUNCTION TO CALCULATE THE GRADIENTS
def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets)
  return loss_value, tape.gradient(loss_value, model.trainable_variables)

# APPLIES THE COMPUTED GRADIENTS TO THE MODEL'S VARIABLES TO MINIMIZE THE LOSS FUNCTION
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

# Keep results for plotting
train_loss_results = []
train_accuracy_results = []

num_epochs = 201

def training_step(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets)

  grads = dist.all_reduce(tape.gradient(loss_value, model.trainable_variables))
  
  optimizer.apply_gradients(zip(grads, model.trainable_variables))

  return loss_value

# EPOCH LOOP
for epoch in range(num_epochs):
  
  # COMPUTES THE (WEIGHTED) MEAN OF THE GIVEN VALUES
  epoch_loss_avg = tf.keras.metrics.Mean()
  
  # CALCULATES HOW OFTEN PREDICTIONS MATCHES INTEGER LABELS
  epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

  # TRAINING LOOP
  for x, y in dist_train_dataset:
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

  if epoch % 50 == 0:
    print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,epoch_loss_avg.result(),epoch_accuracy.result()))
