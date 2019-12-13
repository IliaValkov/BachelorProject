from __future__ import absolute_import, division, print_function, unicode_literals 
import os 
import matplotlib.pyplot as plt 

import tensorflow as tf 
import horovod.tensorflow as hvd

print(f"Tensorflow vesion: {tf.__version__}") 
print(f"Eager execition: {tf.executing_eagerly()}")

hvd.init()
config = tf.compat.v1.ConfigProto()
config.gpu_options.visible_device_list = str(hvd.local_rank())

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
model = tf.keras.Sequential([
  tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,)),  # input shape required
  tf.keras.layers.Dense(10, activation=tf.nn.relu),
  tf.keras.layers.Dense(3)
])

# DECALARE A LOSS FUNCTION
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# GET A BATCH OF FEATURES AND LABELS
features, labels = next(iter(train_dataset))

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

optimizer = hvd.DistributedOptimizer(optimizer)

hooks = [hvd.BroadcastGlobalVariablesHook(0)]

@tf.function
def training_step(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
        
    tape = hvd.DistributedGradientTape(tape)
    grads = tape.gradient(loss_value, model.trainable_variables)
    
    # apply gradients to model
    
    optimizer.get_gradients(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
    return loss_value



# Keep results for plotting
train_loss_results = []
train_accuracy_results = []

num_epochs = 201


# EPOCH LOOP
for epoch in range(num_epochs):
  # COMPUTES THE (WEIGHTED) MEAN OF THE GIVEN VALUES
  epoch_loss_avg = tf.keras.metrics.Mean()
  
  # CALCULATES HOW OFTEN PREDICTIONS MATCHES INTEGER LABELS
  epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

  # TRAINING LOOP - using batches of 32
  for x, y in train_dataset:
    
    loss_value = training_step(model, x, y)


    # Track progress
    epoch_loss_avg(loss_value)  # Add current batch loss
    # Compare predicted label to actual label
    epoch_accuracy(y, model(x))

  # End epoch
  train_loss_results.append(epoch_loss_avg.result())
  train_accuracy_results.append(epoch_accuracy.result())

  if epoch % 50 == 0:
    print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,epoch_loss_avg.result(),epoch_accuracy.result()))

# VISUALIZE THE ACCURACY AND LOSS OVER THE EPOCHS
fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
fig.suptitle('Training Metrics')

axes[0].set_ylabel("Loss", fontsize=14)
axes[0].plot(train_loss_results)

axes[1].set_ylabel("Accuracy", fontsize=14)
axes[1].set_xlabel("Epoch", fontsize=14)
axes[1].plot(train_accuracy_results)
plt.show()

# GET THE TEST SET
test_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv"

test_fp = tf.keras.utils.get_file(fname=os.path.basename(test_url),
                                  origin=test_url)

# SETUP A DATASET  
test_dataset = tf.data.experimental.make_csv_dataset(
    test_fp,
    batch_size,
    column_names=column_names,
    label_name='species',
    num_epochs=1,
    shuffle=False)

test_dataset = test_dataset.map(pack_features_vector)

# EVALUATE THE MODEL ON THE TEST DATASET
test_accuracy = tf.keras.metrics.Accuracy()

for (x, y) in test_dataset:
  logits = model(x)
  prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
  test_accuracy(prediction, y)

print("Test set accuracy: {:.3%}".format(test_accuracy.result()))

#USE THE MODEL TO MAKE PREDICTIONS 

predict_dataset = tf.convert_to_tensor([
    [5.1, 3.3, 1.7, 0.5,],
    [5.9, 3.0, 4.2, 1.5,],
    [6.9, 3.1, 5.4, 2.1]
])

predictions = model(predict_dataset)

for i, logits in enumerate(predictions):
  class_idx = tf.argmax(logits).numpy()
  p = tf.nn.softmax(logits)[class_idx]
  name = class_names[class_idx]
  print("Example {} prediction: {} ({:4.1f}%)".format(i, name, 100*p))


  