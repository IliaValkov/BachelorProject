from __future__ import absolute_import, division, print_function, unicode_literals 
import matplotlib.pyplot as plt 
import tensorflow as tf 
from tools import csv_splitter
import math
import os

from mpi4py import MPI 


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

print(f"Tensorflow vesion: {tf.__version__}") 
print(f"Eager execition: {tf.executing_eagerly()}")


if rank == 0: 
    subsets_dir_path = "./subsets"
    for file in os.listdir(subsets_dir_path):
        file_path = os.path.join(subsets_dir_path, file)
        os.remove(file_path)
    print(os.listdir(subsets_dir_path))

    # GET THE DATA
    train_dataset_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"
    train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(train_dataset_url),
                                             origin=train_dataset_url)
    print("Local copy of the dataset file: {}".format(train_dataset_fp))

    with open(train_dataset_fp, 'r') as data_file:
        n_rows = sum(1 for row in data_file )
        row_limit = math.ceil(n_rows/ (size-1))
    
    csv_splitter.split(open(train_dataset_fp, 'r'), row_limit = row_limit, output_path = './subsets')
    subset_names = os.listdir(subsets_dir_path)

    for i in range(len(subset_names)): 
        path = "./subsets/"+subset_names[i]
        comm.send(path, dest = (i+1))



    while True: 
        # TODO: Use gather 
        g1 = comm.recv( source = 1)
        g2 = comm.recv( source = 2)

        g_reduced = []
        
        # TODO: maybe use zip, see how you'll implement the recieving
        for i in range(len(g1)):
            g_reduced.append( tf.math.add(g1[i] , g2[i]))

        for g in g_reduced:
            g = g / 2 # TODO: make dynamic 
        
        for i in range(2): # TODO: use broadcast; make dynamic
            comm.send(g_reduced, dest = (i+1))

else: 

    train_subset_fp = comm.recv(source = 0)
    print(f"Process {rank} recieved name: {train_subset_fp}")
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
      train_subset_fp,
      batch_size,
      column_names = column_names,
      label_name = label_name,
      num_epochs = 1)

    #print number of batches 
    print(f"train_dataset: {train_dataset} rank: {rank}")
   
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

    print("Loss test: {}".format(l))

    # EPOCH LOOP
    for epoch in range(num_epochs):
      
        # COMPUTES THE (WEIGHTED) MEAN OF THE GIVEN VALUES
        epoch_loss_avg = tf.keras.metrics.Mean()

        # CALCULATES HOW OFTEN PREDICTIONS MATCHES INTEGER LABELS
        epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

        # TRAINING LOOP - using batches of 32
        for x, y in train_dataset:
        # Optimize the model

            # Compute loss value and gradients
            loss_value, grads = grad(model, x, y)

            # apply gradients to model
            # print(f"Sending grads from rank: {rank}")
            comm.send(grads, dest = 0)

            g = comm.recv(source = 0)
            #print(f"recieved g: {g}" )
            optimizer.apply_gradients(zip(g, model.trainable_variables))

            # Track progress
            epoch_loss_avg(loss_value)  # Add current batch loss
            epoch_accuracy(y, model(x)) # Compare predicted label to actual label


        # End epoch
        train_loss_results.append(epoch_loss_avg.result())
        train_accuracy_results.append(epoch_accuracy.result())

        if epoch % 50 == 0:
            print("Process: {:01d}  Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(rank,epoch,epoch_loss_avg.result(),epoch_accuracy.result()))
    
    comm.send("Exit", dest = 0)
