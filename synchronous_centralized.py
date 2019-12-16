from __future__ import absolute_import, division, print_function, unicode_literals 
import tensorflow as tf 
from mpi4py import MPI 

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

gpus = tf.config.experimental.list_physical_devices('GPU')

device_name = gpus[rank].name 

print(device_name)
with tf.device(device_name):

    #import matplotlib.pyplot as plt 
    from tools import csv_splitter
    import math
    import os
    import time

    tf.debugging.set_log_device_placement(True)



    print(f"Tensorflow vesion: {tf.__version__}") 
    print(f"Eager execition: {tf.executing_eagerly()}")


    if rank == 0: 
        subsets_dir_path = "./subsets"
        for file in os.listdir(subsets_dir_path):
            file_path = os.path.join(subsets_dir_path, file)
            os.remove(file_path)

        # GET THE DATA
        train_dataset_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"
        train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(train_dataset_url),
                                                 origin=train_dataset_url)
        print("Local copy of the dataset file: {}".format(train_dataset_fp))

        #
        with open(train_dataset_fp, 'r') as data_file:
            n_rows = sum(1 for row in data_file )
            row_limit = math.ceil(n_rows/ (size))
        
        csv_splitter.split(open(train_dataset_fp, 'r'), row_limit = row_limit, output_path = './subsets')

        subset_names = os.listdir(subsets_dir_path)
        subset_names = [os.path.join(subsets_dir_path, name) for name in subset_names]
        print(f"Subsets list : {subset_names}")

        
        train_subset_fp = subset_names

    else: 
        row_limit = None
        train_subset_fp = None

    max_n_examples = comm.bcast(row_limit, root = 0 )
    train_subset_fp = comm.scatter(train_subset_fp, root = 0)

    print(f"Process {rank} recieved name: {train_subset_fp}")

    def getMaxNumberOfBatches (batch_size, maxNumberOfExamples) : 
        return math.ceil(maxNumberOfExamples / batch_size)

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

    if rank == 0: 
        print(f"Batch size is: {batch_size}")
        
    train_dataset = tf.data.experimental.make_csv_dataset(
      train_subset_fp,
      batch_size,
      column_names = column_names,
      label_name = label_name,
      num_epochs = 1)

    # Get number of batches 
    max_n_batches = getMaxNumberOfBatches(batch_size , max_n_examples)

    def pack_features_vector(features, labels):
        """Pack the features into a single array."""
        features = tf.stack(list(features.values()), axis=1)
        return features, labels

    # CREATE SUITABLE FEATURES-LABEL PAIRS
    train_dataset = train_dataset.map(pack_features_vector)

    # DECLARE THE MODEL

    layers = [  tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,)),
                tf.keras.layers.Dense(10, activation=tf.nn.relu),
                tf.keras.layers.Dense(3)]

    model = tf.keras.Sequential(layers)

    if rank == 0:
        weights = []
        for l in model.layers: 
            weights.append(l.get_weights())
    else: 
        weights = None 

    weights_to_use = comm.bcast(weights, root = 0 )

    for i, l in enumerate(model.layers): 
        l.set_weights(weights_to_use[i])

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
    print("Process {} Loss test: {}".format(rank,l))


    start = time.perf_counter()
    # EPOCH LOOP


    for epoch in range(num_epochs):
      
        # COMPUTES THE (WEIGHTED) MEAN OF THE GIVEN VALUES
        epoch_loss_avg = tf.keras.metrics.Mean()

        # CALCULATES HOW OFTEN PREDICTIONS MATCHES INTEGER LABELS
        epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        
        set_iter = iter(train_dataset)

        # TRAINING LOOP - using batches
        for i in range(max_n_batches):
            try: 
                x, y = next(set_iter)
                # Compute loss value and gradients
                loss_value, grads = grad(model, x, y)
                
                # send to master node for reducing
                payload = grads
            
            except StopIteration as e: 
                # signalize you dont have anything to send 
                payload = "EOS"
            
            # Gather a list of all gradients or "EOS" messages
            #print(f"Sending {type(payload)} from rank {rank} i : {i}")
            payload_gather = comm.gather(payload, root = 0) 
            
            if rank == 0 : 
                filtered = filter(lambda n: n != "EOS", payload_gather)

                reduced = []

                for grad_tuple in zip(*filtered): 
                    reduced.append(tf.math.add_n (list(grad_tuple)) / size) 
                #https://tech.preferred.jp/en/blog/technologies-behind-distributed-deep-learning-allreduce/
            else: 
                reduced = 0

            recieved_gradients = comm.bcast(reduced, root = 0)

            optimizer.apply_gradients(zip(recieved_gradients, model.trainable_variables))

            # Track progress
            epoch_loss_avg(loss_value)  # Add current batch loss
            epoch_accuracy(y, model(x)) # Compare predicted label to actual label


        # End epoch
        train_loss_results.append(epoch_loss_avg.result())
        train_accuracy_results.append(epoch_accuracy.result())

        statistics = {"rank": rank,"loss": train_loss_results,"accuracy":train_accuracy_results}

        statistics_gather = comm.gather(statistics, root = 0)

        if epoch % 50 == 0:
            print("Process: {:01d}  Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(rank,epoch,epoch_loss_avg.result(),epoch_accuracy.result()))

    finish = time.perf_counter() 

    print(f"Process {rank} finished training loop in {round(finish-start,2)} second(s).")

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

    # GET THE TEST SET
    test_dataset = test_dataset.map(pack_features_vector)

    # EVALUATE THE MODEL ON THE TEST DATASET
    test_accuracy = tf.keras.metrics.Accuracy()

    for (x, y) in test_dataset:
        logits = model(x)
        prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
        test_accuracy(prediction, y)

    print("Test {:01d} set accuracy: {:.3%}".format(rank,test_accuracy.result()))

    # if rank == 0:
        
    # # VISUALIZE THE ACCURACY AND LOSS OVER THE EPOCHS
    #     fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
    #     fig.suptitle('Training Metrics')
    #     axes[0].set_ylabel("Loss", fontsize=14)
    #     axes[1].set_ylabel("Acurracy", fontsize=14)
    #     axes[1].set_xlabel("Epoch", fontsize=14)

    #     for index, statistics in enumerate(statistics_gather):
    #         axes[0].plot(statistics["loss"])
    #         axes[1].plot(statistics["accuracy"])

    #     plt.show()

    # # USE THE MODEL TO MAKE PREDICTIONS 

    # predict_dataset = tf.convert_to_tensor([
    #     [5.1, 3.3, 1.7, 0.5,],
    #     [5.9, 3.0, 4.2, 1.5,],
    #     [6.9, 3.1, 5.4, 2.1]
    # ])

    # predictions = model(predict_dataset)

    # for i, logits in enumerate(predictions):
    #     class_idx = tf.argmax(logits).numpy()
    #     p = tf.nn.softmax(logits)[class_idx]
    #     name = class_names[class_idx]
    #     print("Example {} prediction: {} ({:4.1f}%)".format(i, name, 100*p))

