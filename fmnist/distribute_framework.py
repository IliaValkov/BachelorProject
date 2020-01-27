from mpi4py import MPI 
import tensorflow as tf 
import numpy as np
import math
import time
import os

class Distribute():
    '''
        This class delivers methods for distributing the training of a deep neural network 
    '''

    # Variable for the MPI communication object
    comm = MPI.COMM_WORLD
    # Variable holding the rank of a process
    rank = comm.Get_rank()
    # Variable holding number of processes working on a task
    size = comm.Get_size()

    # Variables for reconstructing the shapes of gradients in ring allreduce
    gradients_sizes = None
    gradients_shapes = None
    gradients_ranks = None
    tensors_dtype = None

    # Variables for coordinating the communication between processes during ring allreduce
    recv_from_p = (rank - 1) % size
    send_to_p = (rank + 1) % size
    chunk_to_send = rank

    # Variables for timing the reduction phase in both allreduce implementations,
    # as well as the reconstruction and deconstruction phases in ring allreduce
    time_spend_re = 0
    time_spend_de = 0
    time_spend_reduction = 0

    def __init__(self):
        ''' init method for the Distribute object;
            When a Distribute object is created, it will check if there are available
            GPUs, and assign one to each process, if the number of processes specified 
            in the mpirun command does not exceed the number of available devices.

            If the number of processes specified is greater than the number of available 
            GPUs, the system will use only the CPU and remove all GPUs from the list 
            of visible devices 
        '''
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:    
                if len(gpus) < self.size:
                    tf.config.experimental.set_visible_devices([], 'GPU')
                else:    
                    tf.config.experimental.set_visible_devices(gpus[self.rank], 'GPU')
            except RuntimeError as e:
              # Visible devices must be set before GPUs have been initialized
                print(e)

    def distribute_dataset(self, dataset, batch_size=None):
        ''' Function for providing each process with a part of the original dataset.
            It is based on tensorflow's Dataset.shard() function and provides a 
            wrapper for it. 
            It is recommended, that the shard() function is to be performed on a 
            Dataset before any randomizing operation(like shuffle) is used on the Dataset.
            Some methods for creating a Dataset may require from users to specify a
            batch size, this can be a problem when sharding the Dataset. For this 
            purpose the function takes arguments is_batched and batch_size, to 
            manually unbatch the Dataset before it gets sharded, and batch it back 
            after sharding it. 

            Arguments: 
            dataset - A Dataset object to be distributed
            is_batched - a boolean that specifies if the Dataset is batched
            batch_size - an integer that specifies the batch size

            Returns: 
            The sharded Dataset object
        '''
        if batch_size:
            dataset = dataset.unbatch()
        
        dist_dataset = dataset.shard(num_shards=self.size, index=self.rank)
        
        if batch_size:
            dist_dataset = dist_dataset.repeat(5)
            dist_dataset = dist_dataset.batch(batch_size)

        return dist_dataset
  
    def replicate_model(self, model):
        ''' Function for initializing a model with the same starting parameters
            in all processes

            Arguments: 
            model - a Model object with defined layers

            Returns: 
            A Model object with the same variables as the ones initialized in the
            root process
        '''
        if self.rank == 0:
            dist_w = [w for w in [l.get_weights() for l in model.layers]]
        else: 
            dist_w = None

        weights = self.comm.bcast(dist_w, root = 0) 

        for i, l in enumerate(model.layers): 
            l.set_weights(weights[i])
        
        return model

    def simple_all_reduce(self, grads):
        ''' Function to accumulate the locally computed gradients from all processes,
            perform the reducing operation and distributing back the reduced gradients
            back to all processes. 
            This method uses one process to accumulate, compute 
            and distribute back the gradients.

            Arguments: 
            grads - A list of Tensors, that represent the gradient values for each 
            layer of a deep neural network's model  

            Returns: 
            The reduced list of Tensors
        '''
        s = time.perf_counter()

        # Accumulate local gradients in process 0
        accumulated_grads = self.comm.gather(grads, root=0)
        
        # Perform reduce operation in process 0
        if self.rank == 0: 
            reduced_grads = []
            for grad_tuple in zip(*accumulated_grads):
                reduced_grads.append(tf.math.add_n(list(grad_tuple)) / self.size)
        else: 
            reduced_grads = None      

        # Distribute the result to all processes
        reduced_grads = self.comm.bcast(reduced_grads, root=0)
        e = time.perf_counter()
        self.time_spend_reduction = self.time_spend_reduction + (e - s)
        return reduced_grads

    def ring_all_reduce(self, grads): 
        ''' Function to accumulate the locally computed gradients from all processes,
            perform the reducing operation and distributing back the reduced gradients
            back to all processes. 
            This method distributes the computation of the gradient values and the 
            interprocess communication between the participating processes. 

            Note that this implementation only works with gradients, whose
            total amount of elements does not exceed 400 elements. 
            For communication of lists larger than 400 elements use the 
            simple_all_reduce method. 
            For implemntation of the ring allreduce that supports the reuction of 
            large lists see the ring_all_reduce_larger_lists() method.

            Arguments: 
            grads - A list of Tensors, that represent the gradient values for each 
            layer of a deep neural network's model

            Returns: 
            The reduced list of Tensors
        '''
        s = time.perf_counter()
        
        # Deconstruct grads 
        tensor_list = self.deconstruct(grads)

        # Reduce phase
        for i in range(self.size): 
            self.comm.send((self.chunk_to_send, tensor_list[self.chunk_to_send]), dest = self.send_to_p)
            received_chunk = self.comm.recv(source = self.recv_from_p)
            index = received_chunk[0]
            # Sum the received chunk with the own chunk at the same index
            tensor_list[index] = (tensor_list[index] + received_chunk[1]) 
            self.chunk_to_send = (self.chunk_to_send - 1) % self.size

        # Divide by the number of processes in the group 
        tensor_list = [t/self.size for t in tensor_list]

        # Share phase
        for i in range(self.size - 1): 
            self.comm.send((self.chunk_to_send, tensor_list[self.chunk_to_send]), dest = self.send_to_p)
            received_chunk = self.comm.recv(source = self.recv_from_p)
            index = received_chunk[0] 
            tensor_list[index] = received_chunk[1]

        
        # Reconstruct grads and return
        reduced_grads = self.reconstruct(tensor_list)
        
        e = time.perf_counter()
        self.time_spend_reduction = self.time_spend_reduction + (e - s)
        return reduced_grads

    def deconstruct(self, grads): 
        ''' Function to deconstruct the list of gradient values into a list with
            length equal to the number of processes in the group.   

            Arguments: 
            grads - list of Tensors representing the gradient values for each 
            layer of a deep neural network's model to be deconstructed

            Returns: 
            A list of Tensors represnting the deconstructed grads 
        '''
        s = time.perf_counter()
        # Save values for reconstruction
        if self.gradients_sizes is None:
            self.gradients_sizes = [tf.size(g) for g in grads]
        if self.gradients_shapes is None: 
            self.gradients_shapes = tf.shape_n(grads) 
        if self.tensors_dtype is None:
            self.tensors_dtype = grads[0].dtype

        # Create flat list from all gradient Tensors
        flat_list = []
        for g in grads:
            flat_list += [value for value in g.numpy().flatten()]

        # Split the flat list into a list of n Tensors, with n equal to the number of processes 
        chunk_len = round(len(flat_list)/self.size)
        tensor_list = [tf.constant(t, dtype=self.tensors_dtype) for t in self.chunks(flat_list,chunk_len)]
        
        e = time.perf_counter()
        self.time_spend_de = self.time_spend_de + (e - s)
        return tensor_list

    def reconstruct(self, tensor_list): 
        ''' Function to deconstruct the list of gradient values into a list with
            length equal to the number of processes in the group.   

            Arguments: 
            tensor_list - a list of Tensors to be reconstructed

            Returns: 
            A list of Tensors, each of which corresponds to the original gradiens
            shape. 
        '''
        s = time.perf_counter()
        
        # Create flat list from the tensor list
        flat_list = []
        for t in tensor_list:
            flat_list += [value for value in t.numpy().flatten()]

        # Reconstruct the list to the original gradient shapes
        reconstruncted_grads = []
        for g in self.generate_reconstructed_gradients(flat_list):
            reconstruncted_grads.append(g)

        e = time.perf_counter()
        self.time_spend_re = self.time_spend_re + (e - s)
        return reconstruncted_grads 

    @staticmethod
    def chunks(l, n):
        ''' A generator function that splits a list into chunks of size n

            Arguments: 
            l - list to be split
            n - length of one chunk

            Yields: 
            A list of n elemnts 
        '''
        for i in range(0, len(l), n):
            yield l[i:i+n]
    
    def generate_reconstructed_gradients(self, flat_list):
        ''' A generator function that restructures a flat list into a certain
            shape.

            Arguments: 
            flat_list - a list to be restructured 

            Yields: 
            A restructured element of the whole list
        '''
        begin = 0
        for size, shape in zip(self.gradients_sizes, self.gradients_shapes):
            g = flat_list[begin : size + begin]
            g = tf.constant(g, dtype = self.tensors_dtype)
            yield tf.reshape(g, shape)
            begin += size

    def ring_all_reduce_large_lists(self, grads): 
        ''' Function to accumulate the locally computed gradients from all processes,
            perform the reducing operation and distributing back the reduced gradients
            back to all processes. This method distributes the computation of the 
            gradient values and the interprocess communication between the participating
            processes. 

            Note that this method is only for the purpose of showing how the ring all
            reduce would have to be implemented with the given set of tools.(Python,mpi4py).
            If you have to reduce large lists use the simple_all_reduce() method.

            Arguments: 
            grads - A list of Tensors, that represent the gradient values for each 
            layer of a deep neural network's model

            Returns: 
            The reduced list of Tensors
        '''
        s = time.perf_counter()
        
        # Deconstruct grads 
        tensor_list = self.deconstruct(grads)

        # Reduce grads
        for i in range(self.size): 
            self.send_large_list(tensor_list[self.chunk_to_send], self.chunk_to_send )
            received_chunk = self.recv_large_list()
            
            index = received_chunk[0]
            tensor_list[index] = (tensor_list[index] + received_chunk[1]) 
            self.chunk_to_send = (self.chunk_to_send - 1) % self.size
       
        for i in range(self.size - 1): 
            self.send_large_list(tensor_list[self.chunk_to_send], self.chunk_to_send )
            received_chunk = self.recv_large_list()

            index = received_chunk[0] 
            tensor_list[index] = received_chunk[1]
      
        # Reconstruct grads and return
        reduced_grads = self.reconstruct(tensor_list)
        e = time.perf_counter()
        self.time_spend_reduction = self.time_spend_reduction + (e - s)
        return reduced_grads 


    def send_large_list(self, tensor_to_send, index):
        ''' A wrapper method to send a chunks index and Tensors, if the chunk is 
            longer than 400 elements

            Arguments: 
            tensor_to_send - a Tensor representing the long chunk
            index - the chunks index

            Returns: 
            This method does not return a value.    
        '''

        # Convert the tensor to a lis
        list_to_send = tensor_to_send.numpy().tolist()
        
        # Calculate how many iterations the tensor will need to be sent in
        if self.iterations == None:
            self.iterations = math.ceil(len(list_to_send) / self.max_elements)
        
        # Send the index
        self.comm.send(index, dest = self.send_to_p, tag=0)

        # Send the list slice by slice
        for i in range(self.iterations):
            start = i*self.max_elements
            end = start + self.max_elements  
            
            self.comm.send(list_to_send[start:end], dest = self.send_to_p, tag=i)
        

    def recv_large_list(self):
        ''' A wrapper method to receive large Tensors
            
            Arguments: 
            The method does not take argumnets 

            Returns: 
            A tuple containig the index of a chunk and the received chunk
        ''' 
        # receive the index of the chunk
        index = self.comm.recv(source = self.recv_from_p,tag=0)
            
        # receive the chunk
        recv_list = []
        for i in range(self.iterations): 
            recv = self.comm.recv(source = self.recv_from_p, tag=i)
            recv_list += recv
        
        return (index, tf.constant(recv_list, dtype=self.tensors_dtype))