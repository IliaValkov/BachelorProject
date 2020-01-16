from mpi4py import MPI 
import tensorflow as tf 
import numpy as np
import math
import time
from operator import add

class Dist():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    # Variables for ring all reduce
    gradients_sizes = None
    gradients_shapes = None
    gradients_ranks = None
    recv_from_p = (rank - 1) % size
    send_to_p = (rank + 1) % size
    chunk_to_send = rank
    tensors_dtype = None

    time_spend_re = 0
    time_spend_de = 0
    time_spend_reduction = 0

    def __init__(self):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.experimental.set_visible_devices(gpus[self.rank], 'GPU')
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            except RuntimeError as e:
              # Visible devices must be set before GPUs have been initialized
                print(e)

    def distribute_dataset(self, dataset, batch_size):
        dataset = dataset.unbatch()
        dataset = dataset.shard(self.size, self.rank).shuffle(100)
        dataset = dataset.batch(batch_size)

        return dataset
  
    def replicate_model(self, model):
        if self.rank == 0:
            dist_w = [w for w in [l.get_weights() for l in model.layers]]
        else:
            dist_w = None 
        
        weights = self.comm.bcast(dist_w, root = 0) 

        for i, l in enumerate(model.layers): 
            l.set_weights(weights[i])
        
        return model

    def all_reduce(self, grads):
        s = time.perf_counter()
        accumulated_grads = self.comm.gather(grads, root=0)
        if self.rank == 0: 
            reduced_grads = []
            for grad_tuple in zip(*accumulated_grads):
                reduced_grads.append(tf.math.add_n (list(grad_tuple)) / self.size)
        else: 
            reduced_grads = None      

        reduced_grads = self.comm.bcast(reduced_grads, root=0)
        e = time.perf_counter()
        self.time_spend_reduction = self.time_spend_reduction + (e - s)
        return reduced_grads

    def ring_all_reduce(self, grads): 
        # Deconstruct grads 
        s = time.perf_counter()
        
        tensor_list = self.deconstruct(grads)

        # Reduce grads
        for i in range(self.size): 
            self.comm.send((self.chunk_to_send, tensor_list[self.chunk_to_send]), dest = self.send_to_p)
            received_chunk = self.comm.recv(source = self.recv_from_p)
            index = received_chunk[0]
            # Reduce the received chunk with the own chunk at the same index
            tensor_list[index] = (tensor_list[index] + received_chunk[1]) / 2
            self.chunk_to_send = (self.chunk_to_send - 1) % self.size

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
        s = time.perf_counter()
        if self.gradients_sizes is None:
            self.gradients_sizes = [tf.size(g) for g in grads]
            #print(self.gradients_sizes)
        if self.gradients_shapes is None: 
            self.gradients_shapes = tf.shape_n(grads) 
            #print(self.gradients_shapes)
        if self.gradients_ranks is None: 
            self.gradients_ranks = [tf.rank(g) for g in grads]
           # print(self.gradients_ranks)
        
        flat_tensor = []

        for i, g in enumerate(grads): 
            if self.gradients_ranks[i] > 1:
                g = tf.reshape(g, [self.gradients_sizes[i]])
            flat_tensor.append(g)

        flat_tensor = tf.concat(flat_tensor, 0)

        ft_np_arr = np.array_split(flat_tensor.numpy(), self.size)
        tensor_list = [tf.convert_to_tensor(arr) for arr in ft_np_arr]

        e = time.perf_counter()
        self.time_spend_de = self.time_spend_de + (e - s)
        return tensor_list

    def reconstruct(self, tensor_list): 
        s = time.perf_counter()

        flat_tensor = tf.concat(tensor_list, 0)
        reconstruncted_grads = []
        offset = 0

        for size, shape in zip (self.gradients_sizes, self.gradients_shapes): 
            g = tf.slice(flat_tensor, [offset], [size])
            g = tf.reshape(g, shape)
            offset = offset + size 
            reconstruncted_grads.append(g)

        e = time.perf_counter()
        self.time_spend_re = self.time_spend_re + (e - s)
        return reconstruncted_grads       

    @staticmethod
    def chunks(l, n):
        for i in range(0, len(l), n):
            yield l[i:i+n]


    def deconstruct_faster(self, grads): 
        s = time.perf_counter()
        if self.gradients_sizes is None:
            self.gradients_sizes = [tf.size(g) for g in grads]
        if self.gradients_shapes is None: 
            self.gradients_shapes = tf.shape_n(grads) 
        if self.tensors_dtype is None:
            self.tensors_dtype = grads[0].dtype

        flat_list = []
        for g in grads:
            flat_list += [value for value in g.numpy().flatten()]

        chunk_len = round(len(flat_list)/self.size)
        # tensor_list = list(self.chunks(flat_list,chunk_len))
        tensor_list = [tf.constant(t, dtype=self.tensors_dtype) for t in self.chunks(flat_list,chunk_len)]
        
        e = time.perf_counter()
        self.time_spend_de = self.time_spend_de + (e - s)
        return tensor_list

    
    def generate_reconstructed_gradients(self, flat_list):
        begin = 0
        for size, shape in zip(self.gradients_sizes, self.gradients_shapes):
            g = flat_list[begin : size + begin]
            g = tf.constant(g, dtype = self.tensors_dtype)
            yield tf.reshape(g, shape)
            begin += size

    def reconstruct_faster(self, tensor_list): 
        s = time.perf_counter()

        flat_list = []
        for t in tensor_list:
            flat_list += [value for value in t.numpy().flatten()]

        reconstruncted_grads = []
        for g in self.generate_reconstructed_gradients(flat_list):
            reconstruncted_grads.append(g)

        e = time.perf_counter()
        self.time_spend_re = self.time_spend_re + (e - s)
        return reconstruncted_grads 

    def ring_all_reduce_faster(self, grads): 
        # Deconstruct grads 
        s = time.perf_counter()
        
        tensor_list = self.deconstruct_faster(grads)

        # Reduce grads
        for i in range(self.size): 
            self.comm.send((self.chunk_to_send, tensor_list[self.chunk_to_send]), dest = self.send_to_p)
            received_chunk = self.comm.recv(source = self.recv_from_p)
            index = received_chunk[0]
            # Reduce the received chunk with the own chunk at the same index

            # tensor_list[index] = [(a+b)/2 for a,b in zip(tensor_list[index], received_chunk[1])]
            # tensor_list[index] = list(map(add, tensor_list[index], received_chunk[1]))
            # tensor_list[index] = [a for a in self.gen(tensor_list[index], received_chunk[1])]
            tensor_list[index] = (tensor_list[index] + received_chunk[1]) / 2
            self.chunk_to_send = (self.chunk_to_send - 1) % self.size

        # tensor_list[self.chunk_to_send] = [ e/2 for e in tensor_list[self.chunk_to_send]]
        # DIVIDE elements by the size after accumulated
        for i in range(self.size - 1): 
            self.comm.send((self.chunk_to_send, tensor_list[self.chunk_to_send]), dest = self.send_to_p)
            received_chunk = self.comm.recv(source = self.recv_from_p)
            index = received_chunk[0] 
            tensor_list[index] = received_chunk[1]

        
        # Reconstruct grads and return
        reduced_grads = self.reconstruct_faster(tensor_list)
        e = time.perf_counter()
        self.time_spend_reduction = self.time_spend_reduction + (e - s)
        return reduced_grads 