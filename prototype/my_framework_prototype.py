from mpi4py import MPI 
import tensorflow as tf 
import numpy as np
import time

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

    time_spend_re_de = 0
    time_spend_reduction = 0
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
        flat_tensor = self.deconstruct(grads)

        # Reduce grads
        ft_np_arr = np.array_split(flat_tensor.numpy(), self.size)
        tensor_list = [tf.convert_to_tensor(arr) for arr in ft_np_arr]

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

        reduced_flat_tensor = tf.concat(tensor_list, 0)
        # Reconstruct grads and return
        reduced_grads = self.reconstruct(reduced_flat_tensor)
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
        # Maybe i cant directly concat
        flat_tensor = tf.concat(flat_tensor, 0)
        e = time.perf_counter()
        self.time_spend_re_de = self.time_spend_re_de + (e - s)
        return flat_tensor

    def reconstruct(self, flat_tensor): 
        s = time.perf_counter()
        reconstruncted_grads = []
        offset = 0
        for size, shape in zip (self.gradients_sizes, self.gradients_shapes): 
            g = tf.slice(flat_tensor, [offset],[size])
            g = tf.reshape(g, shape)
            offset = offset + size 
            reconstruncted_grads.append(g)

        e = time.perf_counter()
        self.time_spend_re_de = self.time_spend_re_de + (e - s)
        return reconstruncted_grads       
