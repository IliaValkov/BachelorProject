from mpi4py import MPI 
import tensorflow as tf 

class Dist():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
  
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
        accumulated_grads = self.comm.gather(grads, root=0)
        if self.rank == 0: 
            reduced_grads = []
            for grad_tuple in zip(*accumulated_grads):
                reduced_grads.append(tf.math.add_n (list(grad_tuple)) / self.size)
        else: 
            reduced_grads = None      

        reduced_grads = self.comm.bcast(reduced_grads, root=0)
        
        return reduced_grads