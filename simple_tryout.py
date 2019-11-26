from __future__ import absolute_import, division, print_function 
import tensorflow as tf 
import numpy as np
from mpi4py import MPI 

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0: 
    x = tf.matmul([[1]], [[2, 3]])
    comm.send(x ,dest= 1 )
    print(f"\n Sent from process: {rank} \n")

elif rank == 1 : 
    x = comm.recv(source = 0)
    print(f"\n Recieved in process: {rank} \n")
    print(x)
    print(x.shape)
    print(x.dtype)

    