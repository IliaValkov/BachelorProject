from __future__ import absolute_import, division, print_function, unicode_literals 
import matplotlib.pyplot as plt 
import tensorflow as tf 
from tools import csv_splitter
import math
import os
import time
from mpi4py import MPI 

tf.debugging.set_log_device_placement(True)


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0: 
    device_name = ""

elif rank == 1: 
    device_name = ""

print(f"Tensorflow vesion: {tf.__version__}") 
print(f"Eager execition: {tf.executing_eagerly()}")

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    print("Name:", gpu.name, "  Type:", gpu.device_type)


