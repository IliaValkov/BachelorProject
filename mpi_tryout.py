from mpi4py import MPI 

comm = MPI.COMM_WORLD 

print(f"hello from process {comm.rank}")