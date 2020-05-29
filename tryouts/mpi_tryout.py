from mpi4py import MPI 

comm = MPI.COMM_WORLD 
comm_self = MPI.COMM_SELF
rank = comm.rank

print(f"hello from process {rank}")

comm.Disconnect()
comm_self.Disconnect()