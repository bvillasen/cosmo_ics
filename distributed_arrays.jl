using Printf
using MPI
using PencilArrays
using LinearAlgebra: transpose!

MPI.Init()
comm = MPI.COMM_WORLD       # we assume MPI.Comm_size(comm) == 12
rank = MPI.Comm_rank(comm)  # rank of local process, in 0:11
n_procs = MPI.Comm_size(comm)


@printf( "proc_id: %d   size: %d\n", rank, n_procs )


# Define MPI Cartesian topology: distribute processes on a 3×4 grid.
topology = MPITopology(comm, (2, 2))


dims_global = ( 16, 16, 16)  # global dimensions of the array
decomp_dims = (2, 3)
pen_x = Pencil(topology, dims_global, decomp_dims)


Ax = PencilArray{Float64}(undef, pen_x)

fill!(Ax, rank * π)  # each process locally fills its part of the array
parent(Ax)           # parent array holding the local data (here, an Array{Float64,3})
local_size = size(Ax)             # size of local part
global_size = size_global(Ax)      # total size of the array = (42, 31, 29)

if rank == 0
  print( "global_size: $(global_size)   local_size: $(local_size) \n" )
end

