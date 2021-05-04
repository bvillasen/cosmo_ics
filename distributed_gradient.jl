using MPI
using PencilFFTs
using AbstractFFTs: fftfreq, rfftfreq
using Random
include("./tools.jl")
using .tools


MPI.Init()

# Input data dimensions (Nx × Ny × Nz)
grid_size = (256, 256, 256)
box_size = (1, 1, 1)  # Lx, Ly, Lz

# Apply a 3D real-to-complex (r2c) FFT.
transform = Transforms.RFFT()

# MPI topology information
comm = MPI.COMM_WORLD  # we assume MPI.Comm_size(comm) == 12
rank = MPI.Comm_rank(comm)  # rank of local process, in 0:11
n_procs = MPI.Comm_size(comm)

proc_dims = (8, 8)     # 3 processes along `y`, 4 along `z`

# Create plan
print_mpi( "Generating FFT Plan:  dims: $(grid_size)   proc_dims: $(proc_dims) \n", rank )
plan = PencilFFTPlan(grid_size, transform, proc_dims, comm)

# Allocate data and initialise field
print_mpi( "Allocating Input   \n", rank )
field = allocate_input(plan)
local_size = size(field)             # size of local part
global_size = size_global(field)   
print_mpi( "Initializing Input:  local_size: $(local_size) \n", rank )
randn!(field)

# Perform distributed FFT
print_mpi( "Peforming FFT:  global_size: $(global_size)   local_size: $(local_size) \n", rank )
field_transformed = plan * field
local_transform_size = size(field_transformed)
global_transform_size = size_global(field_transformed)
print_mpi( " FFT Size:  global_size: $(global_transform_size)   local_size: $(local_transform_size) \n", rank )

local_range = range_local( field_transformed )
# print( "proc_id: $(rank)  Local Range: $(local_range) \n", rank )


print_mpi( "Generating K vector grid\n", rank )
sample_rate = 2π .* grid_size ./ box_size

# In our case (Lx = 2π and Nx even), this gives kx = [0, 1, 2, ..., Nx/2].
kx_global = rfftfreq(grid_size[1], sample_rate[1])
# In our case (Ly = 2π and Ny even), this gives
# ky = [0, 1, 2, ..., Ny/2-1, -Ny/2, -Ny/2+1, ..., -1] (and similarly for kz).
ky_global = fftfreq(grid_size[2], sample_rate[2])
kz_global = fftfreq(grid_size[3], sample_rate[3])

kvec_global = (kx_global, ky_global, kz_global)
kvec_local = getindex.(kvec_global, local_range )


print_mpi( "Allocating Gradient   \n", rank )
grad_field = allocate_output(plan, Val(3))
local_gradient_size = size(grad_field[2])
print_mpi( " Gradient Size:    local_size: 3x$(local_gradient_size) \n", rank )



@inbounds for (n, I) in enumerate(CartesianIndices(field_transformed))
  i, j, k = Tuple(I)  # local indices

  local kx, ky, kz
  kx = kvec_local[1][i]
  ky = kvec_local[2][j]
  kz = kvec_local[3][k]  
  
  u = im * field_transformed[i, j, k]

  grad_field[1][i, j, k] = kx * u
  grad_field[2][i, j, k] = ky * u
  grad_field[3][i, j, k] = kz * u
  
end



print_mpi( "Peforming Inverse FFT:  ", rank )
grad_field = plan \ grad_field
size_output_local = size( grad_field[2] )
size_output_global = size_global( grad_field[2] )
print_mpi( " Output Size:   global_size: 3x$(size_output_global)   local_size: 3x$(size_output_local)   \n", rank )


print_mpi( "Finished Successfully!\n", rank )