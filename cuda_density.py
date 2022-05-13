import sys, time, os
from os import listdir
from os.path import isfile, join
import h5py as h5
import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray

root_dir = os.getcwd()
subDirectories = [x[0] for x in os.walk(root_dir)]
sys.path.extend(subDirectories)
from cudaTools import setCudaDevice

#Select CUDA Device
useDevice = 0
#initialize pyCUDA context
cudaDevice = setCudaDevice( devN=useDevice, usingAnimation=False )

cudaCodeFile = open( 'tools/cuda_kernels.cu', 'r')
cudaCodeString = cudaCodeFile.read()
cudaCodeStringComplete = cudaCodeString
cudaCode = SourceModule(cudaCodeStringComplete, no_extern_c=True, include_dirs=[] )
get_density_kernel = cudaCode.get_function("Get_Density_CIC_Kernel")

#set thread grid for CUDA kernels
block_size = 1024
block1D = ( block_size, 1,  1)


N = 256
N_particles = N**3
L = 50000
dx = L / N
n_ghost = 1
N_total = N + 2*n_ghost 
density = np.zeros( [N_total, N_total, N_total])
pos_x = np.random.rand( N_particles ) * L
pos_y = np.random.rand( N_particles ) * L
pos_z = np.random.rand( N_particles ) * L


d_density = gpuarray.to_gpu( density.astype(np.float64) )
d_pos_x = gpuarray.to_gpu( pos_x.astype(np.float64) )
d_pos_y = gpuarray.to_gpu( pos_y.astype(np.float64) )
d_pos_z = gpuarray.to_gpu( pos_z.astype(np.float64) )

grid_size = ( N_particles - 1 ) // block_size + 1
grid1D = ( grid_size, 1, 1 )

particle_mass = 1.0
nx , ny, nz = N, N, N
dy, dz = dx, dx
xMin, yMin, zMin = 0, 0, 0
xMax, yMax, zMax = L, L, L

get_density_kernel( np.int32( N_particles),  np.float64(particle_mass), d_density, d_pos_x, d_pos_y, d_pos_z,
                    np.float64(xMin), np.float64(yMin), np.float64(zMin),
                    np.float64(xMax), np.float64(yMax), np.float64(zMax), 
                    np.float64(dx), np.float64(dy), np.float64(dz),
                    np.int32(nx), np.int32(ny), np.int32(nz), np.int32(n_ghost),  grid=grid1D, block=block1D )
                    
density = d_density.get()
  