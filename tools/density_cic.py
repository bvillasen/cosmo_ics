import numpy as np




def get_density( pos_x, pos_y, pos_z, p_mass, nx, Lx ):
  N_particles = len(pos_x)
  n_ghost = 1
  n_total = nx + 2*n_ghost  
  density = np.zeros( [n_total, n_total, n_total])
  
  particle_mass = p_mass
  nx , ny, nz = nx, nx, nx
  dx = Lx / nx
  dy, dz = dx, dx
  xMin, yMin, zMin = 0, 0, 0
  xMax, yMax, zMax = Lx, Lx, Lx

  density[1,:,:] += density[-1,:,:]
  density[-2,:,:] += density[0,:,:]
  density[:,1,:] += density[:,-1,:]
  density[:,-2,:] += density[:,0,:]
  density[:,:,1] += density[:,:,-1]
  density[:,:,-2] += density[:,:,0]
  density = density[1:-1,1:-1,1:-1]
  return density


def get_density_cic_cuda( pos_x, pos_y, pos_z, p_mass, nx, Lx ):
  import pycuda.driver as cuda
  from pycuda.compiler import SourceModule
  import pycuda.gpuarray as gpuarray
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

  N_particles = len(pos_x)
  n_ghost = 1
  n_total = nx + 2*n_ghost  
  density = np.zeros( [n_total, n_total, n_total])
  d_density = gpuarray.to_gpu( density.astype(np.float64) )
  d_pos_x = gpuarray.to_gpu( pos_x.astype(np.float64) )
  d_pos_y = gpuarray.to_gpu( pos_y.astype(np.float64) )
  d_pos_z = gpuarray.to_gpu( pos_z.astype(np.float64) )#set thread grid for CUDA kernels
  
  block_size = 512
  block1D = ( block_size, 1,  1)
  grid_size = ( N_particles - 1 ) // block_size + 1
  grid1D = ( grid_size, 1, 1 )

  particle_mass = p_mass
  nx , ny, nz = nx, nx, nx
  dx = Lx / nx
  dy, dz = dx, dx
  xMin, yMin, zMin = 0, 0, 0
  xMax, yMax, zMax = Lx, Lx, Lx

  get_density_kernel( np.int32( N_particles),  np.float64(particle_mass), d_density, d_pos_x, d_pos_y, d_pos_z,
                      np.float64(xMin), np.float64(yMin), np.float64(zMin),
                      np.float64(xMax), np.float64(yMax), np.float64(zMax), 
                      np.float64(dx), np.float64(dy), np.float64(dz),
                      np.int32(nx), np.int32(ny), np.int32(nz), np.int32(n_ghost),  grid=grid1D, block=block1D )
                      
  density = d_density.get()
  density[1,:,:] += density[-1,:,:]
  density[-2,:,:] += density[0,:,:]
  density[:,1,:] += density[:,-1,:]
  density[:,-2,:] += density[:,0,:]
  density[:,:,1] += density[:,:,-1]
  density[:,:,-2] += density[:,:,0]
  density = density[1:-1,1:-1,1:-1]
  return density
    
