import os, sys, time
import numpy as np
root_dir = os.getcwd()
subDirectories = [x[0] for x in os.walk(root_dir)]
sys.path.extend(subDirectories)
from tools import *
from cosmology import Cosmology
from ics_functions import Generate_Fourier_Amplitudes, get_k_grid, get_inv_FFT
from constants_cosmo import Myear, Mpc
from internal_energy import get_internal_energy
from ics_particles import generate_ics_particles
from ics_grid import expand_data_grid_to_cholla
from plotting_functions import Plot_transfer_function, Plot_power_spectrum, plot_densities

np.random.seed(123456789)

output_dir = data_dir + 'cosmo_sims/test_ics/ics_python/hydro/256_50Mpc/ics_8_z100/'
# output_dir =  data_dir + 'figures/cosmo_ics/'
create_directory( output_dir )


figs_dir = data_dir + 'cosmo_sims/test_ics/figures/'
create_directory( figs_dir ) 

cosmo = Cosmology( init_time_array=True )

current_z = 100
L = 50.0 #Mpc/h
N = 256
L_kpc = L * 1e3
N_particles = N**3 
dx = L / N
current_a = 1 / ( current_z + 1 )
V = L**3
V_kpc = L_kpc**3
data_ics = { 'dm':{}, 'gas':{} }
data_ics['current_a'] = current_a
data_ics['current_z'] = current_z

# Load the transfer function
components = [ 'cdm', 'baryon', 'total' ]
# components = [ 'cdm', 'baryon',  ]
# components = [ 'cdm' ]
class_tf_file_name = 'plank_2018/out00_tk.dat'
class_pk_file_name = 'plank_2018/out00_pk.dat'
cosmo.load_transfer_function( class_tf_file_name, tf_type='CLASS'  )
cosmo.load_power_spectrum( class_pk_file_name, type='CLASS'  )
cosmo.init_transfer_function( components=components )
cosmo.init_power_spectrum()


input_pk_file = data_dir + 'cosmo_sims/test_ics/ics_music/dmo/input_powerspec.txt'
input_pk_data = np.loadtxt( input_pk_file ).T
input_pk = {'k_vals':input_pk_data[0], 'pk':input_pk_data[1], 'label':'Input Music' }
cosmo.load_input_power_spectrum( input_pk_file )


k_vals = cosmo.transfer_function_data['k']
tk_total = cosmo.transfer_function( k_vals, component='total')
pk_total = cosmo.power_spectrum( k_vals, current_z, component='total' )
pk_diff = ( pk_total - cosmo.power_spectrum_data['pk'] ) / cosmo.power_spectrum_data['pk'] 

Plot_transfer_function( k_vals, components, cosmo, output_dir )
Plot_power_spectrum( k_vals, current_z, components, cosmo, output_dir, other_pk=input_pk )

shift_FFT = False

a = current_a  
# amplitudes, FT_amplitudes = Generate_Fourier_Amplitudes( N, dx, current_z, cosmo.power_spectrum )
amplitudes, FT_amplitudes = Generate_Fourier_Amplitudes( N, dx, current_z, cosmo.input_power_spectrum, shift=shift_FFT )
nx, ny, nz = FT_amplitudes.shape

# Growth factor
D = cosmo.linear_growth_factor( a )
D_dot = cosmo.linear_growth_factor_derivative( a )



kx, ky, kz, k_mag = get_k_grid( nx, dx=dx, shift=shift_FFT )
# kz, ky, kx, k_mag = get_k_grid( nx, dx=dx, shift=False )
#constant mode = 0
# k_mag[0,0,0] = 1 
k2 = ( kx * kx + ky * ky + kz * kz )
zero_indx = np.where( k2 == 0 )
k2[zero_indx] = 1
kx[zero_indx] = 0
ky[zero_indx] = 0
kz[zero_indx] = 0
ft_sx = -1j / D * kx / k2 * FT_amplitudes
ft_sy = -1j / D * ky / k2 * FT_amplitudes
ft_sz = -1j / D * kz / k2 * FT_amplitudes
factor = 1
sx = get_inv_FFT( ft_sx, shift=shift_FFT ).real * factor
sy = get_inv_FFT( ft_sy, shift=shift_FFT ).real * factor
sz = get_inv_FFT( ft_sz, shift=shift_FFT ).real * factor
# #Q: How to go from complex to real?
disp_x = D * sx
disp_y = D * sy
disp_z = D * sz
vel_x = a * D_dot * sx * Mpc * 1e-3 / cosmo.h
vel_y = a * D_dot * sy * Mpc * 1e-3 / cosmo.h
vel_z = a * D_dot * sz * Mpc * 1e-3 / cosmo.h
particle_mass = cosmo.rho_cdm_mean * V_kpc / N_particles
data_ics['dm']['p_mass'] = particle_mass
# data_ics['dm']['p_mass'] = cosmo.rho_mean * V_kpc / N_particles
print( f"Particle Mass: {data_ics['dm']['p_mass']}")
data_ics['dm']['vel_x'] = vel_x.flatten()
data_ics['dm']['vel_y'] = vel_y.flatten()
data_ics['dm']['vel_z'] = vel_z.flatten()

pos_uniform_1D = ( np.linspace( 0, nx-1, nx) + 0.5 ) * dx 
pos_x, pos_y, pos_z = np.meshgrid( pos_uniform_1D, pos_uniform_1D, pos_uniform_1D )
# pos_z, pos_y, pos_x = np.meshgrid( pos_uniform_1D, pos_uniform_1D, pos_uniform_1D )
pos_x += disp_x
pos_y += disp_y 
pos_z += disp_z
pos_x *= 1e3 # kpc/h
pos_y *= 1e3 # kpc/h
pos_z *= 1e3 # kpc/h
pos_x[pos_x<0]     += L_kpc
pos_x[pos_x>L_kpc] -= L_kpc
pos_y[pos_y<0]     += L_kpc
pos_y[pos_y>L_kpc] -= L_kpc
pos_z[pos_z<0]     += L_kpc
pos_z[pos_z>L_kpc] -= L_kpc

pos_x = pos_x.flatten()
pos_y = pos_y.flatten()
pos_z = pos_z.flatten()

data_ics['dm']['pos_z'] = pos_x
data_ics['dm']['pos_y'] = pos_y
data_ics['dm']['pos_x'] = pos_z



def get_particles_density( pos_x, pos_y, pos_z, p_mass, nx, Lx, ):
  N_particles = len( pos_x )
  density  = np.zeros( [nx, nx, nx] )
  dx  = Lx / nx
  indices_x = ( pos_x / dx ).astype( int )
  indices_y = ( pos_y / dx ).astype( int )
  indices_z = ( pos_z / dx ).astype( int )
  for i in range(N_particles):
    density[indices_x[i], indices_y[i], indices_z[i]] += 1
  density *= p_mass / dx**3
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
    
dm_density =  get_density_cic_cuda( pos_z, pos_y, pos_x, particle_mass, N, L_kpc )

# Baryons
delta = get_inv_FFT( FT_amplitudes, shift=shift_FFT ).real * factor
gas_density = cosmo.rho_baryion_mean * ( 1 + delta )
# gas_density = gas_density.T


# gas_density = dm_density / cosmo.rho_cdm_mean * cosmo.rho_baryion_mean
gas_vel_x = vel_x
gas_vel_y = vel_y
gas_vel_z = vel_z
temperature = 231.44931976   #k
gas_U = get_internal_energy( temperature ) * gas_density
gas_E = gas_U + 0.5 * gas_density *  ( gas_vel_x*gas_vel_x + gas_vel_y*gas_vel_y + gas_vel_z*gas_vel_z  )  
data_ics['gas']['density'] = gas_density
data_ics['gas']['momentum_x'] = gas_density * gas_vel_z
data_ics['gas']['momentum_y'] = gas_density * gas_vel_y
data_ics['gas']['momentum_z'] = gas_density * gas_vel_x
data_ics['gas']['GasEnergy'] = gas_U
data_ics['gas']['Energy'] = gas_E

  
plot_densities( dm_density, gas_density, figs_dir )


# 
# n_snapshot = 0
# Lbox = L_kpc
# n_points = N
# box_size = [ Lbox, Lbox, Lbox ]
# grid_size = [ n_points, n_points, n_points ]
# proc_grid  = [ 2, 2, 2 ]
# output_base_name = '{0}_particles.h5'.format( n_snapshot )
# generate_ics_particles(data_ics, output_dir, output_base_name, proc_grid, box_size, grid_size)
# 
# output_base_name = '{0}.h5'.format( n_snapshot )
# expand_data_grid_to_cholla( proc_grid, data_ics['gas'], output_dir, output_base_name, loop_complete_files=True )


# 
# 
# 
# T_0 = 2.725 #k
# a_td = 1./137 * ( cosmo.Omega_b * cosmo.h**2 / 0.022 )**(-2./5)
# gas_T_aprox = T_0 * a_td / current_a**2 
# beta = 1.73
# gas_T = T_0 / current_a * a_td / ( current_a**beta + a_td**beta )**(1/beta)
