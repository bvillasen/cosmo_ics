import os, sys, time
import numpy as np
import scipy.integrate as integrate
root_dir = os.getcwd()
subDirectories = [x[0] for x in os.walk(root_dir)]
sys.path.extend(subDirectories)
from tools import *
from internal_energy import get_internal_energy
from density_cic import get_density_cic_cuda
from plotting_functions import  plot_densities
from ics_particles import generate_ics_particles
from ics_grid import expand_data_grid_to_cholla


np.random.seed(123456789)

G =  4.300927161e-06 # kpc km^2 s^-2 Msun^-1
Myear = 365 * 24 * 3600 * 1e6
pc = 3.0857e13  #km
kpc = 1e3 * pc
Mpc = 1e6 * pc

def get_Hubble( a, cosmo ):
  H = cosmo['H0'] * np.sqrt( cosmo['Om_R']/a**4 + cosmo['Om_M']/a**3 + cosmo['Om_L'] )
  return H

def get_time_integrand( a, cosmo ):
  H = get_Hubble( a, cosmo )
  return 1 / ( H * a ) 

def get_time( a, cosmo, a_start=1e-50 ):
  integral, err = integrate.quad( get_time_integrand, a_start, a, args=(cosmo)  )
  # print( err / integral )
  return integral * Mpc
  
def growth_factor_integrand( a, cosmo ):
  H = get_Hubble( a, cosmo )
  return 1 / ( H * a )**3


def get_linear_growth_factor( a, cosmo, a_start=1e-50 ):
  H = get_Hubble( a, cosmo )
  integral = integrate.romberg( growth_factor_integrand, a_start, a, args=[cosmo]  )
  D =  5./2 * cosmo['H0']**2 * cosmo['Om_M'] * H * integral
  return D 


def get_linear_growth_factor_colossus(  a, cosmo, a_start=1e-50 ):
  from colossus.cosmology import cosmology
  params = {'flat': True, 'H0':cosmo['H0'], 'Om0':cosmo['Om_M'], 'Ob0':cosmo['Om_b'], 'sigma8': cosmo['sigma_8'], 'ns': cosmo['n_s']}
  cosmology = cosmology.setCosmology('myCosmo', params)
  z = 1./a - 1
  D = cosmology.growthFactorUnnormalized( z )
  return D    

def get_linear_growth_factor_deriv( a, cosmo, a_start=1e-50, delta_a=1e-5, use_colossus=False ):
  D_func = get_linear_growth_factor
  if use_colossus: D_func = get_linear_growth_factor_colossus
  D_l = D_func( a - delta_a, cosmo, a_start=a_start )
  D_r = D_func( a + delta_a, cosmo, a_start=a_start )
  time_l = get_time( a - delta_a, cosmo )
  time_r = get_time( a + delta_a, cosmo )
  delta_t = time_r - time_l
  D_dot = ( D_r - D_l ) / ( delta_t )
  return D_dot
  
    

def interpolate_power_spectrum( k_to_inerp, k_vals, pk_vals, log=True ):
  if log: k_to_inerp, k_vals, pk_vals = np.log10(k_to_inerp), np.log10(k_vals), np.log10(pk_vals)
  pk = np.interp( k_to_inerp, k_vals, pk_vals )
  if log: pk = 10**pk
  return pk

baryons = True
if baryons: type = 'hydro'
else: type = 'dmo'  

figs_dir = data_dir + 'cosmo_sims/test_ics/figures/'
output_dir = data_dir + f'cosmo_sims/test_ics/ics_python/{type}/256_50Mpc/ics_8_z100/'
create_directory( output_dir )

# input_pk_file = data_dir + 'cosmo_sims/test_ics/ics_music/dmo/input_powerspec.txt'
input_pk_file = 'input_power_spectrum.txt'
input_pk_data = np.loadtxt( input_pk_file ).T
input_pk = {'k_vals':input_pk_data[0], 'pk':input_pk_data[1] }



# Box parameters  
L = 50.0 #Mpc/h
n_grid = 256
dx = L / n_grid
N_particles = n_grid**3
L_kpc = L * 1e3
V_kpc = L_kpc**3

# Cosmological parameters
current_z = 100
current_a = 1 / ( current_z + 1)
cosmology = {}
cosmology['H0'] = 67.66
cosmology['h'] = cosmology['H0'] / 100
cosmology['Om_M'] = 0.3111
cosmology['Om_L'] = 0.6889
cosmology['Om_b'] = 0.0497
cosmology['Om_R'] = 4.166e-5/cosmology['h']**2
cosmology['sigma_8'] = 0.8102
cosmology['n_s'] = 0.9665
rho_crit =  3*(cosmology['H0']*1e-3)**2/(8*np.pi* G) / cosmology['h']**2
if baryons: Om_cdm = cosmology['Om_M'] - cosmology['Om_b']
else: Om_cdm = cosmology['Om_M']
rho_cdm_mean = rho_crit * Om_cdm
rho_gas_mean = rho_crit * cosmology['Om_b']

data_ics = { 'dm':{}, 'gas':{} }
data_ics['current_a'] = current_a
data_ics['current_z'] = current_z

# Assign random gaussian amplitudes in real space
lambda_vals = np.random.randn( n_grid, n_grid, n_grid )

# Take the FFT of the amplitudes 
FT_lambda_vals = np.fft.fftn( lambda_vals )
# Compute the k values
k_1d = 2*np.pi*np.fft.fftfreq( n_grid, d=dx )
# ky, kz, kx = np.meshgrid( k_1d, k_1d, k_1d )
kz, ky, kx = np.meshgrid( k_1d, k_1d, k_1d, indexing='ij' )
    
k2 = kx*kx + ky*ky + kz*kz
k_grid = np.sqrt( k2 )
# Make the k=0 freq equal to 1
k2[0,0,0] = 1

# Evaluate the power spectrum on the grid 
# and multiply it by the amplitudes in Fourier space  
pk_grid = interpolate_power_spectrum( k_grid, input_pk['k_vals'], input_pk['pk'], log=False )
delta_vals = np.sqrt(pk_grid) * FT_lambda_vals

# Linear growth factor
D = get_linear_growth_factor( current_a, cosmology )
D_dot = get_linear_growth_factor_deriv( current_a, cosmology )


ft_sx = 1j / D * kx / k2 * delta_vals
ft_sy = 1j / D * ky / k2 * delta_vals
ft_sz = 1j / D * kz / k2 * delta_vals
sx = np.fft.ifftn( ft_sx ).real 
sy = np.fft.ifftn( ft_sy ).real 
sz = np.fft.ifftn( ft_sz ).real 
disp_x = D * sx
disp_y = D * sy
disp_z = D * sz
vel_x = current_a * D_dot * sx * Mpc / cosmology['h']
vel_y = current_a * D_dot * sy * Mpc / cosmology['h']
vel_z = current_a * D_dot * sz * Mpc / cosmology['h']

pos_uniform_1D = ( np.linspace( 0, n_grid-1, n_grid) + 0.5 ) * dx 
# pos_y, pos_z, pos_x = np.meshgrid( pos_uniform_1D, pos_uniform_1D, pos_uniform_1D )
pos_z, pos_y, pos_x = np.meshgrid( pos_uniform_1D, pos_uniform_1D, pos_uniform_1D, indexing='ij' )

pos_x += disp_x
pos_y += disp_y 
pos_z += disp_z
pos_x *= 1e3 # kpc/h
pos_y *= 1e3 # kpc/h
pos_z *= 1e3 # kpc/h
# Periodic boundaries
pos_x[pos_x<0]     += L_kpc
pos_x[pos_x>L_kpc] -= L_kpc
pos_y[pos_y<0]     += L_kpc
pos_y[pos_y>L_kpc] -= L_kpc
pos_z[pos_z<0]     += L_kpc
pos_z[pos_z>L_kpc] -= L_kpc
pos_x = pos_x.flatten()
pos_y = pos_y.flatten()
pos_z = pos_z.flatten()

particle_mass = rho_cdm_mean * V_kpc / N_particles
data_ics['dm']['p_mass'] = particle_mass
print( f"Particle Mass: {data_ics['dm']['p_mass']}")
data_ics['dm']['vel_x'] = vel_x.flatten()
data_ics['dm']['vel_y'] = vel_y.flatten()
data_ics['dm']['vel_z'] = vel_z.flatten()
data_ics['dm']['pos_x'] = pos_x
data_ics['dm']['pos_y'] = pos_y
data_ics['dm']['pos_z'] = pos_z

dm_density =  get_density_cic_cuda( pos_x, pos_y, pos_z, particle_mass, n_grid, L_kpc )

n_snapshot = 0
Lbox = L_kpc
box_size = [ Lbox, Lbox, Lbox ]
grid_size = [ n_grid, n_grid, n_grid ]
proc_grid  = [ 2, 2, 2 ]
output_base_name = '{0}_particles.h5'.format( n_snapshot )
generate_ics_particles(data_ics, output_dir, output_base_name, proc_grid, box_size, grid_size)

if baryons:

  # Baryons
  delta = np.fft.ifftn( delta_vals ).real 
  gas_density = rho_gas_mean * ( 1 + delta )
  gas_density = gas_density.T
  # gas_density = dm_density / cosmo.rho_cdm_mean * cosmo.rho_baryion_mean
  gas_vel_x = vel_x.T
  gas_vel_y = vel_y.T
  gas_vel_z = vel_z.T
  temperature = 231.44931976   #k
  gas_U = get_internal_energy( temperature ) * gas_density
  gas_E = gas_U + 0.5 * gas_density *  ( gas_vel_x*gas_vel_x + gas_vel_y*gas_vel_y + gas_vel_z*gas_vel_z  )  
  data_ics['gas']['density'] = gas_density
  data_ics['gas']['momentum_x'] = gas_density * gas_vel_x
  data_ics['gas']['momentum_y'] = gas_density * gas_vel_y
  data_ics['gas']['momentum_z'] = gas_density * gas_vel_z
  data_ics['gas']['GasEnergy'] = gas_U
  data_ics['gas']['Energy'] = gas_E

  plot_densities( dm_density, gas_density, figs_dir )

  output_base_name = '{0}.h5'.format( n_snapshot )
  expand_data_grid_to_cholla( proc_grid, data_ics['gas'], output_dir, output_base_name, loop_complete_files=True )
