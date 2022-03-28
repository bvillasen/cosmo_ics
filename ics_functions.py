import os, sys, time
import numpy as np
root_dir = os.getcwd()
subDirectories = [x[0] for x in os.walk(root_dir)]
sys.path.extend(subDirectories)
from tools import *

def get_inv_FFT( x, shift=False ):
  iFT = np.fft.ifftn( x )
  if shift: iFT = np.fft.fftshift(iFT)
  return iFT

def get_k_grid( nx, dx=1, shift=False ):
  k = 2*np.pi*np.fft.fftfreq( nx, d=dx )
  if shift: k = np.fft.fftshift( k )
  kx, ky, kz = np.meshgrid( k, k, k )
  k_mag = np.sqrt( kx*kx + ky*ky + kz*kz )
  return kx, ky, kz, k_mag
       
def get_FFT( x, dx=1, shift=False, inverse=False  ):
  nx, ny, nz = x.shape
  if nx != ny or nx != nz: 
    print( f'ERROR: Only cubic domains accepted, shape: {x.shape}') 
  FT = np.fft.fftn( x )
  k = 2*np.pi*np.fft.fftfreq( nx, d=dx )
  if shift:
    FT = np.fft.fftshift(FT)
    k = np.fft.fftshift( k )
  kx, ky, kz = np.meshgrid( k, k, k )
  k = np.sqrt( kx*kx + ky*ky + kz*kz )
  return FT, k
  

def Load_Transfer_Function( input_dir, file_base_name, format='CLASS' ):
  file_name = input_dir + f'{file_base_name}_tk.dat'
  print( f'Loading File: {file_name}' )
  tk_data = np.loadtxt( file_name )
  # k is in [h/Mpc]
  if format == 'CAMB':
    k, tk_cdm, tk_b, tk_g, tk_ur, tk_ncdm, tk_tot = tk_data.T
    k2 = -k**2
    tk_cdm, tk_b, tk_g, tk_ur, tk_ncdm, tk_tot = tk_cdm*k2, tk_b*k2, tk_g*k2, tk_ur*k2, tk_ncdm*k2, tk_tot*k2
  elif format == 'CLASS':
    k, tk_g, tk_b, tk_cdm, tk_ur, tk_ncdm, tk_tot, phi, psi = tk_data.T
  else: 
    print(f'ERROR: Transfer function format {format} is not supported')
  tf_data = { 'k':k, 'cdm':tk_cdm, 'baryion':tk_b, 'total':tk_tot }
  return tf_data
  
def Load_Power_Spectrum( input_dir, file_base_name, rescale_by_k2=True ):
  file_name = input_dir + f'{file_base_name}_pk.dat'
  print( f'Loading File: {file_name}' )
  pk_data = np.loadtxt( file_name )
  k, pk = pk_data.T
  # k is in [h/Mpc]
  pk_data = { 'k':k, 'pk':pk }
  return pk_data
  



def Generate_Fourier_Amplitudes( N, dx, z, power_spectrum_function ):
  print( 'Generating Fourier Amplitudes')
  L_box  = N * dx
  V_box = L_box**3
  # Assign random amplitudes in real space
  lambda_vals = np.random.randn( N, N, N )

  # Fourier transform them
  FT_lambda, k_grid = get_FFT( lambda_vals, dx=dx, shift=False ) 
  k_grid[0,0,0] = 1e-6

  # Scale amplitudes with the power spectrum
  pk_vals = power_spectrum_function( k_grid, z ) 
  delta_vals = np.sqrt(pk_vals) * FT_lambda 
  # delta_vals = np.zeros_like( FT_lambda )
  # for i in range(N):
  #   for j in range(N):
  #     for k in range(N):
  #       k_local = k_grid[i,j,k]
  #       # P_local = 1
  #       P_local = power_spectrum_function( k_local)
  #       delta_vals[i,j,k] = np.sqrt( P_local ) * FT_lambda[i,j,k]
  #       # delta_vals[i,j,k] = alpha * k_local**(n_s/2) * transfer_function(k_grid) * FT_lambda[i,j,k] #eq 2 from music paper
  #       #Q: What is the normalization constant alpha for P(k) = alpha * P_0(k) * D(a)**2 * T(a)**2
  return lambda_vals, delta_vals



def Generate_Initial_Conditions( a, dx, FT_amplitudes ):
  nx, ny, nz = FT_amplitudes.shape
  # Growth factor
  D = cosmo.compute_linear_growth_factor( a )
  D_dot = cosmo.compute_linear_growth_factor_derivative( a )
  kx, ky, kz, k_mag = get_k_grid( nx, dx=dx, shift=False )
  #constant mode = 0
  k_mag[0,0,0] = 1 
  kx[0,0,0] = ky[0,0,0] = kz[0,0,0] = 0 
  ft_sx = -1j / D * kx / k_mag**2 * FT_amplitudes
  ft_sy = -1j / D * ky / k_mag**2 * FT_amplitudes
  ft_sz = -1j / D * kz / k_mag**2 * FT_amplitudes
  sx = get_inv_FFT( ft_sx, shift=False )
  sy = get_inv_FFT( ft_sy, shift=False )
  sz = get_inv_FFT( ft_sz, shift=False )
  #Q: How to go from complex to real?
  disp_x = D * sx
  disp_y = D * sy
  disp_z = D * sz
  vx = a * D_dot * sx
  vy = a * D_dot * sy
  vz = a * D_dot * sz

  ics = {}
  return ics
  

def Generate_Linear_Density( amplitudes ):
  delta_vals = get_inv_FFT( amplitudes, shift=False )
  return delta_vals


def Generate_Linear_Velocity( a, dx, FT_amplitudes ):
  nx, ny, nz = FT_amplitudes.shape
  # Growth factor
  D = cosmo.compute_linear_growth_factor( a )
  D_dot = cosmo.compute_linear_growth_factor_derivative( a )
  kx, ky, kz, k_mag = get_k_grid( nx, dx=dx, shift=False )
  #constant mode = 0
  k_mag[0,0,0] = 1 
  kx[0,0,0] = ky[0,0,0] = kz[0,0,0] = 0 
  vel_coeff = -1j * a * D_dot / D / k_mag**2 * FT_amplitudes
  ft_vx, ft_vy, ft_vz = kx*vel_coeff, ky*vel_coeff, kz*vel_coeff
  vx = get_inv_FFT( ft_vx, shift=False )
  vy = get_inv_FFT( ft_vy, shift=False )
  vz = get_inv_FFT( ft_vz, shift=False )
  return vx, vy, vz
