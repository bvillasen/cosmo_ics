import numpy as np
import scipy.integrate as integrate
from constants_cosmo import Mpc, Myear, Gcosmo, Msun, kpc
from rk4 import RK4_step
from tools import get_root_dir


class Cosmology:
  
  def __init__(self, z_start=10000, init_time_array=False):
    # Initializa Planck 2018 parameters
    self.H0 = 67.66
    self.Omega_M = 0.3111
    self.Omega_L = 0.6889
    self.Omega_b = 0.0497
    self.Omega_cdm = self.Omega_M - self.Omega_b
    self.sigma8  = 0.8102
    self.n_s     = 0.9665
    self.T_cmb   = 2.726
    self.h = self.H0 / 100.
    self.rho_crit =  3*(self.H0*1e-3)**2/(8*np.pi* Gcosmo) / self.h**2
    self.rho_baryion_mean = self.rho_crit * self.Omega_b   #kg cm^-3
    self.rho_cdm_mean = self.rho_crit * self.Omega_cdm   #kg cm^-3
    self.z_start = z_start
    # self.k_pivot =  0.02/self.h
    self.k_pivot =  1
    
    # For time-z correspondance
    self.z_array = None
    self.a_array = None
    self.t_array = None
    if init_time_array: self.integrate_scale_factor()
    
    self.linear_growth = None
    
    # Tranfer function
    root_dir = get_root_dir()
    self.tf_type = "CLASS"
    self.class_dir = root_dir + '/class_output/'
    self.tf_func = {}
    
    #Power spectrum
    self.pk_amlitude = {}
     
    
    
    
  def get_Hubble( self, current_a ):
      a_dot = self.H0 * np.sqrt( self.Omega_M/current_a + self.Omega_L*current_a**2  )  
      H = a_dot / current_a
      return H
  
  def get_dt( self, current_a, delta_a ):
    a_dot = self.H0 * np.sqrt( self.Omega_M/current_a + self.Omega_L*current_a**2  ) * 1000 / Mpc
    dt = delta_a / a_dot
    return dt  
    
  def get_delta_a( self, current_a, dt ):
    a_dot = self.H0 * np.sqrt( self.Omega_M/current_a + self.Omega_L*current_a**2  ) * 1000 / Mpc
    delta_a = dt * a_dot 
    return delta_a  
    
  def a_deriv( self, time, current_a, kargs=None ):
    a_dot = self.H0 * np.sqrt( self.Omega_M/current_a + self.Omega_L*current_a**2  ) * 1000 / Mpc
    return a_dot
    
  def get_current_a( self, time ):
    current_a = np.interp( time, self.t_vals, self.a_vals )
    return current_a
    
  def integrate_scale_factor(self):
    print( 'Integrating Scale Factor')
    z_vals = [self.z_start]
    a_vals = [1./(self.z_start+1) ]
    t_vals = [0]
    dt = 0.1 * Myear
    while a_vals[-1] < 1.0:
      a = RK4_step( self.a_deriv, t_vals[-1], a_vals[-1], dt )
      a_vals.append( a )
      z_vals.append( 1/a - 1 )
      t_vals.append( t_vals[-1] + dt)
    self.t_vals = np.array( t_vals )
    self.a_vals = np.array( a_vals )
    self.z_vals = np.array( z_vals )
  
    
  def get_time( self, a,  ):
    time = np.interp( a, self.a_vals, self.t_vals)
    return time
    
  
  def compute_linear_growth( self, a_start=1e-10, z_end=99, n_samples=100000  ):
    print( f'Computing linear growth factor' )
    a_end = 1 / ( z_end + 1 )
    a_vals = np.linspace( a_start, a_end, n_samples )
    H_vals = self.get_Hubble( a_vals )
    integrand = 1 / ( a_vals * H_vals )**3
    integral = integrate.cumtrapz( integrand, x=a_vals, initial=0 )
    D_vals = 5./2 * self.H0**2 * self.Omega_M * H_vals * integral
    self.linear_growth = {}
    self.linear_growth['D(a)'] = D_vals
    self.linear_growth['a'] = a_vals
  
  def growth_integrand_func( self, a ):
    H = self.get_Hubble( a )
    return 1 / ( H * a )**3
    
  def linear_growth_factor( self, a, a_start=1e-100 ):
    H = self.get_Hubble( a )
    integral = integrate.romberg( self.growth_integrand_func, a_start, a  )
    D = 5./2 * self.H0**2 * self.Omega_M * H * integral    
    return D
  
  def linear_growth_factor_colossus( self, a ):
    from colossus.cosmology import cosmology
    params = {'flat': True, 'H0': self.H0, 'Om0': self.Omega_M, 'Ob0': self.Omega_b, 'sigma8': 0, 'ns': 0}
    cosmology = cosmology.setCosmology('myCosmo', params)
    z = 1./a - 1
    D = cosmology.growthFactorUnnormalized( z )
    return D
  
  def linear_growth_factor_derivative( self, a, a_start=1e-100, delta_a=1e-5, use_colossus=False):
    if use_colossus:
      D   = self.linear_growth_factor_colossus( a )
      D_l = self.linear_growth_factor_colossus( a-delta_a )
      D_r = self.linear_growth_factor_colossus( a+delta_a )
    else:
      D   = self.linear_growth_factor( a, a_start=a_start )
      D_l = self.linear_growth_factor( a-delta_a, a_start=a_start )
      D_r = self.linear_growth_factor( a+delta_a, a_start=a_start )
    time_l = self.get_time( a-delta_a )
    time_r = self.get_time( a+delta_a )
    delta_t = time_r - time_l
    D_dot = ( D_r - D_l ) / ( delta_t )
    return D_dot
  
  def load_transfer_function( self, file_name, tf_type='CLASS', format='CAMB'  ):
    if tf_type == 'CLASS':
      # file_name = input_dir + f'{file_base_name}_tk.dat'
      in_file_name = self.class_dir + file_name 
      print( f'Loading TF File: {in_file_name}' )
      tk_data = np.loadtxt( in_file_name )
      if format == 'CAMB':
        k, tk_cdm, tk_b, tk_g, tk_ur, tk_ncdm, tk_tot = tk_data.T
        # k2 = -k**2
        # tk_cdm, tk_b, tk_g, tk_ur, tk_ncdm, tk_tot = tk_cdm*k2, tk_b*k2, tk_g*k2, tk_ur*k2, tk_ncdm*k2, tk_tot*k2
      elif format == 'CLASS':
        k, tk_g, tk_b, tk_cdm, tk_ur, tk_ncdm, tk_tot, phi, psi = tk_data.T
        tk_cdm, tk_b, tk_tot = np.abs(tk_cdm), np.abs(tk_b), np.abs(tk_tot)
      else: 
        print(f'ERROR: Transfer function format {format} is not supported')      
      tf_data = { 'k':k, 'cdm':tk_cdm, 'baryon':tk_b, 'total':tk_tot }
    else:
      print( f'ERROR: {tf_type} is not supported ')
    self.transfer_function_data = tf_data

  def load_power_spectrum( self, file_name, type='CLASS', ):
    if type == 'CLASS':
      # file_name = input_dir + f'{file_base_name}_tk.dat'
      in_file_name = self.class_dir + file_name 
      print( f'Loading Pk File: {in_file_name}' )
      pk_data = np.loadtxt( in_file_name )
      k, pk = pk_data.T
      pk_data = { 'k':k, 'pk':pk }
    else:
      print( f'ERROR: {tf_type} is not supported ')
    self.power_spectrum_data = pk_data
  
  def init_transfer_function( self, components=[ 'cdm', 'baryon', 'total' ], interp_type='cubic' ):
    from scipy import interpolate
    for component in components:
      if component not in self.transfer_function_data:
        print(f'ERROR: This component ({component}) is not supported by this transfer function. ')
        return None
      k_vals  = self.transfer_function_data['k']
      tf_vals = self.transfer_function_data[component]
      log_k  = np.log10(k_vals).copy()
      log_tf = np.log10(tf_vals).copy()
      log_transfer_function = interpolate.interp1d( log_k, log_tf, kind=interp_type, fill_value='extrapolate' )
      # self.tf_func[component] = lambda k : 10**log_transfer_function(np.log10(k))
      self.tf_func[component] = log_transfer_function

  
  def transfer_function( self, k, component='total' ):
    if self.tf_func is None:
      print('ERROR: First you need to initialize a transfer_function')
      return None
    if component not in self.tf_func:
      print(f'ERROR: No tranfer function for component: {component} ')
      return None
    transfer_function = self.tf_func[component]
    return 10**transfer_function(np.log10(k))
  
  def transfer_function_eisenstein98( k_vals ):
    from colossus.cosmology import cosmology
    tk_vals = cosmology.power_spectrum.transferFunction(k_vals, cosmo.h, cosmo.Omega_M, cosmo.Omega_b, cosmo.T_cmb, model='eisenstein98')
    return tk_vals 
    
  def power_spectrum( self, k, z, component='total' ):
    a = 1. / ( z + 1 )
    D_plus = self.linear_growth_factor( a ) 
    amp = self.pk_amlitude['total']
    T = self.transfer_function( k, component=component )
    ns = self.n_s
    return amp * (k/self.k_pivot)**ns * ( D_plus * T )**2 
    

  def W_R( self, k, R=8 ):
    # print( k)
    x = k * R
    # if x < 1e-3: W = 1.0 - x**2/10
    # else:W = 3. / x**3 * ( np.sin(x) - x * np.cos(x) )
    W = 3. / x**3 * ( np.sin(x) - x * np.cos(x) ) 
    return W

  def Sigma_R_integrand( self, k, R, z, component ):
    pk = self.power_spectrum(k, z, component=component )
    W = self.W_R( k, R )
    # print( f'{k} {R} {component} {pk} {W}')
    return pk * W**2 * k**2
    
  def SigmaR( self, R, z, component='total' ):
    from scipy import integrate
    k_max = self.transfer_function_data['k'].max()
    k_min = self.transfer_function_data['k'].min()
    # integral, err = integrate.quadrature( self.Sigma_R_integrand, k_min, k_max, args=(R, z, component),  maxiter=1000 )
    # integral = integrate.romberg( self.Sigma_R_integrand, k_min, k_max, args=(R, z, component)  )
    n_samples = 10000
    k_vals = np.logspace( np.log10(k_min), np.log10(k_max), n_samples )
    integrand = np.zeros_like(k_vals)
    for i in range( n_samples ): integrand[i] = self.Sigma_R_integrand( k_vals[i], R, z, component )
    integral = integrate.simps( integrand, x=k_vals )
    return np.sqrt(1/(2*np.pi**2) * integral) 
  
  def init_power_spectrum( self ):
    D_plus = self.linear_growth_factor( 1.0 ) 
    components = [ 'total' ]
    z = 0.0
    for component in components:
      if component not in self.tf_func: continue
      self.pk_amlitude[component] = 1.0
      print( f'Computing P(k) amplitude component: {component}')
      s8 = self.SigmaR( 8.0, z, component=component )
      self.pk_amlitude[component] = (self.sigma8/s8)**2
    
    
    

