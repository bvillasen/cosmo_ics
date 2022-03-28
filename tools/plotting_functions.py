import sys, os
import numpy as np
import h5py as h5
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
root_dir = os.path.dirname(os.getcwd()) + '/'
subDirectories = [x[0] for x in os.walk(root_dir)]
sys.path.extend(subDirectories)
from tools import *

tick_size_major, tick_size_minor = 4, 3
tick_label_size_major, tick_label_size_minor = 12, 12
tick_width_major, tick_width_minor = 1.5, 1
label_size = 14

text_color = 'black'

matplotlib.rcParams['font.sans-serif'] = "Helvetica"
matplotlib.rcParams['font.family'] = "sans-serif"
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['mathtext.rm'] = 'serif'


def Plot_transfer_function( k_vals, components, cosmo, output_dir ):
  
  nrows, ncols = 1, 1
  fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8*ncols,6*nrows))
  
  for component in components:
    tk_vals = cosmo.transfer_function( k_vals, component=component )
    tk_vals /= tk_vals[0]
    ax.plot( k_vals, tk_vals, label=component )
  
  from colossus.cosmology import cosmology
  tk_vals = cosmology.power_spectrum.transferFunction(k_vals, cosmo.h, cosmo.Omega_M, cosmo.Omega_b, 2.726, model='eisenstein98')
  ax.plot( k_vals, tk_vals, label='Eisenstein' )
  
  ax.legend( frameon=False )
  
  ax.set_xlabel(  r'$k$   $[h \mathregular{Mpc^{-1}}]$', fontsize=label_size) 
  ax.set_ylabel(  r'$T\,(k, z=100)$', fontsize=label_size) 
  ax.tick_params(axis='both', which='major', direction='in', color=text_color, labelcolor=text_color, labelsize=tick_label_size_major, size=tick_size_major, width=tick_width_major  )
  ax.tick_params(axis='both', which='minor', direction='in', color=text_color, labelcolor=text_color, labelsize=tick_label_size_minor, size=tick_size_minor, width=tick_width_minor  )

  
  ax.set_xscale('log')
  ax.set_yscale('log')
  ax.set_xlim(1e-3, None)
  # ax.set_ylim(1e-1, None)
  
  figure_name = output_dir + 'transfer_function.png'
  fig.savefig( figure_name, bbox_inches='tight', dpi=300, facecolor=fig.get_facecolor() )
  print( f'Saved Figure: {figure_name}' )



def Plot_power_spectrum( k_vals, z, components, cosmo, output_dir, other_pk=None ):
  
  nrows, ncols = 1, 1
  fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8*ncols,6*nrows))
  
  for component in components:
    pk_vals = cosmo.power_spectrum( k_vals, z, component=component )
    ax.plot( k_vals, pk_vals, label=component )
  
  if other_pk is not None:
    k_vals = other_pk['k_vals']
    pk_vals = other_pk['pk']
    label = other_pk['label']
    print( label )
    ax.plot( k_vals, pk_vals, label=label )
  
  ax.legend( frameon=False )
  
  ax.set_xlabel(  r'$k$   $[h \mathregular{Mpc^{-1}}]$', fontsize=label_size) 
  ax.set_ylabel(  r'$P\,(k, z=100)$', fontsize=label_size) 
  ax.tick_params(axis='both', which='major', direction='in', color=text_color, labelcolor=text_color, labelsize=tick_label_size_major, size=tick_size_major, width=tick_width_major  )
  ax.tick_params(axis='both', which='minor', direction='in', color=text_color, labelcolor=text_color, labelsize=tick_label_size_minor, size=tick_size_minor, width=tick_width_minor  )

  
  ax.set_xscale('log')
  ax.set_yscale('log')
  ax.set_xlim(1e-4, 5)
  # ax.set_ylim(1e-3, None)
  
  figure_name = output_dir + 'power_spectrum.png'
  fig.savefig( figure_name, bbox_inches='tight', dpi=300, facecolor=fig.get_facecolor() )
  print( f'Saved Figure: {figure_name}' )

