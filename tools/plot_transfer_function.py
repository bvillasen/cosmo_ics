import os, sys, time
import numpy as np
import matplotlib.pyplot as plt
from tools import *

input_dir  = home_dir + 'cosmo_ics/class_output/plank_2018/'
output_dir = data_dir + 'cosmo_sims/ics/class/'
create_directory( output_dir )

file_base_name = 'out00'
rescale_by_k2 = True

file_name = input_dir + file_base_name + '_tk.dat'
tk_data = np.loadtxt( file_name )
k, tk_cdm, tk_b, tk_g, tk_ur, tk_ncdm, tk_tot = tk_data.T
# k is in [h/Mpc]
if rescale_by_k2:
  k2 = -k**2
  tk_cdm, tk_b, tk_g, tk_ur, tk_ncdm, tk_tot = tk_cdm*k2, tk_b*k2, tk_g*k2, tk_ur*k2, tk_ncdm*k2, tk_tot*k2




nrows = 1
ncols = 1

tick_size_major, tick_size_minor = 6, 4
tick_label_size_major, tick_label_size_minor = 14, 12
tick_width_major, tick_width_minor = 1.5, 1

font_size = 18
label_size = 16
alpha = 0.7

line_width = 0.6

border_width = 1.5

text_color  = 'black'

  
fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8*ncols,6*nrows))




ax.plot( k, tk_cdm**2, ls='--', label='cdm' )
ax.plot( k, tk_b**2, ls='--', label='baryon' )

leg = ax.legend(loc=2, frameon=False, fontsize=16 )


ax.set_xscale('log')
ax.set_yscale('log')

ax.set_xlabel( r'$k$  [$h/\mathregular{Mpc}$]')
ax.set_ylabel( r'$T^2(k)$')





figure_name = output_dir + 'transfer_function.png'
fig.savefig( figure_name, bbox_inches='tight', dpi=300, facecolor=fig.get_facecolor() )
print( f'Saved Figure: {figure_name}' )

