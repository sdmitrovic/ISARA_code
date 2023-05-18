#########################################################################################################################
# mopsmap_SD_run.py                                   by:  Joseph Schlosser
#                                                revised:  2 Feb 2022 
#                                    language (revision):  python3 (3.8.2-0ubuntu2)
#
# description: procedure to interface with the Modeled Optical Properties of ensembles of Aerosol Particles (MOPSMAP) 
# package [1]
#
# each of the required inputs for this code are described as follows:
# 1) wvl: wavelength (a single number or a list of numbers)
# 2) size_equ: size equivalence 
# 3) dndlogdp,dpg: log-normal size distribution and geometric mean diameter of each bin (single parameter each or lists 
#    with same lengths describing the parameters of each mode)
# 4) m: refractive index, given as a string which is added in input file after 'refrac' (if single string it is applied 
#    to all modes, otherwise if list it is applied to each mode)
# 5) nonabs_fraction: ratio of non-absorbing particles (if single parameter it is applied to all modes, otherwise if list
#    it is applied to each mode)
# 6) shape: particle shape, given as a string which is added in input file after 'shape' (if single string it is applied 
#    to all modes, otherwise if list it is applied to each mode)
# 7) density: particle density (if single parameter it is applied to all modes, otherwise if list it is applied to each 
#    mode)
# 8) rH: relative humidity used for particle resizing (if entered as zero particle is considered dry and no resizing will
#    be performed)
# 9) kappa: particle hygroscopicity parameter (if single parameter it is applied to all modes, otherwise if list it is
#    applied to each mode)
# 10) num_theta: number of scattering angles in output
#
# the output of this code is results, which is a python3 dictionary containing column-arrays for each of the MOPSMAP 
# ouput parameters.
# -> each column corresponds to the wavelegnths provided in wvl
#
# Example:
#       output_dictionary = mopsmap_SD_run.Model(wvl,size_equ,dndlogdp,dpg,m,nonabs_fraction,shape,density,RH,kappa,num_theta)
#
#       print(output_dictionary)
#
#       output_dictionary =
#
#         {'wvl': array([0.355, 0.532, 1.064]), 'ext_coeff': array([2.25173169e-05, 1.25377600e-05, 2.69045229e-06]), 
#         'ssa': array([0.99211268, 0.99163201, 0.98555103]), 'g': array([0.69079052, 0.63824596, 0.47263837]), 
#         ...
#         'delta_l': array([-8.88178420e-16, -4.55191440e-15, -2.44249065e-15]), 
#         'back_angstrom': array([       nan, 1.72583002, 1.20915354])}
#
# WARNINGS:
# 1) numpy, subprocess, and collections must be installed to the python environment
# 2) the MOPSMAP package must be installed first
# 
# References
# [1] Gasteiger, J., & Wiegner, M. (2018). MOPSMAP v1.0: a versatile tool for the modeling of aerosol optical properties. 
# Geosci. Model Dev., 11(7), 2739-2762. doi:10.5194/gmd-11-2739-2018
#########################################################################################################################

import numpy as np
import subprocess
import collections
import time 
from os.path import isfile
from os import remove
# needs to be adjusted to your installation
path_optical_dataset='../optical_dataset/'
path_mopsmap_executable='../mopsmap'

#def pause():
#    programPause = input("Press the <ENTER> key to continue...")

def Model(wvl,size_equ,dndlogdp,dpg,m,nonabs_fraction,shape,density,RH,kappa,num_theta):
  filename = f'tmp_mopsmap_{time.time()}_{np.random.randn()}_{np.random.randn()}_{np.random.randn()}'
  
  # create a input file for the Fortran code and a wavelength file
  mopsmap_input_file = open(f'{filename}.inp', 'w')

  if not isfile('tmp_mopsmap.wvl'):
    mopsmap_wvl_file = open('tmp_mopsmap.wvl', 'w')
    # write wavelength file
    wvl = np.array(wvl,ndmin = 1)
    for i_wvl in range(wvl.shape[0]):
      mopsmap_wvl_file.write('%10.8f \n'%wvl[i_wvl])
    mopsmap_wvl_file.close()

  # write wvls
  mopsmap_input_file.write("wavelength file tmp_mopsmap.wvl \n")

  # write size_equ
  mopsmap_input_file.write('size_equ %s\n'%size_equ)

  # write modes
  dndlogdp = np.array(dndlogdp,ndmin = 1)  
  dpg = np.array(dpg,ndmin = 1)
  if dndlogdp.shape != dpg.shape:
    print("shapes of n and dpg do not agree")
    raise SystemExit()

  dpg_dnlogdp = [dpg,dndlogdp]
  dpg_dnlogdp = np.reshape(dpg_dnlogdp, 2*dpg.shape[0], order='F')
  listToStr = ' '.join(["%0.2E"%elem for elem in dpg_dnlogdp])
  mopsmap_input_file.write('mode %d size distr_list dndlogr %s\n'%(1,listToStr))
  mopsmap_input_file.write('mode %d density %f\n'%(1,density))
  mopsmap_input_file.write('mode %d kappa %f\n'%(1,kappa))
  mopsmap_input_file.write('mode %d refrac %s\n'%(1,m))
  mopsmap_input_file.write('mode %d refrac nonabs_fraction %f\n'%(1,nonabs_fraction))
  mopsmap_input_file.write('mode %d shape %s\n'%(1,shape))
  mopsmap_input_file.write('rH %f\n'%(RH))
  mopsmap_input_file.write('diameter\n')
  mopsmap_input_file.write('scatlib \'%s\'\n'%path_optical_dataset)
  mopsmap_input_file.write('output integrated\n')
  mopsmap_input_file.write('output scattering_matrix\n')
  mopsmap_input_file.write('output volume_scattering_function\n')
  mopsmap_input_file.write('output num_theta %i\n'%num_theta)
  mopsmap_input_file.write('output lidar\n')
  mopsmap_input_file.write('output digits 15\n')
  mopsmap_input_file.write('output ascii_file %s\n'%filename)

  mopsmap_input_file.close()

  # after writing the input file now start mopsmap
  p = subprocess.Popen([path_mopsmap_executable, f'{filename}.inp'], stdout = subprocess.PIPE, stderr = subprocess.STDOUT, close_fds = True)
  stdout1,stderr1 = p.communicate()

  if stdout1 or stderr1:
    if stdout1:
      print(stdout1)
    if stderr1:
      print(stderr1)
    raise SystemExit()

  # read the mopsmap output files into numpy arrays
  output_integrated = np.loadtxt(f'{filename}.integrated',ndmin = 1,dtype = [('wvl', 'f8'),('ext_coeff', 'f8'), ('ssa','f8'),('g','f8'),('r_eff','f8'),('n','f8'),('a','f8'),('v','f8'),('m','f8'),('ext_angstrom','f8'),('sca_angstrom','f8'),('abs_angstrom','f8')])
  output_matrix = np.loadtxt(f'{filename}.scattering_matrix',ndmin = 1,dtype = [('wvl', 'f8'),('angle', 'f8'), ('a1','f8'),('a2','f8'),('a3','f8'),('a4','f8'),('b1','f8'),('b2','f8')])
  output_vol_scat = np.loadtxt(f'{filename}.volume_scattering_function',ndmin = 1,dtype = [('wvl', 'f8'),('angle', 'f8'), ('a1_vol','f8')])
  output_lidar = np.loadtxt(f'{filename}.lidar',ndmin = 1,dtype = [('wvl', 'f8'),('ext_coeff', 'f8'), ('back_coeff','f8'), ('S','f8'), ('delta_l','f8'),('ext_angstrom','f8'),('back_angstrom','f8')])


  # store the results in an easier-to-use way
  num_wvl = output_integrated['wvl'].shape[0]
  num_angles = output_matrix['angle'].shape[0]//num_wvl

  results = {}
  results['wvl'] = output_integrated['wvl']
  results['ext_coeff'] = output_integrated['ext_coeff']
  results['ssa'] = output_integrated['ssa']
  results['g'] = output_integrated['g']
  results['r_eff'] = output_integrated['r_eff']
  results['n'] = output_integrated['n']
  results['a'] = output_integrated['a']
  results['v'] = output_integrated['v']
  results['m'] = output_integrated['m']
  results['ext_angstrom'] = output_integrated['ext_angstrom']
  results['sca_angstrom'] = output_integrated['sca_angstrom']
  results['abs_angstrom'] = output_integrated['abs_angstrom']
  results['angle'] = output_matrix['angle'][0:num_angles]
  results['a1'] = output_matrix['a1'].reshape((num_wvl,num_angles))
  results['a2'] = output_matrix['a2'].reshape((num_wvl,num_angles))
  results['a3'] = output_matrix['a3'].reshape((num_wvl,num_angles))
  results['a4'] = output_matrix['a4'].reshape((num_wvl,num_angles))
  results['b1'] = output_matrix['b1'].reshape((num_wvl,num_angles))
  results['b2'] = output_matrix['b2'].reshape((num_wvl,num_angles))
  results['a1_vol'] = output_vol_scat['a1_vol'].reshape((num_wvl,num_angles))
  results['back_coeff'] = output_lidar['back_coeff']
  results['S'] = output_lidar['S']
  results['delta_l'] = output_lidar['delta_l']
  results['back_angstrom'] = output_lidar['back_angstrom']

  remove(f'{filename}.inp')
  remove(f'{filename}.integrated')
  remove(f'{filename}.scattering_matrix')
  remove(f'{filename}.volume_scattering_function')
  remove(f'{filename}.lidar')
  
  return results
