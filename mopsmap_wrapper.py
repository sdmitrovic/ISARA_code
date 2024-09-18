import numpy as np
import subprocess
import collections
import time 
import os
import sys
from os.path import isfile
from os import remove

def Model(wvl,size_equ,dndlogdp,dpg,RRI,IRI,nonabs_fraction,shape,density,RH,kappa,num_theta,path_optical_dataset,path_mopsmap_executable):

  """
  Interface with the Modeled Optical Properties of ensembles of Aerosol Particles (MOPSMAP) package to return forward modeled aerosol optical properties [1]. [1] Gasteiger, J., & Wiegner, M. (2018). MOPSMAP v1.0: a versatile tool for the modeling of aerosol optical properties. Geosci. Model Dev., 11(7), 2739-2762. doi:10.5194/gmd-11-2739-2018

  :param wvl: Array of wavelengths in nm (e.g., 450, 470, 532, 550, 660, 700)
  :type wvl: numpy array
  :param size_equ: size equivalence (e.g., 'cs')
  :type size_equ: numpy dictionary with keys for each mode as str 
  :param dndlogdp: log-normal concentration of each size distribution bin of each mode
  :type dndlogdp: numpy dictionary with keys for each mode as numpy array
  :param dpg: geometric mean diameter of each bin of each mode
  :type dpg: numpy dictionary with keys for each mode as numpy array
  :param RRI: refractive index, given as a string which is added in input file after 'refrac' (if single string it is applied  to all modes, otherwise if list it is applied to each mode)
  :type RRI: numpy dictionary with keys for each mode as double
  :param IRI: refractive index, given as a string which is added in input file after 'refrac' (if single string it is applied  to all modes, otherwise if list it is applied to each mode)
  :type IRI: numpy dictionary with keys for each mode as double
  :param nonabs_fraction: ratio of non-absorbing particles 
  :type nonabs_fraction: numpy dictionary with keys for each mode as int
  :param shape: particle shape, given as a string which is added in input file after 'shape' (if single string it is applied to all modes, otherwise if list it is applied to each mode)
  :type shape: numpy dictionary with keys for each mode as str
  :param density: particle density (if single parameter it is applied to all modes, otherwise if list it is applied to each mode)
  :type density: numpy dictionary with keys for each mode as double
  :param RH: relative humidity used for particle resizing (if entered as zero particle is considered dry and no resizing will be performed)
  :type RH: numpy dictionary with keys for each mode as double
  :param kappa: particle hygroscopicity parameter (if single parameter it is applied to all modes, otherwise if list it is applied to each mode)
  :type kappa: numpy dictionary with keys for each mode as double
  :param num_theta: number of scattering angles in output
  :type num_theta: int                   
  :param optical_dataset: String indicating the path for the optical dataset required for MOPSMAP.
  :type optical_dataset: str
  :param path_mopsmap_executable: String indicating the path for the mopsmap.exe file.
  :type path_mopsmap_executable: str    
  :return: dictionary of all forward modeled aerosol properties available from MOSPMAP.
  :rtype: numpy dictionary
  """  

  filename = f'tmp_mopsmap_{time.time()}_{np.random.randn()}_{np.random.randn()}_{np.random.randn()}'
  dndlogdp_ary_filename =f'{filename}_dndp'
  # create a input file for the Fortran code and a wavelength file
  mopsmap_input_file = open(f'{filename}.inp', 'w')

  if not isfile('tmp_mopsmap.wvl'):
    mopsmap_wvl_file = open('tmp_mopsmap.wvl', 'w')
    # write wavelength file
    wvl = np.array(wvl,ndmin = 1)
    for i_wvl in np.arange(len(wvl)):
      mopsmap_wvl_file.write('%10.8f \n'%wvl[i_wvl])
    mopsmap_wvl_file.close()

  # write modes
  ikey = 1
  for key in dndlogdp:
    dndlogdp_ary = np.array(dndlogdp[key],ndmin = 1)  
    dpg_ary = np.array(dpg[key],ndmin = 1)
    dndlogdp_ary_file = open(dndlogdp_ary_filename, 'w')
    # write wavelength file
    wvl = np.array(wvl,ndmin = 1)
    for i in np.arange(dndlogdp_ary.shape[0]):
      if i < dndlogdp_ary.shape[0]:
        dndlogdp_ary_file.write('%10.4f %i\n'%(dpg_ary[i],dndlogdp_ary[i]))
      else:
        dndlogdp_ary_file.write('%10.4f %i'%(dpg_ary[i],dndlogdp_ary[i]))
    dndlogdp_ary_file.close()

    if dndlogdp_ary.shape != dpg_ary.shape:
      print("shapes of n and dpg do not agree")
      raise SystemExit()  
    mopsmap_input_file.write("mode %d wavelength file tmp_mopsmap.wvl \n"%ikey) # write wvls
    mopsmap_input_file.write('mode %d size_equ %s\n'%(ikey,size_equ[key])) # write size_equ
    #dpg_dnlogdp = [dpg_ary,dndlogdp_ary]
    #dpg_dnlogdp = np.reshape(dpg_dnlogdp, 2*dpg_ary.shape[0], order='F')
    #listToStr = ' '.join(["%0.4E"%elem for elem in dpg_dnlogdp])
    #mopsmap_input_file.write('mode %d size distr_list dndlogr %s\n'%(ikey,listToStr))
    mopsmap_input_file.write('mode %d size distr_file dndlogr %s\n'%(ikey,dndlogdp_ary_filename))
    mopsmap_input_file.write('mode %d density %f\n'%(ikey,density[key]))
    mopsmap_input_file.write('mode %d kappa %f\n'%(ikey,kappa))
    mopsmap_input_file.write('mode %d refrac %0.4f %0.4f\n'%(ikey,RRI[key],IRI[key]))
    mopsmap_input_file.write('mode %d refrac nonabs_fraction %f\n'%(ikey,nonabs_fraction[key]))
    mopsmap_input_file.write('mode %d shape %s\n'%(ikey,shape[key]))
    ikey += 1

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
  output_lidar = np.loadtxt(f'{filename}.lidar',ndmin = 1,usecols = range(0,7),dtype = [('wvl', 'f8'),('ext_coeff', 'f8'), ('back_coeff','f8'), ('S','f8'), ('delta_l','f8'),('ext_angstrom','f8'),('back_angstrom','f8')])


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
