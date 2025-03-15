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

  filename_sfx = f'tmp_mopsmap_{time.time()}_{np.random.randn()}_{np.random.randn()}_{np.random.randn()}'
  
  # create a input file for the Fortran code and a wavelength file
  mopsmap_input_file = open(f'{filename_sfx}.inp', 'w')

  #if not isfile('tmp_mopsmap.wvl'):
  #if not isfile('tmp_mopsmap.wvl'):
  mopsmap_wvl_file = open(f'{filename_sfx}.wvl', 'w')
  wvl = np.array(wvl,ndmin = 1)
  for i_wvl in np.arange(len(wvl)):
    mopsmap_wvl_file.write('%10.8f \n'%(wvl[i_wvl]/1000))
  mopsmap_wvl_file.close()

  # write modes
  ikey = 1
  modeflag = {}
  for key in dndlogdp:
    modeflag[key] = 1
    dndlogdp_ary_filename =f'{filename_sfx}_{key}'
    dndlogdp_ary = np.array(dndlogdp[key],ndmin = 1)  
    dpg_ary = np.array(dpg[key],ndmin = 1)
    dndlogdp_ary_file = open(dndlogdp_ary_filename, 'w')
    for i in np.arange(dndlogdp_ary.shape[0]):
      if i < dndlogdp_ary.shape[0]:
        dndlogdp_ary_file.write('%0.04E %0.04E\n'%(dpg_ary[i],dndlogdp_ary[i]))
      else:
        dndlogdp_ary_file.write('%0.04E %0.04E'%(dpg_ary[i],dndlogdp_ary[i]))
    dndlogdp_ary_file.close()

    if dndlogdp_ary.shape != dpg_ary.shape:
      print("shapes of n and dpg do not agree")
      raise SystemExit()  
    mopsmap_input_file.write("mode %d wavelength file %s.wvl \n"%(ikey,filename_sfx)) # write wvls
    mopsmap_input_file.write('mode %d size_equ %s\n'%(ikey,size_equ[key])) # write size_equ
    dpg_dnlogdp = [dpg_ary,dndlogdp_ary]
    #dpg_dnlogdp = np.reshape(dpg_dnlogdp, 2*dpg_ary.shape[0], order='F')
    #listToStr = ' '.join(["%0.04E %i"%elem for elem in dpg_dnlogdp])
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
  mopsmap_input_file.write('output ascii_file %s\n'%filename_sfx)

  mopsmap_input_file.close()

  # after writing the input file now start mopsmap
  p = subprocess.Popen([path_mopsmap_executable, f'{filename_sfx}.inp'], stdout = subprocess.PIPE, stderr = subprocess.STDOUT, close_fds = True)
  stdout1,stderr1 = p.communicate()

  if stdout1 or stderr1:
    if stdout1:
      print(stdout1)
    if stderr1:
      print(stderr1)
    raise SystemExit()

  # read the mopsmap output files into numpy arrays
  output_integrated = np.loadtxt(f'{filename_sfx}.integrated',ndmin = 1,dtype = [('wvl', 'f8'),('ext_coeff', 'f8'), ('ssa','f8'),('g','f8'),('r_eff','f8'),('n','f8'),('a','f8'),('v','f8'),('m','f8'),('ext_angstrom','f8'),('sca_angstrom','f8'),('abs_angstrom','f8')])
  output_matrix = np.loadtxt(f'{filename_sfx}.scattering_matrix',ndmin = 1,dtype = [('wvl', 'f8'),('angle', 'f8'), ('a1','f8'),('a2','f8'),('a3','f8'),('a4','f8'),('b1','f8'),('b2','f8')])
  output_vol_scat = np.loadtxt(f'{filename_sfx}.volume_scattering_function',ndmin = 1,dtype = [('wvl', 'f8'),('angle', 'f8'), ('a1_vol','f8')])
  output_lidar = np.loadtxt(f'{filename_sfx}.lidar',ndmin = 1,usecols = range(0,7),dtype = [('wvl', 'f8'),('ext_coeff', 'f8'), ('back_coeff','f8'), ('S','f8'), ('delta_l','f8'),('ext_angstrom','f8'),('back_angstrom','f8')])


  # store the results in an easier-to-use way
  num_wvl = output_integrated['wvl'].shape[0]
  num_angles = output_matrix['angle'].shape[0]//num_wvl

  results = {}
  results['r_eff'] = output_integrated['r_eff'][0]  
  results['n'] = output_integrated['n'][0]
  results['a'] = output_integrated['a'][0]
  results['v'] = output_integrated['v'][0]
  a1 = output_matrix['a1'].reshape((num_wvl,num_angles))
  a2 = output_matrix['a2'].reshape((num_wvl,num_angles))
  a3 = output_matrix['a3'].reshape((num_wvl,num_angles))
  a4 = output_matrix['a4'].reshape((num_wvl,num_angles))
  b1 = output_matrix['b1'].reshape((num_wvl,num_angles))
  b2 = output_matrix['b2'].reshape((num_wvl,num_angles))
  a1_vol = output_vol_scat['a1_vol'].reshape((num_wvl,num_angles))  

  if num_wvl > 0:
    for i1 in range(num_wvl): 
      results[f'ext_coeff_{wvl[i1]}_m-1'] = output_integrated['ext_coeff'][i1]
      results[f'm_{wvl[i1]}'] = output_integrated['m'][i1]  
      results[f'ssa_{wvl[i1]}'] = output_integrated['ssa'][i1]
      results[f'g_{wvl[i1]}'] = output_integrated['g'][i1]
      results[f'ext_angstrom_{wvl[i1]}'] = output_integrated['ext_angstrom'][i1]
      results[f'sca_angstrom_{wvl[i1]}'] = output_integrated['sca_angstrom'][i1]
      results[f'abs_angstrom_{wvl[i1]}'] = output_integrated['abs_angstrom'][i1]
      results[f'back_coeff_{wvl[i1]}_m-1'] = output_lidar['back_coeff'][i1]
      results[f'lidar_ratio_{wvl[i1]}'] = output_lidar['S'][i1]
      results[f'delta_l_{wvl[i1]}'] = output_lidar['delta_l'][i1]
      results[f'back_angstrom_{wvl[i1]}'] = output_lidar['back_angstrom'][i1]
      for iangle in range(num_angles):    
        ang = output_matrix['angle'][iangle]
        results[f'a1_{wvl[i1]}_{ang}'] = a1[i1,iangle]
        results[f'a2_{wvl[i1]}_{ang}'] = a2[i1,iangle]
        results[f'a3_{wvl[i1]}_{ang}'] = a3[i1,iangle]
        results[f'a4_{wvl[i1]}_{ang}'] = a4[i1,iangle]
        results[f'b1_{wvl[i1]}_{ang}'] = b1[i1,iangle]
        results[f'b2_{wvl[i1]}_{ang}'] = b2[i1,iangle]
        results[f'a1_vol_{wvl[i1]}_{ang}'] = a1_vol[i1,iangle]
 
  else:
    results[f'ext_coeff_{wvl[0]}_m-1'] = output_integrated['ext_coeff'][0]
    results[f'm_{wvl[0]}'] = output_integrated['m'][0]  
    results[f'ssa_{wvl[0]}'] = output_integrated['ssa'][0]
    results[f'g_{wvl[0]}'] = output_integrated['g'][0]
    results[f'ext_angstrom_{wvl[0]}'] = output_integrated['ext_angstrom'][0]
    results[f'sca_angstrom_{wvl[0]}'] = output_integrated['sca_angstrom'][0]
    results[f'abs_angstrom_{wvl[0]}'] = output_integrated['abs_angstrom'][0]
    results[f'back_coeff_{wvl[0]}_m-1'] = output_lidar['back_coeff'][0]
    results[f'lidar_ratio_{wvl[0]}'] = output_lidar['S'][0]
    results[f'delta_l_{wvl[0]}'] = output_lidar['delta_l'][0]
    results[f'back_angstrom_{wvl[0]}'] = output_lidar['back_angstrom'][0]    
    results['angle'] = output_matrix['angle'][0:num_angles] 
    for iangle in range(num_angles):    
      ang = output_matrix['angle'][iangle]
      results[f'a1_{wvl[0]}_{ang}'] = a1[0,iangle]
      results[f'a2_{wvl[0]}_{ang}'] = a2[0,iangle]
      results[f'a3_{wvl[0]}_{ang}'] = a3[0,iangle]
      results[f'a4_{wvl[0]}_{ang}'] = a4[0,iangle]
      results[f'b1_{wvl[0]}_{ang}'] = b1[0,iangle]
      results[f'b2_{wvl[0]}_{ang}'] = b2[0,iangle]
      results[f'a1_vol_{wvl[0]}_{ang}'] = a1_vol[0,iangle]
  
  for key in dndlogdp:
    if modeflag[key]==1:
      remove(f'{filename_sfx}_{key}')
  remove(f'{filename_sfx}.wvl')    
  remove(f'{filename_sfx}.inp')
  remove(f'{filename_sfx}.integrated')
  remove(f'{filename_sfx}.scattering_matrix')
  remove(f'{filename_sfx}.volume_scattering_function')
  remove(f'{filename_sfx}.lidar')
  
  return results
