#########################################################################################################################
# mopsmap_UI.py                                       by:  Joseph Schlosser
#                                                revised:  19 Feb 2022 
#                                    language (revision):  python3 (3.8.2-0ubuntu2)
#
# Description: user interface used to select LAS size distribution (SD) data from a ICT file and format the data into a
# python3 dictionary
#
# Implementation: this interface can be run from using the following ubuntu syntax: user:$ python3 mopsmap_UI.py
#
# the output of this code is output_dictionary, which is a python3 dictionary containing column-arrays for each of the 
# parameters in the .ict file and row-column matricies for each of the MOPSMAP ouputs
# -> each column corresponds to a line in the provided .ict file
# -> each row of the MOPSMAP ouputs corresponds to each of the desired output wavelengths
#
# WARNINGS:
# 1) numpy, csv, and tqdm must be installed to the python environment
# 2) importICT.py, mopsmap_SD_run.py, LAS_bin_sizes.csv, and and file with the corresponding filename must be present in 
#    a directory that is in your PATH
#########################################################################################################################

import mopsmap_SD_run_par
import S_e
import numpy as np
import csv
from tqdm import tqdm
import os

def pause():
    programPause = input("Press the <ENTER> key to continue...")


def Retr_CRI(wvl,
  size_equ,
  sd1,dpg1,
  CRI,
  nonabs_fraction,
  shape,
  rho,
  RH,
  kappa,
  num_theta,
  measured_coef,
  Acur,
  Uncr,
  cFct,
  chi2_criteria,
  Hz
):
  L1 = len(CRI[:,0,0])
  L2 = len(CRI[0,:,0]) 
  L3 = len(wvl)
  chi2 = np.zeros((L1,L2))
  cal_coef = np.zeros((L1,L2,L3))
  sig1 = Uncr/np.sqrt(Hz)
  for i1 in range(L1):
    for i2 in range(L2):
      results = mopsmap_SD_run_par.Model(wvl,size_equ,sd1,dpg1,"%0.4f %0.4f"%(CRI[i1,i2,0],CRI[i1,i2,1]),nonabs_fraction,shape,rho,RH,kappa,num_theta)
      scat_coef = results['ssa'][0:3]*results['ext_coeff'][0:3]
      abs_coef = results['ext_coeff'][3:6]-results['ssa'][3:6]*results['ext_coeff'][3:6]
      cal_coef[i1,i2,:] = np.hstack((scat_coef,abs_coef))
      Se = S_e.Cal(measured_coef,Acur,sig1,cFct)
      Cdif = np.matrix(np.add(np.squeeze(measured_coef),-np.squeeze(cal_coef[i1,i2,:])))
      a = np.sqrt(0.5*((Cdif)*(np.linalg.inv(Se))*Cdif.T))/L3
      chi2[i1,i2] = a
#  idx2 = chi2<=chi2_criteria 
#  RRI = CRI[:,:,0]
#  IRI = CRI[:,:,1]  
#  CRIdry = [np.sum(RRI[idx2]*chi2[idx2])/np.sum(chi2[idx2]),np.sum(IRI[idx2]*chi2[idx2])/np.sum(chi2[idx2])]
  RRI = CRI[:,:,0]
  IRI = CRI[:,:,1]
  idx2 = np.array(np.where(chi2==np.min(chi2)))
  CRIdry = [RRI[idx2[0][0],idx2[1][0]],IRI[idx2[0][0],idx2[1][0]]]
  Cal_coef = cal_coef[idx2[0][0],idx2[1][0],:]
  return CRIdry, Cal_coef

def Retr_IRI(wvl,
  size_equ,
  sd1,dpg1,
  CRI,
  nonabs_fraction,
  shape,
  rho,
  RH,
  kappa,
  num_theta,
  measured_coef,
):
  L1 = len(CRI[:,0])
  L2 = len(wvl)

  IRI = np.zeros(L1)
  RRI = np.zeros(L1)
  Results = dict()  
  Results["RRIdry"] = None
  Results["IRIdry"] = None
  Results["scat_coef"] = None
  Results["abs_coef"] = None
  Results["ext_coef"]= None
  Results["SSA"] = None

  for i1 in range(L1):
    results = mopsmap_SD_run_par.Model(wvl,size_equ,sd1,dpg1,"%0.4f %0.4f"%(CRI[i1,0],CRI[i1,1]),nonabs_fraction,shape,rho,RH,kappa,num_theta)
    scat_coef = results['ssa'][[0,3,5]]*results['ext_coeff'][[0,3,5]]
    abs_coef = results['ext_coeff'][[1,2,4]]-results['ssa'][[1,2,4]]*results['ext_coeff'][[1,2,4]]   
    Cdif1 = abs(measured_coef[0:3]-scat_coef)/measured_coef[0:3]
    Cdif2 = abs(measured_coef[3:]-abs_coef)
    a1 = ((Cdif1)<0.2).astype('int')
    a2 = ((Cdif2)<1*pow(10,-6)).astype('int')
    if np.sum(a1)==3 & np.sum(a2)==3:
      RRI[i1] = CRI[i1,0]
      IRI[i1] = CRI[i1,1]

  if np.sum(RRI[RRI>0])>0:      
    Results["RRIdry"] = np.mean(RRI[RRI>0])
    Results["IRIdry"] = np.mean(IRI[RRI>0])

    results = mopsmap_SD_run_par.Model(wvl,size_equ,sd1,dpg1,"%0.4f %0.4f"%(Results["RRIdry"],Results["IRIdry"]),nonabs_fraction,shape,rho,RH,kappa,num_theta) 
    Results["scat_coef"] = results['ssa'][[0,3,5]]*results['ext_coeff'][[0,3,5]]
    Results["abs_coef"] = results['ext_coeff'][[1,2,4]]-results['ssa'][[1,2,4]]*results['ext_coeff'][[1,2,4]]
    Results["SSA"] = results['ssa']
    Results["ext_coef"] = results['ext_coeff']

  return Results

def Retr_kappa(wvl,
  size_equ,
  sd1,dpg1,
  CRI,
  nonabs_fraction,
  shape,
  rho,
  RH,
  kappa,
  num_theta,
  measured_coef,
):
  L1 = len(kappa)
  L3 = len(wvl) 
  Results = dict()
  Results["Kappa"] = None
  Results["Cal_coef"] = None
  Results["Cal_SSA"] = None
  Results["Cal_ext_coef"] = None
  stop_indx = 0
  for i1 in range(L1):
    if stop_indx == 0:
      results = mopsmap_SD_run_par.Model(wvl,size_equ,sd1,dpg1,"%0.4f %0.4f"%(CRI[0],CRI[1]),nonabs_fraction,shape,rho,RH,kappa[i1],num_theta)
      scat_coef = results['ssa']*results['ext_coeff']
      abs_coef = results['ext_coeff']-results['ssa']*results['ext_coeff']
      Cdif = (measured_coef-scat_coef[3])/measured_coef
      a = ((Cdif)<0.01).astype('int')
      if np.sum(a)==1:
        Results["Kappa"] = kappa[i1]
        Results["Cal_coef"] = scat_coef
        Results["Cal_SSA"] = results['ssa']
        Results["Cal_ext_coef"] = results['ext_coeff']
        stop_indx = 1

  return Results
