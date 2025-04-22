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
import numpy as np
import pandas as pd
import powerfunction as pwrfn
from scipy.optimize import curve_fit
f_model = pwrfn.f_model

def Align(D_optic,N_optic,D_aero,N_aero):
    rho = None
    peak = None
    ro =  np.arange(1.3, 3.0, 0.1, dtype=float)
    d_optic = np.squeeze(D_optic[77:])
    N_optic = np.squeeze(N_optic[77:])
    N_aero = np.squeeze(N_aero[2:])
    D_aero = np.squeeze(D_aero[2:])
    ihu = np.where(np.logical_not(np.isnan(N_optic)))
    ihu2 = np.where(np.logical_not(np.isnan(N_aero)))
    if (len(N_optic[ihu])>1)&(len(N_aero[ihu2])>1):
        Noptic = N_optic[ihu]
        Doptic = d_optic[ihu]
        Naero = N_aero[ihu2]
        Daero = D_aero[ihu2]        
        JJ = np.where(Noptic==np.nanmax(Noptic))[0]
        if len(JJ)>1:
            JJ = JJ[0]
        JJ = np.squeeze(JJ)    
        NOPT = Noptic[JJ:]
        DOPT = Doptic[JJ:]
        
        if len(NOPT)>3:
            popt, pcov = curve_fit(
                f=f_model,       # model function
                xdata=DOPT,   # x data
                ydata=NOPT,   # y data
                p0=(0.24,-3.08),      # initial value of the parameters
                maxfev = 1000000,
                sigma=np.multiply(NOPT,0.2)   # uncertainties on y
            )
            a_opt, c_opt = popt
            if a_opt == 0:
                popt, pcov = curve_fit(
                f=f_model,       # model function
                xdata=DOPT,   # x data
                ydata=NOPT,   # y data
                p0=(0.24,-3.08),      # initial value of the parameters
                sigma=np.multiply(NOPT,0.2)   # uncertainties on y
                )
                a_opt, c_opt = popt  
            if c_opt != 0:
                s2 = np.empty(len(ro))
                jj = np.where(Naero==np.nanmax(Naero))[0]
                #print(jj)
                if len(jj) == 1:
                    jj = np.arange(np.squeeze(jj),len(Naero)) 
                else:
                    jj = np.arange(np.squeeze(jj[0]),len(Naero))
                naero = Naero[jj]
               
                for i2 in range(len(ro)):
                    daero = np.divide(Daero[jj],np.sqrt(ro[i2]))
                    jij = np.where((daero>=DOPT[1])&(daero<=DOPT[-1]))[0]
                    d_aero = daero[jij]
                    n_aero = naero[jij]
                    dlogdn_optic = a_opt*d_aero**(c_opt)
                    if len(d_aero)>1:
                        s2[i2] = ((len(d_aero))**(-1))*np.sum((np.log(dlogdn_optic)-np.log(n_aero))**2) 
                idx = np.where((s2>0)*(np.isfinite(s2)))   
                if len(idx)>1:
                    s3 = s2[idx]
                    r0 = ro[idx]
                    jj2 = np.where(s3==np.min(s3))
                    if len(jj2)>0:
                        rho = r0[jj2]
                        peak = np.divide(Daero[jj[0]],np.sqrt(rho))

    #avg =[np.mean(rho[np.logical_not(np.isnan(n_optic))]),np.std(rho[rho>0])]
    #rho[rho==0] = avg[0]
    output = dict();
    if rho is not None:
        output['rho'] = rho
        output['peak'] = peak
    else:
        output['rho'] = 1.00
        output['peak'] = np.nan        
    return output 
    ##
    return output