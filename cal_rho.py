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
    ro =  np.arange(1.3, 3.0, 0.1, dtype=float)

    d_optic = D_optic[19:]
    N_optic = N_optic[19:,:]

    N_aero = N_aero[2:,:]
    D_aero = D_aero[2:]

    rho = np.empty(len(N_optic[0,:]))
    peak = np.empty(len(N_optic[0,:]))
    for i1 in range(len(N_optic[0,:])):
        n_optic = N_optic[:,i1]
        ihu = np.where(n_optic>0)
        if len(n_optic[ihu])>1:
            Noptic = n_optic[ihu]
            Doptic = d_optic[ihu]
            JJ = np.where(Noptic==np.max(Noptic))
            if len(JJ)>1:
                JJ = JJ[0]
            NOPT = Noptic[JJ]
            DOPT = Doptic[JJ]
            Naero = N_aero[:,i1]
            #if sum(np.where(n_optic>0)).all()>3:
            #print(n_optic)
            if len(n_optic)>1:
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
                    s2 = np.zeros(len(ro))
                    jj = np.array(np.where(Naero==np.max(Naero)))
                    if len(jj[0]) == 1:
                        jj = np.arange(jj[0],len(Naero)) 
                    else:
                        jj = np.arange(jj[0,0],len(Naero))
                    naero = Naero[jj]
                   
                    for i2 in range(len(ro)):
                        daero = np.divide(D_aero[jj],np.sqrt(ro[i2]))
                        jij = np.where((daero>=DOPT[1])&(daero<=DOPT[-1]))
                        d_aero = daero[jij]
                        n_aero = naero[jij]
                        dlogdn_optic = a_opt*d_aero**(c_opt)
                        if len(d_aero)>1:
                            s2[i2] = ((len(d_aero))**(-1))*np.sum((np.log(dlogdn_optic)-np.log(n_aero))**2) 
 
                    idx = np.where((s2>0)*(np.isfinite(s2)))   
                    if len(idx[0])>1:
                        s3 = s2[idx[0]]
                        r0 = ro[idx[0]]
                        jj2 = np.where(s3==np.min(s3))
                        if len(jj2[0])>0:
                            rho[i1] = r0[jj2]
                            peak[i1] = np.divide(D_aero[jj[0]],np.sqrt(rho[i1]))

    avg =[np.mean(rho[rho>0]),np.std(rho[rho>0])]
    rho[rho==0] = avg[0]
    output = dict();
    output['rho'] = rho
    output['peak'] = peak
    return output 
    ##
    return output