import mopsmap_SD_run_par
import ISARA2_par
import importICARTT
import os
import numpy as np
from pathos.multiprocessing import ProcessPool


# Number of cores you want to use
number_of_cores = 6
# This should be at the start of the code to minimize the fork size
pool = ProcessPool(ncpus=number_of_cores)

def pause():
    programPause = input("Press the <ENTER> key to continue...")

def grab_ICT_Data(filename):
    data = importICARTT.imp(filename)

    def grab_keydata(key_starts_with):
        for key in data.keys():
            if key.startswith(key_starts_with):
                return data[key]

    time = np.array(grab_keydata('Time_Start'))
    frmttime = np.array(grab_keydata('fmtdatetime'))
    date = grab_keydata('date')
    alt = np.array(grab_keydata('GPS_Altitude'))
    lat = np.array(grab_keydata('Latitude'))
    lon = np.array(grab_keydata('Longitude'))
    sd1 = np.array([v for k, v in data.items() if k.startswith('LAS_')])
    RH_amb = np.array(grab_keydata('RHw'))
    RH_sp = np.array(grab_keydata('RH_Sc'))
    Sc = np.array([v for k, v in data.items() if k.startswith('Sc')])
    Abs = np.array([v for k, v in data.items() if k.startswith('Abs')])
    Ext = np.array([v for k, v in data.items() if k.startswith('Ext')])
    SSA = np.array([v for k, v in data.items() if k.startswith('SSA')])
    fRH = np.array(grab_keydata('fRH'))
    return (data, time, frmttime, date, alt, lat, lon, sd1, RH_amb, RH_sp, Sc, Abs, Ext, SSA, fRH)


OP_Dictionary = {}
# set desired output wavelengths in micrometer
wvl = [0.450, 0.470, 0.532, 0.550, 0.660, 0.700]
#wvl_amb = [0.450, 0.470, 0.532, 0.550, 0.660, 0.700]
#wvl = [0.355, 0.450, 0.470, 0.532, 0.550, 0.660, 0.700, 0.750, 1.064]

size_equ = 'cs'

# refractive index
# RRI = np.arange(1.45,1.55,0.01).reshape(-1)
RRI = 1.55
# IRI = np.arange(0.0,0.08,0.01).reshape(-1)
IRI = np.hstack((0, np.logspace(np.log10(pow(10, -4)), np.log10(0.08),
                                num=20).reshape(-1)))
# CRI = np.zeros((len(RRI),len(IRI),2))
# for i1 in range(len(RRI)):
#   for i2 in range(len(IRI)):
#     CRI[i1,i2,:] = [RRI[i1],IRI[i2]]
CRI = np.zeros((len(IRI), 2))
for i1 in range(len(IRI)):
    CRI[i1, :] = [RRI, IRI[i1]]
# set the non-absorbing fraction of the aerosol SD
nonabs_fraction = 0

# set shape or shape distribution of particles
# shape='spheroid oblate 1.7'
# shape='spheroid distr_file '../data/ar_kandler''
shape = 'sphere'

kappa = np.arange(0.0, 1.40, 0.01).reshape(-1)
num_theta = 2
# Acur1 = [0.1,0.1,0.1,0.05,0.05,0.05]
# Uncr1 = np.multiply([4.5,3.0,3.0,1.0,1.0,1.0],pow(10,-6))
Acur1 = np.array([0.05, 0.05, 0.05])
Uncr1 = np.multiply(np.array([1.0, 1.0, 1.0]), pow(10, -6))
cFct1 = 0.05
chi2_criteria1 = 0.001

rho_dry = 2.63

Acur2 = 0.2
Uncr2 = 4.5 * pow(10, -6)
cFct2 = 0.05
chi2_criteria2 = 0.01

rho_amb = 1.63

resolution = 60
#ioo = 1

i0 = 1  # index used to skip header row
G = open('LAS_bin_sizes.csv', 'r')  # open .csv
g = G.read().splitlines()  # read .csv
hdrs = g[0].split(',')  # define headers
# create zeros array to be filled iteratively
Dp = np.zeros((len(g) - 1, len(hdrs)))
for i1 in range(len(g) - 1):
    # split string into array and define as number array
    Dp[i1, :] = np.array(list(eval(g[i0])))
    i0 += 1
# geometric mean diameter in micrometer
dpg = np.array(Dp[:, 2]) / 1000
# transpose dpg to make column-array of geometric mean diameters from row-array
# of geometric mean diameters
dpg1 = dpg.T
G.close()  # close .csv


def handle_line(sd1, measured_coef_dry, measured_ext_coef_dry, measured_ssa_dry,
                    measured_coef_amb, measured_ext_coef_amb, measured_ssa_amb, measured_fRH,
                    wvl, size_equ, dpg1, CRI, nonabs_fraction, shape, rho_dry,
                    RH_sp, kappa, num_theta, RH_amb, rho_amb):#Acur1, Uncr1, cFct1, chi2_criteria1, Acur2, Uncr2,cFct2, chi2_criteria2, resolution, 
                
    # So this code may look a bit funky, but we are doing what is called
    # currying. This is simply the idea of returning a function inside of
    # a function. It may look weird doing this, but this is actually required
    # so that each worker has the necessary data. What ends up happening is
    # each worker is passed a full copy of all the data contained within this
    # function, so it has to know what data needs to be copied. Anyhow, the
    # inner `curry` function is what is actually being called for each
    # iteration of the for loop.
    def curry(i1):
        meas_coef = np.multiply(measured_coef_dry[:, i1], pow(10, -6))
        dndlogdp = np.multiply(sd1[:, i1], pow(10, 6))
        dndlogdp[np.where(dndlogdp.astype(str) == 'nan')] = 0
        # This is where things become a pain :( Since we are spreading the
        # work across multiple cores, there is a copy of the data in each core.
        # Therefore, we are not able to easily make updates to the numpy
        # arrays, so instead we will obtain the results for each line then
        # join them together after the multiprocessing occurs.
        IRI_dry = None
        RRI_dry = None
        CalScatCoef_dry = None
        CalAbsCoef_dry = None
        CalExtCoef_dry = None
        CalSSA_dry = None
        meas_coef_dry = None
        meas_ext_coef_dry = None
        meas_ssa_dry = None
        Kappa = None
        CalCoef_amb = None
        CalExtCoef_amb = None
        CalSSA_amb = None
        CalfRH = None
        meas_coef_amb = None
        meas_ext_coef_amb = None
        meas_ssa_amb = None
        meas_fRH = None
        
        #results = None
        # You will notice that in the code, instead of doing things like
        # CRI_dry[:, i1] = ..., we are instead just assining the value for this
        # row instead and then they will be merged later
        if np.sum(dndlogdp) > 0 and int(np.sum(meas_coef[0:3] > 0)) == 3:
            Results = ISARA2_par.Retr_IRI(
                wvl, size_equ, dndlogdp, dpg1, CRI, nonabs_fraction, shape,
                rho_dry, RH_sp[i1], kappa[0], num_theta, meas_coef)#, Acur1, Uncr1, cFct1, chi2_criteria1, resolution)
            if Results["RRIdry"] is not None:
                RRI_dry = Results["RRIdry"]
                IRI_dry = Results["IRIdry"]
                CRI_dry = np.array([RRI_dry,IRI_dry])
                CalScatCoef_dry = Results["scat_coef"]
                CalAbsCoef_dry = Results["abs_coef"]
                CalExtCoef_dry = Results["ext_coef"]
                CalSSA_dry = Results["SSA"]
                meas_coef_dry = measured_coef_dry[:, i1]
                meas_ext_coef_dry = measured_ext_coef_dry[i1]
                meas_ssa_dry = measured_ssa_dry[:, i1]
                if RH_amb[i1] > 0 and measured_coef_amb[i1] != 'nan':
                    meas_coef = np.multiply(measured_coef_amb[i1], pow(10, -6))
                    Results = ISARA2_par.Retr_kappa(
                        wvl, size_equ, dndlogdp, dpg1, CRI_dry,
                        nonabs_fraction, shape, rho_amb, RH_amb[i1], kappa,
                        num_theta, meas_coef)#, Acur2, Uncr2, cFct2,chi2_criteria2, resolution)
                    if Results["Kappa"] is not None:
                        Kappa = Results["Kappa"]
                        CalCoef_amb = Results["Cal_coef"]
                        CalExtCoef_amb = Results["Cal_ext_coef"]
                        CalSSA_amb = Results["Cal_SSA"]
                        meas_coef_amb = measured_coef_amb[i1]
                        meas_ext_coef_amb = measured_ext_coef_amb[i1]
                        meas_ssa_amb = measured_ssa_amb[i1]
                        meas_fRH = measured_fRH[i1]
                        CalfRH = np.zeros(len(CalCoef_amb))
                        for i3 in range(len(CalCoef_amb)):
                            if CalExtCoef_dry[i3]>0:
                                CalfRH[i3] = CalCoef_amb[i3]/(CalExtCoef_dry[i3]*CalSSA_dry[i3])
        return (RRI_dry, IRI_dry, CalScatCoef_dry, CalAbsCoef_dry, CalExtCoef_dry, CalSSA_dry, meas_coef_dry, 
                meas_ext_coef_dry, meas_ssa_dry, Kappa, CalCoef_amb, CalExtCoef_amb, CalSSA_amb, CalfRH,
                meas_coef_amb, meas_ext_coef_amb, meas_ssa_amb, meas_fRH)#, results)

    return curry

IFN = [f for f in os.listdir(r'../ACTIVATE/') if f.endswith('.ict')]
for input_filename in IFN[0:40]:
    print(input_filename)
    # import the .ict data into a dictonary
    (output_dict, time, frmttime, date, alt, lat, lon, sd1, RH_amb, RH_sp, Sc,
     Abs, Ext, SSA, fRH) = grab_ICT_Data(f'../ACTIVATE/{input_filename}')
    Abs[Abs < 0] = 0
    Sc[Sc <= 0] = 0
    Ext[Ext <= 0] = 0
    RH_amb[RH_amb > 99] = 99
    RH_amb[RH_amb < 0] = 0
    RH_sp[RH_sp < 0] = 0 

    measured_coef_dry = np.vstack((Sc[1:, :], Abs))
    measured_ext_coef_dry = Ext[1, :]
    measured_ssa_dry = SSA[0:3, :]
    measured_coef_amb = Sc[0, :]
    measured_ext_coef_amb = Ext[0, :]
    measured_ssa_amb = SSA[-1, :]
    #measured_ssa_amb[Abs[1,:] == 0] = 1
    measured_fRH = fRH

    Lwvl = len(wvl)
    Lwvl_s = int(Lwvl/2)
    L1 = len(sd1[0, :])
    RRI_dry = np.zeros((1, L1))
    IRI_dry = np.zeros((1, L1))
    CalScatCoef_dry = np.zeros((Lwvl_s, L1))
    CalAbsCoef_dry = np.zeros((Lwvl_s, L1))
    CalExtCoef_dry = np.zeros((Lwvl, L1))
    CalSSA_dry = np.zeros((Lwvl, L1))
    meas_coef_dry = np.zeros((Lwvl, L1))
    meas_coef_amb = np.zeros((1, L1))
    meas_ext_coef_dry = np.zeros((1, L1))
    meas_ssa_dry = np.zeros((3, L1))
    Kappa = np.zeros((1,L1))
    CalCoef_amb = np.zeros((Lwvl, L1))
    CalfRH = np.zeros((Lwvl, L1))
    CalExtCoef_amb = np.zeros((Lwvl, L1))
    CalSSA_amb = np.zeros((Lwvl, L1))
    meas_ext_coef_amb = np.zeros((1, L1))
    meas_ssa_amb = np.zeros((1, L1)) 
    meas_fRH = np.zeros((1, L1))  
    # Loop through each of the rows here using multiprocessing. This will
    # split the rows across multiple different cores. Each row will be its
    # own index in `line_data` with a tuple full of information. So, for
    # instance, line_data[0] will contain (CRI_dry, CalCoef_dry, meas_coef_dry,
    # Kappa, CalCoef_amb, meas_coef_amb, results) for the first line of data
    line_data = pool.map(
        # This is a pain, I know, but all the data has to be cloned and
        # accessible within each worker
        handle_line(sd1, measured_coef_dry, measured_ext_coef_dry, measured_ssa_dry,
                    measured_coef_amb, measured_ext_coef_amb, measured_ssa_amb, measured_fRH,
                    wvl, size_equ, dpg1, CRI, nonabs_fraction, shape, rho_dry,
                    RH_sp, kappa, num_theta, RH_amb, rho_amb),#,wvl,Acur1, Uncr1, cFct1,  chi2_criteria1, Acur2, Uncr2, cFct2, chi2_criteria2,resolution, 
        range(L1),
    )
    # Now that the data has been fetched, we have to join together all the
    # results into aggregated arrays. The `enumerate` function simply loops
    # through the elements in the array and attaches the associated array
    # index to it.
    for i1, line_data in enumerate(line_data):
        (RRI_dry_line, IRI_dry_line, CalScatCoef_dry_line, CalAbsCoef_dry_line, CalExtCoef_dry_line, 
            CalSSA_dry_line, meas_coef_dry_line, meas_ext_coef_dry_line, meas_ssa_dry_line, Kappa_line, 
            CalCoef_amb_line, CalExtCoef_amb_line, CalSSA_amb_line, CalfRH_line, meas_coef_amb_line,
            meas_ext_coef_amb_line, meas_ssa_amb_line, meas_fRH_line) = line_data#, results_line) = line_data

        # The general trend for merging the values is pretty simple. If the
        # value is not None, that means that it has a value set because it
        # was reached conditionally. Therefore, if it does have a value, we
        # will just update that part of the array. Now, I know you're probably
        # thinking "why are we doing all this work again." Well, true, it is
        # repeated work, but this will allow for much faster times overall
        # (well, that's the hope anyhow).
        def merge_in(line_val, merged_vals):
            merged_vals[:, i1] = line_val
#            if line_val is not None:
#                merged_vals[:, i1] = line_val


        merge_in(RRI_dry_line, RRI_dry)
        merge_in(IRI_dry_line, IRI_dry)
        merge_in(CalScatCoef_dry_line, CalScatCoef_dry)
        merge_in(CalAbsCoef_dry_line, CalAbsCoef_dry)        
        merge_in(CalExtCoef_dry_line, CalExtCoef_dry)
        merge_in(CalSSA_dry_line, CalSSA_dry)
        merge_in(meas_coef_dry_line, meas_coef_dry) 
        merge_in(meas_ext_coef_dry_line, meas_ext_coef_dry)
        merge_in(meas_ssa_dry_line, meas_ssa_dry)
        merge_in(Kappa_line, Kappa)
        merge_in(CalCoef_amb_line, CalCoef_amb)
        merge_in(CalExtCoef_amb_line, CalExtCoef_amb)
        merge_in(CalSSA_amb_line, CalSSA_amb)
        merge_in(CalfRH_line, CalfRH)
        merge_in(meas_coef_amb_line, meas_coef_amb)
        merge_in(meas_ext_coef_amb_line, meas_ext_coef_amb)
        merge_in(meas_ssa_amb_line, meas_ssa_amb)
        merge_in(meas_fRH_line, meas_fRH)
        
    # From here on out, everything can continue as normal :)

    output_dict['RRI_dry'] = RRI_dry
    output_dict['IRI_dry'] = IRI_dry
    output_dict['Kappa'] = Kappa

    i0 = 0
    for i1 in [0,3,5]:
        output_dict[f'Cal_Sca_Coef_dry_{wvl[i1]}'] = CalScatCoef_dry[i0, :]
        output_dict[f'Meas_Sca_Coef_dry_{wvl[i1]}'] = meas_coef_dry[i0, :]
        output_dict[f'Meas_SSA_dry_{wvl[i1]}'] = meas_ssa_dry[i0, :]
        i0 += 1
    i0 = 0    
    for i1 in [1,2,4]:
        output_dict[f'Cal_Abs_Coef_dry_{wvl[i1]}'] = CalAbsCoef_dry[i0, :]
        output_dict[f'Meas_Abs_Coef_dry_{wvl[i1]}'] = meas_coef_dry[i0+3, :]
        i0 += 1
    i0 = 0
    for i1 in range(0, Lwvl, 1):
        output_dict[f'Cal_Sca_Coef_amb_{wvl[i1]}'] = CalCoef_amb[i0, :]
        output_dict[f'Cal_Ext_Coef_dry_{wvl[i1]}'] = CalExtCoef_dry[i0, :]
        output_dict[f'Cal_Ext_Coef_amb_{wvl[i1]}'] = CalExtCoef_amb[i0, :]
        output_dict[f'Cal_SSA_dry_{wvl[i1]}'] = CalSSA_dry[i0, :]
        output_dict[f'Cal_SSA_amb_{wvl[i1]}'] = CalSSA_amb[i0, :]
        output_dict[f'Cal_fRH_{wvl[i1]}'] = CalfRH[i0, :] 
        i0 += 1  
#    CalCoef_amb[np.where(CalCoef_amb == 0)] = 'nan'
#    meas_coef_amb[np.where(meas_coef_amb == 0)] = 'nan'
    
    output_dict['Meas_Ext_Coef_dry_0.532'] = meas_ext_coef_dry   
    output_dict['Meas_Sca_Coef_amb_0.55'] = meas_coef_amb
    output_dict['Meas_Ext_Coef_amb_0.532'] = meas_ext_coef_amb
    output_dict['Meas_SSA_amb_0.55'] = meas_ssa_amb
    output_dict['Meas_fRH_0.55'] = meas_fRH   
    output_filename = np.array(input_filename.split('.ict'))
    output_filename = output_filename[0]
    np.save(f'{output_filename}.npy', output_dict)
       
#    ioo += 1

# Close the pool to any new jobs and remove it
pool.close()
pool.clear()
