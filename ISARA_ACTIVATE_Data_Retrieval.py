import ISARA
import importICARTT
import load_sizebins
import numpy as np
import os
import sys
from pathos.multiprocessing import ProcessPool
def RunISARA():

    """
    Returns a dictionary of each of the ACTIVATE merged Falcon data files in directory as well as the retrieved particle complex refractive index and hygroscopicity


    :return: dictionary of all merged data and the retrieved parameters.
    :rtype: numpy dictionary

    >>> import ISARA_ACTIVATE_Data_Retrieval
    >>> ISARA_ACTIVATE_Data_Retrieval.RunISARA()
    activate-mrg-activate-large-smps_hu25_20200214_R0_20230831T150854.ict
    182
    182
    182
    """   

    sys.path.insert(0, os.path.abspath("../"))  

    # Number of cores you want to use
    number_of_cores = 16    

    # This should be at the start of the code to minimize the fork size
    pool = ProcessPool(ncpus=number_of_cores)   

    def pause():
        programPause = input

        ("Press the <ENTER> key to continue...")    

    def grab_ICT_Data(filename):
        data = importICARTT.imp(filename,2) 

        def grab_keydata(key_starts_with):
            for key in data.keys():
                if key.startswith(key_starts_with):
                    return data[key]    

        time = np.array(grab_keydata('Time_Start'))
        frmttime = np.array(grab_keydata('fmtdatetime_Start'))
        print(len(frmttime))
        date = grab_keydata('date')
        alt = np.array(grab_keydata('gpsALT'))
        lat = np.array(grab_keydata('Latitude'))
        print(len(lat))
        lon = np.array(grab_keydata('Longitude'))
        sd2 = np.array([v for k, v in data.items() if k.startswith('LAS_')])
        sd1 = np.array([v for k, v in data.items() if k.startswith('SMPS_')])
        RH_amb = np.array(grab_keydata('RHw_DLH_DISKIN_ '))
        print(RH_amb.size)
        RH_sp = np.array(grab_keydata('RH_Sc'))
        Sc = np.array([v for k, v in data.items() if k.startswith('Sc')])
        Abs = np.array([v for k, v in data.items() if k.startswith('Abs')])
        Ext = np.array([v for k, v in data.items() if k.startswith('Ext')])
        SSA = np.array([v for k, v in data.items() if k.startswith('SSA')])
        fRH = np.array(grab_keydata('fRH'))
        return (data, time, date, alt, lat, lon, sd1, sd2, RH_amb, RH_sp, Sc, Abs, Ext, SSA, fRH)   
    

    OP_Dictionary = {}  

    # set desired output wavelengths in micrometer
    wvl = [0.450, 0.470, 0.532, 0.550, 0.660, 0.700]    

    size_equ = 'cs' 

    RRIp = np.array([1.53])#np.arange(1.45,2.01,0.01).reshape(-1)
    IRIp = np.hstack((0,10**(-7),10**(-6),10**(-5),10**(-4),np.arange(0.001,0.0801,0.001).reshape(-1)))#np.hstack((0,10**(-7),10**(-6),10**(-5),10**(-4),np.arange(0.001,0.101,0.001).reshape(-1),np.arange(0.1,0.96,0.01).reshape(-1)))#np.arange(0.0,0.08,0.001).reshape(-1)
    CRI = np.empty((len(IRIp)*len(RRIp), 2))
    io = 0
    for i1 in range(len(IRIp)):
        for i2 in range(len(RRIp)):
            CRI[io, :] = [RRIp[i2], IRIp[i1]]
            io += 1 

    kappa = np.arange(0.0, 1.40, 0.01).reshape(-1)  

    # set the non-absorbing fraction of the aerosol SD
    nonabs_fraction = 0 

    # set shape or shape distribution of particles
    # shape='spheroid oblate 1.7'
    # shape='spheroid distr_file '../data/ar_kandler''
    shape = 'sphere'    

    num_theta = 2   

    rho_dry = 2.63
    rho_amb = 1.63  

    resolution = 60 

    dpg2 = load_sizebins.Load('LAS_bin_sizes.csv')*pow(10,-3) 
    dpg1 = load_sizebins.Load('SMPS_bin_sizes.csv')*pow(10,-3)  

    def handle_line(sd1, sd2, measured_coef_dry, measured_ext_coef_dry, measured_ssa_dry,
                        measured_coef_amb, measured_ext_coef_amb, measured_ssa_amb, measured_fRH,
                        wvl, size_equ, dpg1, dpg2, CRI, nonabs_fraction, shape, rho_dry,
                        RH_sp, kappa, num_theta, RH_amb, rho_amb):
                    
        # So this code may look a bit funky, but we are doing what is called currying. This is simply the idea of returning a function inside of a function. It may look weird doing this, but this is actually required so that each worker has the necessary data. What ends up happening is each worker is passed a full copy of all the data contained within this function, so it has to know what data needs to be copied. Anyhow, the inner `curry` function is what is actually being called for each iteration of the for loop.
        def curry(i1):  

            meas_coef = np.multiply(measured_coef_dry[:, i1], pow(10, -6))
            dndlogdp1 = np.multiply(sd1[:, i1], pow(10, 6))
            dndlogdp1[np.where(dndlogdp1 ==0)[0]] = np.nan
            dndlogdp2 = np.multiply(sd2[:, i1], pow(10, 6))
            dndlogdp2[np.where(dndlogdp2 ==0)[0]] = np.nan
    

            # This is where things become a pain :( Since we are spreading the work across multiple cores, there is a copy of the data in each core. Therefore, we are not able to easily make updates to the numpy arrays, so instead we will obtain the results for each line then join them together after the multiprocessing occurs.
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
            
            # You will notice that in the code, instead of doing things like CRI_dry[:, i1] = ..., we are instead just assining the value for this row instead and then they will be merged later
            dpflg1 = np.where(np.logical_not(np.isnan(dndlogdp1)))[0]
            dpflg2 = np.where(np.logical_not(np.isnan(dndlogdp2)))[0]
            measflg = np.where(np.logical_not(np.isnan(meas_coef)))[0]
            if (len(dndlogdp1[dpflg1])>3) & (len(dndlogdp2[dpflg2])>3) &(len(meas_coef[measflg]) == 6):
                Dpg1 = dpg1[dpflg1]
                dndlogdp1 = dndlogdp1[dpflg1]  

                Dpg2 = dpg2[dpflg2]
                dndlogdp2 = dndlogdp2[dpflg2]
                Results = ISARA.Retr_CRI(wvl, meas_coef[0:3], meas_coef[3:], dndlogdp1, dndlogdp2, Dpg1, Dpg2, CRI, size_equ, size_equ, 
                    nonabs_fraction, nonabs_fraction, shape, shape, rho_dry, rho_dry, num_theta)    

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

                    #if (RH_amb[i1].astype(str) != 'nan') and (measured_coef_amb[i1].astype(str) != 'nan'):
                    if measured_coef_amb[i1].astype(str) != 'nan':
                        meas_coef = np.multiply(measured_coef_amb[i1], pow(10, -6))
                        Results = ISARA.Retr_kappa(wvl, meas_coef, dndlogdp1, dndlogdp2, Dpg1, Dpg2, 80, kappa, CRI_dry, CRI_dry,
                            size_equ, size_equ, nonabs_fraction, nonabs_fraction, shape, shape, rho_amb, rho_amb, num_theta)
                        if Results["Kappa"] is not None:
                            Kappa = Results["Kappa"]
                            CalCoef_amb = Results["Cal_coef"]
                            CalExtCoef_amb = Results["Cal_ext_coef"]
                            CalSSA_amb = Results["Cal_SSA"]
                            meas_coef_amb = measured_coef_amb[i1]
                            meas_ext_coef_amb = measured_ext_coef_amb[i1]
                            meas_ssa_amb = measured_ssa_amb[i1]
                            meas_fRH = measured_fRH[i1]
                            CalfRH = np.empty(len(CalCoef_amb))
                            CalfRH[:] = np.nan
                            for i3 in range(len(CalCoef_amb)):
                                if CalExtCoef_dry is not None:
                                    CalfRH[i3] = CalCoef_amb[i3]/(CalExtCoef_dry[i3]*CalSSA_dry[i3])
            return (RRI_dry, IRI_dry, CalScatCoef_dry, CalAbsCoef_dry, CalExtCoef_dry, CalSSA_dry, meas_coef_dry, 
                    meas_ext_coef_dry, meas_ssa_dry, Kappa, CalCoef_amb, CalExtCoef_amb, CalSSA_amb, CalfRH,
                    meas_coef_amb, meas_ext_coef_amb, meas_ssa_amb, meas_fRH)#, results)    

        return curry    
    

    IFN = [f for f in os.listdir(r'./misc/ACTIVATE/FalconSMPS/') if f.endswith('.ict')]
    for input_filename in IFN:#[156:]:
        print(input_filename)
        # import the .ict data into a dictonary
        (output_dict, time, date, alt, lat, lon, sd1, sd2, RH_amb, RH_sp, Sc,
         Abs, Ext, SSA, fRH) = grab_ICT_Data(f'./misc/ACTIVATE/FalconSMPS/{input_filename}')
        if RH_amb.size > 1:
            print(RH_amb)
            #RH_amb[RH_amb > 99] = 99    

            measured_coef_dry = np.vstack((Sc[1:, :], Abs))
            measured_ext_coef_dry = Ext[1, :]
            measured_ssa_dry = SSA[0:3, :]
            measured_coef_amb = measured_coef_dry[1,:]*fRH #Sc[0, :]
            measured_ext_coef_amb = Ext[0, :]
            measured_ssa_amb = SSA[-1, :]
            measured_fRH = fRH      

            Lwvl = len(wvl)
            Lwvl_s = int(Lwvl/2)
            L1 = len(sd1[0, :])
            RRI_dry = np.empty((1, L1))
            RRI_dry[:]=np.nan
            IRI_dry = np.empty((1, L1))
            IRI_dry[:]=np.nan
            CalScatCoef_dry = np.empty((Lwvl_s, L1))
            CalScatCoef_dry[:]=np.nan
            CalAbsCoef_dry = np.empty((Lwvl_s, L1))
            CalAbsCoef_dry[:]=np.nan
            CalExtCoef_dry = np.empty((Lwvl, L1))
            CalExtCoef_dry[:]=np.nan
            CalSSA_dry = np.empty((Lwvl, L1))
            CalSSA_dry[:]=np.nan
            meas_coef_dry = np.empty((Lwvl, L1))
            meas_coef_dry[:]=np.nan
            meas_coef_amb = np.empty((1, L1))
            meas_coef_amb[:]=np.nan
            meas_ext_coef_dry = np.empty((1, L1))
            meas_ext_coef_dry[:]=np.nan
            meas_ssa_dry = np.empty((3, L1))
            meas_ssa_dry[:]=np.nan
            Kappa = np.empty((1,L1))
            Kappa[:]=np.nan
            CalCoef_amb = np.empty((Lwvl, L1))
            CalCoef_amb[:]=np.nan
            CalfRH = np.empty((Lwvl, L1))
            CalfRH[:]=np.nan
            CalExtCoef_amb = np.empty((Lwvl, L1))
            CalExtCoef_amb[:]=np.nan
            CalSSA_amb = np.empty((Lwvl, L1))
            CalSSA_amb[:]=np.nan
            meas_ext_coef_amb = np.empty((1, L1))
            meas_ext_coef_amb[:]=np.nan
            meas_ssa_amb = np.empty((1, L1)) 
            meas_ssa_amb[:]=np.nan
            meas_fRH = np.empty((1, L1))  
            meas_fRH[:]=np.nan
            # Loop through each of the rows here using multiprocessing. This will split the rows across multiple different cores. Each row will be its own index in `line_data` with a tuple full of information. So, for instance, line_data[0] will contain (CRI_dry, CalCoef_dry, meas_coef_dry, Kappa, CalCoef_amb, meas_coef_amb, results) for the first line of data
            line_data = pool.map(
                # This is a pain, I know, but all the data has to be cloned and accessible within each worker
                handle_line(sd1, sd2, measured_coef_dry, measured_ext_coef_dry, measured_ssa_dry,
                            measured_coef_amb, measured_ext_coef_amb, measured_ssa_amb, measured_fRH,
                            wvl, size_equ, dpg1, dpg2, CRI, nonabs_fraction, shape, rho_dry,
                            RH_sp, kappa, num_theta, RH_amb, rho_amb),
                range(L1),
            )
            # Now that the data has been fetched, we have to join together all the results into aggregated arrays. The `enumerate` function simply loops through the elements in the array and attaches the associated array index to it.
            for i1, line_data in enumerate(line_data):
                (RRI_dry_line, IRI_dry_line, CalScatCoef_dry_line, CalAbsCoef_dry_line, CalExtCoef_dry_line, 
                    CalSSA_dry_line, meas_coef_dry_line, meas_ext_coef_dry_line, meas_ssa_dry_line, Kappa_line, 
                    CalCoef_amb_line, CalExtCoef_amb_line, CalSSA_amb_line, CalfRH_line, meas_coef_amb_line,
                    meas_ext_coef_amb_line, meas_ssa_amb_line, meas_fRH_line) = line_data#, results_line) = line_data       

                # The general trend for merging the values is pretty simple. If the value is not None, that means that it has a value set because it was reached conditionally. Therefore, if it does have a value, we will just update that part of the array. Now, I know you're probably thinking "why are we doing all this work again." Well, true, it is repeated work, but this will allow for much faster times overall (well, that's the hope anyhow).
                def merge_in(line_val, merged_vals):
                    merged_vals[:, i1] = line_val   

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
                
            # From here on out, everything can continue as normal
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
            
            output_dict['Meas_Ext_Coef_dry_0.532'] = meas_ext_coef_dry   
            output_dict['Meas_Sca_Coef_amb_0.55'] = meas_coef_amb
            output_dict['Meas_Ext_Coef_amb_0.532'] = meas_ext_coef_amb
            output_dict['Meas_SSA_amb_0.55'] = meas_ssa_amb
            output_dict['Meas_fRH_0.55'] = meas_fRH   
            output_filename = np.array(input_filename.split('.ict'))
            output_filename = output_filename[0]
            np.save(f'{output_filename}.npy', output_dict)  

    # Close the pool to any new jobs and remove it
    pool.close()
    pool.clear()
