import ISARA2
import APS_rho
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
    path_optical_dataset='./optical_dataset/'
    path_mopsmap_executable=r'./mopsmap'

    sys.path.insert(0, os.path.abspath("../"))  

    # Number of cores you want to use
    number_of_cores = 32    

    # This should be at the start of the code to minimize the fork size
    pool = ProcessPool(ncpus=number_of_cores)   

    def pause():
        programPause = input("Press the <ENTER> key to continue...")    

    def grab_ICT_Data(filename,modelist):
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
        sd = {}
        for imode in modelist:
            if imode == "FIMS":
                sd[imode] = np.array([v for k, v in data.items() if k.startswith(f'n_Dp_')])
            else:
                sd[imode] = np.array([v for k, v in data.items() if k.startswith(f'{imode}_')])
            #print(sd[imode].size(1,2))
        RH_amb = np.array(grab_keydata('RHw_'))
        print(RH_amb.size)
        RH_sp = np.array(grab_keydata('RH_Sc'))
        Sc = np.array([v for k, v in data.items() if (k.startswith('Sc')&k.__contains__('total'))])
        Abs = np.array([v for k, v in data.items() if k.startswith('Abs')])
        Ext = np.array([v for k, v in data.items() if k.startswith('Ext')])
        SSA = np.array([v for k, v in data.items() if (k.startswith('SSA')&k.__contains__('dry'))])
        SSAa =np.array(grab_keydata('SSA_amb_550nm_ZIEMBA'))
        print(SSAa.size)
        fRH = np.array(grab_keydata('fRH'))
        return (data, time, date, alt, lat, lon, sd, RH_amb, RH_sp, Sc, Abs, Ext, SSA, SSAa, fRH)   
    

    OP_Dictionary = {}  

    # set desired output wavelengths in micrometer
    wvl = [0.450, 0.470, 0.532, 0.550, 0.660, 0.700]    

    size_equ = 'cs' 

    RRIp = np.arange(1.52,1.54,0.01).reshape(-1)#np.array([1.53])#np.arange(1.45,2.01,0.01).reshape(-1)
    IRIp = np.hstack((0,10**(-7),10**(-6),10**(-5),10**(-4),np.arange(0.001,0.0801,0.001).reshape(-1)))
    #np.hstack((0,10**(-7),10**(-6),10**(-5),10**(-4),np.arange(0.001,0.101,0.001).reshape(-1),np.arange(0.1,0.96,0.01).reshape(-1)))
    #np.arange(0.0,0.08,0.001).reshape(-1)
    CRI_p = np.empty((len(IRIp)*len(RRIp), 2))
    io = 0
    for i1 in range(len(IRIp)):
        for i2 in range(len(RRIp)):
            CRI_p[io, :] = [RRIp[i2], IRIp[i1]]
            io += 1 

    kappa_p = np.arange(0.0, 1.40, 0.001).reshape(-1)  

    # set the non-absorbing fraction of the aerosol SD
    nonabs_fraction = 0 

    # set shape or shape distribution of particles
    # shape='spheroid oblate 1.7'
    # shape='spheroid distr_file '../data/ar_kandler''
    shape = 'sphere'    

    num_theta = 2   

    #rho_dry = 2.63
    rho_amb = 1.63  

    def handle_line(modelist, sd, dpg, UBcutoff, LBcutoff, measured_coef_dry, measured_ext_coef_dry, measured_ssa_dry,
                        measured_coef_amb, measured_ext_coef_amb, measured_ssa_amb, measured_fRH,
                        wvl, size_equ, CRI_p, nonabs_fraction, shape,
                        RH_sp, kappa_p, num_theta, RH_amb, rho_amb, path_optical_dataset, path_mopsmap_executable):
                    
        # So this code may look a bit funky, but we are doing what is called currying. This is simply the idea of returning a function inside of a function. It may look weird doing this, but this is actually required so that each worker has the necessary data. What ends up happening is each worker is passed a full copy of all the data contained within this function, so it has to know what data needs to be copied. Anyhow, the inner `curry` function is what is actually being called for each iteration of the for loop.
        def curry(i1):  

            meas_coef = np.multiply(measured_coef_dry[:, i1], pow(10, -6))
            dndlogdp = {}
            for imode in sd:
                dndlogdp[imode] = np.multiply(sd[imode][:, i1], pow(10, 6))
                dndlogdp[imode][np.where(dndlogdp[imode] == 0)[0]] = np.nan

            if "APS" in modelist[:]:
                output_dictionary_1 = APS_rho.Align(dpg["UHSAS"],dndlogdp["UHSAS"],dpg["APS"],dndlogdp["APS"])
                rho_dry = output_dictionary_1["rho"]
                peak = output_dictionary_1["peak"]
            else:
                rho_dry = np.full((1, L1), 2.63)
                peak = np.full((1, L1), np.nan)

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
            attempt_count = np.zeros(2)
            # You will notice that in the code, instead of doing things like CRI_dry[:, i1] = ..., we are instead just assining the value for this row instead and then they will be merged later
            dpflg = 0
            icount = 0
            Dpg = {}
            Dndlogdp = {}
            Size_equ = {}
            Nonabs_fraction = {}
            Shape = {}
            Rho_dry = {}
            Rho_amb = {}  
            for imode in sd:
                icount =+ 1
                modeflg = np.where(np.logical_not(np.isnan(dndlogdp[imode]))&(dpg[imode]>=LBcutoff[imode])&(dpg[imode]<UBcutoff[imode]))[0]
                if len(dpg[imode]) > 3:
                    dpflg =+ 1
                    Dndlogdp[imode] = dndlogdp[imode][modeflg] 
                    Size_equ[imode] = size_equ
                    Nonabs_fraction[imode] = nonabs_fraction
                    Shape[imode] = shape
                    Rho_dry[imode] = rho_dry
                    Rho_amb[imode] = rho_amb
                    if imode == "APS":
                        Dpg[imode] = np.divide(dpg[imode][modeflg],np.sqrt(Rho_dry[imode]))
                    else:
                        Dpg[imode] = dpg[imode][modeflg]

            measflg = np.where((np.logical_not(np.isnan(meas_coef))&(meas_coef>10**(-6))))[0]
            if (dpflg==icount) & (len(meas_coef[measflg]) == 6):
                attempt_count[0] = 1
                Results = ISARA2.Retr_CRI(wvl, meas_coef[0:3], meas_coef[3:], Dndlogdp, Dpg, CRI_p, Size_equ, 
                    Nonabs_fraction, Shape, Rho_dry, num_theta, path_optical_dataset, path_mopsmap_executable)    

                if Results["RRIdry"] is not None:
                    #print(Results["RRIdry"])
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
                    if np.logical_not(np.isnan(measured_coef_amb[i1])):
                        attempt_count[1] = 1
                        meas_coef = np.multiply(measured_coef_amb[i1], pow(10, -6))
                        Results = ISARA2.Retr_kappa(wvl, meas_coef, Dndlogdp, Dpg, 80, kappa_p, CRI_dry,
                            Size_equ, Nonabs_fraction, Shape, Rho_amb, num_theta,
                            path_optical_dataset, path_mopsmap_executable)
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
                    meas_coef_amb, meas_ext_coef_amb, meas_ssa_amb, meas_fRH, attempt_count)#, results)    

        return curry    
    
    DN = input("Enter the campaign name (e.g., ACTIVATE): ")   
    nummodes = int(input("Enter number of size distributions measured: "))
    modelist = np.empty(nummodes).astype(str)  
    UBcutoff = {}    
    LBcutoff = {}   
    dpg = {}
    for i1 in range(nummodes):
        keyname = input(f"Enter the instrument name for mode {i1+1} data (e.g., LAS): ")
        modelist[i1] = keyname
        ifn = [f for f in os.listdir(f'./misc/{DN}/SDBinInfo/') if f.__contains__(keyname)]
        dpData = load_sizebins.Load(f'./misc/{DN}/SDBinInfo/{ifn[0]}')
        dpg[keyname] = dpData["Mid Points"]*pow(10,-3) 
        UBcutoff[keyname] = float(input(f"Enter the upper bound of particle sizes\nfor {keyname} data in nm (e.g., 125): "))*pow(10,-3)
        LBcutoff[keyname] = float(input(f"Enter the lower bound of particle sizes\nfor {keyname} data in nm (e.g., 10): "))*pow(10,-3)

    IFN = [f for f in os.listdir(f'./misc/{DN}/InsituData/') if f.endswith('.ict')]
    for input_filename in IFN:#[156:]:
        print(input_filename)
        # import the .ict data into a dictonary
        (output_dict,time,date,alt,lat,lon,sd,
            RH_amb,RH_sp,Sc,Abs,Ext,SSA,SSAa,fRH) = grab_ICT_Data(f'./misc/{DN}/InsituData/{input_filename}', modelist)
        if RH_amb.size > 1:
            #print(RH_amb)
            #RH_amb[RH_amb > 99] = 99    

            measured_coef_dry = np.vstack((Sc, Abs))
            measured_ext_coef_dry = Ext[1, :]
            measured_ssa_dry = SSA
            measured_coef_amb = Sc[0,:]*fRH #Sc[0, :]
            measured_ext_coef_amb = Ext[0, :]
            measured_ssa_amb = SSAa
            measured_fRH = fRH      
            #print(measured_coef_amb[np.where(np.logical_not(np.isnan(measured_coef_amb)))])
            Lwvl = len(wvl)
            Lwvl_s = int(Lwvl/2)
            L1 = RH_amb.size

            RRI_dry = np.full((1, L1),np.nan)
            IRI_dry =  np.full((1, L1),np.nan)
            CalScatCoef_dry = np.full((Lwvl_s, L1),np.nan)
            CalAbsCoef_dry = np.full((Lwvl_s, L1),np.nan)
            CalExtCoef_dry = np.full((Lwvl, L1),np.nan)
            CalSSA_dry = np.full((Lwvl, L1),np.nan)
            meas_coef_dry = np.full((Lwvl, L1),np.nan)
            meas_coef_amb = np.full((1, L1),np.nan)
            meas_ext_coef_dry = np.full((1, L1),np.nan)
            meas_ssa_dry = np.full((3, L1),np.nan)
            Kappa = np.full((1, L1),np.nan)
            CalCoef_amb = np.full((Lwvl, L1),np.nan)
            CalfRH = np.full((Lwvl, L1),np.nan)
            CalExtCoef_amb = np.full((Lwvl, L1),np.nan)
            CalSSA_amb = np.full((Lwvl, L1),np.nan)
            meas_ext_coef_amb = np.full((1, L1),np.nan)
            meas_ssa_amb = np.full((1, L1),np.nan)
            meas_fRH = np.full((1, L1),np.nan) 
            atmpt_cnt =  np.full((2, L1),np.nan)       
            # Loop through each of the rows here using multiprocessing. This will split the rows across multiple different cores. Each row will be its own index in `line_data` with a tuple full of information. So, for instance, line_data[0] will contain (CRI_dry, CalCoef_dry, meas_coef_dry, Kappa, CalCoef_amb, meas_coef_amb, results) for the first line of data
            line_data = pool.map(
                # This is a pain, I know, but all the data has to be cloned and accessible within each worker
                handle_line(modelist, sd, dpg, UBcutoff, LBcutoff, measured_coef_dry, measured_ext_coef_dry, measured_ssa_dry,
                            measured_coef_amb, measured_ext_coef_amb, measured_ssa_amb, measured_fRH,
                            wvl, size_equ, CRI_p, nonabs_fraction, shape,
                            RH_sp, kappa_p, num_theta, RH_amb, rho_amb, path_optical_dataset, path_mopsmap_executable),
                range(L1),
            )
            # Now that the data has been fetched, we have to join together all the results into aggregated arrays. The `enumerate` function simply loops through the elements in the array and attaches the associated array index to it.
            for i1, line_data in enumerate(line_data):
                (RRI_dry_line, IRI_dry_line, CalScatCoef_dry_line, CalAbsCoef_dry_line, CalExtCoef_dry_line, 
                    CalSSA_dry_line, meas_coef_dry_line, meas_ext_coef_dry_line, meas_ssa_dry_line, Kappa_line, 
                    CalCoef_amb_line, CalExtCoef_amb_line, CalSSA_amb_line, CalfRH_line, meas_coef_amb_line,
                    meas_ext_coef_amb_line, meas_ssa_amb_line, meas_fRH_line, attempt_count_line) = line_data#, results_line) = line_data       

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
                merge_in(attempt_count_line, atmpt_cnt)
                
            # From here on out, everything can continue as normal
            output_dict['RRI_dry'] = RRI_dry
            output_dict['IRI_dry'] = IRI_dry
            output_dict['Kappa'] = Kappa     
            output_dict['attempt_count_CRI'] = atmpt_cnt[0,:]
            output_dict['attempt_count_kappa'] = atmpt_cnt[1,:]
            #print(Kappa)
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
