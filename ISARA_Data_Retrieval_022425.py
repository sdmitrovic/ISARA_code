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

    >>> import ISARA_Data_Retrieval
    >>> ISARA_Data_Retrieval.RunISARA()
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

    def grab_ICT_Data(filename,modelist,full_wvl):
        data = importICARTT.imp(filename,2) 
        def grab_keydata(key_starts_with):
            for key in data.keys():
                if key.startswith(key_starts_with):
                    return data[key]    

        time = np.array(grab_keydata('Time_Start'))
        frmttime = np.array(grab_keydata('fmtdatetime_Start'))
        print(len(frmttime))
        date = grab_keydata('date')
        alt = np.array(grab_keydata('GPS_Altitude_BUCHOLTZ'))#alt = np.array(grab_keydata('gpsALT'))#alt = np.array(grab_keydata('GPS_alt'))
        lat = np.array(grab_keydata('Latitude'))
        print(len(lat))
        lon = np.array(grab_keydata('Longitude'))
        sd = {}
        for imode in modelist:
            if imode == "FIMS":
                sd[imode] = np.array([v for k, v in data.items() if k.startswith(f'n_Dp_')])
            else:
                sd[imode] = np.array([v for k, v in data.items() if k.startswith(f'{imode}_')])

        Sc = dict()
        Abs = dict()
        #SSA = {}
        Lwvl = len(full_wvl["Sc"])
        for iwvl in range(Lwvl):
            #print(iwvl)
            Sc[f'{full_wvl["Sc"][iwvl]}'] = np.array(grab_keydata(f'Sc{full_wvl["Sc"][iwvl]}_total_ZIEMBA'))
            Abs[f'{full_wvl["Abs"][iwvl]}'] = np.array(grab_keydata(f'Abs{full_wvl["Abs"][iwvl]}_total_ZIEMBA'))
        fRH = np.array(grab_keydata('fRH550_RH20to80_ZIEMBA'))
        print(fRH.size)
        return (data, time, date, alt, lat, lon, sd, Sc, Abs, fRH)   
    

    def handle_line(modelist, sd, dpg, dpu, dpl, UBcutoff, LBcutoff, measured_Sc_dry, measured_Abs_dry, measured_fRH,
                        wvl, size_equ, CRI_p, nonabs_fraction, shape,
                        kappa_p, num_theta, rho_wet, path_optical_dataset, path_mopsmap_executable, full_dp):
                    
        # So this code may look a bit funky, but we are doing what is called currying. This is simply the idea of returning a function inside of a function. 
        # It may look weird doing this, but this is actually required so that each worker has the necessary data. What ends up happening is each worker is 
        # passed a full copy of all the data contained within this function, so it has to know what data needs to be copied. Anyhow, the inner `curry` 
        # function is what is actually being called for each iteration of the for loop.
        def curry(i1):  
            finalout = {}
            #finalout['full_wvl'] = full_wvl
            measflg = 0 
            Lwvl = len(full_wvl["Sc"])
            for iwvl in range(Lwvl): 
                finalout[f'Meas_sca_coef_dry_{full_wvl["Sc"][iwvl]}'] = np.multiply(measured_Sc_dry[f'{full_wvl["Sc"][iwvl]}'][i1], pow(10, -6))
                finalout[f'Meas_abs_coef_dry_{full_wvl["Abs"][iwvl]}'] = np.multiply(measured_Abs_dry[f'{full_wvl["Abs"][iwvl]}'][i1], pow(10, -6))

                if (np.logical_not(np.isnan(finalout[f'Meas_sca_coef_dry_{full_wvl["Sc"][iwvl]}']))&(finalout[f'Meas_sca_coef_dry_{full_wvl["Sc"][iwvl]}']>10**(-6))):
                    measflg += 1
                if (np.logical_not(np.isnan(finalout[f'Meas_abs_coef_dry_{full_wvl["Abs"][iwvl]}']))&(finalout[f'Meas_abs_coef_dry_{full_wvl["Abs"][iwvl]}']>10**(-6))):
                    measflg += 1 

            finalout[f'Meas_sca_coef_wet_550'] = finalout[f'Meas_sca_coef_dry_550']*measured_fRH[i1]
            finalout[f'Meas_ext_coef_wet_550'] = finalout[f'Meas_sca_coef_wet_550']+finalout[f'Meas_abs_coef_dry_520']
            full_wvl2 = {}
            full_wvl2["Sc"] = [550]

            dndlogdp = {}
            for imode in sd:
                dndlogdp[imode] = np.multiply(sd[imode][:, i1], pow(10, 6))
                dndlogdp[imode][np.where(dndlogdp[imode] == 0)[0]] = np.nan
            if "APS" in modelist[:]:
                output_dictionary_1 = APS_rho.Align(dpg["UHSAS"],dndlogdp["UHSAS"],dpg["APS"],dndlogdp["APS"])
                rho_dry = output_dictionary_1["rho"]
                peak = output_dictionary_1["peak"]
            else:
                rho_dry = np.full((1, L1), 1)
                peak = np.full((1, L1), np.nan)

            # This is where things become a pain :( Since we are spreading the work across multiple cores, there is a copy of the data in each core. Therefore, we are not 
            # able to easily make updates to the numpy arrays, so instead we will obtain the results for each line then join them together after the multiprocessing occurs.
            finalout['attempt_count_CRI'] = 0
            finalout['attempt_count_kappa'] = 0
            # You will notice that in the code, instead of doing things like CRI_dry[:, i1] = ..., we are instead just assining the value for this row instead and then they will be merged later
            dpflg = 0
            icount = 0
            Dpg = {}
            Dpu = {}
            Dpl = {}  
            Dndlogdp = {}
            Size_equ = {}
            Nonabs_fraction = {}
            Shape = {}
            Rho_dry = {}
            Rho_wet = {}  
            fullsd = None
            fulldpg = None
            fulldpu = None
            fulldpl = None
            for imode in sd:
                icount += 1
                if len(dpg[imode]) > 3:
                    if imode == "APS":
                        a = np.divide(dpl[imode],np.sqrt(rho_dry))
                        b = np.divide(dpu[imode],np.sqrt(rho_dry))
                        modeflg = np.where(np.logical_not(np.isnan(dndlogdp[imode]))&(a>=LBcutoff[imode])&(b<=UBcutoff[imode]))[0]
                    else:
                        modeflg = np.where(np.logical_not(np.isnan(dndlogdp[imode]))&(dpl[imode]>=LBcutoff[imode])&(dpu[imode]<=UBcutoff[imode]))[0]
                
                    dpflg += 1
                    Dndlogdp[imode] = dndlogdp[imode][modeflg] 
                    Size_equ[imode] = size_equ
                    Nonabs_fraction[imode] = nonabs_fraction
                    Shape[imode] = shape
                    Rho_dry[imode] = rho_dry
                    Rho_wet[imode] = rho_wet
                    if imode == "APS":
                        Dpg[imode] = np.divide(dpg[imode],np.sqrt(Rho_dry[imode]))[modeflg]
                        Dpu[imode] = np.divide(dpu[imode],np.sqrt(Rho_dry[imode]))[modeflg]
                        Dpl[imode] = np.divide(dpl[imode],np.sqrt(Rho_dry[imode]))[modeflg]
                    else:
                        Dpg[imode] = dpg[imode][modeflg]
                        Dpu[imode] = dpu[imode][modeflg]
                        Dpl[imode] = dpl[imode][modeflg]
                    if dpflg == 1:
                        fullsd = Dndlogdp[imode]
                        fulldpg = Dpg[imode]
                        fulldpu = Dpu[imode]
                        fulldpl = Dpl[imode]
                    else:
                        fullsd = np.hstack((fullsd,Dndlogdp[imode]))
                        fulldpg = np.hstack((fulldpg,Dpg[imode]))
                        fulldpu = np.hstack((fulldpu,Dpu[imode]))
                        fulldpl = np.hstack((fulldpl,Dpl[imode]))
            full_sd = np.full(len(full_dp["dpg"]),np.nan)
            for idpg in range(len(full_dp["dpg"])):
                fulldpflg = np.where((fulldpg>=full_dp["dpl"][idpg])&(fulldpg<=full_dp["dpu"][idpg]))[0]
                if len(fulldpflg)>0:
                    full_sd[idpg] = fullsd[fulldpflg]
            dpgcount = 0
            for idpg in full_dp["dpg"]:
                finalout[f'full_dndlogdp_{idpg}'] = full_sd[dpgcount]
                dpgcount += 1        

            #measflg = np.where((np.logical_not(np.isnan(meas_coef))&(meas_coef>10**(-6))))[0]
            #print(len(meas_coef))
            if (dpflg==icount) & (measflg == 6):
                finalout['attempt_count_CRI'] = 1
                Results = ISARA2.Retr_CRI(full_wvl, finalout, Dndlogdp, Dpg, CRI_p, Size_equ, 
                    Nonabs_fraction, Shape, Rho_dry, num_theta, path_optical_dataset, path_mopsmap_executable)    

                if Results["RRI_dry"] is not None:
                    #print(Results["RRIdry"])
                    CRI_dry = np.array([Results["RRI_dry"],Results["IRI_dry"]])
                    for key in Results:
                        finalout[key] = Results[key]

                    #if (RH_amb[i1].astype(str) != 'nan') and (measured_coef_wet[i1].astype(str) != 'nan'):
                    if np.logical_not(np.isnan(finalout[f'Meas_sca_coef_wet_550'])):
                        finalout['attempt_count_kappa'] = 1
                        Results = ISARA2.Retr_kappa(full_wvl2, finalout, Dndlogdp, Dpg, 80, kappa_p, CRI_dry,
                            Size_equ, Nonabs_fraction, Shape, Rho_wet, num_theta,
                            path_optical_dataset, path_mopsmap_executable)
                        
                        if Results["Kappa"] is not None:
                            for key in Results:
                                finalout[key] = Results[key]
                            #print(finalout["kappa"])    
                            finalout[f'Cal_fRH'] = finalout[f'Cal_sca_coef_wet_550']/finalout[f'Meas_sca_coef_dry_550']
                        else:
                            finalout[f'Cal_fRH'] = np.nan
                            finalout[f'Kappa'] = np.nan
                            for i2 in range(len(full_wvl2["Sc"])):
                                finalout[f'Cal_sca_coef_wet_{full_wvl2["Sc"][i2]}'] = np.nan
                                finalout[f'Cal_SSA_wet_{full_wvl2["Sc"][i2]}'] = np.nan
                                finalout[f'Cal_ext_coef_wet_{full_wvl2["Sc"][i2]}'] = np.nan
                else:
                    finalout["RRI_dry"] = np.nan
                    finalout["IRI_dry"] = np.nan
                    for i2 in range(Lwvl):
                        finalout[f'Cal_sca_coef_dry_{full_wvl["Sc"][i2]}'] = np.nan
                        finalout[f'Cal_abs_coef_dry_{full_wvl["Abs"][i2]}'] = np.nan
                        finalout[f'Cal_SSA_dry_{full_wvl["Sc"][i2]}'] = np.nan
                        finalout[f'Cal_SSA_dry_{full_wvl["Abs"][i2]}'] = np.nan
                        finalout[f'Cal_ext_coef_dry_{full_wvl["Sc"][i2]}'] = np.nan
                        finalout[f'Cal_ext_coef_dry_{full_wvl["Abs"][i2]}'] = np.nan
            else:
                    finalout["RRI_dry"] = np.nan
                    finalout["IRI_dry"] = np.nan
                    for i2 in range(Lwvl):
                        finalout[f'Cal_sca_coef_dry_{full_wvl["Sc"][i2]}'] = np.nan
                        finalout[f'Cal_abs_coef_dry_{full_wvl["Abs"][i2]}'] = np.nan
                        finalout[f'Cal_SSA_dry_{full_wvl["Sc"][i2]}'] = np.nan
                        finalout[f'Cal_SSA_dry_{full_wvl["Abs"][i2]}'] = np.nan
                        finalout[f'Cal_ext_coef_dry_{full_wvl["Sc"][i2]}'] = np.nan
                        finalout[f'Cal_ext_coef_dry_{full_wvl["Abs"][i2]}'] = np.nan

                    finalout[f'Cal_fRH'] = np.nan
                    finalout[f'Kappa'] = np.nan
                    for i2 in range(len(full_wvl2["Sc"])):
                        finalout[f'Cal_sca_coef_wet_{full_wvl2["Sc"][i2]}'] = np.nan
                        finalout[f'Cal_SSA_wet_{full_wvl2["Sc"][i2]}'] = np.nan
                        finalout[f'Cal_ext_coef_wet_{full_wvl2["Sc"][i2]}'] = np.nan                        
            return (finalout)   

        return curry    

    OP_Dictionary = {}  

    # set desired output wavelengths in micrometer
    #wvl = [0.450, 0.470, 0.532, 0.550, 0.660, 0.700]    
    #wvl = [0.450, 0.465, 0.520, 0.550, 0.640, 0.700] 
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
    rho_wet = 1.63  

    DN = input("Enter the campaign name (e.g., ACTIVATE): ")   
    nummodes = int(input("Enter number of size distributions measured: "))
    modelist = np.empty(nummodes).astype(str)  
    UBcutoff = {}    
    LBcutoff = {}   
    dpg = {}
    dpu = {}
    dpl = {}
    maxdpglength = 0
    full_dp = {}
    full_dp["dpg"] = None
    full_dp["dpu"] = None
    full_dp["dpl"] = None
    for i1 in range(nummodes):
        keyname = input(f"Enter the instrument name for mode {i1+1} data (e.g., LAS): ")
        modelist[i1] = keyname
        ifn = [f for f in os.listdir(f'./misc/{DN}/SDBinInfo/') if f.__contains__(keyname)]
        dpData = load_sizebins.Load(f'./misc/{DN}/SDBinInfo/{ifn[0]}')
        dpg[keyname] = dpData["Mid Points"]*pow(10,-3) 
        dpu[keyname] = dpData["Upper Bounds"]*pow(10,-3) 
        dpl[keyname] = dpData["Lower Bounds"]*pow(10,-3) 
        UBcutoff[keyname] = float(input(f"Enter the upper bound of particle sizes\nfor {keyname} data in nm (e.g., 125): "))*pow(10,-3)
        LBcutoff[keyname] = float(input(f"Enter the lower bound of particle sizes\nfor {keyname} data in nm (e.g., 10): "))*pow(10,-3)
        dpcutoffflg = np.where((dpl[keyname]>=LBcutoff[keyname])&(dpu[keyname]<=UBcutoff[keyname]))[0]
        maxdpglength += len(dpcutoffflg)
        if i1 == 0:
            full_dp["dpg"] = dpg[keyname][dpcutoffflg]
            full_dp["dpu"] = dpu[keyname][dpcutoffflg]
            full_dp["dpl"] = dpl[keyname][dpcutoffflg]
        else:
            full_dp["dpg"] = np.hstack((full_dp["dpg"],dpg[keyname][dpcutoffflg]))
            full_dp["dpu"] = np.hstack((full_dp["dpu"],dpu[keyname][dpcutoffflg]))
            full_dp["dpl"] = np.hstack((full_dp["dpl"],dpl[keyname][dpcutoffflg]))
    numwvl = int(input("Enter number of spectral channels measured: "))
    full_wvl = {}
    full_wvl["Sc"] = np.full(numwvl,np.nan).astype(int)
    full_wvl["Abs"] = np.full(numwvl,np.nan).astype(int)
    for iwvl in range(numwvl):
        full_wvl["Sc"][iwvl]  = input(f"Enter scattering wavelength associated with channel {iwvl+1} in nm (e.g., 450): ")
        full_wvl["Abs"][iwvl] = input(f"Enter absorption wavelength associated with channel {iwvl+1} in nm (e.g., 465): ")
    IFN = [f for f in os.listdir(f'./misc/{DN}/InsituData/') if f.endswith('.ict')]
    for input_filename in IFN:#[156:]:
        print(input_filename)
        # import the .ict data into a dictonary
        (output_dict, time, date, alt, lat, lon, sd, Sc, Abs, fRH)  = grab_ICT_Data(f'./misc/{DN}/InsituData/{input_filename}', modelist, full_wvl)

        if ((fRH.size > 1)&(Sc[f'{full_wvl["Sc"][0]}'].size > 1)):
            L1 = fRH.size
            output_dict['full_dp'] = full_dp
            output_dict["dpg"] = dpg
            output_dict["dpu"] = dpu
            output_dict["dpl"] = dpl
            output_dict["UBcutoff"] = UBcutoff
            output_dict["LBcutoff"] = LBcutoff
            output_dict["dpcutoffflg"] = dpcutoffflg
            output_dict["maxdpglength"] = maxdpglength            

            # Loop through each of the rows here using multiprocessing. This will split the rows across multiple different cores. Each row will be its own index in `line_data` 
            # with a tuple full of information. So, for instance, line_data[0] will contain (CRI_dry, CalCoef_dry, meas_coef_dry, Kappa, CalCoef_wet, meas_coef_wet, results) 
            # for the first line of data
            line_data = pool.map(
                # This is a pain, I know, but all the data has to be cloned and accessible within each worker
                handle_line(modelist, sd, dpg, dpu, dpl, UBcutoff, LBcutoff, Sc, Abs,
                            fRH, full_wvl, size_equ, CRI_p, nonabs_fraction, shape,
                            kappa_p, num_theta, rho_wet, path_optical_dataset, path_mopsmap_executable, full_dp),
                range(L1),
            )
            #output_dict = dict()
            # Now that the data has been fetched, we have to join together all the results into aggregated arrays. The `enumerate` function simply loops through the elements in 
            # the array and attaches the associated array index to it.
            # The general trend for merging the values is pretty simple. If the value is not None, that means that it has a value set because it was reached conditionally. 
            #Therefore, if it does have a value, we will just update that part of the array. Now, I know you're probably thinking "why are we doing all this work again." Well, 
            # true, it is repeated work, but this will allow for much faster times overall (well, that's the hope anyhow).
            # def merge_in(line_val, merged_vals):
            for i1, line_data in enumerate(line_data):
                (results_line) = line_data   
                for key2 in results_line:
                    if key2 in output_dict:
                        output_dict[key2][i1] = results_line[key2]
                    else:
                        output_dict[key2] = np.full((L1),np.nan)
                        #print(results_line[key2])
                        output_dict[key2][i1] = results_line[key2]
            print(output_dict["Kappa"].size)            
            output_filename = np.array(input_filename.split('.ict'))
            output_filename = output_filename[0]
            np.save(f'{output_filename}.npy', output_dict)  

    # Close the pool to any new jobs and remove it
    pool.close()
    pool.clear()
