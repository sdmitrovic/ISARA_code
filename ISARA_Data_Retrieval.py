import ISARA
import APS_rho
import importICARTT
import load_sizebins
import numpy as np
import os
import sys
from pathos.multiprocessing import ProcessPool
def RunISARA():

    """
    Saves a dictionary file of each of the merged data files in source directory that includes ISARA retrievals of CRI and kappa. Dictionary includes metadata for netCDF compliancy.

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
    def grab_data(data,key_name):
        for key in data.keys():
            if key.__contains__(key_name):
                return data[key]    
    def grab_ICT_Data(filename,modelist,full_wvl):
        data = importICARTT.imp(filename,2) 
        def grab_keydata(key_starts_with,does_not_contain=None):
            for key in data.keys():
                if does_not_contain is None:
                    if key.startswith(key_starts_with):
                        return data[key], key    
                else:
                    if key.startswith(key_starts_with)&np.logical_not(key.__contains__(does_not_contain)):
                        return data[key], key                             
        Sc = dict()
        Abs = dict()
        #SSA = {}
        Lwvl = len(full_wvl["Sc"])
        for iwvl in range(Lwvl):
            #print(iwvl)
            Scat, ScatKey = grab_keydata(f'Sc{full_wvl["Sc"][iwvl]}','amb')
            Sc[ScatKey]=Scat
            Absor, AbsorKey = grab_keydata(f'Abs{full_wvl["Abs"][iwvl]}','amb')
            Abs[AbsorKey]=Absor
        RHsc,kynmRH = grab_keydata('RH_Sc')
        RHsc= np.array(RHsc)
        gamma,kynmgamma = grab_keydata('gamma550')
        gamma= np.array(gamma)
        print(RHsc.size,gamma.size)

        time,keynmtime = grab_keydata('Time_Start')
        time= np.array(time)        
        frmttime,kynm = grab_keydata('datetime_Start')
        frmttime= np.array(frmttime) 
        print(len(frmttime))
        date,knmdate = grab_keydata('date')
        sd = {}
        for imode in modelist:
            if imode == "FIMS":
                sd[imode] = np.array([v for k, v in data.items() if k.startswith(f'n_Dp_')])
            else:
                sd[imode] = np.array([v for k, v in data.items() if k.startswith(f'{imode}_')])


        #pause()
        return (data, time, date, sd, Sc, Abs, RHsc, gamma)   
    

    def handle_line(modelist, sd, dpg, dpu, dpl, UBcutoff, LBcutoff, measured_Sc_dry, measured_Abs_dry, RHsc, gamma,
                        full_wvl, full_wvl2, val_wvl, size_equ, CRI_p, nonabs_fraction, shape,
                        kappa_p, num_theta, rho_wet, path_optical_dataset, path_mopsmap_executable, full_dp):
                    
        # So this code may look a bit funky, but we are doing what is called currying. This is simply the idea of returning a function inside of a function. 
        # It may look weird doing this, but this is actually required so that each worker has the necessary data. What ends up happening is each worker is 
        # passed a full copy of all the data contained within this function, so it has to know what data needs to be copied. Anyhow, the inner `curry` 
        # function is what is actually being called for each iteration of the for loop.
        # You will notice that in the code we are assining the value for this row and they will be merged later
        def curry(i1):  
            finalout = {}
            #finalout['full_wvl'] = full_wvl
            measflg = 0 
            Lwvl = len(full_wvl["Sc"])

            iwvl = 0        
            for kwvl in measured_Abs_dry: 
                finalout[f'dry_meas_abs_coef_{full_wvl["Abs"][iwvl]}_m-1'] = np.multiply(measured_Abs_dry[kwvl][i1], pow(10, -6))
                if finalout[f'dry_meas_abs_coef_{full_wvl["Abs"][iwvl]}_m-1']>=0:
                    measflg += 1
                iwvl += 1    

            iwvl = 0
            keycheck = None                          
            for kwvl in measured_Sc_dry: 
                if RHsc[i1]>40:
                    finalout[f'dry_meas_sca_coef_{full_wvl["Sc"][iwvl]}_m-1'] = np.multiply(measured_Sc_dry[kwvl][i1], pow(10, -6))/(np.exp(gamma[i1]*np.log((100-40)/(100-RHsc[i1]))))#scat_calc=scat_rh=measured(e^(GAMMA*ln((100-calcRH)/(100-measRH))))
                else:
                    finalout[f'dry_meas_sca_coef_{full_wvl["Sc"][iwvl]}_m-1'] = np.multiply(measured_Sc_dry[kwvl][i1], pow(10, -6))
                keycheck = kwvl     
                if finalout[f'dry_meas_sca_coef_{full_wvl["Sc"][iwvl]}_m-1']>=10**(-6):#
                    measflg += 1
                if kwvl.__contains__(str(full_wvl["Sc"][1])):
                    finalout[f'wet_meas_sca_coef_{full_wvl["Sc"][1]}_m-1'] = np.multiply(measured_Sc_dry[kwvl][i1], pow(10, -6))/(np.exp(gamma[i1]*np.log((100-80)/(100-RHsc[i1]))))
                    finalout[f'wet_meas_ext_coef_{full_wvl["Sc"][1]}_m-1'] = finalout[f'wet_meas_sca_coef_{full_wvl["Sc"][1]}_m-1']+finalout[f'dry_meas_abs_coef_{full_wvl["Abs"][1]}_m-1']
                    finalout[f'meas_fRH_{full_wvl["Sc"][1]}_unitless'] = finalout[f'wet_meas_sca_coef_{full_wvl["Sc"][1]}_m-1']/finalout[f'dry_meas_sca_coef_{full_wvl["Sc"][1]}_m-1']
                iwvl += 1     
    


            dndlogdp = {}
            for imode in sd:
                dndlogdp[imode] = np.multiply(sd[imode][:, i1], pow(10, 6))
            if "APS" in modelist[:]:
                output_dictionary_1 = APS_rho.Align(dpg["UHSAS"],dndlogdp["UHSAS"],dpg["APS"],dndlogdp["APS"])
                rho_dry = output_dictionary_1["rho"]
                peak = output_dictionary_1["peak"]
            else:
                rho_dry = 1
                peak = np.nan
            finalout['dry_rho_g.cm-3'] = rho_dry
            finalout['peak_diameter_um'] = peak
            finalout['attempt_flag_CRI_unitless'] = 0
            finalout['attempt_flag_kappa_unitless'] = 0
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
                    
                    Dndlogdp[imode] = dndlogdp[imode][modeflg] 
                    if len(Dndlogdp[imode]) > 3:
                        dpflg += 1
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
            if (dpflg==icount) & (measflg == 6):        
                full_sd = np.full(len(full_dp["dpg"]),np.nan)
                full_dpl= np.full(len(full_dp["dpg"]),np.nan)
                full_dpg = np.full(len(full_dp["dpg"]),np.nan)
                full_dpu = np.full(len(full_dp["dpg"]),np.nan)          
                for idpg in range(len(full_dp["dpg"])):
                    fulldpflg = np.where((fulldpg>=full_dp["dpl"][idpg])&(fulldpg<=full_dp["dpu"][idpg]))[0]
                    if len(fulldpflg)>0:
                        full_sd[idpg] = fullsd[fulldpflg]
                        full_dpl[idpg] = fulldpl[fulldpflg]
                        full_dpg[idpg] = fulldpg[fulldpflg]
                        full_dpu[idpg] = fulldpu[fulldpflg]
                for idpg in range(len(full_dp["dpg"])):
                    finalout[f'dndlogdp_bin{idpg}_cm-3'] = full_sd[idpg]       
                    finalout[f'dpl_bin{idpg}_um'] = full_dpl[idpg]
                    finalout[f'dpg_bin{idpg}_um'] = full_dpg[idpg]
                    finalout[f'dpu_bin{idpg}_um'] = full_dpu[idpg]          
            
                finalout['attempt_flag_CRI_unitless'] = 1
                if (keycheck.__contains__('submicron')&(UBcutoff[imode]>1)):
                    for imode in Dpg:
                        submicronfilter = np.where(Dpu[imode]<=1)[0]
                        Dndlogdp[imode]= Dndlogdp[imode][submicronfilter]
                        Dpg[imode] = Dpg[imode][submicronfilter]
                        Dpl[imode] = Dpl[imode][submicronfilter]
                        Dpu[imode] = Dpu[imode][submicronfilter]


                Results = ISARA.Retr_CRI(full_wvl, val_wvl, finalout, Dndlogdp, Dpg, CRI_p, Size_equ, 
                    Nonabs_fraction, Shape, Rho_dry, num_theta, path_optical_dataset, path_mopsmap_executable)    
                if Results["dry_RRI_unitless"] is not None:
                    finalout['attempt_flag_CRI_unitless'] = 2
                    #print(Results["RRIdry"])
                    CRI_dry = np.array([Results["dry_RRI_unitless"],Results["dry_IRI_unitless"]])
                    for key in Results:
                        finalout[key] = Results[key]
                    #if (RH_amb[i1].astype(str) != 'nan') and (measured_coef_wet[i1].astype(str) != 'nan'):
                    if np.logical_not(np.isnan(finalout[f'wet_meas_sca_coef_550_m-1'])):
                        finalout['attempt_flag_kappa_unitless'] = 1
                        Results = ISARA.Retr_kappa(full_wvl2, val_wvl, finalout, Dndlogdp, Dpg, 80, kappa_p, CRI_dry,
                            Size_equ, Nonabs_fraction, Shape, Rho_wet, num_theta,
                            path_optical_dataset, path_mopsmap_executable)
                        
                        if Results["kappa_unitless"] is not None:
                            finalout['attempt_flag_kappa_unitless'] = 2
                            for key in Results:
                                finalout[key] = Results[key]
                            #print(finalout["kappa"])    
                            finalout[f'cal_fRH_550_unitless'] = finalout[f'wet_cal_sca_coef_550_m-1']/finalout[f'dry_meas_sca_coef_550_m-1']
                        else:
                            finalout[f'cal_fRH_550_unitless'] = np.nan
                            finalout[f'kappa_unitless'] = np.nan
                            for i2 in range(len(full_wvl2["Sc"])):
                                finalout[f'wet_cal_sca_coef_{full_wvl2["Sc"][i2]}_m-1'] = np.nan
                                finalout[f'wet_cal_SSA_{full_wvl2["Sc"][i2]}_unitless'] = np.nan
                                finalout[f'wet_cal_ext_coef_{full_wvl2["Sc"][i2]}_m-1'] = np.nan
                            if val_wvl is not None:
                                for i2 in range(len(val_wvl)):
                                    finalout[f'wet_cal_sca_coef_{val_wvl[i2]}_m-1'] = np.nan
                                    finalout[f'wet_cal_SSA_{val_wvl[i2]}_unitless'] = np.nan
                                    finalout[f'wet_cal_ext_coef_{val_wvl[i2]}_m-1'] = np.nan                                     
                    else:
                        finalout["dry_RRI_unitless"] = np.nan
                        finalout["dry_IRI_unitless"] = np.nan
                        for i2 in range(Lwvl):
                            finalout[f'dry_cal_sca_coef_{full_wvl["Sc"][i2]}_m-1'] = np.nan
                            finalout[f'dry_cal_abs_coef_{full_wvl["Abs"][i2]}_m-1'] = np.nan
                            finalout[f'dry_cal_SSA_{full_wvl["Sc"][i2]}_unitless'] = np.nan
                            finalout[f'dry_cal_SSA_{full_wvl["Abs"][i2]}_unitless'] = np.nan
                            finalout[f'dry_cal_ext_coef_{full_wvl["Sc"][i2]}_m-1'] = np.nan
                            finalout[f'dry_cal_ext_coef_{full_wvl["Abs"][i2]}_m-1'] = np.nan
                        if val_wvl is not None:
                            for i2 in range(len(val_wvl)):
                                finalout[f'dry_cal_sca_coef_{val_wvl[i2]}_m-1'] = np.nan
                                finalout[f'dry_cal_SSA_{val_wvl[i2]}_unitless'] = np.nan
                                finalout[f'dry_cal_ext_coef_{val_wvl[i2]}_m-1'] = np.nan                       
            else:
                finalout["dry_RRI_unitless"] = np.nan
                finalout["dry_IRI_unitless"] = np.nan
                for i2 in range(Lwvl):
                    finalout[f'dry_cal_sca_coef_{full_wvl["Sc"][i2]}_m-1'] = np.nan
                    finalout[f'dry_cal_abs_coef_{full_wvl["Abs"][i2]}_m-1'] = np.nan
                    finalout[f'dry_cal_SSA_{full_wvl["Sc"][i2]}_unitless'] = np.nan
                    finalout[f'dry_cal_SSA_{full_wvl["Abs"][i2]}_unitless'] = np.nan
                    finalout[f'dry_cal_ext_coef_{full_wvl["Sc"][i2]}_m-1'] = np.nan
                    finalout[f'dry_cal_ext_coef_{full_wvl["Abs"][i2]}_m-1'] = np.nan

                finalout[f'cal_fRH_550_unitless'] = np.nan
                finalout[f'kappa_unitless'] = np.nan
                for i2 in range(len(full_wvl2["Sc"])):
                    finalout[f'wet_cal_sca_coef_{full_wvl2["Sc"][i2]}_m-1'] = np.nan
                    finalout[f'wet_cal_SSA_{full_wvl2["Sc"][i2]}_unitless'] = np.nan
                    finalout[f'wet_cal_ext_coef_{full_wvl2["Sc"][i2]}_m-1'] = np.nan   
                if val_wvl is not None:
                    for i2 in range(len(val_wvl)):
                        finalout[f'dry_cal_sca_coef_{val_wvl[i2]}_m-1'] = np.nan
                        finalout[f'dry_cal_SSA_{val_wvl[i2]}_unitless'] = np.nan
                        finalout[f'dry_cal_ext_coef_{val_wvl[i2]}_m-1'] = np.nan    
                        finalout[f'wet_cal_sca_coef_{val_wvl[i2]}_m-1'] = np.nan
                        finalout[f'wet_cal_SSA_{val_wvl[i2]}_unitless'] = np.nan
                        finalout[f'wet_cal_ext_coef_{val_wvl[i2]}_m-1'] = np.nan                                                           
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
    rho_wet = 1.00  

    DN = input("Enter the campaign name (e.g., ACTIVATE): ")   
    #dryorSP = input("Is the dry RH specified? Enter yes or no: ")
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
        #print(dpData)
        dpg[keyname] = grab_data(dpData,"Mid Points")*pow(10,-3) 
        dpu[keyname] = grab_data(dpData,"Upper Bounds")*pow(10,-3) 
        dpl[keyname] = grab_data(dpData,"Lower Bounds")*pow(10,-3) 
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
    numwvl = int(input("Enter number of dry spectral channels measured (e.g., 3): "))
    full_wvl = {}
    full_wvl["Sc"] = np.full(numwvl,np.nan).astype(int)
    full_wvl["Abs"] = np.full(numwvl,np.nan).astype(int)
    dry_channel_color = np.full(numwvl,np.nan).astype(str) 
    for iwvl in range(numwvl):
        full_wvl["Sc"][iwvl] = input(f"Enter scattering wavelength associated with channel {iwvl+1} in nm (e.g., 450): ")
        full_wvl["Abs"][iwvl] = input(f"Enter absorption wavelength associated with channel {iwvl+1} in nm (e.g., 465): ")
        dry_channel_color[iwvl] = input(f"Enter the wavelength color to represent channel {iwvl+1} (e.g., Blue, Green, or Red): ")

    numwvl = int(input("Enter number of humidified spectral channels measured (e.g., 1): "))
    full_wvl2 = {}
    full_wvl2["Sc"] = np.full(numwvl,np.nan).astype(int)
    wet_channel_color = np.full(numwvl,np.nan).astype(str) 
    for iwvl in range(numwvl):
        full_wvl2["Sc"][iwvl]  = input(f"Enter scattering wavelength associated with channel {iwvl+1} in nm (e.g., 450): ")
        wet_channel_color[iwvl] = input(f"Enter the wavelength color to represent channel {iwvl+1} (e.g., Blue, Green, or Red): ")

    addwvl = input(f"Are there any additional wavelengths needed? (yes or no): ")
    if addwvl == "yes":
        valwvl = input(f"Enter the additional wavelength channels speparated\nby a comma and a space (e.g., 370, 530, 1060): ")
        val_wvl =  np.array(valwvl.split(", ")).astype(int)
        valcolor = input(f"Enter the additional wavelength channels colors speparated\nby a comma and a space (e.g., Blue, Green, Red): ")
        val_channel_color =  np.array(valcolor.split(", ")).astype(str)
    else:
        val_wvl = None
    data_directory = input("Enter the name of the directory that contains\nin-situ measurements (e.g., InsituData): ")
    IFN = [f for f in os.listdir(f'./misc/{DN}/{data_directory}/') if f.endswith('.ict')]
    #b = np.array([39,126,138]).astype(int)#
    #IFN2 = [IFN[i] for i in b]
    for input_filename in IFN:#IFN2:#
        print(input_filename)
        # import the .ict data into a dictonary
        (output_dict, time, date, sd, Sc, Abs, RHsc, gamma)  = grab_ICT_Data(f'./misc/{DN}/{data_directory}/{input_filename}', modelist, full_wvl)
        output_dict['SourceFlag'] = {}
        output_dict['Dims'] = {}
        for key in output_dict['VariableAttributes'].keys():
            #if np.logical_not(isinstance(output_dict[key],dict)):
            output_dict['SourceFlag'][key] = 'source'
            output_dict['Dims'][key] = 'time'

        if ((gamma.size > 1)&(len(Sc.keys()) > 1)):
            L1 = gamma.size
            output_dict['full_dp'] = full_dp
            output_dict["dpg"] = dpg
            output_dict["dpu"] = dpu
            output_dict["dpl"] = dpl
            output_dict["UBcutoff"] = UBcutoff
            output_dict["LBcutoff"] = LBcutoff 
            output_dict['VariableAttributes']["dry_RRI_unitless"] = {}
            output_dict['VariableAttributes']["dry_RRI_unitless"]['short_name'] = 'dry_RRI'
            output_dict['VariableAttributes']["dry_RRI_unitless"]['units'] = '1'
            output_dict['VariableAttributes']["dry_RRI_unitless"]['long_name'] = 'Real refractive index of BULK particles at DRY relative humidity of 20% and STANDARD temperature and pressure derived from ISARA.'
            output_dict['VariableAttributes']["dry_RRI_unitless"]['ACVSNC_standard_name'] = 'AerOpt_m_InSitu_BluetoRed_RHd_Bulk_STP'
            output_dict['VariableAttributes']["dry_IRI_unitless"] = {}
            output_dict['VariableAttributes']["dry_IRI_unitless"]['short_name'] = 'dry_IRI'
            output_dict['VariableAttributes']["dry_IRI_unitless"]['units'] = '1'
            output_dict['VariableAttributes']["dry_IRI_unitless"]['long_name'] = 'Imaginary refractive index of BULK particles at DRY relative humidity of 20% and STANDARD temperature and pressure derived from ISARA.'
            output_dict['VariableAttributes']["dry_IRI_unitless"]['ACVSNC_standard_name'] = 'AerOpt_k_InSitu_BluetoRed_RHd_Bulk_STP'
            output_dict['VariableAttributes']['dry_rho_g.cm-3'] = {}
            output_dict['VariableAttributes']["dry_rho_g.cm-3"]['short_name'] = 'dry_rho'
            output_dict['VariableAttributes']["dry_rho_g.cm-3"]['units'] = 'g.cm-3'
            output_dict['VariableAttributes']["dry_rho_g.cm-3"]['long_name'] = 'Effective particle density of BULK particles at DRY relative humidity of 20% and STANDARD temperature and pressure derived from ISARA.'
            output_dict['VariableAttributes']["dry_rho_g.cm-3"]['ACVSNC_standard_name'] = 'none'    
            output_dict['VariableAttributes']['peak_diameter_um'] = {}
            output_dict['VariableAttributes']["peak_diameter_um"]['short_name'] = 'peak_diameter'
            output_dict['VariableAttributes']["peak_diameter_um"]['units'] = 'um'
            output_dict['VariableAttributes']["peak_diameter_um"]['long_name'] = 'Peak dry diameter of APS size distribution.'
            output_dict['VariableAttributes']["peak_diameter_um"]['ACVSNC_standard_name'] = 'none'   
            output_dict['VariableAttributes']['attempt_flag_CRI_unitless'] = {}
            output_dict['VariableAttributes']["attempt_flag_CRI_unitless"]['short_name'] = 'attempt_flag_CRI'
            output_dict['VariableAttributes']["attempt_flag_CRI_unitless"]['long_name'] = 'Flags points where all measurements required for ISARA CRI retrieval and whether or not CRI was successfully retrieved.'
            output_dict['VariableAttributes']["attempt_flag_CRI_unitless"]['flag_values'] = '0 1 2'   
            output_dict['VariableAttributes']["attempt_flag_CRI_unitless"]['flag_meanings'] = 'no_attempt attempt success' 
            output_dict['VariableAttributes']['attempt_flag_kappa_unitless'] = {}
            output_dict['VariableAttributes']["attempt_flag_kappa_unitless"]['short_name'] = 'attempt_flag_kappa'
            output_dict['VariableAttributes']["attempt_flag_kappa_unitless"]['long_name'] = 'Flags points where all measurements required for ISARA CRI and kappa retrieval and whether or not kappa was successfully retrieved.'
            output_dict['VariableAttributes']["attempt_flag_kappa_unitless"]['flag_values'] = '0 1 2'   
            output_dict['VariableAttributes']["attempt_flag_kappa_unitless"]['flag_meanings'] = 'no_attempt attempt success' 
            output_dict['VariableAttributes']["cal_fRH_550_unitless"] = {}    
            output_dict['VariableAttributes']["cal_fRH_550_unitless"]['short_name'] = 'cal_fRH'
            output_dict['VariableAttributes']["cal_fRH_550_unitless"]['units'] = '1'
            output_dict['VariableAttributes']["cal_fRH_550_unitless"]['long_name'] = 'Optical hygrsocopic growth factor at 550 nm of BULK particles derived from ISARA.'
            output_dict['VariableAttributes']["cal_fRH_550_unitless"]['ACVSNC_standard_name'] = 'AerOpt_fRHScat_InSitu_Green_RHd_Bulk_None'
            output_dict['VariableAttributes']["meas_fRH_550_unitless"] = {}    
            output_dict['VariableAttributes']["meas_fRH_550_unitless"]['short_name'] = 'meas_fRH'
            output_dict['VariableAttributes']["meas_fRH_550_unitless"]['units'] = '1'
            output_dict['VariableAttributes']["meas_fRH_550_unitless"]['long_name'] = 'Optical hygrsocopic growth factor at 550 nm of BULK particles derived from gamma measurement.'
            output_dict['VariableAttributes']["meas_fRH_550_unitless"]['ACVSNC_standard_name'] = 'AerOpt_fRHScat_InSitu_Green_RHd_Bulk_None'
            output_dict['VariableAttributes']["kappa_unitless"] = {}
            output_dict['VariableAttributes']["kappa_unitless"]['short_name'] = 'kappa'
            output_dict['VariableAttributes']["kappa_unitless"]['units'] = '1'
            output_dict['VariableAttributes']["kappa_unitless"]['long_name'] = 'Hygroscopicity of BULK particles derived from ISARA.'
            output_dict['VariableAttributes']["kappa_unitless"]['ACVSNC_standard_name'] = 'AerMP_gRH_InSitu_None_Optical_Bulk_None' 
            for i2 in range(len(full_wvl["Sc"])):
                sc_wvl = full_wvl["Sc"][i2]
                abs_wvl = full_wvl["Abs"][i2]
                color_dry_wvl = dry_channel_color[i2]
                output_dict['VariableAttributes'][f'dry_meas_sca_coef_{sc_wvl}_m-1'] = {}
                output_dict['VariableAttributes'][f'dry_meas_sca_coef_{sc_wvl}_m-1']['short_name'] = f'dry_meas_sca_coef_{sc_wvl}'
                output_dict['VariableAttributes'][f'dry_meas_sca_coef_{sc_wvl}_m-1']['units'] = 'm-1'
                output_dict['VariableAttributes'][f'dry_meas_sca_coef_{sc_wvl}_m-1']['long_name'] = f'Scattering coefficient at {sc_wvl} nm of BULK particles at DRY relative humidity of 20% and STANDARD temperature and pressure derived from gamma and scattering measurement at specified relative humidity.'
                output_dict['VariableAttributes'][f'dry_meas_sca_coef_{sc_wvl}_m-1']['ACVSNC_standard_name'] = f'AerOpt_Scattering_InSitu_{color_dry_wvl}_RHd_Bulk_STP'
                output_dict['VariableAttributes'][f'dry_meas_abs_coef_{abs_wvl}_m-1'] = {}
                output_dict['VariableAttributes'][f'dry_meas_abs_coef_{abs_wvl}_m-1']['short_name'] = f'dry_meas_abs_coef_{abs_wvl}'
                output_dict['VariableAttributes'][f'dry_meas_abs_coef_{abs_wvl}_m-1']['units'] = 'm-1'
                output_dict['VariableAttributes'][f'dry_meas_abs_coef_{abs_wvl}_m-1']['long_name'] = f'Absorption coefficient at {abs_wvl} nm of BULK particles at DRY relative humidity of 20% and STANDARD temperature and pressure derived from absorption measurement.'
                output_dict['VariableAttributes'][f'dry_meas_abs_coef_{abs_wvl}_m-1']['ACVSNC_standard_name'] = f'AerOpt_Absorption_InSitu_{color_dry_wvl}_RHd_Bulk_STP'
                output_dict['VariableAttributes'][f'dry_cal_sca_coef_{sc_wvl}_m-1'] = {}
                output_dict['VariableAttributes'][f'dry_cal_sca_coef_{sc_wvl}_m-1']['short_name'] = f'dry_cal_sca_coef_{sc_wvl}'
                output_dict['VariableAttributes'][f'dry_cal_sca_coef_{sc_wvl}_m-1']['units'] = 'm-1'
                output_dict['VariableAttributes'][f'dry_cal_sca_coef_{sc_wvl}_m-1']['long_name'] = f'Scattering coefficient at {sc_wvl} nm of BULK particles at DRY relative humidity of 20% and STANDARD temperature and pressure derived from ISARA.'
                output_dict['VariableAttributes'][f'dry_cal_sca_coef_{sc_wvl}_m-1']['ACVSNC_standard_name'] = f'AerOpt_Scattering_InSitu_{color_dry_wvl}_RHd_Bulk_STP'
                output_dict['VariableAttributes'][f'dry_cal_abs_coef_{abs_wvl}_m-1'] = {}
                output_dict['VariableAttributes'][f'dry_cal_abs_coef_{abs_wvl}_m-1']['short_name'] = f'dry_cal_abs_coef_{abs_wvl}'
                output_dict['VariableAttributes'][f'dry_cal_abs_coef_{abs_wvl}_m-1']['units'] = 'm-1'
                output_dict['VariableAttributes'][f'dry_cal_abs_coef_{abs_wvl}_m-1']['long_name'] = f'Absorption coefficient at {abs_wvl} nm of BULK particles at DRY relative humidity of 20% and STANDARD temperature and pressure derived from ISARA.'
                output_dict['VariableAttributes'][f'dry_cal_abs_coef_{abs_wvl}_m-1']['ACVSNC_standard_name'] = f'AerOpt_Absorption_InSitu_{color_dry_wvl}_RHd_Bulk_STP'
                output_dict['VariableAttributes'][f'dry_cal_SSA_{sc_wvl}_unitless'] = {}
                output_dict['VariableAttributes'][f'dry_cal_SSA_{sc_wvl}_unitless']['short_name'] = f'dry_cal_SSA_{sc_wvl}'
                output_dict['VariableAttributes'][f'dry_cal_SSA_{sc_wvl}_unitless']['units'] = '1'
                output_dict['VariableAttributes'][f'dry_cal_SSA_{sc_wvl}_unitless']['long_name'] = f'Single scattering albedo at {sc_wvl} nm of BULK particles at DRY relative humidity of 20% and STANDARD temperature and pressure derived from ISARA.'
                output_dict['VariableAttributes'][f'dry_cal_SSA_{sc_wvl}_unitless']['ACVSNC_standard_name'] = f'AerOpt_SSA_InSitu_{color_dry_wvl}_RHd_Bulk_None'
                output_dict['VariableAttributes'][f'dry_cal_SSA_{abs_wvl}_unitless'] = {}
                output_dict['VariableAttributes'][f'dry_cal_SSA_{abs_wvl}_unitless']['short_name'] = f'dry_cal_SSA_{abs_wvl}'
                output_dict['VariableAttributes'][f'dry_cal_SSA_{abs_wvl}_unitless']['units'] = '1'
                output_dict['VariableAttributes'][f'dry_cal_SSA_{abs_wvl}_unitless']['long_name'] = f'Single scattering albedo at {abs_wvl} nm of BULK particles at DRY relative humidity of 20% derived from ISARA.'
                output_dict['VariableAttributes'][f'dry_cal_SSA_{abs_wvl}_unitless']['ACVSNC_standard_name'] = f'AerOpt_SSA_InSitu_{color_dry_wvl}_RHd_Bulk_None'
                output_dict['VariableAttributes'][f'dry_cal_ext_coef_{sc_wvl}_m-1'] = {}
                output_dict['VariableAttributes'][f'dry_cal_ext_coef_{sc_wvl}_m-1']['short_name'] = f'dry_cal_ext_coef_{sc_wvl}'
                output_dict['VariableAttributes'][f'dry_cal_ext_coef_{sc_wvl}_m-1']['units'] = 'm-1'
                output_dict['VariableAttributes'][f'dry_cal_ext_coef_{sc_wvl}_m-1']['long_name'] = f'Extinction coefficient at {sc_wvl} nm of BULK particles at DRY relative humidity of 20% and STANDARD temperature and pressure derived from ISARA.'
                output_dict['VariableAttributes'][f'dry_cal_ext_coef_{sc_wvl}_m-1']['ACVSNC_standard_name'] = f'AerOpt_Extinction_InSitu_{color_dry_wvl}_RHd_Bulk_STP'
                output_dict['VariableAttributes'][f'dry_cal_ext_coef_{abs_wvl}_m-1'] = {}
                output_dict['VariableAttributes'][f'dry_cal_ext_coef_{abs_wvl}_m-1']['short_name'] = f'dry_cal_ext_coef_{abs_wvl}'
                output_dict['VariableAttributes'][f'dry_cal_ext_coef_{abs_wvl}_m-1']['units'] = 'm-1'
                output_dict['VariableAttributes'][f'dry_cal_ext_coef_{abs_wvl}_m-1']['long_name'] = f'Extinction coefficient at {abs_wvl} nm of BULK particles at DRY relative humidity of 20% and STANDARD temperature and pressure derived from ISARA.'
                output_dict['VariableAttributes'][f'dry_cal_ext_coef_{abs_wvl}_m-1']['ACVSNC_standard_name'] = f'AerOpt_Extinction_InSitu_{color_dry_wvl}_RHd_Bulk_STP'          

            for i2 in range(len(full_wvl2["Sc"])):
                wet_wvl = full_wvl2["Sc"][i2]
                color_wet_wvl = wet_channel_color[i2]
                output_dict['VariableAttributes'][f'wet_meas_sca_coef_{wet_wvl}_m-1'] = {}
                output_dict['VariableAttributes'][f'wet_meas_sca_coef_{wet_wvl}_m-1']['short_name'] = f'wet_meas_sca_coef_{wet_wvl}'
                output_dict['VariableAttributes'][f'wet_meas_sca_coef_{wet_wvl}_m-1']['units'] = 'm-1'
                output_dict['VariableAttributes'][f'wet_meas_sca_coef_{wet_wvl}_m-1']['long_name'] = f'Scattering coefficient at {wet_wvl} nm of BULK particles at WET relative humidity of 80% and STANDARD temperature and pressure derived from gamma and scattering measurement at specified relative humidity.'
                output_dict['VariableAttributes'][f'wet_meas_sca_coef_{wet_wvl}_m-1']['ACVSNC_standard_name'] = f'AerOpt_Scattering_InSitu_{color_wet_wvl}_RHsp_Bulk_STP'
                output_dict['VariableAttributes'][f'wet_meas_ext_coef_{wet_wvl}_m-1'] = {}
                output_dict['VariableAttributes'][f'wet_meas_ext_coef_{wet_wvl}_m-1']['short_name'] = f'wet_meas_ext_coef_{wet_wvl}'
                output_dict['VariableAttributes'][f'wet_meas_ext_coef_{wet_wvl}_m-1']['units'] = 'm-1'
                output_dict['VariableAttributes'][f'wet_meas_ext_coef_{wet_wvl}_m-1']['long_name'] = f'Extinction coefficient at {wet_wvl} nm of BULK particles at WET relative humidity of 80% and STANDARD temperature and pressure derived from humidified scattering and dry absorption.'
                output_dict['VariableAttributes'][f'wet_meas_ext_coef_{wet_wvl}_m-1']['ACVSNC_standard_name'] = f'AerOpt_Extinction_InSitu_{color_wet_wvl}_RHsp_Bulk_STP'
                output_dict['VariableAttributes'][f'wet_cal_sca_coef_{wet_wvl}_m-1'] = {}
                output_dict['VariableAttributes'][f'wet_cal_sca_coef_{wet_wvl}_m-1']['short_name'] = f'wet_cal_sca_coef_{wet_wvl}'
                output_dict['VariableAttributes'][f'wet_cal_sca_coef_{wet_wvl}_m-1']['units'] = 'm-1'
                output_dict['VariableAttributes'][f'wet_cal_sca_coef_{wet_wvl}_m-1']['long_name'] = f'Scattering coefficient at {wet_wvl} nm of BULK particles at WET relative humidity of 80% and STANDARD temperature and pressure derived derived from ISARA.'
                output_dict['VariableAttributes'][f'wet_cal_sca_coef_{wet_wvl}_m-1']['ACVSNC_standard_name'] = f'AerOpt_Scattering_InSitu_{color_wet_wvl}_RHsp_Bulk_STP'
                output_dict['VariableAttributes'][f'wet_cal_SSA_{wet_wvl}_unitless'] = {}
                output_dict['VariableAttributes'][f'wet_cal_SSA_{wet_wvl}_unitless']['short_name'] = f'wet_cal_SSA_{wet_wvl}'
                output_dict['VariableAttributes'][f'wet_cal_SSA_{wet_wvl}_unitless']['units'] = '1'
                output_dict['VariableAttributes'][f'wet_cal_SSA_{wet_wvl}_unitless']['long_name'] = f'Single scattering albedo at {wet_wvl} nm of BULK particles at WET relative humidity of 80% and STANDARD temperature and pressure derived derived from ISARA.'
                output_dict['VariableAttributes'][f'wet_cal_SSA_{wet_wvl}_unitless']['ACVSNC_standard_name'] = f'AerOpt_SSA_InSitu_{color_wet_wvl}_RHsp_Bulk_None'
                output_dict['VariableAttributes'][f'wet_cal_ext_coef_{wet_wvl}_m-1'] = {}
                output_dict['VariableAttributes'][f'wet_cal_ext_coef_{wet_wvl}_m-1']['short_name'] = f'wet_cal_ext_coef_{wet_wvl}'
                output_dict['VariableAttributes'][f'wet_cal_ext_coef_{wet_wvl}_m-1']['units'] = 'm-1'
                output_dict['VariableAttributes'][f'wet_cal_ext_coef_{wet_wvl}_m-1']['long_name'] = f'Extinction coefficient at {wet_wvl} nm of BULK particles at WET relative humidity of 80% and STANDARD temperature and pressure derived from ISARA.'
                output_dict['VariableAttributes'][f'wet_cal_ext_coef_{wet_wvl}_m-1']['ACVSNC_standard_name'] = f'AerOpt_Extinction_InSitu_{color_wet_wvl}_RHsp_Bulk_STP'

            if val_wvl is not None:
                for i2 in range(len(val_wvl)):
                    valwvl = val_wvl[i2]
                    color_val_wvl = val_channel_color[i2]
                    output_dict['VariableAttributes'][f'dry_cal_sca_coef_{valwvl}_m-1'] = {}
                    output_dict['VariableAttributes'][f'dry_cal_sca_coef_{valwvl}_m-1']['short_name'] = f'dry_cal_sca_coef_{valwvl}'
                    output_dict['VariableAttributes'][f'dry_cal_sca_coef_{valwvl}_m-1']['units'] = 'm-1'
                    output_dict['VariableAttributes'][f'dry_cal_sca_coef_{valwvl}_m-1']['long_name'] =  f'Scattering coefficient at {valwvl} nm of BULK particles at DRY relative humidity of 20% and STANDARD temperature and pressure derived from ISARA.'
                    output_dict['VariableAttributes'][f'dry_cal_sca_coef_{valwvl}_m-1']['ACVSNC_standard_name'] = f'AerOpt_Scattering_InSitu_{color_val_wvl}_RHd_Bulk_STP'
                    output_dict['VariableAttributes'][f'dry_cal_SSA_{valwvl}_unitless'] = {}
                    output_dict['VariableAttributes'][f'dry_cal_SSA_{valwvl}_unitless']['short_name'] = f'dry_cal_SSA_{valwvl}'
                    output_dict['VariableAttributes'][f'dry_cal_SSA_{valwvl}_unitless']['units'] = '1'
                    output_dict['VariableAttributes'][f'dry_cal_SSA_{valwvl}_unitless']['long_name'] = f'Single scattering albedo at {valwvl} nm of BULK particles at DRY relative humidity of 20% and STANDARD temperature and pressure derived from ISARA.'
                    output_dict['VariableAttributes'][f'dry_cal_SSA_{valwvl}_unitless']['ACVSNC_standard_name'] = f'AerOpt_SSA_InSitu_{color_val_wvl}_RHd_Bulk_None'
                    output_dict['VariableAttributes'][f'dry_cal_ext_coef_{valwvl}_m-1'] = {}
                    output_dict['VariableAttributes'][f'dry_cal_ext_coef_{valwvl}_m-1']['short_name'] = f'dry_cal_ext_coef_{valwvl}'
                    output_dict['VariableAttributes'][f'dry_cal_ext_coef_{valwvl}_m-1']['units'] = 'm-1'
                    output_dict['VariableAttributes'][f'dry_cal_ext_coef_{valwvl}_m-1']['long_name'] = f'Extinction coefficient at {valwvl} nm of BULK particles at DRY relative humidity of 20% and STANDARD temperature and pressure derived from ISARA.'
                    output_dict['VariableAttributes'][f'dry_cal_ext_coef_{valwvl}_m-1']['ACVSNC_standard_name'] = f'AerOpt_Extinction_InSitu_{color_val_wvl}_RHd_Bulk_STP'
                    output_dict['VariableAttributes'][f'wet_cal_sca_coef_{valwvl}_m-1'] = {}
                    output_dict['VariableAttributes'][f'wet_cal_sca_coef_{valwvl}_m-1']['short_name'] = f'wet_cal_sca_coef_{valwvl}'
                    output_dict['VariableAttributes'][f'wet_cal_sca_coef_{valwvl}_m-1']['units'] = 'm-1'
                    output_dict['VariableAttributes'][f'wet_cal_sca_coef_{valwvl}_m-1']['long_name'] = f'Scattering coefficient at {valwvl} nm of BULK particles at WET relative humidity of 80% and STANDARD temperature and pressure derived derived from ISARA.'
                    output_dict['VariableAttributes'][f'wet_cal_sca_coef_{valwvl}_m-1']['ACVSNC_standard_name'] = f'AerOpt_Scattering_InSitu_{color_val_wvl}_RHsp_Bulk_STP'
                    output_dict['VariableAttributes'][f'wet_cal_SSA_{valwvl}_unitless'] = {}
                    output_dict['VariableAttributes'][f'wet_cal_SSA_{valwvl}_unitless']['short_name'] = f'wet_cal_SSA_{valwvl}'
                    output_dict['VariableAttributes'][f'wet_cal_SSA_{valwvl}_unitless']['units'] = '1'
                    output_dict['VariableAttributes'][f'wet_cal_SSA_{valwvl}_unitless']['long_name'] = f'Single scattering albedo at {valwvl} nm of BULK particles at WET relative humidity of 80% and STANDARD temperature and pressure derived derived from ISARA.'
                    output_dict['VariableAttributes'][f'wet_cal_SSA_{valwvl}_unitless']['ACVSNC_standard_name'] = f'AerOpt_SSA_InSitu_{color_val_wvl}_RHsp_Bulk_None'
                    output_dict['VariableAttributes'][f'wet_cal_ext_coef_{valwvl}_m-1'] = {}
                    output_dict['VariableAttributes'][f'wet_cal_ext_coef_{valwvl}_m-1']['short_name'] = f'wet_cal_ext_coef_{valwvl}'
                    output_dict['VariableAttributes'][f'wet_cal_ext_coef_{valwvl}_m-1']['units'] = 'm-1'
                    output_dict['VariableAttributes'][f'wet_cal_ext_coef_{valwvl}_m-1']['long_name'] = f'Extinction coefficient at {valwvl} nm of BULK particles at WET relative humidity of 80% and STANDARD temperature and pressure derived from ISARA.'
                    output_dict['VariableAttributes'][f'wet_cal_ext_coef_{valwvl}_m-1']['ACVSNC_standard_name'] = f'AerOpt_Extinction_InSitu_{color_val_wvl}_RHsp_Bulk_STP'  

            # Loop through each of the rows here using multiprocessing. This will split the rows across multiple different cores. Each row will be its own index in `line_data` 
            # with a tuple full of information. So, for instance, line_data[0] will contain (CRI_dry, CalCoef_dry, meas_coef_dry, Kappa, CalCoef_wet, meas_coef_wet, results) 
            # for the first line of data
            line_data = pool.map(
                # This is a pain, I know, but all the data has to be cloned and accessible within each worker
                handle_line(modelist, sd, dpg, dpu, dpl, UBcutoff, LBcutoff, Sc, Abs, RHsc, gamma, 
                            full_wvl, full_wvl2, val_wvl, size_equ, CRI_p, nonabs_fraction, shape,
                            kappa_p, num_theta, rho_wet, path_optical_dataset, path_mopsmap_executable, full_dp),
                range(L1),
            )

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
                        output_dict['SourceFlag'][key2] = 'derived' 
                        output_dict['Dims'][key2] = 'time'
                        if key2 in output_dict['VariableAttributes']:
                            output_dict['VariableAttributes'][key2]['_FillValue'] = np.nan
                        else:
                            output_dict['VariableAttributes'][key2] = {}
                            output_dict['VariableAttributes'][key2]['_FillValue'] = np.nan
                        output_dict[key2][i1] = results_line[key2]
            print(output_dict["kappa_unitless"].size)           
            output_filename = np.array(input_filename.split('.ict'))
            output_filename = output_filename[0]
            np.save(f'{output_filename}.npy', output_dict)  

    # Close the pool to any new jobs and remove it
    pool.close()
    pool.clear()
