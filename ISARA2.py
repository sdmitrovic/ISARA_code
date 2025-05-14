import mopsmap_wrapper
MMModel = mopsmap_wrapper.Model
import numpy as np

def Retr_CRI(wvl_dict, 
  val_wvl, 
  optical_measurements,
  sd,
  dpg,
  CRI_p,
  size_equ,
  nonabs_fraction,
  shape,
  rho,
  num_theta,
  path_optical_dataset, 
  path_mopsmap_executable,
):

  """
  Returns aerosol particle real and imaginary refractive index from three scattering coefficeint measurements, three absorption coefficient measurements, a measured number concentration for an aerosol size distribution. WARNINGS: 1) numpy must be installed to the python environment 2) mopsmap_wrapper.py must be present in a directory that is in your PATH
  :param wvl_dict: Dictionary of wavelengths associated with each of the scattering and absorption measurements
  :type wvl_dict: numpy dictionary
  :param val_wvl: Dictionary of wavelengths associated with validation measurements
  :type val_wvl: numpy dictionary
  :param optical_measurements: Dictionary containing measured dry scattering and absorption coefficients in m^-1; NOTE: There should be one key per channel (e.g., optical_measurements['dry_meas_sca_coef_450_m-1'], optical_measurements['dry_meas_abs_coef_470_m-1'], etc.)
  :type optical_measurements: numpy dictionary       
  :param sd: Dictionary containing the modal size resolved number concentrations in m^-3; NOTE: there should be one key for each measurement mode
  :type sd: numpy dictionary  
  :param dpg: Dictionary containing the modal geometric mean particle diameters of each size bin in nm; NOTE: there should be one key for each measurement mode
  :type dpg: numpy dictionary      
  :param CRI_p: 2-D array containing the prescribed RRI and IRI range to be searched
  :type CRI_p: numpy array
  :param nonabs_fraction: Dictionary of integers indicating the desired non-absorbing fraction for each size mode; NOTE: there should be one key for each measurement mode
  :type nonabs_fraction: numpy dictionary
  :param shape: String indicating the desired particle shape(s) for each size mode; NOTE: there should be one key for each measurement mode
  :type shape: numpy dictionary
  :param rho: Double indicating the desired particle density in g m^-3 for each size mode; NOTE: there should be one key for each measurement mode
  :type rho: numpy dictionary 
  :param num_theta: Integer indicating the number of phase function angles to provide
  :type num_theta: numpy int     
  :param path_optical_dataset: String indicating the path for the optical dataset required for MOPSMAP
  :type path_optical_dataset: str
  :param path_mopsmap_executable: String indicating the path for the mopsmap.exe file
  :type path_mopsmap_executable: str                                  
  :return: Dictionary (Results) with the retrieved complex refractive index, calculated scattering and absorption coefficients in native measurements, and calculated single scattering albedo and extinction coefficients in measured and validation wavelengths
  :rtype: numpy dictionary
  """

  L1 = len(CRI_p[:,0]) # length of array with all possible cri values
  L2 = len(wvl_dict["Sc"]) # number of measured scattering (Sc) coefficient channels
  ## collect scattering and absorption (Abs) coefficient channel wavelengths (wvl) into array
  wvl = None
  for iwvl in range(L2):
    if iwvl == 0:
      wvl = np.array([wvl_dict["Sc"][iwvl],wvl_dict["Abs"][iwvl]])
    else:
      wvl = np.hstack((wvl,np.array([wvl_dict["Sc"][iwvl],wvl_dict["Abs"][iwvl]])))
  ##
  wvl = np.sort(wvl, axis=None) # sort array of wavelengths in ascending order
  ## Prepare output arrays and dictionary
  iri = np.full((L1), np.nan)
  rri = np.full((L1), np.nan)
  Results = dict()  
  Results["dry_RRI_unitless"] = None
  Results["dry_IRI_unitless"] = None
  for i2 in range(L2):
    Results[f'dry_cal_sca_coef_{wvl_dict["Sc"][i2]}_m-1'] = None
    Results[f'dry_cal_abs_coef_{wvl_dict["Abs"][i2]}_m-1'] = None
    Results[f'dry_cal_SSA_{wvl_dict["Sc"][i2]}_unitless'] = None
    Results[f'dry_cal_SSA_{wvl_dict["Abs"][i2]}_unitless'] = None
    Results[f'dry_cal_ext_coef_{wvl_dict["Sc"][i2]}_m-1'] = None
    Results[f'dry_cal_ext_coef_{wvl_dict["Abs"][i2]}_m-1'] = None
  ##
  for i1 in range(L1): # initiate loop through possible cri values
    ## assign the rri and iri for to each size mode.
    RRI_p = {}
    IRI_p = {}
    for imode in sd:
      RRI_p[imode] = CRI_p[i1,0]
      IRI_p[imode] = CRI_p[i1,1]
    ##  
    results = MMModel(wvl,size_equ,sd,dpg,RRI_p,IRI_p,nonabs_fraction,shape,rho,0,0,num_theta,path_optical_dataset,path_mopsmap_executable) # calculate microphysical properties for a given cri
    ref_scat_coef = np.full((L2),np.nan)## prepare arrays of measured scattering and absorption coefficients
    ref_abs_coef = np.full((L2),np.nan)
    scat_coef = np.full((L2),np.nan)## prepare arrays of calculated scattering and absorption coefficients
    abs_coef = np.full((L2),np.nan)
    ## Assign values to prepared measured and calculated coefficients
    for i2 in range(L2):
      scat_coef[i2] = results[f'ssa_{wvl_dict["Sc"][i2]}']*results[f'ext_coeff_{wvl_dict["Sc"][i2]}_m-1']
      abs_coef[i2] = results[f'ext_coeff_{wvl_dict["Abs"][i2]}_m-1']-results[f'ssa_{wvl_dict["Abs"][i2]}']*results[f'ext_coeff_{wvl_dict["Abs"][i2]}_m-1'] 
      ref_scat_coef[i2] = optical_measurements[f'dry_meas_sca_coef_{wvl_dict["Sc"][i2]}_m-1']
      ref_abs_coef[i2] = optical_measurements[f'dry_meas_abs_coef_{wvl_dict["Abs"][i2]}_m-1']
    ##
    Cdif1 = abs(ref_scat_coef-scat_coef)/ref_scat_coef # calculate absolute relative difference of scattering coefficients in each channel
    Cdif2 = abs(ref_abs_coef-abs_coef)# calculate absolute difference of absoprtion coefficients in each channel
    ## check if relative difference in scattering coefficient is within 20% for all and channels that the difference in absorption coefficient is within 1 Mm-1 for all channels
    a1 = ((Cdif1)<0.2).astype('int')
    a1[np.isinf(a1)]=0
    a2 = ((Cdif2)<pow(10,-6)).astype('int')#
    if np.sum(a1)==3 & np.sum(a2)==3:
      iri[i1] = CRI_p[i1,1]
      rri[i1] = CRI_p[i1,0]
    ##    
  flgs = np.logical_not(np.isnan(rri)) # flag valid solutions
  if np.sum(rri[flgs])>0: # check to see if any valid solutions exist 
    ## take mean rri and iri of all valid solutions and recalculate aerosol properties with mean cri values.    
    rri = np.mean(rri[flgs]) 
    iri = np.mean(iri[flgs])
    RRI_d = {}
    IRI_d = {}
    for imode in sd:
      RRI_d[imode] = rri
      IRI_d[imode] = iri
    results = MMModel(wvl,size_equ,sd,dpg,RRI_d,IRI_d,nonabs_fraction,shape,rho,0,0,num_theta,path_optical_dataset,path_mopsmap_executable) 
    ##
    ## same as before, check for to ensure recalculated scattering coefficients are within 20% and absorption coefficients are with 1 Mm-1 when using mean cri
    scat_coef = np.full((L2),np.nan)
    abs_coef = np.full((L2),np.nan)
    for i2 in range(L2):
      scat_coef[i2] = results[f'ssa_{wvl_dict["Sc"][i2]}']*results[f'ext_coeff_{wvl_dict["Sc"][i2]}_m-1']
      abs_coef[i2] = results[f'ext_coeff_{wvl_dict["Abs"][i2]}_m-1']-results[f'ssa_{wvl_dict["Abs"][i2]}']*results[f'ext_coeff_{wvl_dict["Abs"][i2]}_m-1'] 
    Cdif1 = abs(ref_scat_coef-scat_coef)/ref_scat_coef
    Cdif2 = abs(ref_abs_coef-abs_coef)
    a1 = ((Cdif1)<0.2).astype('int')
    a1[np.isinf(a1)]=0
    a2 = ((Cdif2)<pow(10,-6)).astype('int')
    ##
    if np.sum(a1)==L2 & np.sum(a2)==L2: # if solution is valid, store dry cri and dry calculated extinction, scattering, and absorption coefficients and SSA in all measured wavelengths
      Results["dry_RRI_unitless"] = rri
      Results["dry_IRI_unitless"] = iri
      for i2 in range(L2):
        Results[f'dry_cal_sca_coef_{wvl_dict["Sc"][i2]}_m-1'] = results[f'ssa_{wvl_dict["Sc"][i2]}']*results[f'ext_coeff_{wvl_dict["Sc"][i2]}_m-1']
        Results[f'dry_cal_abs_coef_{wvl_dict["Abs"][i2]}_m-1'] = results[f'ext_coeff_{wvl_dict["Abs"][i2]}_m-1']-results[f'ssa_{wvl_dict["Abs"][i2]}']*results[f'ext_coeff_{wvl_dict["Abs"][i2]}_m-1'] 
        Results[f'dry_cal_SSA_{wvl_dict["Sc"][i2]}_unitless'] = results[f'ssa_{wvl_dict["Sc"][i2]}']
        Results[f'dry_cal_SSA_{wvl_dict["Abs"][i2]}_unitless'] = results[f'ssa_{wvl_dict["Abs"][i2]}']
        Results[f'dry_cal_ext_coef_{wvl_dict["Sc"][i2]}_m-1'] = results[f'ext_coeff_{wvl_dict["Sc"][i2]}_m-1']
        Results[f'dry_cal_ext_coef_{wvl_dict["Abs"][i2]}_m-1'] = results[f'ext_coeff_{wvl_dict["Abs"][i2]}_m-1']
      if val_wvl is not None: # if validation wavelengths are requested, provide outputs for those wavelengths as well
        wvl2 = None
        for iwvl in range(len(val_wvl)):
          if iwvl == 0:
            wvl2 = val_wvl
          else:
            wvl2 = np.hstack((wvl2,val_wvl))
        results = MMModel(wvl2,size_equ,sd,dpg,RRI_d,IRI_d,nonabs_fraction,shape,rho,0,0,num_theta,path_optical_dataset,path_mopsmap_executable) 
        for iwvl in range(len(val_wvl)):
          Results[f'dry_cal_sca_coef_{val_wvl[iwvl]}_m-1'] = results[f'ssa_{val_wvl[iwvl]}']*results[f'ext_coeff_{val_wvl[iwvl]}_m-1']
          Results[f'dry_cal_SSA_{val_wvl[iwvl]}_unitless'] = results[f'ssa_{val_wvl[iwvl]}']
          Results[f'dry_cal_ext_coef_{val_wvl[iwvl]}_m-1'] = results[f'ext_coeff_{val_wvl[iwvl]}_m-1']        
    
  return Results # return dictionary (Results) of dry cri and dry calculated extinction, scattering, and absorption coefficients and SSA in all measured and validation wavelengths

def Retr_kappa(wvl_dict,
  val_wvl, 
  optical_measurements,
  sd,
  dpg,
  RH,
  kappa_p,
  CRI_d,
  size_equ,
  nonabs_fraction,
  shape,
  rho,
  num_theta,
  path_optical_dataset, 
  path_mopsmap_executable,
):
  """
  Returns aerosol particle hygroscopic growth factor from a humdified scattering coefficeint measurement, dry complex refractive index, and a measured number concentration for an aerosol size distribution. WARNINGS: 1) numpy must be installed to the python environment 2) mopsmap_wrapper.py must be present in a directory that is in your PATH.
  :param wvl_dict: Dictionary of wavelengths associated with each of the scattering and absorption measurements
  :type wvl_dict: numpy dictionary
  :param val_wvl: Dictionary of wavelengths associated with validation measurements
  :type val_wvl: numpy dictionary
  :param optical_measurements: Dictionary containing measured dry scattering and absorption coefficients in m^-1; NOTE: There should be one key per channel (e.g., optical_measurements['wet_meas_sca_coef_450_m-1'] etc.)
  :type optical_measurements: numpy dictionary       
  :param sd: Dictionary containing the modal size resolved number concentrations in m^-3; NOTE: there should be one key for each measurement mode
  :type sd: numpy dictionary  
  :param dpg: Dictionary containing the modal geometric mean particle diameters of each size bin in nm; NOTE: there should be one key for each measurement mode
  :type dpg: numpy dictionary   
  :param RH: Array containing the percent relative humidity associated with the measured humidified scattering coefficients
  :type RH: int   
  :param kappa_p: Array containing the desired kappa range to be searched.
  :type kappa_p: numpy array    
  :param CRI_d: Array containing the desired dry RRI and IRI.
  :type CRI_d: numpy array
  :param nonabs_fraction: Dictionary of integers indicating the desired non-absorbing fraction for each size mode; NOTE: there should be one key for each measurement mode
  :type nonabs_fraction: numpy dictionary
  :param shape: String indicating the desired particle shape(s) for each size mode; NOTE: there should be one key for each measurement mode
  :type shape: numpy dictionary
  :param rho: Double indicating the desired particle density in g m^-3 for each size mode; NOTE: there should be one key for each measurement mode
  :type rho: numpy dictionary 
  :param num_theta: Integer indicating the number of phase function angles to provide
  :type num_theta: int       
  :param path_optical_dataset: String indicating the path for the optical dataset required for MOPSMAP
  :type path_optical_dataset: str
  :param path_mopsmap_executable: String indicating the path for the mopsmap.exe file
  :type path_mopsmap_executable: str                                         
  :return: Real refractive index, imaginary refractive index, calculated scattering and absorption coefficients in native measurements, and calculated single scattering albedo and extinction coefficients in all wavelengths
  :rtype: numpy dictionary
  """

  L1 = len(kappa_p) # length of array with all possible kappa values
  L2 = len(wvl_dict["Sc"]) # number of measured scattering (Sc) coefficient channels

  ## collect scattering coefficient channel wavelengths (wvl) into array and sort in ascending order
  wvl = None
  for i1 in range(L2):
    if i1 == 0:
      wvl = np.array([wvl_dict["Sc"][i1]])
    else:
      wvl = np.hstack((wvl,np.array([wvl_dict["Sc"][i1]])))
  wvl = np.sort(wvl, axis=None)
  ##
  ## Prepare output dictionary
  Results = dict()
  Results["kappa_unitless"] = None
  for i2 in range(L2):
    Results[f'wet_cal_sca_coef_{wvl_dict["Sc"][i2]}_m-1'] = None
    Results[f'wet_cal_SSA_{wvl_dict["Sc"][i2]}_unitless'] = None
    Results[f'wet_cal_ext_coef_{wvl_dict["Sc"][i2]}_m-1'] = None
  ##  
  stop_indx = 0 # initate stop index for first valid solution
  RRIw = 1.33 # set rri of water 
  IRIw = 0 # set iri of water 
  for i1 in range(L1): # loop through each possible kappa value 
    dpg_w = {}
    RRI_w = {}
    IRI_w = {}
    for imode in sd:
      gf = np.power((1+kappa_p[i1]*RH/(100-RH)),1/3) # calculate growth factor given the incrimental kappa value and the measurement relative humidity for each size mode
      dpg_w[imode] = np.squeeze(np.multiply(gf,dpg[imode])) # adjust the size distribution by multplying the growth factor by each dry particle diameter in each size mode
      RRI_w[imode] = (CRI_d[0]+((gf**3)-1)*RRIw)/(gf**3) # volume weighted humidified rri for each size mode
      IRI_w[imode] = (CRI_d[1]+((gf**3)-1)*IRIw)/(gf**3) # volume weighted humidified iri for each size mode
    if stop_indx == 0: # stop if last solution was valid (Cdif<0.01)
      results = MMModel(wvl,size_equ,sd,dpg_w,RRI_w,IRI_w,nonabs_fraction,shape,rho,0,0,num_theta,path_optical_dataset,path_mopsmap_executable) # calculate microphysical properties for a given kappa
      scat_coef = np.full((L2),np.nan)# prepare array of calculated scattering coefficients
      ref_scat_coef = np.full((L2),np.nan) # prepare array of measured scattering coefficients
      ## Assign values to prepared measured and calculated coefficients
      for i2 in range(L2):
        scat_coef[i2] = results[f'ssa_{wvl_dict["Sc"][i2]}']*results[f'ext_coeff_{wvl_dict["Sc"][i2]}_m-1']
        ref_scat_coef[i2] = optical_measurements[f'wet_meas_sca_coef_{wvl_dict["Sc"][i2]}_m-1']
      ##  
      Cdif = abs(ref_scat_coef-scat_coef)/ref_scat_coef # calculate absolute relative difference of scattering coefficients in each channel
      if Cdif<0.01: # solution is valid if scattering coefficients are within 1%
        Results["kappa_unitless"] = kappa_p[i1] # store retrieved kappa
        ## store calculated scattering and extinction coefficients and SSA for measured and validation wavelengths
        for i2 in range(L2):
          Results[f'wet_cal_sca_coef_{wvl_dict["Sc"][i2]}_m-1'] = results[f'ssa_{wvl_dict["Sc"][i2]}']*results[f'ext_coeff_{wvl_dict["Sc"][i2]}_m-1'] 
          Results[f'wet_cal_SSA_{wvl_dict["Sc"][i2]}_unitless'] = results[f'ssa_{wvl_dict["Sc"][i2]}']
          Results[f'wet_cal_ext_coef_{wvl_dict["Sc"][i2]}_m-1'] = results[f'ext_coeff_{wvl_dict["Sc"][i2]}_m-1']
        if val_wvl is not None:
          wvl2 = None
          for iwvl in range(len(val_wvl)):
            if iwvl == 0:
              wvl2 = val_wvl
            else:
              wvl2 = np.hstack((wvl2,val_wvl))
          results = MMModel(wvl2,size_equ,sd,dpg_w,RRI_w,IRI_w,nonabs_fraction,shape,rho,0,0,num_theta,path_optical_dataset,path_mopsmap_executable) 
          for iwvl in range(len(val_wvl)):
            Results[f'wet_cal_sca_coef_{val_wvl[iwvl]}_m-1'] = results[f'ssa_{val_wvl[iwvl]}']*results[f'ext_coeff_{val_wvl[iwvl]}_m-1']
            Results[f'wet_cal_SSA_{val_wvl[iwvl]}_unitless'] = results[f'ssa_{val_wvl[iwvl]}']
            Results[f'wet_cal_ext_coef_{val_wvl[iwvl]}_m-1'] = results[f'ext_coeff_{val_wvl[iwvl]}_m-1'] 
        ##        
        stop_indx = 1 # change stop index when first valid solution is reached
  return Results # return dictionary (Results) of kappa and wet calculated extinction, scattering, and absorption coefficients and SSA in all measured and validation wavelengths
