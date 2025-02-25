import mopsmap_wrapper_022425
MMModel = mopsmap_wrapper_022425.Model
import numpy as np

def Retr_CRI(full_wvl,optical_measurements,
  sd,
  dpg,
  CRI_p,
  size_equ,
  nonabs_fraction,
  shape,
  rho,
  num_theta,
  optical_dataset, 
  path_mopsmap_executable,
):

  """
  Returns aerosol particle real and imaginary refractive index from three scattering coefficeint measurements, three absorption coefficient measurements, a measured number concentration for an aerosol size distribution. WARNINGS: 1) numpy must be installed to the python environment 2) mopsmap_SD_run.py must be present in a directory that is in your PATH

  :param wvl: Array of 3 scattering and 3 absorption wavelengths in nm (e.g., 450, 470, 532, 550, 660, 700).
  :type wvl: numpy array
  :raise ISARA.InvalidNumberOfWavelengths: If the length of wvl is not 6.
  :param measured_coef: Array containing the 3 measured scattering coefficients in m^-1.
  :type measured_coef: numpy array  
  :param measured_coef: Array containing the 3 measured absorption coefficients in m^-1.
  :type measured_coef: numpy array      
  :param sd: Array containing the fist mode number concentrations in m^-3.
  :type sd: numpy array  
  :param dpg: Array containing the fist mode particle diameters in nm.
  :type dpg: numpy array      
  :param CRI_p: 2-D array containing the prescribed RRI and IRI range to be searched.
  :type CRI_p: numpy array
  :param nonabs_fraction: Integer indicating the desired non-absorbing fraction for first mode.
  :type nonabs_fraction: int
  :param shape: String indicating the desired particle shape(s) for first mode.
  :type shape: str
  :param rho: Double indicating the desired particle density in g m^-3 for first mode.
  :type rho: double  
  :param num_theta: Integer indicating the number of phase function angles to provide.
  :type num_theta: int       
  :param optical_dataset: String indicating the path for the optical dataset required for MOPSMAP.
  :type optical_dataset: str
  :param path_mopsmap_executable: String indicating the path for the mopsmap.exe file.
  :type path_mopsmap_executable: str                                  
  :return: Real refractive index, imaginary refractive index, calculated scattering and absorption coefficients in native measurements, and calculated single scattering albedo and extinction coefficients in all wavelengths.
  :rtype: numpy dictionary
  """

  L1 = len(CRI_p[:,0])
  L2 = len(full_wvl["Sc"])
  wvl = None
  for i1 in range(L2):
    if i1 == 0:
      wvl = np.array([full_wvl["Sc"][i1],full_wvl["Abs"][i1]])
    else:
      wvl = np.hstack((wvl,np.array([full_wvl["Sc"][i1],full_wvl["Abs"][i1]])))

  wvl = np.sort(wvl, axis=None)
  iri = np.full((L1), np.nan)
  rri = np.full((L1), np.nan)
  Results = dict()  
  Results["RRI_dry"] = None
  Results["IRI_dry"] = None
  Results["SSA"] = None
  for i2 in range(L2):
    Results[f'Cal_sca_coef_dry_{full_wvl["Sc"][i2]}'] = None
    Results[f'Cal_abs_coef_dry_{full_wvl["Abs"][i2]}'] = None
    Results[f'Cal_SSA_dry_{full_wvl["Sc"][i2]}'] = None
    Results[f'Cal_SSA_dry_{full_wvl["Abs"][i2]}'] = None
    Results[f'Cal_ext_coef_dry_{full_wvl["Sc"][i2]}'] = None
    Results[f'Cal_ext_coef_dry_{full_wvl["Abs"][i2]}'] = None

  for i1 in range(L1):
    RRI_p = {}
    IRI_p = {}
    for imode in sd:
      RRI_p[imode] = CRI_p[i1,0]
      IRI_p[imode] = CRI_p[i1,1]
    results = MMModel(wvl,size_equ,sd,dpg,RRI_p,IRI_p,nonabs_fraction,shape,rho,0,0,num_theta,optical_dataset,path_mopsmap_executable)
    ref_scat_coef = np.full((L2),np.nan)
    ref_abs_coef = np.full((L2),np.nan)
    scat_coef = np.full((L2),np.nan)
    abs_coef = np.full((L2),np.nan)
    for i2 in range(L2):
      scat_coef[i2] = results[f'ssa_{full_wvl["Sc"][i2]}']*results[f'ext_coeff_{full_wvl["Sc"][i2]}']
      abs_coef[i2] = results[f'ext_coeff_{full_wvl["Abs"][i2]}']-results[f'ssa_{full_wvl["Abs"][i2]}']*results[f'ext_coeff_{full_wvl["Abs"][i2]}'] 
      ref_scat_coef[i2] = optical_measurements[f'Meas_sca_coef_dry_{full_wvl["Sc"][i2]}']
      ref_abs_coef[i2] = optical_measurements[f'Meas_abs_coef_dry_{full_wvl["Abs"][i2]}']
    Cdif1 = abs(ref_scat_coef-scat_coef)/ref_scat_coef
    Cdif2 = abs(ref_abs_coef-abs_coef)
    a1 = ((Cdif1)<0.2).astype('int')
    a1[np.isinf(a1)]=0
    a2 = ((Cdif2)<pow(10,-6)).astype('int')#
    if np.sum(a1)==3 & np.sum(a2)==3:
      iri[i1] = CRI_p[i1,1]
      rri[i1] = CRI_p[i1,0]

  flgs = np.logical_not(np.isnan(rri)) 
  if np.sum(rri[flgs])>0:      
    rri = np.mean(rri[flgs])
    iri = np.mean(iri[flgs])

    RRI_d = {}
    IRI_d = {}
    for imode in sd:
      RRI_d[imode] = rri
      IRI_d[imode] = iri
    results = MMModel(wvl,size_equ,sd,dpg,RRI_d,IRI_d,nonabs_fraction,shape,rho,0,0,num_theta,optical_dataset,path_mopsmap_executable) 
    scat_coef = np.full((L2),np.nan)
    abs_coef = np.full((L2),np.nan)
    for i2 in range(L2):
      scat_coef[i2] = results[f'ssa_{full_wvl["Sc"][i2]}']*results[f'ext_coeff_{full_wvl["Sc"][i2]}']
      abs_coef[i2] = results[f'ext_coeff_{full_wvl["Abs"][i2]}']-results[f'ssa_{full_wvl["Abs"][i2]}']*results[f'ext_coeff_{full_wvl["Abs"][i2]}'] 
    Cdif1 = abs(ref_scat_coef-scat_coef)/ref_scat_coef
    Cdif2 = abs(ref_abs_coef-abs_coef)
    a1 = ((Cdif1)<0.2).astype('int')
    a1[np.isinf(a1)]=0
    a2 = ((Cdif2)<pow(10,-6)).astype('int')#
    if np.sum(a1)==L2 & np.sum(a2)==L2:
      Results["RRI_dry"] = rri
      Results["IRI_dry"] = iri
      for i2 in range(L2):
        Results[f'Cal_sca_coef_dry_{full_wvl["Sc"][i2]}'] = results[f'ssa_{full_wvl["Sc"][i2]}']*results[f'ext_coeff_{full_wvl["Sc"][i2]}']
        Results[f'Cal_abs_coef_dry_{full_wvl["Abs"][i2]}'] = results[f'ext_coeff_{full_wvl["Abs"][i2]}']-results[f'ssa_{full_wvl["Abs"][i2]}']*results[f'ext_coeff_{full_wvl["Abs"][i2]}'] 
        Results[f'Cal_SSA_dry_{full_wvl["Sc"][i2]}'] = results[f'ssa_{full_wvl["Sc"][i2]}']
        Results[f'Cal_SSA_dry_{full_wvl["Abs"][i2]}'] = results[f'ssa_{full_wvl["Abs"][i2]}']
        Results[f'Cal_ext_coef_dry_{full_wvl["Sc"][i2]}'] = results[f'ext_coeff_{full_wvl["Sc"][i2]}']
        Results[f'Cal_ext_coef_dry_{full_wvl["Abs"][i2]}'] = results[f'ext_coeff_{full_wvl["Abs"][i2]}']
  return Results

def Retr_kappa(full_wvl,optical_measurements,
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
  optical_dataset, 
  path_mopsmap_executable,
):
  """
  Returns aerosol particle hygroscopic growth factor from three dry scattering coefficeint measurements, three humidifide scattering coefficient measurements, a measured number concentration for an aerosol size distribution. WARNINGS: 1) numpy must be installed to the python environment 2) mopsmap_SD_run.py must be present in a directory that is in your PATH

  :param wvl: Array of 3 scattering and 3 absorption wavelengths in nm (e.g., 450, 470, 532, 550, 660, 700).
  :type wvl: numpy array
  :raise ISARA.InvalidNumberOfWavelengths: If the length of wvl is not 6.
  :param measured_coef: Array containing the 3 measured humidified scattering coefficients in m^-1.
  :type measured_coef: numpy array      
  :param sd: Array containing the fist mode number concentrations in m^-3.
  :type sd: numpy array  
  :param dpg: Array containing the fist mode particle diameters in nm.
  :type dpg: numpy array   
  :param RH: Array containing the percent relative humidity associated with the measured humidified scattering coefficients.
  :type RH: int   
  :param kappa_p: Array containing the desired kappa range to be searched.
  :type kappa_p: numpy array    
  :param CRI_d: Array containing the desired dry RRI and IRI.
  :type CRI_d: numpy array
  :param nonabs_fraction: Integer indicating the desired non-absorbing fraction for first mode.
  :type nonabs_fraction: int
  :param shape: String indicating the desired particle shape(s) for first mode.
  :type shape: str
  :param rho: Double indicating the desired particle density in g m^-3 for first mode.
  :type rho: double  
  :param num_theta: Integer indicating the number of phase function angles to provide.
  :type num_theta: int       
  :param optical_dataset: String indicating the path for the optical dataset required for MOPSMAP.
  :type optical_dataset: str
  :param path_mopsmap_executable: String indicating the path for the mopsmap.exe file.
  :type path_mopsmap_executable: str                                         
  :return: Real refractive index, imaginary refractive index, calculated scattering and absorption coefficients in native measurements, and calculated single scattering albedo and extinction coefficients in all wavelengths.
  :rtype: numpy dictionary
  """
  L1 = len(kappa_p)
  L2 = len(full_wvl["Sc"])
  wvl = None
  for i1 in range(L2):
    if i1 == 0:
      wvl = np.array([full_wvl["Sc"][i1]])
    else:
      wvl = np.hstack((wvl,np.array([full_wvl["Sc"][i1]])))

  wvl = np.sort(wvl, axis=None)
  #print(wvl)
  Results = dict()
  Results["Kappa"] = None
  for i2 in range(L2):
    Results[f'Cal_sca_coef_wet_{full_wvl["Sc"][i2]}'] = None
    Results[f'Cal_SSA_wet_{full_wvl["Sc"][i2]}'] = None
    Results[f'Cal_ext_coef_wet_{full_wvl["Sc"][i2]}'] = None
    
  stop_indx = 0

  RRIw = 1.33
  IRIw = 0

  for i1 in range(L1):
    dpg_w = {}
    RRI_w = {}
    IRI_w = {}
    for imode in sd:
      gf = np.power((1+kappa_p[i1]*RH/(100-RH)),1/3)
      dpg_w[imode] = np.squeeze(np.multiply(gf,dpg[imode]))
      RRI_w[imode] = (CRI_d[0]+((gf**3)-1)*RRIw)/(gf**3)
      IRI_w[imode] = (CRI_d[1]+((gf**3)-1)*IRIw)/(gf**3)#CRI1[1]
    if stop_indx == 0:
      results = MMModel(wvl,size_equ,sd,dpg_w,RRI_w,IRI_w,nonabs_fraction,shape,rho,0,0,num_theta,optical_dataset,path_mopsmap_executable)
      scat_coef = np.full((L2),np.nan)
      ref_scat_coef = np.full((L2),np.nan)
      for i2 in range(L2):
        scat_coef[i2] = results[f'ssa_{full_wvl["Sc"][i2]}']*results[f'ext_coeff_{full_wvl["Sc"][i2]}']
        ref_scat_coef[i2] = optical_measurements[f'Meas_sca_coef_wet_{full_wvl["Sc"][i2]}']
      Cdif = abs(ref_scat_coef-scat_coef)/ref_scat_coef
      #a = ((Cdif)<0.01).astype('int')
      if Cdif<0.01:
        Results["Kappa"] = kappa_p[i1]
        for i2 in range(L2):
          Results[f'Cal_sca_coef_wet_{full_wvl["Sc"][i2]}'] = results[f'ssa_{full_wvl["Sc"][i2]}']*results[f'ext_coeff_{full_wvl["Sc"][i2]}']
          Results[f'Cal_SSA_wet_{full_wvl["Sc"][i2]}'] = results[f'ssa_{full_wvl["Sc"][i2]}']
          Results[f'Cal_ext_coef_wet_{full_wvl["Sc"][i2]}'] = results[f'ext_coeff_{full_wvl["Sc"][i2]}']
        stop_indx = 1

  return Results

class InvalidNumberOfWavelengths(Exception):
    """Raised if the length of wvl is not 6."""
    pass   