import mopsmap_wrapper
import numpy as np

def Retr_CRI(wvl,
  measured_sca_coef,
  measured_abs_coef,
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
  L2 = len(wvl)

  iri = np.full((L1), np.nan)
  rri = np.full((L1), np.nan)
  Results = dict()  
  Results["RRIdry"] = None
  Results["IRIdry"] = None
  Results["SSA"] = None
  Results["ext_coef"] = None
  Results["abs_coef"] = None
  Results["scat_coef"] = None

  for i1 in range(L1):
    RRI_p = {}
    IRI_p = {}
    for imode in sd:
      RRI_p[imode] = CRI_p[i1,0]
      IRI_p[imode] = CRI_p[i1,1]
    results = mopsmap_wrapper.Model(wvl,size_equ,sd,dpg,RRI_p,IRI_p,nonabs_fraction,shape,rho,0,0,num_theta,optical_dataset,path_mopsmap_executable)
    scat_coef = results['ssa'][[0,3,5]]*results['ext_coeff'][[0,3,5]]
    abs_coef = results['ext_coeff'][[1,2,4]]-results['ssa'][[1,2,4]]*results['ext_coeff'][[1,2,4]]   
    Cdif1 = abs(measured_sca_coef-scat_coef)/measured_sca_coef
    Cdif2 = abs(measured_abs_coef-abs_coef)
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
    results = mopsmap_wrapper.Model(wvl,size_equ,sd,dpg,RRI_d,IRI_d,nonabs_fraction,shape,rho,0,0,num_theta,optical_dataset,path_mopsmap_executable) 
    scat_coef = results['ssa'][[0,3,5]]*results['ext_coeff'][[0,3,5]]
    abs_coef = results['ext_coeff'][[1,2,4]]-results['ssa'][[1,2,4]]*results['ext_coeff'][[1,2,4]]   
    Cdif1 = abs(measured_sca_coef-scat_coef)/measured_sca_coef
    Cdif2 = abs(measured_abs_coef-abs_coef)
    a1 = ((Cdif1)<0.2).astype('int')
    a1[np.isinf(a1)]=0
    a2 = ((Cdif2)<pow(10,-6)).astype('int')#
    if np.sum(a1)==3 & np.sum(a2)==3:
      Results["RRIdry"] = rri
      Results["IRIdry"] = iri
      Results["scat_coef"] = results['ssa'][[0,3,5]]*results['ext_coeff'][[0,3,5]]
      Results["abs_coef"] = results['ext_coeff'][[1,2,4]]-results['ssa'][[1,2,4]]*results['ext_coeff'][[1,2,4]]
      Results["SSA"] = results['ssa']
      Results["ext_coef"] = results['ext_coeff']

  return Results

def Retr_kappa(wvl,
  measured_wet_sca_coef,
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
  L3 = len(wvl) 
  Results = dict()
  Results["Kappa"] = None
  Results["Cal_coef"] = None
  Results["Cal_SSA"] = None
  Results["Cal_ext_coef"] = None
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
      results = mopsmap_wrapper.Model(wvl,size_equ,sd,dpg_w,RRI_w,IRI_w,nonabs_fraction,shape,rho,0,0,num_theta,optical_dataset,path_mopsmap_executable)
      scat_coef = results['ssa']*results['ext_coeff']
      abs_coef = results['ext_coeff']-results['ssa']*results['ext_coeff']
      Cdif = abs(measured_wet_sca_coef-scat_coef[3])/measured_wet_sca_coef
      #a = ((Cdif)<0.01).astype('int')
      if Cdif<0.01:
        Results["Kappa"] = kappa_p[i1]
        Results["Cal_coef"] = scat_coef
        Results["Cal_SSA"] = results['ssa']
        Results["Cal_ext_coef"] = results['ext_coeff']
        stop_indx = 1

  return Results

class InvalidNumberOfWavelengths(Exception):
    """Raised if the length of wvl is not 6."""
    pass   