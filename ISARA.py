import mopsmap_wrapper
import numpy as np

def Retr_CRI(wvl,
  measured_sca_coef,
  measured_abs_coef,
  sd1,sd2,
  dpg1,dpg2,
  CRI,
  size_equ1,
  size_equ2,
  nonabs_fraction1,
  nonabs_fraction2,
  shape1,
  shape2,
  rho1,
  rho2,
  num_theta,
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
  :param sd1: Array containing the fist mode number concentrations in m^-3.
  :type sd1: numpy array  
  :param sd2: Array containing the second mode number concentrations in m^-3.
  :type sd2: numpy array
  :param dpg1: Array containing the fist mode particle diameters in nm.
  :type dpg1: numpy array  
  :param dpg2: Array containing the second mode particle diameters in nm.
  :type dpg2: numpy array      
  :param CRI: 2-D array containing the desired RRI and IRI range to be searched.
  :type CRI: numpy array
  :param nonabs_fraction1: Integer indicating the desired non-absorbing fraction for first mode.
  :type nonabs_fraction1: int
  :param nonabs_fraction2: Integer indicating the desired non-absorbing fraction for second mode.
  :type nonabs_fraction2: int
  :param shape1: String indicating the desired particle shape(s) for first mode.
  :type shape1: str
  :param shape2: String indicating the desired particle shape(s) for second mode.
  :type shape2: str  
  :param rho1: String indicating the desired particle density in g m^-3 for first mode.
  :type rho1: str
  :param rho2: String indicating the desired particle density in g m^-3 for second mode.
  :type rho2: str                                        
  :return: Real refractive index, imaginary refractive index, calculated scattering and absorption coefficients in native measurements, and calculated single scattering albedo and extinction coefficients in all wavelengths.
  :rtype: numpy dictionary
  """
  L1 = len(CRI[:,0])
  L2 = len(wvl)

  iri = np.zeros(L1)-1
  rri = np.zeros(L1)-1
  Results = dict()  
  Results["RRIdry"] = None
  Results["IRIdry"] = None
  Results["SSA"] = None
  Results["ext_coef"] = None
  Results["abs_coef"] = None
  Results["scat_coef"] = None
  size_equ = dict()
  sd = dict()
  dpg = dict()
  RRI = dict()
  IRI = dict()
  nonabs_fraction = dict()
  rho = dict()
  shape = dict()
  kappa = dict()  
  size_equ[1] = size_equ1
  sd[1] = sd1
  dpg[1] = dpg1
  nonabs_fraction[1] = nonabs_fraction1
  rho[1] = rho1
  shape[1] = shape1
  kappa[1] = 0
  size_equ[2] = size_equ2
  sd[2] = sd2
  dpg[2] = dpg2
  nonabs_fraction[2] = nonabs_fraction2
  rho[2] = rho2
  shape[2] = shape2
  kappa[2] = 0

  for i1 in range(L1):

    RRI[1] = CRI[i1,0]
    IRI[1] = CRI[i1,1]

    RRI[2] = CRI[i1,0]
    IRI[2] = CRI[i1,1]

    results = mopsmap_wrapper.Model(wvl,size_equ,sd,dpg,RRI,IRI,nonabs_fraction,shape,rho,0,kappa,num_theta)
    scat_coef = results['ssa'][[0,3,5]]*results['ext_coeff'][[0,3,5]]
    abs_coef = results['ext_coeff'][[1,2,4]]-results['ssa'][[1,2,4]]*results['ext_coeff'][[1,2,4]]   
    Cdif1 = abs(measured_sca_coef-scat_coef)/measured_sca_coef
    Cdif2 = abs(measured_abs_coef-abs_coef)
    a1 = ((Cdif1)<0.2).astype('int')
    a1[np.isinf(a1)]=0
    a2 = ((Cdif2)<pow(10,-6)).astype('int')#
    if np.sum(a1)==3 & np.sum(a2)==3:
      iri[i1] = CRI[i1,1]
      rri[i1] = CRI[i1,0]

  flgs = rri>=0 
  if np.sum(rri[flgs])>0:      
    Results["RRIdry"] = np.mean(rri[flgs])
    Results["IRIdry"] = np.mean(iri[flgs])
    RRI[1] = np.mean(rri[flgs])
    IRI[1] = np.mean(iri[flgs])
    RRI[2] = np.mean(rri[flgs])
    IRI[2] = np.mean(iri[flgs])    
    results = mopsmap_wrapper.Model(wvl,size_equ,sd,dpg,RRI,IRI,nonabs_fraction,shape,rho,0,kappa,num_theta) 

    scat_coef = results['ssa'][[0,3,5]]*results['ext_coeff'][[0,3,5]]
    abs_coef = results['ext_coeff'][[1,2,4]]-results['ssa'][[1,2,4]]*results['ext_coeff'][[1,2,4]]   
    Cdif1 = abs(measured_sca_coef-scat_coef)/measured_sca_coef
    Cdif2 = abs(measured_abs_coef-abs_coef)
    a1 = ((Cdif1)<0.2).astype('int')
    a1[np.isinf(a1)]=0
    a2 = ((Cdif2)<pow(10,-6)).astype('int')#
    if np.sum(a1)==3 & np.sum(a2)==3:
      Results["scat_coef"] = results['ssa'][[0,3,5]]*results['ext_coeff'][[0,3,5]]
      Results["abs_coef"] = results['ext_coeff'][[1,2,4]]-results['ssa'][[1,2,4]]*results['ext_coeff'][[1,2,4]]
      Results["SSA"] = results['ssa']
      Results["ext_coef"] = results['ext_coeff']

  return Results

def Retr_kappa(wvl,
  measured_wet_sca_coef,
  sd1,sd2,
  dpg1,dpg2,
  RH,
  Kappa,
  CRI1,CRI2,
  size_equ1,
  size_equ2,
  nonabs_fraction1,
  nonabs_fraction2,
  shape1,
  shape2,
  rho1,
  rho2,
  num_theta,
):
  """
  Returns aerosol particle hygroscopic growth factor from three dry scattering coefficeint measurements, three humidifide scattering coefficient measurements, a measured number concentration for an aerosol size distribution. WARNINGS: 1) numpy must be installed to the python environment 2) mopsmap_SD_run.py must be present in a directory that is in your PATH

  :param wvl: Array of 3 scattering and 3 absorption wavelengths in nm (e.g., 450, 470, 532, 550, 660, 700).
  :type wvl: numpy array
  :raise ISARA.InvalidNumberOfWavelengths: If the length of wvl is not 6.
  :param measured_coef: Array containing the 3 measured humidified scattering coefficients in m^-1.
  :type measured_coef: numpy array      
  :param sd1: Array containing the fist mode number concentrations in m^-3.
  :type sd1: numpy array  
  :param sd2: Array containing the second mode number concentrations in m^-3.
  :type sd2: numpy array
  :param dpg1: Array containing the fist mode particle diameters in nm.
  :type dpg1: numpy array  
  :param dpg2: Array containing the second mode particle diameters in nm.
  :type dpg2: numpy array  
  :param RH: Array containing the percent relative humidity associated with the measured humidified scattering coefficients.
  :type RH: int   
  :param Kappa: Array containing the desired kappa range to be searched.
  :type Kappa: numpy array    
  :param CRI: Array containing the desired dry RRI and IRI.
  :type CRI: numpy array
  :param nonabs_fraction1: Integer indicating the desired non-absorbing fraction for first mode.
  :type nonabs_fraction1: int
  :param nonabs_fraction2: Integer indicating the desired non-absorbing fraction for second mode.
  :type nonabs_fraction2: int
  :param shape1: String indicating the desired particle shape(s) for first mode.
  :type shape1: str
  :param shape2: String indicating the desired particle shape(s) for second mode.
  :type shape2: str  
  :param rho1: String indicating the desired particle density in g m^-3 for first mode.
  :type rho1: str
  :param rho2: String indicating the desired particle density in g m^-3 for second mode.
  :type rho2: str                                        
  :return: Real refractive index, imaginary refractive index, calculated scattering and absorption coefficients in native measurements, and calculated single scattering albedo and extinction coefficients in all wavelengths.
  :rtype: numpy dictionary
  """
  L1 = len(Kappa)
  L3 = len(wvl) 
  Results = dict()
  Results["Kappa"] = None
  Results["Cal_coef"] = None
  Results["Cal_SSA"] = None
  Results["Cal_ext_coef"] = None
  stop_indx = 0

  size_equ = dict()
  sd = dict()
  dpg = dict()
  RRI = dict()
  IRI = dict()
  nonabs_fraction = dict()
  rho = dict()
  shape = dict()
  kappa = dict()  
  size_equ[1] = size_equ1
  sd[1] = sd1
  kappa[1]= 0
  nonabs_fraction[1] = nonabs_fraction1
  rho[1] = rho1
  shape[1] = shape1
  size_equ[2] = size_equ1
  sd[2] = sd2
  kappa[2]= 0

  nonabs_fraction[2] = nonabs_fraction1
  rho[2] = rho1
  shape[2] = shape1
  
  RRIw = 1.33
  IRIw = 0

  for i1 in range(L1):
    gf = np.power((1+Kappa[i1]*RH/(100-RH)),1/3)#D/Ddry = (1+kappa*RH/(100-RH))**(1/3)
    D_1 = np.squeeze(np.multiply(gf,dpg1))
    D_2 = np.squeeze(np.multiply(gf,dpg2))
    #IRIf = (CRI[0]+((gf**3)-1)*IRIw)/(gf**3)
    dpg[1] = D_1
    RRI[1] = (CRI1[0]+((gf**3)-1)*RRIw)/(gf**3)
    IRI[1] = CRI1[1]
    dpg[2] = D_2
    RRI[2] = (CRI2[0]+((gf**3)-1)*RRIw)/(gf**3)
    IRI[2] = CRI2[1]

    if stop_indx == 0:
      results = mopsmap_wrapper.Model(wvl,size_equ,sd,dpg,RRI,IRI,nonabs_fraction,shape,rho,0,kappa,num_theta)
      scat_coef = results['ssa']*results['ext_coeff']
      abs_coef = results['ext_coeff']-results['ssa']*results['ext_coeff']
      Cdif = (measured_wet_sca_coef-scat_coef[3])/measured_wet_sca_coef
      a = ((Cdif)<0.01).astype('int')
      if np.sum(a)==1:
        Results["Kappa"] = Kappa[i1]
        Results["Cal_coef"] = scat_coef
        Results["Cal_SSA"] = results['ssa']
        Results["Cal_ext_coef"] = results['ext_coeff']
        stop_indx = 1

  return Results

class InvalidNumberOfWavelengths(Exception):
    """Raised if the length of wvl is not 6."""
    pass   