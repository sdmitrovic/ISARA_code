Usage
=====
Acknowledgements
----------------

ISARA is being developed in collaboration with the Office of Naval Research, Oak Ridge Associated Universities, NASA Langley Research Center, and the University of Arizona.


Copyright
---------
MIT License

Copyright 2023 Joseph Schlosser, Snorre Stamnes, Sanja Dmitrovic

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


Installation
------------

To use ISARA, first install it using pip:

.. code-block:: console

   (.venv) $ pip3 install ISARA


Retrieve Complex Refractive Index
---------------------------------

.. autofunction:: ISARA.Retr_CRI

>>> import ISARA
>>> import numpy as np
>>> import os
>>> import sys
>>> RRI = 1.53
>>> IRI = np.hstack((0.0001, np.arange(0.001,0.041,0.001).reshape(-1)))
>>> wvl = np.array([0.450, 0.470, 0.532, 0.550, 0.660, 0.700])
>>> measured_sca_coef =  np.array([26.7769, 22.9139, 21.2927])*pow(10,-6) 
>>> measured_abs_coef =  np.array([0.91078, 0.72244, 0.36833])*pow(10,-6) 
>>> sd1 =  np.array([0.00,2527.74,0.00E+00,0.00E+00,2005.82,0.00E+00,1207.24,899.07,718.81,899.1,1000.6,874.72,1241.7,1655.18,376.54,2159.19,2626.72,3489.7,2749.07,2378.41,2015.2,2089.09,2676,3190.64,3483.72,3839.27,3667.73,4412.98,4220.87,3913.1])*pow(10,6) 
>>> sd2 =  np.array([4464.27, 4307.44, 3495.04, 3198.41, 1972.54, 1523.9, 1173, 907.394, 696.4407, 474.6136, 274.8, 212.104, 105.9756, 64.9613, 6.35556, 5.82689, 4.23289, 1.59156, 2.64978, 0.70533, 1.41356, 0.42222, 0.42311, 0.00E+00, 1.05978, 0.63622])*pow(10,6) 
>>> dpg1 =  np.array([3.16,3.55,3.98,4.47,5.01,5.62,6.31,7.08,7.94,8.91,10.0,11.2,12.6,14.1,15.8,17.8,20.0,22.4,25.1,28.2,31.6,35.5,39.8,44.7,50.1,56.2,63.1,70.8,79.4,89.1])*pow(10,-3) 
>>> dpg2 =  np.array([100.0,112.2,125.9,141.3,158.5,177.8,199.5,223.9,251.2,281.8,316.2,354.8,398.1,446.7,501.2,562.3,631.0,707.9,794.3,891.3,1000.0,1258.9,1584.9,1995.3,2511.9,3162.3])*pow(10,-3) 
>>> CRI = np.zeros((len(IRI), 2))
>>> for i1 in range(len(IRI)): CRI[i1, :] = [RRI, IRI[i1]]
>>> size_equ1 = 'cs'
>>> size_equ2 = 'cs'
>>> nonabs_fraction1 = 0
>>> nonabs_fraction2 = 0
>>> shape1 = 'sphere'
>>> shape2 = 'sphere'
>>> rho1 = 2.63
>>> rho2 = 2.63
>>> num_theta = 2
>>> Results = ISARA.Retr_CRI(wvl,measured_sca_coef,measured_abs_coef,sd1,sd2,dpg1,dpg2,CRI,size_equ1,size_equ2,nonabs_fraction1,nonabs_fraction2,shape1,shape2,rho1,rho2,num_theta)

Retrieve Hygroscopicity
-----------------------

.. autofunction:: ISARA.Retr_kappa

>>> import ISARA
>>> import numpy as np
>>> import os
>>> import sys
>>> RRI = 1.53
>>> IRI = np.hstack((0.0001, np.arange(0.001,0.041,0.001).reshape(-1)))
>>> wvl = np.array([0.450, 0.470, 0.532, 0.550, 0.660, 0.700])
>>> measured_sca_coef =  np.array([26.7769, 22.9139, 21.2927])*3*pow(10,-6) 
>>> measured_abs_coef =  np.array([0.91078, 0.72244, 0.36833])*pow(10,-6) 
>>> sd1 =  np.array([0.00,2527.74,0.00E+00,0.00E+00,2005.82,0.00E+00,1207.24,899.07,718.81,899.1,1000.6,874.72,1241.7,1655.18,376.54,2159.19,2626.72,3489.7,2749.07,2378.41,2015.2,2089.09,2676,3190.64,3483.72,3839.27,3667.73,4412.98,4220.87,3913.1])*pow(10,6) 
>>> sd2 =  np.array([4464.27, 4307.44, 3495.04, 3198.41, 1972.54, 1523.9, 1173, 907.394, 696.4407, 474.6136, 274.8, 212.104, 105.9756, 64.9613, 6.35556, 5.82689, 4.23289, 1.59156, 2.64978, 0.70533, 1.41356, 0.42222, 0.42311, 0.00E+00, 1.05978, 0.63622])*pow(10,6) 
>>> dpg1 =  np.array([3.16,3.55,3.98,4.47,5.01,5.62,6.31,7.08,7.94,8.91,10.0,11.2,12.6,14.1,15.8,17.8,20.0,22.4,25.1,28.2,31.6,35.5,39.8,44.7,50.1,56.2,63.1,70.8,79.4,89.1])*pow(10,-3) 
>>> dpg2 =  np.array([100.0,112.2,125.9,141.3,158.5,177.8,199.5,223.9,251.2,281.8,316.2,354.8,398.1,446.7,501.2,562.3,631.0,707.9,794.3,891.3,1000.0,1258.9,1584.9,1995.3,2511.9,3162.3])*pow(10,-3) 
>>> CRI = np.zeros((len(IRI), 2))
>>> for i1 in range(len(IRI)): CRI[i1, :] = [RRI, IRI[i1]]
>>> size_equ1 = 'cs'
>>> size_equ2 = 'cs'
>>> nonabs_fraction1 = 0
>>> nonabs_fraction2 = 0
>>> shape1 = 'sphere'
>>> shape2 = 'sphere'
>>> rho1 = 2.63
>>> rho2 = 2.63
>>> num_theta = 2
>>> Results = ISARA.Retr_CRI(wvl,measured_sca_coef,measured_abs_coef,sd1,sd2,dpg1,dpg2,CRI,size_equ1,size_equ2,nonabs_fraction1,nonabs_fraction2,shape1,shape2,rho1,rho2,num_theta)
>>> RRI_dry = Results["RRIdry"]
>>> IRI_dry = Results["IRIdry"]
>>> CRI1 = np.array([RRI_dry,IRI_dry])
>>> CRI2 = np.array([RRI_dry,IRI_dry])
>>> measured_wet_sca_coef =  np.array([22.9139])*8*pow(10,-6) 
>>> RH = 85
>>> Kappa = np.arange(0.0, 1.40, 0.01).reshape(-1)
>>> Results2 = ISARA.Retr_kappa(wvl,measured_sca_coef,measured_wet_sca_coef,sd1,sd2,dpg1,dpg2,RH,Kappa,CRI1,CRI2,size_equ1,size_equ2,nonabs_fraction1,nonabs_fraction2,shape1,shape2,rho1,rho2,num_theta)

Wavelength Excemption
---------------------

.. autoexception:: ISARA.InvalidNumberOfWavelengths


Forward Model for Aerosol Optial Properties
-------------------------------------------

.. autofunction:: mopsmap_wrapper.Model


Import ICARTT Files
-------------------

.. autofunction:: importICARTT.imp


Import Diameters of Aerosol Size Distribution 
---------------------------------------------

.. autofunction:: load_sizebins.Load


ISARA ACTIVATE Data Retrievals 
------------------------------

.. autofunction:: ISARA_ACTIVATE_Data_Retrieval.RunISARA

