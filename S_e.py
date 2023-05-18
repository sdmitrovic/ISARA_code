###############################################################################
#  The following code is used to generate the covariance matrix (covM) of 6
#  optical coefficient measurements that are correlated by the correlation
#  factor (i.e, cr). Each of the required inputs for this code are
#  described as follows:
#  1) y: 6 measured optical coefficients
#     -> dimensions: [1,6]
#     -> Units: # Mm-1
#  2) Acur: 3 Scattering and 3 absorption measurement accuracy values used
#     to calculate Chi'.
#     -> dimensions: [1,6]
#     -> FORMAT: [accuracy of scattering channels 1 to 3,accuracy of 
#                 absportion channels 1 to 3]
#     -> EXAPMLE: [0.1 0.1 0.1 0.05 0.05 0.05]
#     -> Units: unitless
#  3) Sig1: 3 Scattering and 3 absorption measurement precision values
#     divided by the square root of the data resolution  (i.e. number of
#     native optical measurments averaged to create data set). 
#     -> dimensions: [1,6] 
#     -> FORMAT: [precision of scattering channels 1 to 3,precision of
#                 absportion channels 1 to 3] 
#     -> Units: Mm-1
#  4) cFct: Optical measurement correlation factor.
#     -> dimensions: [1,1]
#     -> EXAPMLE: 0.3
#     -> Units: unitless
# The output of this code is covM that is described as follows:
#     -> dimensions: [6,6]
#     -> Units: mM-1
# *Designed for MATLAB 2021a*
###############################################################################
import numpy as np

def Cal(y,Acur,sig1,cFct):
    # define the diagonal positons of covM
    if type(y) == type(np.float64(1.0)):
        covM = y*Acur+sig1  
    else:
        L0 = len(y)
        covm = np.zeros((L0))
        for i1 in range(L0):
            covm[i1] = y[i1]*Acur[i1]+sig1[i1]      
        # initialize the loop to calculate the upper and lower triangls of covM
        # based off of the diagonal values
        covM = np.zeros((L0,L0));           
        # calculate the non-diagonal positions of covM
        for io in range(L0):
            for io2 in range(L0):
                if io2 == io:
                    covM[io,io2] = covm[io];
                else:
                    covM[io,io2] = cFct*np.sqrt(covm[io])*np.sqrt(covm[io2]);
    return covM