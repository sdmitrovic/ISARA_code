#########################################################################################################################
# mopsmap_UI.py                                       by:  Joseph Schlosser
#                                                revised:  19 Feb 2022 
#                                    language (revision):  python3 (3.8.2-0ubuntu2)
#
# Description: user interface used to select LAS size distribution (SD) data from a ICT file and format the data into a
# python3 dictionary
#
# Implementation: this interface can be run from using the following ubuntu syntax: user:$ python3 mopsmap_UI.py
#
# the output of this code is output_dictionary, which is a python3 dictionary containing column-arrays for each of the 
# parameters in the .ict file and row-column matricies for each of the MOPSMAP ouputs
# -> each column corresponds to a line in the provided .ict file
# -> each row of the MOPSMAP ouputs corresponds to each of the desired output wavelengths
#
# WARNINGS:
# 1) numpy, csv, and tqdm must be installed to the python environment
# 2) importICT.py, mopsmap_SD_run.py, LAS_bin_sizes.csv, and and file with the corresponding filename must be present in 
#    a directory that is in your PATH
#########################################################################################################################
import numpy as np
def f_model(x,a,c): 
	return np.multiply(a,pow(x, c))