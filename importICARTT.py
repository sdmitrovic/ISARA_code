#########################################################################################################################
# importICARTT.py                                        by:  Joseph Schlosser
#                                                revised:  20 Feb 2022 
#                                    language (revision):  python3 (3.8.2-0ubuntu2)
# 
# Description: procedure to open ICARTT file and format the data and headers into the output (output_dictionary) python 
# dictionary. Each of the required inputs for this code are described as follows:
# 1) filename: String or character containing the name of the desired ICARTT v2.0 (.ict) file.
#
# the output of this code is output_dictionary, which is a python3 dictionary containing column-arrays for each of the parameters
# in the .ict file
# -> each column corresponds to a line in the provided .ict file
#
#   EXAMPLES:
#       output_dictionary = importICT.imp("activate-mrg1_hu25_20200214_R0.ict")
#
#       print(output_dictionary)
#
#       output_dictionary =
#
#           {'Time_Start_seconds': array([61301., 61302., 61303., ..., 72258., 72259., 72260.]), 
#           'Time_Stop_seconds': array([61302., 61303., 61304., ..., 72259., 72260., 72261.]), 
#           'Latitude_THORNHILL_ deg': array([37.085528, 37.085798, 37.086065, ..., 37.126424, 37.126694, ...
#
# WARNINGS:
# 1) numpy must be installed to the python environment
# 2) importICART.py and file with the corresponding filename must be present in a directory that is in your PATH
#########################################################################################################################

import numpy as np
import datetime

def imp(filename = None): 
    G = open(filename, 'r') # open .ict file
    g = G.readlines() # read .ict file
    DATEinfo = np.array(g[6].split(",")) 
    DATE = DATEinfo[0:3] # save date to add to file

    Fv = g[11]  # locate line with fill values, wich are located on line 11 of .ict file, to replace with 'nan'
    fv = Fv.split(",") # create array of fill values
    varend = int(g[9]) # locate line with number of variables, which is located on line 9 of .ict file

    full_var_titles = ["" for x in range(np.add(varend,1))] # create empty string array for full variable titles

    # fill full variable title string array with full variable titles starting on line 12 of .ict file
    i2 = 12
    for i1 in range(varend):
        full_var_titles[np.add(i1,1)] = g[i2]
        i2 = np.add(i2,1)

    # fill first variable title string array position with full variable title of first variable on line 8 of .ict file    
    starttime = g[8]
    full_var_titles[0] = starttime[0:len(starttime)-1]    

    # locate data header and data start rows of .ict file using line 0 of .ict file
    st = g[0]
    Vr_id = np.array(st.split(","))
    vr_id = int(Vr_id[0])

    rawdata = g[vr_id::] # create raw data array starting at data start row

    # create empty string arrays for final variable names, units, and extra info
    var_names = ["" for x in range(len(full_var_titles))]
    var_units = ["" for x in range(len(full_var_titles))]
    
    # iteratively fill string arrays with final variable names, units, and extra info
    for i1 in np.arange(0,len(full_var_titles)).reshape(-1):
        fvt = full_var_titles[i1]
        FVT = fvt.split(",")
        var_names[i1] = FVT[0]
        var_units[i1] = FVT[1]

    # create data arraw with length of dataset and width of number of variables       
    data = np.zeros((len(rawdata),len(full_var_titles)))

    # iteratively fill data array with data and replace fv values with NAN
    for i1 in np.arange(0,len(rawdata)).reshape(-1):
        processdata1 = rawdata[i1]
        processdata2 = processdata1.split(",")
        if processdata2 == ['\n']:
            data[i1,:] =  np.full((1,len(full_var_titles)), 'nan')
        else:
            for i2 in np.arange(0,len(processdata2)).reshape(-1):
                if not processdata2[i2]:
                    processdata2[i2] = 'nan'
                elif processdata2[i2] == fv[np.add(i2,- 1)]:
                    processdata2[i2] = 'nan'
            data[i1,:] = processdata2
    
    # creat empty dictionary
    output_dictionary = {}

    # fill dictionary with data and keys   
    for i1 in range(len(full_var_titles)):
        output_dictionary[("%s_%s"%(var_names[i1],var_units[i1]))] = data[:,i1]


    output_dictionary["date"] = np.array(DATE) # Add date to dictionary  
    dta = data[:,0] # fill dictionary with date 
    frmttimedata = ["" for x in range(len(dta))] # create array of zeros for datetime data
    mattimedata = np.zeros((len(dta),6)) # create array of zeros for datetime data

    # fill empty arrays formated datetime and matix date time
    for i1 in range(len(dta)):
        if np.isnan(dta[i1]):
            mattimedata[i1,:] = np.full((1,6), 'nan')
            frmttimedata[i1] = 'nan'
        else:   
            Hrs = int(np.floor(dta[i1]/(60*60)))
            Mnts = int(np.floor((dta[i1]/(60*60)-np.floor(dta[i1]/(60*60)))*60))
            Secd = int(((dta[i1]/(60*60)-np.floor(dta[i1]/(60*60)))*60-np.floor((dta[i1]/(60*60)-np.floor(dta[i1]/(60*60)))*60))*60)
            Yr = int(DATE[0])
            Mon = int(DATE[1])
            Day = int(DATE[2])
            mattimedata[i1,:]  = [Yr,Mon,Day,Hrs,Mnts,Secd]
            # set day to next dat if hours are greater than 23
            if Hrs > 23:
                Hrs = Hrs - 24
                Day = Day + 1
            frmttimedata[i1] = datetime.datetime(Yr,Mon,Day,Hrs,Mnts,Secd)
    
    # Add frmttimedata and mattimedata to dictionary   
    output_dictionary["fmtdatetime"] = frmttimedata
    output_dictionary["matdatetime"] = mattimedata


    G.close() # close data file      
    return output_dictionary
    ##
    return output_dictionary