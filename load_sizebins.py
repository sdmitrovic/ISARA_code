import numpy as np

def Load(filename):

    """
    Returns an array of bin geometric mean diameters from a specified .csv file
    
    :param filename: String containing the name of the desired .csv file containing the size distribution bin diameters.
    :type filename: str  
    :return: array of bin geometric mean diameter in micrometer.
    :rtype: numpy arra
    """   

    i0 = 1  # index used to skip header row
    G = open(filename, 'r')  # open .csv
    g = G.read().splitlines()  # read .csv
    hdrs = g[0].split(',')  # define headers
    # create zeros array to be filled iteratively
    Dp = np.zeros((len(g) - 1, len(hdrs)))
    for i1 in range(len(g) - 1):
        # split string into array and define as number array
        Dp[i1, :] = np.array(list(eval(g[i0])))
        i0 += 1
    # geometric mean diameter in micrometer
    dpg = np.array(Dp[:, 2]) 
    # transpose dpg to make column-array of geometric mean diameters from row-array
    # of geometric mean diameters
    dpg1 = dpg.T
    G.close()  # close .csv
    return dpg1