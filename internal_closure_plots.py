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
# 2) importICARTT.py, mopsmap_SD_run.py, LAS_bin_sizes.csv, and and file with the corresponding filename must be present 
#    in a directory that is in your PATH
#########################################################################################################################
import numpy as np
import csv
from tqdm import tqdm
import os
import datetime
import StatsCode
import itertools
import matplotlib as mpl
import matplotlib.pyplot as plt 
from matplotlib.ticker import MaxNLocator
from matplotlib.dates import DayLocator, HourLocator, DateFormatter
from matplotlib.colors import LogNorm
from matplotlib import cm
from matplotlib.collections import PolyCollection
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.offsetbox import AnchoredText

from mpl_toolkits.basemap import Basemap
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from pylab import rcParams

def flatten(l):
    return [item for sublist in l for item in sublist]

def Line(m,x,b):
    y = m*x + b
    return y

def grabvalues(
    dictionaryname,
    startofkeyname
  ):
    OP = dict()
    io = 0
    for key in dictionaryname.item():
      if key.startswith(startofkeyname):
        #print(key,io)
        value = dictionaryname.item().get(key)
        OP[io] = value.T
        io += 1
    return OP  

def getPercentileList(
    prctile,
    suffix
  ):
    prctile_lst = np.array([f"{x}_percentile_{suffix}" for x in prctile])
    return prctile_lst
    
bds1 = np.vstack((np.hstack((0,1,2,range(50,600,50))),np.hstack((0,1,2,range(10,120,10)))))
bds2 = np.vstack((np.hstack((0,1,2,range(10,120,10))),np.hstack((0,1,2,range(5,60,5)))))
bds3 = np.vstack((np.hstack((0,1,2,range(10,120,10))),np.hstack((0,1,2,range(5,60,5)))))
fs = 32
wvl=[0.450,0.550,0.700,0.470,0.532,0.660]#
IRI = np.arange(0.0, 0.08, 0.01).reshape(-1) 
kappa = np.arange(0.0, 1.40, 0.1).reshape(-1) 
Bin = dict()
Lst = ["IRI","Kappa"]
Bin["IRI"] = IRI
Bin["Kappa"] = kappa
N = len(bds1[0,:])-1
Jet = plt.get_cmap('jet', N)
newcolors = Jet(np.linspace(0, 1, N))
wht = np.array([1, 1, 1, 1])
gry = np.array([0.75, 0.75, 0.75, 1])
blk = np.array([0, 0, 0, 1])
newcolors[0, :] = wht
newcolors[-1, :] = blk
newcolors = np.vstack((gry,newcolors))
cmap = ListedColormap(newcolors)
resolution = np.array([60]) 

prctile = [0,10,25,50,75,90,100]
prctile_lst_rb = getPercentileList(prctile,"RB")
prctile_lst_arb = getPercentileList(prctile,"ARB")
prctile_lst_x = getPercentileList(prctile,"x")
prctile_lst_y = getPercentileList(prctile,"y")
#cmap = 'jet'
#print(cmap[0])

i0 = 1  # index used to skip header row
G = open('LAS_bin_sizes.csv', 'r')  # open .csv
g = G.read().splitlines()  # read .csv
hdrs = g[0].split(',')  # define headers
# create empty array to be filled iteratively
Dp = np.zeros((len(g) - 1, len(hdrs)))
for i1 in range(len(g) - 1):
    # split string into array and define as number array
    Dp[i1, :] = np.array(list(eval(g[i0])))
    i0 += 1
# geometric mean diameter in micrometer
dpg1 = np.array(Dp[:, 2]) / 1000
# transpose dpg to make column-array of geometric mean diameters from row-array
# of geometric mean diameters
dpg = dpg1.T
G.close()  # close .csv

rcParams['font.size'] = fs
#rcParams['axes.formatter.useoffset'] = False    
plt.rcParams.update({'font.size': fs})
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']   

rsindex = 0
for RS in resolution:
  output_filename=f"ACTIVATE_DataRetrievals_{RS}.npy"
  OP_Dictionary = np.load(output_filename,allow_pickle='TRUE')


  IRI = grabvalues(OP_Dictionary,'IRI_dry')
  RRI = grabvalues(OP_Dictionary,'RRI_dry')
  kappa = grabvalues(OP_Dictionary,'Kappa')
  y2 = np.vstack((IRI[0],kappa[0]))
  stats_y2 = StatsCode.Survey(y2)

  Sc = grabvalues(OP_Dictionary,"Meas_Sca_Coef_dry")
  #Sc[Sc==0]='nan'
  Abs = grabvalues(OP_Dictionary,"Meas_Abs_Coef_dry") 

  #Abs[Abs==0]='nan'
  y0 = dict()
  y0[0] = Sc
  y0[1] = Abs 

  SSA = grabvalues(OP_Dictionary,"Meas_SSA_dry")
  #SSA[SSA==0]='nan'  

  SSA_amb = grabvalues(OP_Dictionary,"Meas_SSA_amb")
  #ix = np.where((Abs[1,:]*pow(10,6)<=2))
  #ix = np.where((SSA_amb==0))
  #SSA_amb[ix]=float(1) 

  Ext_amb = grabvalues(OP_Dictionary,"Meas_Ext_Coef_amb")
  fRH = grabvalues(OP_Dictionary,"Meas_fRH")
  y1 = dict()
  y1[0] = Ext_amb[0]
  y1[1] = SSA_amb[0]
  y1[2] = fRH[0]
  #y1[y1==0]='nan'  

  Cal_Sc = grabvalues(OP_Dictionary,"Cal_Sca_Coef_dry")
  #Cal_Sc[Cal_Sc==0]='nan'
  Cal_Abs = grabvalues(OP_Dictionary,"Cal_Abs_Coef_dry")
  #Abs[Abs==0]='nan'
  x0 = dict()
  x0[0] = Cal_Sc
  x0[1] = Cal_Abs 

  #x0[x0==0]='nan'
  #print(x0[i1,:])
  Cal_ssa = grabvalues(OP_Dictionary,"Cal_SSA_dry")
  Cal_SSA = dict()
  Cal_SSA[0] = Cal_ssa[1][0]
  Cal_SSA[1] = Cal_ssa[2][0]
  Cal_SSA[2] = Cal_ssa[4][0]
  #Cal_SSA[Cal_SSA==0]='nan'  

  Cal_SSA_amb = grabvalues(OP_Dictionary,'Cal_SSA_amb_0.55')  

  Cal_Ext_amb = grabvalues(OP_Dictionary,'Cal_Ext_Coef_amb_0.532')
  Cal_fRH = grabvalues(OP_Dictionary,'Cal_fRH_0.55')
  x1 = dict()
  x1[0] = Cal_Ext_amb[0][0]*pow(10,6)
  x1[1] = Cal_SSA_amb[0][0]
  x1[2] = Cal_fRH[0][0]
  #x1[x1==0]='nan'  

  SD = grabvalues(OP_Dictionary,'LAS_')
  sd = np.zeros((len(SD.keys()),len(SD[0][0])))
  for i1 in range(len(SD)):
    sd[i1,:] = SD[i1]
  sd[sd.astype(str)=='nan']=0

  stats_sd = StatsCode.Survey(sd)
  #mean_sd = np.squeeze(np.nanmean(sd,1,where=sd>0))  

  pltidx = np.zeros((len(wvl),2)).astype(int)
  j1 = 0
  j2 = 0
  for i1 in range(len(wvl)):
    if i1 < 3:
      pltidx[i1,:] = [j1,j2]
      j1 = j1 + 1
    elif i1 == 3:
      j1 = 0
      j2 = 1
      pltidx[i1,:] =[j1,j2]
    else:
      j1 = j1 + 1
      pltidx[i1,:] =[j1,j2]
  
  ttl = ["Dry Scattering", "Dry Absorption"]
  FIGLBLS = np.array([["(a)","(b)"],["(c)","(d)"],["(e)","(f)"]])
  rcParams['figure.figsize'] = 15, 20
  fig,ax2=plt.subplots(3, 2) # create figure and subplot
  #xy0mmax = 0
  #xy1mmax = 0
  #for i1 in x0:
  #  xy0max = np.nanmax([xy0mmax,np.nanmax(np.multiply(x0[0][i1],pow(10,6))),np.nanmax(y0[0][i1])])
  #  xy1max = np.nanmax([xy1mmax,np.nanmax(np.multiply(x0[1][i1],pow(10,6))),np.nanmax(y0[1][i1])])
  #xymax = np.array([xy0max,xy1max])
  xymax = np.array([300,15])

  bounds = bds1[rsindex,:]
  lenbnds = len(bounds)
  boundsLbs = bounds.astype(str)
  boundsLbs[lenbnds-2] = f">{boundsLbs[lenbnds-2]}"
  boundsLbs[lenbnds-1] = ""
  norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

  stats_dict = np.zeros((len(wvl)+6,45))
  #stats_dict = []
  i0 = 0
  for i1 in range(len(wvl)):
    # Create heatmap
    if i1 < 3:
      x = x0[0][i1][0]*pow(10,6)
      y = y0[0][i1][0]
    else:
      x = x0[1][i1-3][0]*pow(10,6)
      y = y0[1][i1-3][0]  

    Npt = len((y[y.astype(str)!='nan']))
    idx = np.where(((y.astype(str)!='nan')&(x.astype(str)!='nan')))[0]
    x = x[idx]
    y = y[idx]
    x = x[y>0]
    y = y[y>0]  
    stats_dict[i0,:] = np.hstack((wvl[i1],StatsCode.Comparison(x,y),Npt))
    i0 += 1   

  #  y = list(itertools.chain(*y))
  #  x = list(itertools.chain(*x))  

    H, xedges, yedges = np.histogram2d(x, y, bins=(64,64),range=([[0, xymax[pltidx[i1,1]]], [0, xymax[pltidx[i1,1]]]]))
    H = H.T
    X, Y = np.meshgrid(xedges, yedges)
    # Plot heatmap
  #  im = ax2[pltidx[i1,0],pltidx[i1,1]].pcolormesh(X,Y,np.where(H == 0, np.nan, H), cmap=cmap, vmin=1, vmax=100)
    im = ax2[pltidx[i1,0],pltidx[i1,1]].pcolormesh(X,Y,H, cmap=cmap, norm=norm)
    ax2[pltidx[i1,0],pltidx[i1,1]].set_facecolor(gry)
    if i1 < 3: 
      ax2[pltidx[i1,0],pltidx[i1,1]].set_ylabel("measured", fontsize=fs) # set yaxis label   
    if pltidx[i1,0] == 0:
      ax2[pltidx[i1,0],pltidx[i1,1]].set_title("%s\n(Mm$^{-1}$)"%(ttl[pltidx[i1,1]]), fontsize=fs) #set title as flight date.
    elif pltidx[i1,0] == 2:
      ax2[pltidx[i1,0],pltidx[i1,1]].set_xlabel("calculated", fontsize=fs) # set xaxis label 
    ax2[pltidx[i1,0],pltidx[i1,1]].set_ylim(0,xymax[pltidx[i1,1]]) # cut y-axis off at zero   
    ax2[pltidx[i1,0],pltidx[i1,1]].set_xlim(0,xymax[pltidx[i1,1]])    
    # set the line widths of the axes
    for axis in ['top','bottom','left','right']:
        ax2[pltidx[i1,0],pltidx[i1,1]].spines[axis].set_linewidth(1.5)     
    ax2[pltidx[i1,0],pltidx[i1,1]].tick_params(direction='in', length=16, width=3) # set inside facing ticks, ticklength, and tick line width
    ax2[pltidx[i1,0],pltidx[i1,1]].tick_params(axis='both', labelsize=fs, rotation=0)  
    for label in ax2[pltidx[i1,0],pltidx[i1,1]].get_xticklabels():
        label.set_horizontalalignment('center')
    at = AnchoredText(FIGLBLS[pltidx[i1,0],pltidx[i1,1]], prop=dict(size=fs), frameon=False, loc='upper left')
    ax2[pltidx[i1,0],pltidx[i1,1]].add_artist(at)
    ytks = ax2[pltidx[i1,0],pltidx[i1,1]].get_yticks()
    ytklbls = ["%i"%ix for ix in ytks]
    xtklbls = ["%i"%ix for ix in ytks]
    xtklbls[0] = ""
    ax2[pltidx[i1,0],pltidx[i1,1]].set_xticks(ytks, xtklbls) 
    ax2[pltidx[i1,0],pltidx[i1,1]].set_yticks(ytks, ytklbls) 
    ax2[pltidx[i1,0],pltidx[i1,1]].plot(ytks,ytks,'--m',lw=3)  
  # display and save figure using the *.ict data filename
  plt.subplots_adjust(bottom=0.1, right=0.77, top=0.9)
  cax = plt.axes([0.8, 0.1, 0.055, 0.8])
  #cbar = plt.colorbar(im,cax=cax,ticks=np.hstack((1, range(10, 100, 10))))
  cbar =  plt.colorbar(im,cax=cax,cmap=cmap, norm=norm,boundaries=bounds,ticks=bounds,format='%1i')
  cbar.set_ticklabels(boundsLbs) 
  cbar.ax.tick_params(length=16, width=3, which="major")
  cbar.outline.set_linewidth(1.5)
  #cbar.ax.get_yaxis().set_ticks([])
  #for j, lab in enumerate(['0','$1$','$10$','$20$','$40$','$50$','$60$','$70$','$80$','$90$','$>100$']):
  cbar.set_label('count',labelpad=-25)
  plt.savefig(f"ACTIVATE-DataRetrievals_1to1_dry_{RS}", dpi=300)
  plt.show() # function to display the plot        
  plt.close() # 

  bounds = bds2[rsindex,:]
  lenbnds = len(bounds)
  boundsLbs = bounds.astype(str)
  boundsLbs[lenbnds-2] = f">{boundsLbs[lenbnds-2]}"
  boundsLbs[lenbnds-1] = ""
  norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

  rcParams['figure.figsize'] = 10, 20
  fig,ax2=plt.subplots(3, 1) # create figure and subplot
  #xymax = np.array([np.nanmax(np.vstack((Cal_SSA,SSA))),
  #                  np.nanmax(np.vstack((Cal_SSA,SSA)))])
  #fig,ax2=plt.subplots(1, 1) # create figure and subplot
  FIGLBLS = ["(a)","(b)","(c)"]
  ttl = "SSA"
  ws = [450,550,700]
  xymin = 0.6  
  xymax = 1.00
    #xymax = 150
  for i2 in [0,1,2]:
    # Create heatmap
    x = SSA[i2][0]
    y = Cal_SSA[i2] 

    Npt = len((y[y.astype(str)!='nan']))  
    idx = np.where((y.astype(str)!='nan')&(x.astype(str)!='nan'))[0]
  #  print(idx)
    x = x[idx]
    y = y[idx]
    x = x[y>0]
    y = y[y>0]  

    stats_dict[i0,:] = np.hstack((ws[i2],StatsCode.Comparison(x,y),Npt))
    i0 += 1
    #y = list(itertools.chain(*y))
    #x = list(itertools.chain(*x))
    H, xedges, yedges = np.histogram2d(x, y, bins=(64,64),range=([[xymin, xymax], [xymin, xymax]]))
    H = H.T
    X, Y = np.meshgrid(xedges, yedges)
    # Plot heatmap
    im = ax2[i2].pcolormesh(X,Y,H, cmap=cmap, norm=norm)
    ax2[i2].set_facecolor(gry)
    if i2 == 2:
      ax2[i2].set_xlabel("calculated", fontsize=fs) # set xaxis label 
    if i1 == 1:
      ax2[i2].set_title(f'{ttl}', fontsize=fs) #set title as flight date.
    ax2[i2].set_ylabel("measured", fontsize=fs) # set yaxis label   
    ax2[i2].set_ylim(xymin,xymax) # cut y-axis off at zero   
    ax2[i2].set_xlim(xymin,xymax)    
    # set the line widths of the axes
    for axis in ['top','bottom','left','right']:
        ax2[i2].spines[axis].set_linewidth(1.5)     
    ax2[i2].tick_params(direction='in', length=16, width=3) # set inside facing ticks, ticklength, and tick line width
    ax2[i2].tick_params(axis='both', labelsize=fs, rotation=0)  
    for label in ax2[i2].get_xticklabels():
        label.set_horizontalalignment('center')
    #display and save figure using the *.ict data filename 
    at = AnchoredText(FIGLBLS[i2], prop=dict(size=fs), frameon=False, loc='upper left')
    ax2[i2].add_artist(at)
    ytks = ax2[i2].get_yticks()
    ytklbls = ["%0.2f"%ix for ix in ytks]
    xtklbls = ["%0.2f"%ix for ix in ytks]
    xtklbls[0] = ""
    ax2[i2].set_xticks(ytks, xtklbls)
    ax2[i2].set_yticks(ytks, ytklbls)
    ax2[i2].plot(ytks,ytks,'--m',lw=3)   
  plt.subplots_adjust(left=0.16, bottom=0.1, right=0.725, top=0.9)
  cax = plt.axes([0.775, 0.1, 0.055, 0.8])
  cbar =  plt.colorbar(im,cax=cax,cmap=cmap, norm=norm,boundaries=bounds,ticks=bounds,format='%1i')
  cbar.set_ticklabels(boundsLbs) 
  cbar.outline.set_linewidth(1.5)
  cbar.ax.tick_params(length=16, width=3, which="major")
  cbar.set_label('count',labelpad=-25)
  plt.savefig(f"ACTIVATE-DataRetrievals_1to1_SSA_{RS}", dpi=300)
  plt.show() # function to display the plot        
  plt.close() # 

  bounds = bds3[rsindex,:]
  lenbnds = len(bounds)
  boundsLbs = bounds.astype(str)
  boundsLbs[lenbnds-2] = f">{boundsLbs[lenbnds-2]}"
  boundsLbs[lenbnds-1] = ""
  norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

  rcParams['figure.figsize'] = 10, 20
  fig,ax2=plt.subplots(3, 1) # create figure and subplot  
  FIGLBLS = ["(a)","(b)","(c)"]
  ttl = ["Ambient Extinction\n(Mm$^{-1}$)","Ambient SSA","f(RH) (80% to 20%)"]
  ws = [532,550,550]
  xymax = np.array([300,1,6])
  xymin = np.array([0,0.75,0])  
  for i2 in [0,1,2]:
    # Create heatmap
    x = x1[i2]
    y = y1[i2][0]
  #  print(x)
  #  print(y)  
    Npt = len((y[y.astype(str)!='nan']))
    idx = np.where((y.astype(str)!='nan')&(x.astype(str)!='nan'))[0]
  #  print(idx)
    x = x[idx]
    y = y[idx]
    x = x[y>0]
    y = y[y>0]  

    stats_dict[i0,:] = np.hstack((ws[i2],StatsCode.Comparison(x,y),Npt))
    i0 += 1 

    #xymax = np.nanmax(np.vstack((x,y)))
  #  y = list(itertools.chain(*y))
  #  x = list(itertools.chain(*x))
    H, xedges, yedges = np.histogram2d(x, y, bins=(64,64),range=([[xymin[i2], xymax[i2]], [xymin[i2], xymax[i2]]]))
    H = H.T
    X, Y = np.meshgrid(xedges, yedges)
    # Plot heatmap
    im = ax2[i2].pcolormesh(X,Y,H, cmap=cmap, norm=norm)
    ax2[i2].set_facecolor(gry)  

    if i2 == 2:
      ax2[i2].set_xlabel("calculated", fontsize=fs) # set xaxis label 
    ax2[i2].set_title(f'{ttl[i2]}', fontsize=fs) #set title as flight date.
    ax2[i2].set_ylabel("measured", fontsize=fs) # set yaxis label   
    ax2[i2].set_ylim(xymin[i2],xymax[i2]) # cut y-axis off at zero   
    ax2[i2].set_xlim(xymin[i2],xymax[i2])    
    # set the line widths of the axes
    for axis in ['top','bottom','left','right']:
        ax2[i2].spines[axis].set_linewidth(1.5)     
    ax2[i2].tick_params(direction='in', length=16, width=3) # set inside facing ticks, ticklength, and tick line width
    ax2[i2].tick_params(axis='both', labelsize=fs, rotation=0)  
    for label in ax2[i2].get_xticklabels():
        label.set_horizontalalignment('center')
    #display and save figure using the *.ict data filename 
    at = AnchoredText(FIGLBLS[i2], prop=dict(size=fs), frameon=False, loc='upper left')
    ax2[i2].add_artist(at)
    ytks = ax2[i2].get_yticks()
    if i2 == 1:
       ytklbls = ["%0.2f"%ix for ix in ytks]
       xtklbls = ["%0.2f"%ix for ix in ytks]
    else:
      ytklbls = ["%i"%ix for ix in ytks]
      xtklbls = ["%i"%ix for ix in ytks]
    xtklbls[0] = ""
    ax2[i2].set_xticks(ytks, xtklbls)
    ax2[i2].set_yticks(ytks, ytklbls)
    ax2[i2].plot(ytks,ytks,'--m',lw=3)   
  plt.subplots_adjust(left=0.16, bottom=0.1, right=0.725, top=0.9)
  cax = plt.axes([0.775, 0.1, 0.055, 0.8])
  cbar =  plt.colorbar(im,cax=cax,cmap=cmap, norm=norm,boundaries=bounds,ticks=bounds,format='%1i')
  cbar.set_ticklabels(boundsLbs) 
  cbar.outline.set_linewidth(1.5)
  cbar.ax.tick_params(length=16, width=3, which="major")
  cbar.set_label('count',labelpad=-25)
  plt.savefig(f"ACTIVATE-DataRetrievals_1to1_amb_{RS}", dpi=300)
  plt.show() # function to display the plot        
  plt.close() # 

  cols = ['param','Dry_Scattering', 'Dry_Scattering', 'Dry_Scattering', 'Dry_Absorption', 
          'Dry_Absorption', 'Dry_Absorption', 'Dry_SSA', 'Dry_SSA', 'Dry_SSA', 'Amb_Extinction', 'Amb_SSA', 'Amb_fRH']
  colnames = ','.join(str(e) for e in cols)
  rows = np.hstack(('wavelength_nm','R','p-value',prctile_lst_rb,"mean_rb","stdev_rb",prctile_lst_arb,
                  "mean_arb","stdev_arb",'NMAD','MAD','NRMSD','RMSD',prctile_lst_x,"mean_x","stdev_x",
                  prctile_lst_y,"mean_y","stdev_y",'count','total_count'))
  str_data = np.char.mod("%10.6f", stats_dict.T)
  str_data= np.column_stack((rows,str_data))
  output_filename = f"ACTIVATE-DataRetrievals_1to1_stats_{RS}.csv"
  with open(output_filename, 'w') as f:
      np.savetxt(f, str_data, delimiter=', ', fmt='%s', header=colnames)  

  rcParams['figure.figsize'] = 9.5, 7.5
  fig,ax2=plt.subplots(1, 1) # create figure and subplot
  x = dpg*1000
  x_min_max = np.array([x[0],x[-1]])
  Y100 = stats_sd[6,:]
  Y90 = stats_sd[5,:] 
  Y75 = stats_sd[4,:]
  Y50 = stats_sd[3,:]
  Y25 = stats_sd[2,:]
  Y10 = stats_sd[1,:]
  Y0 = stats_sd[0,:]
  Ymean = stats_sd[7,:]
  y_min_max = np.array([0.1,100000])  
#  y_min_max = np.array([np.nanmin(Y4[Y4>0]),np.nanmax(Y0[Y0>0])])
  ax2.plot(x,Y100,'--r',lw=6)
  #ax2.plot(x,Y1,':m',lw=3)
  ax2.plot(x,Ymean,'-k',lw=6) 
  #ax2.plot(x,Y3,':m',lw=3)
  ax2.plot(x,Y0,'--r',lw=6)
  ax2.fill_between(x, Y100, Ymean, color='green',alpha=0.5) 
  ax2.fill_between(x, Ymean, Y0, color='green',alpha=0.5)     
  uni = r"$(\rm cm^{-3})$"
  uni2 = r"$(\rm nm)$"
  ax2.set_ylabel("dN/dlog(D) %s"%uni) 
  ax2.set_xlabel("Dry D %s"%uni2) 
  ax2.set_xscale("log")
  ax2.set_yscale("log")      
  ax2.set_ylim(y_min_max[0],y_min_max[1]) # cut y-axis off at zero   
  ax2.set_xlim(x_min_max[0],x_min_max[1])
  xtklbls = ["%i"%(x[ix]) for ix in range(0,len(x)+1,5)]
  ax2.set_xticks(x[range(0,len(x)+1,5)], xtklbls)    
  # set the line widths of the axes
  for axis in ['top','bottom','left','right']:
      ax2.spines[axis].set_linewidth(1.5)     
  ax2.tick_params(direction='in', length=16, width=3) # set inside facing ticks, ticklength, and tick line width
  ax2.tick_params(axis='both', labelsize=fs, rotation=0)  
  ax2.tick_params(axis='both',which="minor",direction='in', length=8, width=1.5) # set inside facing ticks, ticklength, and tick line width
  for label in ax2.get_xticklabels():
      label.set_horizontalalignment('center')
  #display and save figure using the *.ict data filename 
  plt.subplots_adjust(left=0.2, bottom=0.2, right=0.9, top=0.9)
  plt.savefig(f"ACTIVATE-DataRetrievals_LAS_dry_{RS}", dpi=300)
  plt.show() # function to display the plot        
  plt.close() # 

  cols = ["Dp","0","10","25","50","75","90","100","mean","stdev","count"]
  colnames = ','.join(e for e in cols)
  rows = ["%i"%(x[ix]) for ix in range(0,len(x))]
  str_data = np.char.mod("%10.6f", stats_sd.T)
  str_data= np.column_stack((rows,str_data))
  output_filename = f"ACTIVATE-DataRetrievals_LAS_SD_stats_{RS}.csv"
  with open(output_filename, 'w') as f:
      np.savetxt(f, str_data, delimiter=', ', fmt='%s', header=colnames)  

  rcParams['figure.figsize'] = 12, 15
  fig,ax2=plt.subplots(2, 1) # create figure and subplot
  #fig,ax2=plt.subplots(1, 1) # create figure and subplot
  FIGLBLS = ["(a)","(b)"]
  ttl = ["IRI","$\kappa$"]  

  ymin = 0
  for i2 in [0,1]:
    y = y2[i2,:]  

    ymax = np.nanmax(y)
    #xymax = 150
  #  if i2 == 0:
  #    ax2[i2].set_xscale("log")
    ax2[i2].set_yscale("log")   
    ax2[i2].hist(y,bins=Bin[Lst[i2]]) 
    ax2[i2].set_xlabel(f'{ttl[i2]}', fontsize=fs) # set xaxis label 
    #ax2[i2].set_title(f'{ttl[i2]}', fontsize=fs) #set title as flight date.
    ax2[i2].set_ylabel("count", fontsize=fs) # set yaxis label   
    #ax2[i2].set_ylim(xymin[i2],xymax) # cut y-axis off at zero   
    ax2[i2].set_xlim(ymin,ymax)    
    # set the line widths of the axes
    for axis in ['top','bottom','left','right']:
        ax2[i2].spines[axis].set_linewidth(1.5)     
    ax2[i2].tick_params(direction='in', length=16, width=3) # set inside facing ticks, ticklength, and tick line width
    ax2[i2].tick_params(axis='both', labelsize=fs, rotation=0)  
    ax2[i2].tick_params(axis='y',which="minor",direction='in', length=8, width=1.5) # set inside facing ticks, ticklength, and tick line width  

    for label in ax2[i2].get_xticklabels():
        label.set_horizontalalignment('center')
    #display and save figure using the *.ict data filename 
    at = AnchoredText(FIGLBLS[i2], prop=dict(size=fs), frameon=False, loc='upper right')
    ax2[i2].add_artist(at)
  plt.subplots_adjust(bottom=0.1, right=0.75, top=0.9)  

  plt.savefig(f"ACTIVATE-DataRetrievals_histograms_{RS}", dpi=300)
  plt.show() # function to display the plot        
  plt.close() ##
  rsindex += 1

  cols = ["param","0","10","25","50","75","90","100","mean","stdev","count"]
  colnames = ','.join(e for e in cols)  
  rows = Lst
  str_data = np.char.mod("%10.6f", stats_y2.T)
  str_data= np.column_stack((rows,str_data))
  output_filename = f"ACTIVATE-DataRetrievals_Kappa_IRI_stats_{RS}.csv"
  with open(output_filename, 'w') as f:
        np.savetxt(f, str_data, delimiter=', ', fmt='%s', header=colnames)  

#xymax = [np.nanmax(np.vstack((np.squeeze(Sc[1:,:]),np.squeeze(OP_Dictionary["CalCoef_dry"][:,0:3].T)))), np.nanmax(np.vstack((np.squeeze(Abs),np.squeeze(np.multiply(OP_Dictionary["CalCoef_dry"][:,3:6],pow(10,6))).T)))]
#xymax = [np.nanmax(np.vstack((np.squeeze(Sc[1:,:]),np.squeeze(OP_Dictionary["CalCoef_dry"][:,0:3].T)))), np.nanmax(np.multiply(OP_Dictionary["CalCoef_dry"][:,3:6],pow(10,6)))]

#rcParams['figure.figsize'] = 10, 20
#rcParams['font.size'] = fs
##rcParams['axes.formatter.useoffset'] = False    
#plt.rcParams.update({'font.size': fs})
#plt.rcParams['font.family'] = 'serif'
#plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

#for i1 in range(len(xymax)):
#  if i1 == 0:
#    x = np.array([0,xymax[i1]])
#    y = Line(1,x,0)
#    ax2[i1].plot(x,y,'--k',lw=2)
#    for i2 in range(0,3):    
#      x0 = np.multiply(OP_Dictionary["CalCoef_dry"][:,i2],pow(10,6))
#      y0 = y0[i2,:]
#      ax2[i1].plot(x0, y0,ln_stl_clr_shp[i2],lw=2)
#    ax2[i1].set_ylabel("measured", fontsize=fs) # set yaxis label
#  else:
#    x = np.array([0,xymax[i1]])
#    y = Line(1,x,0)
#    ax2[i1].plot(x,y,'--k',lw=2)
#    for i2 in range(3,6):   
#      x0 = np.multiply(OP_Dictionary["CalCoef_dry"][:,i2],pow(10,6))
#      y0 = y0[i2,:]
#      ax2[i1].plot(x0, y0,ln_stl_clr_shp[i2],lw=2)
#    ax2[i1].set_ylabel("measured", fontsize=fs) # set yaxis label
#    ax2[i1].set_xlabel("calculated", fontsize=fs) # set xaxis label 
#          
#  ax2[i1].set_ylim(0,xymax[i1]) # cut y-axis off at zero   
#  ax2[i1].set_xlim(0,xymax[i1])    
#  # set the line widths of the axes
#  for axis in ['top','bottom','left','right']:
#      ax2[i1].spines[axis].set_linewidth(1.5)     
#  ax2[i1].tick_params(direction='in', length=16, width=3) # set inside facing ticks, ticklength, and tick line width
#  ax2[i1].set_title("%s (Mm$^{-1}$)"%(ttl[i1]), fontsize=fs) #set title as flight date.
#  ax2[i1].tick_params(axis='both', labelsize=fs, rotation=0)  
#  for label in ax2[i1].get_xticklabels():
#      label.set_horizontalalignment('center')
## display and save figure using the *.ict data filename 
#plt.savefig("ACTIVATE-DataRetrievals_1to1_dry", dpi=300)
#plt.show() # function to display the plot        
#plt.close() #

#rcParams['figure.figsize'] = 10, 10
#rcParams['font.size'] = fs
##rcParams['axes.formatter.useoffset'] = False    
#plt.rcParams.update({'font.size': fs})
#plt.rcParams['font.family'] = 'serif'
#plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']#

#ln_stl_clr_shp = 'go'
#ttl = "Scattering"
#fig,ax2=plt.subplots(1, 1) # create figure and subplot
#xymax = np.nanmax(np.vstack((y1,OP_Dictionary["CalCoef_amb"])))
#x = np.array([0,xymax])
#y = Line(1,x,0)
#ax2.plot(x,y,'--k',lw=2)
#x0 = np.multiply(OP_Dictionary["CalCoef_amb"],pow(10,6))
#y0 = y1
#ax2.plot(x0, y0,ln_stl_clr_shp,lw=2)
#ax2.set_xlabel("calculated", fontsize=fs) # set xaxis label
#ax2.set_ylabel("measured", fontsize=fs) # set yaxis label        
#ax2.set_ylim(0,xymax) # cut y-axis off at zero   
#ax2.set_xlim(0,xymax)    
## set the line widths of the axes
#for axis in ['top','bottom','left','right']:
#    ax2.spines[axis].set_linewidth(1.5)     
#ax2.tick_params(direction='in', length=16, width=3) # set inside facing ticks, ticklength, and tick line width
#ax2.set_title("%s (Mm$^{-1}$)"%(ttl), fontsize=fs) #set title as flight date.
#ax2.tick_params(axis='both', labelsize=fs, rotation=0)  
#for label in ax2.get_xticklabels():
#    label.set_horizontalalignment('center')
## display and save figure using the *.ict data filename 
#plt.savefig("ACTIVATE-DataRetrievals_1to1_amb", dpi=300)
#plt.show() # function to display the plot        
#plt.close() 
