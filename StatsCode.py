#########################################################################################################################
# StatsCode.py                                   by:  Joseph Schlosser
#                                                revised:  4 April 2022 
#                                    language (revision):  python3 (3.8.2-0ubuntu2)
#
# description: procedure to calculate closure stats between two measurements of the same property
#
# These statistical calculations require two 1-D parameters with the same length and units, x and y, and do not need to
# be filtered for missing values, which is done in the procedure.
#
# the output of this code is StatsTable, which is a array containing on column for each of the following closure 
# statistics:
# R = correlation coefficient 
# p-value = probability that the two parameters are not correlated (i.e., probability that the null-hypothesis is true)
# 10prctile_relative_bias = 10th percentile of relative bias
# 25prctile_relative_bias = 25th percentile of relative bias
# median_relative_bias = 50th percentile of relative bias (i.e., median relative bias)
# 75prctile_relative_bias = 75th percentile of relative bias
# 90prctile_relative_bias = 90th percentile of relative bias
# 10prctile_abs_relative_bias = 10th percentile of absolute relative bias
# 25prctile_abs_relative_bias = 25th percentile of absolute relative bias
# median_abs_relative_bias = 50th percentile of absolute relative bias (i.e., median relative bias)
# 75prctile_abs_relative_bias = 75th percentile of absolute relative bias
# 90prctile_abs_relative_bias = 90th percentile of absolute relative bias
# NMAD = Normalized Mean Absolute Deviation
# MAD_[units] = Mean Absolute Deviation in user provided units
# NRMSD = Normalized Root-Mean Squared Deviation
# RMSD_[units] = Root-Mean Squared Deviation in user provided units
# x_min_[units] = minimum valid value of x
# x_max_[units] = maximum valid value of x
# y_min_[units] = minimum valid value of y
# y_max_[units] = maximum valid value of y
# count = number of points where both x and y had valid values
#
#
# WARNINGS:
# 1) numpy and scipy.stats must be installed to the python environment
# 
#########################################################################################################################

import numpy as np
from scipy.stats import ttest_ind  
from sklearn.feature_selection import f_regression

def Comparison(x,y):
	y = np.squeeze(y)
	x = np.squeeze(x)
	II3 = np.where((x>0)&(y>0))
	x = x[II3]
	y = y[II3]
	xy = np.matrix(np.vstack((x,y)))
	MinMax_xy = [np.min(xy),np.max(xy)]
	rng =	np.add(MinMax_xy[1],-MinMax_xy[0])
	xstdev = np.std(x)
	ystdev = np.std(y)
	x_mean = np.mean(x)
	y_mean = np.mean(y)
	npt = len(y)
	R = np.corrcoef(x,y)
	R = R[0,1]
	#pval = t_test(x,y,alternative='greater')
	freg=f_regression(x.reshape(-1, 1),y)
	log10_pvalues=np.log10(freg[1])
	mean_ary = np.mean(xy,0)
	dif_ary = y - x
	abs_dif_ary = np.absolute(y - x)
	rb = np.divide(dif_ary,mean_ary)
	arb = np.divide(abs_dif_ary,mean_ary)

	rb_mean = np.mean(rb)
	rbstdev = np.std(rb)
	arb_mean = np.mean(arb)
	arbstdev = np.std(arb)
	prctils = [0,10,25,50,75,90,100]
	relative_bias_prctiles = np.zeros((len(prctils)))
	abs_relative_bias_prctiles = np.zeros((len(prctils)))
	x_prctiles = np.zeros((len(prctils)))
	y_prctiles = np.zeros((len(prctils)))
	for i1 in range(len(prctils)):
		relative_bias_prctiles[i1] = np.percentile(rb,prctils[i1],axis=1)
		abs_relative_bias_prctiles[i1] = np.percentile(arb,prctils[i1],axis=1)
		x_prctiles[i1] = np.percentile(x,prctils[i1],axis=0)
		y_prctiles[i1] = np.percentile(y,prctils[i1],axis=0)

	nrmsd = np.sqrt(np.sum((y-x)**2)/npt)/rng
	rmsd  = np.sqrt(np.sum((y-x)**2)/npt)
	mad = np.sum(np.absolute(y-x))/npt
	nmad =  np.sum(np.absolute(y-x))/npt/rng

	Results = np.hstack((R,log10_pvalues,relative_bias_prctiles,rb_mean,rbstdev,abs_relative_bias_prctiles,
						arb_mean,arbstdev,nmad,mad,nrmsd,rmsd,x_prctiles,x_mean,xstdev,y_prctiles,y_mean,ystdev,npt))
	return Results


def Survey(x):
	prctils = [0,10,25,50,75,90,100]
	prctiles = np.zeros((len(prctils),len(x[:,0])))
	npt = np.zeros((1,len(x[:,0])))
	mn = np.squeeze(np.nanmean(x,1,where=x>0))
	xstdev = np.squeeze(np.std(x,1,where=x>0))
	for i2 in range(len(x[:,0])):
		x1 = x[i2,:]
		npt[0,i2] = len(x1[x1>0])
		for i1 in range(len(prctils)):
			prctiles[i1,i2] = np.squeeze(np.percentile(x1[x1>0],prctils[i1],axis=0))
	op = np.vstack((prctiles,mn,xstdev,npt))		
	return op