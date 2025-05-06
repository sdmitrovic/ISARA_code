import numpy as np
def f_model(x,a,c): 
	"""
	function y = a*x^c 
	
	:param x: base of power function
	:type x: double, float, int  
	:param a: scale of power function
	:type a: double, float, int
	:param c: exponent of power function
	:type c: double, float, int
	:return y: 
	:rtype: double, float, int
	"""   
	y = np.multiply(a,pow(x, c))
	return y