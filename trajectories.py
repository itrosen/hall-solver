"""
Created on Dec 16 2021
@author: Ilan Rosen
Trajectories in (s_xy, s_xx) space
functions output lambda = s_xy / s_xx
all conductivities in units e2/h
resistivities in units ohms/sq

RG flow semicircle trajectory:
sxx ** 2 + (sxy - 0.5) ** 2 = 0.25
"""

import numpy as np

def lambda_sxy_const(lower = 100,
				upper = 5000,
				n = 10):
	'''
	trajectory along sxy = 1
	Args:
		lower : first value of p_xx
		upper : final value of p_xx
		n : number of points
	Returns:
		lmbda : lambda along trajectory
	'''
	pxx = np.linspace(lower, upper, n)/25812
	sxx = pxx / ( pxx**2 + 1 )
	sxy = 1 / ( pxx**2 + 1 )
	return sxy / sxx

def pxx_sxy_const(lower = 100,
				upper = 5000,
				n = 10):
	'''
	trajectory along sxy = 1
	Args:
		lower : first value of p_xx
		upper : final value of p_xx
		n : number of points
	Returns:
		pxx : pxx along trajectory
	'''
	pxx = np.linspace(lower, upper, n)
	return pxx

def semicircle_x2y(x):
	'''
	converts sxx to sxy along RG flow trajectory
	Args: x : sxx. Must be between 0 and 0.5
	Returns: y : sxy
	'''
	if np.any(x > 0.5) or np.any(x < 0):
		raise Exception ('x out of bounds')
	return np.sqrt(0.25 - x**2) + 0.5

def semicircle_y2x(y):
	'''
	converts sxx to sxy along RG flow trajectory
	Args: x : sxx
	Returns: y : sxy
	'''
	return np.sqrt(0.25 - (y - 0.5)**2)

def lambda_semicircle(lower = 100,
				upper = 5000,
				n = 10):
	'''
	trajectory along RG semicircle
	Args:
		lower : first value of p_xx
		upper : final value of p_xx
		n : number of points
	Returns:
		lambda : sxy/sxx along trajectory
	'''
	if lower < 0.1 or lower > 10000:
		raise Exception ('lower limit out of bounds')
	if upper < 0.1 or upper > 10000:
		raise Exception ('upper limit out of bounds')
	if lower >= upper:
		raise Exception ('lower limit exceeds upper limit')

	# estimate sxx lower and sxx final
	sxx_lower = 25812 * lower / (lower**2 + 25812**2)
	sxx_upper = 25812 * upper / (upper**2 + 25812**2)

	# make sxx linearly spaced
	sxx = np.linspace(sxx_lower, sxx_upper, n)

	# solve for sxy
	sxy = semicircle_x2y(sxx)

	return sxy / sxx

def sxx_semicircle(lower = 100,
				upper = 5000,
				n = 10):
	'''
	trajectory along RG semicircle
	Args:
		lower : first value of p_xx
		upper : final value of p_xx
		n : number of points
	Returns:
		lambda : sxy/sxx along trajectory
	'''
	if lower < 0.1 or lower > 10000:
		raise Exception ('lower limit out of bounds')
	if upper < 0.1 or upper > 10000:
		raise Exception ('upper limit out of bounds')
	if lower >= upper:
		raise Exception ('lower limit exceeds upper limit')

	# estimate sxx lower and sxx final
	sxx_lower = 25812 * lower / (lower**2 + 25812**2)
	sxx_upper = 25812 * upper / (upper**2 + 25812**2)

	# make sxx linearly spaced
	sxx = np.linspace(sxx_lower, sxx_upper, n)

	# solve for sxy
	sxy = semicircle_x2y(sxx)

	return sxx

def sxy_semicircle(lower = 100,
				upper = 5000,
				n = 10):
	'''
	trajectory along RG semicircle
	Args:
		lower : first value of p_xx
		upper : final value of p_xx
		n : number of points
	Returns:
		lambda : sxy/sxx along trajectory
	'''
	if lower < 0.1 or lower > 10000:
		raise Exception ('lower limit out of bounds')
	if upper < 0.1 or upper > 10000:
		raise Exception ('upper limit out of bounds')
	if lower >= upper:
		raise Exception ('lower limit exceeds upper limit')

	# estimate sxx lower and sxx final
	sxx_lower = 25812 * lower / (lower**2 + 25812**2)
	sxx_upper = 25812 * upper / (upper**2 + 25812**2)

	# make sxx linearly spaced
	sxx = np.linspace(sxx_lower, sxx_upper, n)

	# solve for sxy
	sxy = semicircle_x2y(sxx)

	return sxy

def pxx_semicircle(lower = 100,
				upper = 5000,
				n = 10):
	'''
	trajectory along RG semicircle
	Args:
		lower : first value of p_xx
		upper : final value of p_xx
		n : number of points
	Returns:
		pxx : pxx along trajectory
	'''
	if lower < 0.1 or lower > 10000:
		raise Exception ('lower limit out of bounds')
	if upper < 0.1 or upper > 10000:
		raise Exception ('upper limit out of bounds')
	if lower >= upper:
		raise Exception ('lower limit exceeds upper limit')

	# estimate sxx lower and sxx final
	sxx_lower = (lower / 25812) / (1 + (lower/25812)**2)
	sxx_upper = (upper / 25812) / (0.95 + (upper/25812)**2) # this is super crude but whatever

	# make sxx linearly spaced
	sxx = np.linspace(sxx_lower, sxx_upper, n)

	# solve for sxy
	sxy = semicircle_x2y(sxx)

	# convert to resistance units
	pxx = 25812 * sxx / (sxx**2 + sxy**2)

	return pxx

def lambda2pxx(lmbda):
	return 25812 / lmbda