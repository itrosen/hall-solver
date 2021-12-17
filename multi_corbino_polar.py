"""
Created on Dec 16 2021
@author: Ilan Rosen
Poisson equation solver for the Hall effect in a multi-terminal Corbino geometry.
"""

import time
import math
import numpy as np
import scipy.sparse as sp                 # import sparse matrix library
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve

# import the file where the differentiation matrix operators are defined
from diff_matrices import Diff_mat_1D, Diff_mat_2D   
from diff_matrices_polar import Diff_mat_2D_polar

class multi_corbino_polar():
	"""The class for a Corbino with small voltage contacts
	Use a square device with fixed size to make this easier
	Args:
		Nr : number of points in grid along radius
		Nt : number of points in grid along theta
	"""

	def __init__(self, Nr = 81, Nt = 1000, ri = 50, ro = 100):

		# Define coordinate variables
		self.Nr = Nr # radial
		self.Nt = Nt # theta
		self.ri = ri # inner radius
		self.ro = ro # outer radius
		self.r = np.linspace(ri, ro, num=self.Nr)
		self.t = np.linspace(0, 2*np.pi, endpoint = False, num=self.Nt)
		self.dr = self.r[1] - self.r[0]                # grid spacing
		self.dt = self.t[1] - self.t[0]                # grid spacing

		self.R,self.T = np.meshgrid(self.r,self.t)          # 2D meshgrid
		self.invR = 1/self.R

		# 1D indexing
		self.Ru = self.R.ravel()                  # Unravel 2D meshgrid to 1D array
		self.Tu = self.T.ravel()

		# contact angle
		self.contact_angle = np.pi/20 # lol idk. also this will be half the contact angle.

		###################################
		### Search for boundary indices ###
		###################################
		start_time = time.time()

		# Outer boundary
		self.ind_unravel_o = np.squeeze(np.where(self.Ru==self.r[-1]))

		# Inner boundary
		self.ind_unravel_i = np.squeeze(np.where(self.Ru==self.r[0]))

		# Outer contacts
		self.ind_unravel_oC_L = np.squeeze(np.where( (self.Ru == self.r[-1]) & ( np.abs(self.Tu - np.pi) <= self.contact_angle) ))  # Left
		self.ind_unravel_oC_R = np.squeeze(np.where( (self.Ru == self.r[-1]) & 
							( ( np.abs(self.Tu) <= self.contact_angle) | (np.abs(self.Tu - 2*np.pi) <= self.contact_angle) ) ))  # Right
		self.ind_unravel_oC_T = np.squeeze(np.where( (self.Ru == self.r[-1]) & ( np.abs(self.Tu - np.pi/2) <= self.contact_angle) ))  # Top
		self.ind_unravel_oC_B = np.squeeze(np.where( (self.Ru == self.r[-1]) & ( np.abs(self.Tu - 3/2*np.pi) <= self.contact_angle) ))  # Bottom

		# Inner contacts
		self.ind_unravel_iC_L = np.squeeze(np.where( (self.Ru == self.r[0]) & ( np.abs(self.Tu - np.pi) <= self.contact_angle) ))  # Left
		self.ind_unravel_iC_R = np.squeeze(np.where( (self.Ru == self.r[0]) &
							( ( np.abs(self.Tu) <= self.contact_angle) | (np.abs(self.Tu - 2*np.pi) <= self.contact_angle) ) ))  # Right
		self.ind_unravel_iC_T = np.squeeze(np.where( (self.Ru == self.r[0]) & ( np.abs(self.Tu - np.pi/2) <= self.contact_angle) ))  # Top
		self.ind_unravel_iC_B = np.squeeze(np.where( (self.Ru == self.r[0]) & ( np.abs(self.Tu - 3/2*np.pi) <= self.contact_angle) ))  # Bottom

		print("Boundary search time = %1.4s" % (time.time()-start_time))

		# Load finite difference matrix operators
		self.Dr_2d, self.rDr_2d, self.D2r_2d, self.rDt_2d, self.r2D2t_2d = Diff_mat_2D_polar(self.Nr,self.Nt, self.r)

		# Initiate empty solution matrix
		self.u = 0

		# Contacts not yet implemented
		# self.contacts = []


	def solve(self, lmbda):
		'''
		Constructs matrix problem and solves Poisson equation
		ARGS
			lmbda : sigma_xy / sigma_xx. Must be finite
		RETURNS
			self.u : electric potential
		'''

		self.lmbda = lmbda

		# Construct system matrix without boundary conditions. Laplacian in polar coordinates
		start_time = time.time()
		I_sp = sp.eye(self.Nr*self.Nt).tocsr()
		# L_sys = (1 / self.Ru) * self.Dr_2d/(2 * self.dr) + self.D2r_2d/self.dr**2 + (1 / self.Ru**2) * self.D2t_2d/self.dt**2
		L_sys = self.rDr_2d/(2 * self.dr) + self.D2r_2d/self.dr**2 + self.r2D2t_2d/self.dt**2

		# Boundary operators
		BD = I_sp       # Dirichlet boundary operator
		BNO = self.Dr_2d / (2 * self.dr) + lmbda * self.rDt_2d / (2 * self.dt)     # Mixed boundary operator
		BNI = self.Dr_2d / (2 * self.dr) + lmbda * self.rDt_2d / (2 * self.dt)     # Mixed boundary operator

		# ZERO CURRENT THROUGH EDGES
		L_sys[self.ind_unravel_o,:] = BNO[self.ind_unravel_o,:] # Outer
		L_sys[self.ind_unravel_i,:] = BNI[self.ind_unravel_i,:] # Inner

		# DIRICHLET BOUNDARY CONDITIONS FOR SOURCE/DRAIN CONTACTS
		# Replace above boundary conditions
		L_sys[self.ind_unravel_oC_R,:] = BD[self.ind_unravel_oC_R,:]    # Source at outer right
		L_sys[self.ind_unravel_iC_R,:] = BD[self.ind_unravel_iC_R,:]    # Drain at inner right

		# Source function (right hand side vector)
		g = np.zeros(self.Nr*self.Nt)
		# Insert source function at source terminal
		g[self.ind_unravel_oC_R] = 1 # Dirichlet boundary condition at source

		print("System matrix and right hand vector computation time = %1.6s" % (time.time()-start_time))
		
		start_time = time.time()
		self.u = spsolve(L_sys,g).reshape(self.Nt,self.Nr).T
		print("spsolve() time = %1.6s" % (time.time()-start_time))

	def plot_potential(self):

		if np.all(self.u==0):
			raise Exception('System has not been solved')

		X = self.R * np.cos(self.T)
		Y = self.R * np.sin(self.T)

		fig = plt.figure(figsize = [8,5])
		plt.pcolormesh(X, Y, self.u.T, cmap = 'inferno')
		cbar = plt.colorbar(ticks = np.arange(0, 1.01, 0.2), label = r'$\phi / \phi_s$')
		plt.xlabel(r'x ($\mu$m)');
		plt.ylabel(r'y ($\mu$m)');
		plt.show()

	def measure(self):
		'''
		Measures the voltage at top, left, and bottom of the inner and outer edges
		of the device, averaging over 20 points to avoid aliasing effects from
		the finite mesh size

		RETURNS: list of measurements, in units of potential normalized to source potential,
			in the order [top outside, left outside, bottom outside, top inside, left inside, bottom inside]
		'''

		if np.all(self.u==0):
			raise Exception('System has not been solved')

		U = self.u.T.ravel()

		to = np.mean(U[self.ind_unravel_oC_T]) # top outside
		lo = np.mean(U[self.ind_unravel_oC_L]) # left outside
		bo = np.mean(U[self.ind_unravel_oC_B]) # bottom outside
		ti = np.mean(U[self.ind_unravel_iC_T]) # top inside
		li = np.mean(U[self.ind_unravel_iC_L]) # left inside
		bi = np.mean(U[self.ind_unravel_iC_B]) # bottom inside

		return [to, lo, bo, ti, li, bi]

def main():
	'''
	unit testing
	'''
	cb = multi_corbino_polar()

	cb.solve(4)

	cb.plot_potential()

	print(cb.measure())
















if __name__ == "__main__":
	main()
