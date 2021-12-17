"""
Created on Dec 16 2021
@author: Ilan Rosen
Poisson equation solver for the Hall effect.
Includes classes for Hall bars, Hall bars in a nonlocal geometry, and Corbino disks.
The Hall bar class has build in methods for longitudinal and Hall 4-probe resistance measurements.
Plotting functions assume coordinates are in microns, but the Poisson equation is scale-invariant.
"""

import time
import math
import numpy as np
import scipy.sparse as sp                 # import sparse matrix library
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve

# import the file where the differentiation matrix operators are defined
from diff_matrices import Diff_mat_1D, Diff_mat_2D   

class hallbar():
	"""The class for a Hall bar device
	Source is the left terminal, drain is the right terminal.
	Args:
		Lx : length in x direction
		Ly : length in y direction
		Nx : number of points in grid along x
		Ny : number of points in grid along y
	"""

	def __init__(self, Lx, Ly, Nx = 301, Ny = 201):

		# Initiate with no contacts
		self.contacts = []

		# Define coordinate variables
		self.Nx = Nx
		self.Ny = Ny
		self.Lx = Lx
		self.Ly = Ly
		self.x = np.linspace(0,self.Lx,self.Nx)
		self.y = np.linspace(0,self.Ly,self.Ny) 

		self.dx = self.x[1] - self.x[0]                # grid spacing along x direction
		self.dy = self.y[1] - self.y[0]                # grid spacing along y direction

		self.X,self.Y = np.meshgrid(self.x,self.y)          # 2D meshgrid

		# 1D indexing
		self.Xu = self.X.ravel()                  # Unravel 2D meshgrid to 1D array
		self.Yu = self.Y.ravel()

		# Search for boundary indices
		start_time = time.time()
		self.ind_unravel_L = np.squeeze(np.where(self.Xu==self.x[0]))          # Left boundary
		self.ind_unravel_R = np.squeeze(np.where(self.Xu==self.x[self.Nx-1]))       # Right boundary
		self.ind_unravel_B = np.squeeze(np.where(self.Yu==self.y[0]))          # Bottom boundary
		self.ind_unravel_T = np.squeeze(np.where(self.Yu==self.y[self.Ny-1]))       # Top boundary

		self.ind_boundary_unravel = np.squeeze(np.where((self.Xu==self.x[0]) | (self.Xu==self.x[self.Nx-1]) | (self.Yu==self.y[0]) | (self.Yu==self.y[self.Ny-1])))  # outer boundaries 1D unravel indices
		self.ind_boundary = np.where((self.X==self.x[0]) | (self.X==self.x[self.Nx-1]) | (self.Y==self.y[0]) | (self.Y==self.y[self.Ny-1]))    # outer boundary

		print("Boundary search time = %1.4s" % (time.time()-start_time))

		# Load finite difference matrix operators
		self.Dx_2d, self.Dy_2d, self.D2x_2d, self.D2y_2d = Diff_mat_2D(self.Nx,self.Ny)

		# Initiate empty solution matrix
		self.u = 0

	def solve(self, lmbda):
		# constructs matrix problem and solves Poisson equation
		# Args: lmbda : sigma_xy / sigma_xx. Must be finite
		# Returns: self.u : electric potential

		self.lmbda = lmbda

		# Construct system matrix without boundary conditions
		start_time = time.time()
		I_sp = sp.eye(self.Nx*self.Ny).tocsr()
		L_sys = self.D2x_2d/self.dx**2 + self.D2y_2d/self.dy**2

		# Boundary operators
		BD = I_sp       # Dirichlet boundary operator
		BNx = self.Dx_2d / (2 * self.dx)     # Neumann boundary operator for x component
		BNy = self.Dy_2d / (2 * self.dy)     # Neumann boundary operator for y component

		# DIRICHLET BOUNDARY CONDITIONS FOR CONTACTS
		L_sys[self.ind_unravel_L,:] = BD[self.ind_unravel_L,:]    # Boundaries at the left layer
		L_sys[self.ind_unravel_R,:] = BD[self.ind_unravel_R,:]    # Boundaries at the right edges
		# CURRENT THROUGH EDGES
		L_sys[self.ind_unravel_T,:] = BNy[self.ind_unravel_T,:] - lmbda * BNx[self.ind_unravel_T,:]    # Boundaries at the top layer
		L_sys[self.ind_unravel_B,:] = BNy[self.ind_unravel_B,:] - lmbda * BNx[self.ind_unravel_B,:]    # Boundaries at the bottom layer

		# Source function (right hand side vector)
		g = np.zeros(self.Nx*self.Ny)
		# Insert boundary values at the boundary points
		g[self.ind_unravel_L] = 1 # Dirichlet boundary condition at source
		g[self.ind_unravel_R] = 0 # Dirichlet boundary condition at drain
		g[self.ind_unravel_T] = 0 # No current through top
		g[self.ind_unravel_B] = 0 # No current through bottom

		print("System matrix and right hand vector computation time = %1.6s" % (time.time()-start_time)) 
		
		start_time = time.time()
		self.u = spsolve(L_sys,g).reshape(self.Ny,self.Nx).T
		print("spsolve() time = %1.6s" % (time.time()-start_time))

	def voltage_measurement(self, x1, x2, side='top'):
		# Args: 	x1 : point of V_A
		#			x2 : point of V_B
		#			side ('top', 'bottom', or 'hall') : which side of Hall bar to measure
		# Returns:	V_A - V_B

		if np.all(self.u==0):
			raise Exception('System has not been solved')
		if x1 > self.Lx or x1 < 0 or x2 > self.Lx or x2 < 0:
			raise Exception('Points out of bounds')

		if side=='top':
			ya = self.Ny-1
			yb = self.Ny-1
		elif side=='bottom':
			ya = 0
			yb = 0
		elif side=='hall':
			ya = 0
			yb = self.Ny-1
		else:
			raise Exception('Side must be top or bottom')

		# Find nearest index value to input coordinates
		xa = np.searchsorted(self.x, x1, side='left')
		xb = np.searchsorted(self.x, x2, side='left')

		return self.u[xa, ya] - self.u[xb, yb]

	def plot_potential(self):

		if np.all(self.u==0):
			raise Exception('System has not been solved')

		fig = plt.figure(figsize = [8,5])
		plt.contourf(self.x,self.y,self.u.T,41,cmap = 'inferno')
		cbar = plt.colorbar(ticks = np.arange(0, 1.01, 0.2), label = r'$\phi / \phi_s$')
		plt.xlabel(r'x ($\mu$m)');
		plt.ylabel(r'y ($\mu$m)');
		plt.show()

	def plot_resistance(self):

		if np.all(self.u==0):
			raise Exception('System has not been solved')

		r_top = (self.u[0:-1, -1] - self.u[1:, -1]) * 25812 * self.Ly / self.dx
		r_bottom = (self.u[0:-1, 0] - self.u[1:, 0]) * 25812 * self.Ly / self.dx
		rxx = 25812 / self.lmbda

		fig = plt.figure(figsize = [8,5])
		plt.plot(self.x[0:-1] - self.dx, r_top, 'r', label='top')
		plt.plot(self.x[0:-1] - self.dx, r_bottom, 'b', label='bottom')
		plt.hlines(rxx, self.x[0], self.x[-1], linestyle='dashed', color='grey', label=r'$\rho_{xx}$')
		plt.xlabel(r'x ($\mu$m)');
		plt.ylabel(r'$\rho_{xx}$ $(\Omega)$');
		plt.legend()
		plt.ylim([0, 12000]);
		plt.show()

	def add_contact(self, contact):

		if contact.x1 > self.Lx or contact.x2 > self.Lx:
			raise Exception('Contact out of bounds')

		self.contacts.append(contact)

	def measure_contact_voltageonly(self, contact):
		# Args: 	contact instance
		# Returns:	measured resistivity
		# Voltage is averaged across voltage tap
		# THIS FUNCTION DOES NOT CHECK THE CURRENT!
		# This method assumes 2terminal resistance is h/e2, which in general is wrong

		if np.all(self.u==0):
			raise Exception('System has not been solved')

		if contact.side=='top':
			y = self.Ny-1
		elif contact.side=='bottom':
			y = 0
		else:
			raise Exception('Side must be top or bottom')

		# Average voltage A
		A_indices = np.where(np.abs(self.x - contact.x1) < contact.width)[0]
		A_voltage = self.u[A_indices, y].mean()

		# Average voltage A
		B_indices = np.where(np.abs(self.x - contact.x2) < contact.width)[0]
		B_voltage = self.u[B_indices, y].mean()

		# voltage difference
		v = A_voltage - B_voltage
		# length between contacts
		dx = np.abs(contact.x1 - contact.x2)

		# return apparent resistivity
		return 25812 * v * self.Ly / dx

	def measure_all_contacts_voltageonly(self):
		# Args: none
		# Returns: array; resistivity measurement of all contacts

		if np.all(self.u==0):
			raise Exception('System has not been solved')

		result = []

		for contact in self.contacts:
			result.append(self.measure_contact_voltageonly(contact))
		
		return result

	def measure_contact(self, contact, sxx, sxy):
		'''
		Voltage is averaged across voltage tap
		This method checks the current and outputs resistivity.

		Args:
		contact : contact instance
		sxx : longitudinal
		sxy : hall. sxy/sxx should match self.lmbda

		Returns:	measured resistivity
		'''

		if np.all(self.u==0):
			raise Exception('System has not been solved')

		if contact.side=='top':
			ya = self.Ny-1
			yb = self.Ny-1
		elif contact.side=='bottom':
			ya = 0
			yb = 0
		elif contact.side=='hall':
			ya = 0
			yb = self.Ny-1
		else:
			raise Exception('Side must be top or bottom')

		# Average voltage A
		A_indices = np.where(np.abs(self.x - contact.x1) < contact.width)[0]
		A_voltage = self.u[A_indices, ya].mean()

		# Average voltage B
		B_indices = np.where(np.abs(self.x - contact.x2) < contact.width)[0]
		B_voltage = self.u[B_indices, yb].mean()

		# voltage difference
		v = A_voltage - B_voltage
		# length between contacts
		dx = np.abs(contact.x1 - contact.x2)

		i = self.measure_current(sxx, sxy)

		# return apparent resistivity
		if contact.side=='hall':
			return v / i
		else:
			return v / i * self.Ly / dx

	def measure_all_contacts(self, sxx, sxy):
		# Args: none
		# Returns: array; resistivity measurement of all contacts

		if np.all(self.u==0):
			raise Exception('System has not been solved')

		result = []

		for contact in self.contacts:
			result.append(self.measure_contact(contact, sxx, sxy))
		
		return result

	def measure_current(self, sxx, sxy):
		'''
		ARGS : sxx and sxy : longitudinal and Hall conductivity. units e2/h
		Returns : current moving through device
		''' 

		# choose place to measure: halfway across Hallbar
		ind_x = int(self.Nx/2)

		# calculate electric field using E = -\nabla V
		# x electric field, using second order central finite difference
		E_x = 0.5 * (self.u[ind_x - 1, :] - self.u[ind_x + 1, :]) / self.dx
		# y electric field, need forward/backward differences for edges
		Dy_1d, D2y_1d = Diff_mat_1D(self.Ny)
		E_y = - 0.5 * Dy_1d.dot(self.u[ind_x, :]) / self.dy

		# calculate x current using j = sigma E; integrate and convert to SI units
		current = np.sum(sxx * E_x + sxy * E_y) * self.dy / 25812

		return current


class contact():
	"""The class for a voltage contact
	Args:
		x1 : coordinate location of V_A
		x2 : coordinate location of V_B
		side ('top', 'bottom', or 'hall') : which side of the Hall bar to measure
		width : width of voltage tap in microns
	"""

	def __init__(self, x1, x2, side='top', width=6):

		self.x1 = x1
		self.x2 = x2
		self.side = side
		self.width = width


class nonlocal_hb():
	"""The class for nonlocal measurements
	Contacts are on the bottom edge of the device
	Args:
		Lx : length in x direction
		Ly : length in y direction
		Nx : number of points in grid along x
		Ny : number of points in grid along y
		settings : positions of contacts
	"""

	def __init__(self, Lx, Ly, Nx = 301, Ny = 201, settings = {}):

		# Initiate with no contacts
		self.contacts = []

		# Define coordinate variables
		self.Nx = Nx
		self.Ny = Ny
		self.Lx = Lx
		self.Ly = Ly
		self.x = np.linspace(0,self.Lx,self.Nx)
		self.y = np.linspace(0,self.Ly,self.Ny)

		self.dx = self.x[1] - self.x[0]                # grid spacing along x direction
		self.dy = self.y[1] - self.y[0]                # grid spacing along y direction

		self.X,self.Y = np.meshgrid(self.x,self.y)          # 2D meshgrid

		# 1D indexing
		self.Xu = self.X.ravel()                  # Unravel 2D meshgrid to 1D array
		self.Yu = self.Y.ravel()

		# Nonlocal contacts
		self.source_x1 = settings.get("source_x1", Lx/4)
		self.source_x2 = settings.get("source_x2", Lx/3)
		self.drain_x1 = settings.get("drain_x1", 2*Lx/3)
		self.drain_x2 = settings.get("drain_x2", 3*Lx/4)

		# Search for boundary indices
		start_time = time.time()
		self.ind_unravel_L = np.squeeze(np.where(self.Xu==self.x[0]))          # Left boundary
		self.ind_unravel_R = np.squeeze(np.where(self.Xu==self.x[self.Nx-1]))       # Right boundary
		self.ind_unravel_B = np.squeeze(np.where(self.Yu==self.y[0]))          # Bottom boundary
		self.ind_unravel_T = np.squeeze(np.where(self.Yu==self.y[self.Ny-1]))       # Top boundary

		self.ind_boundary_unravel = np.squeeze(np.where((self.Xu==self.x[0]) | (self.Xu==self.x[self.Nx-1]) | (self.Yu==self.y[0]) | (self.Yu==self.y[self.Ny-1])))  # outer boundaries 1D unravel indices
		self.ind_boundary = np.where((self.X==self.x[0]) | (self.X==self.x[self.Nx-1]) | (self.Y==self.y[0]) | (self.Y==self.y[self.Ny-1]))    # outer boundary

		self.ind_unravel_source = np.squeeze(np.where( (self.Yu==self.y[0]) & (self.Xu >= self.source_x1) & (self.Xu <= self.source_x2) ))	# Source
		self.ind_unravel_drain = np.squeeze(np.where( (self.Yu==self.y[0]) & (self.Xu >= self.drain_x1) & (self.Xu <= self.drain_x2) ))	# Drain

		print("Boundary search time = %1.4s" % (time.time()-start_time))

		# Load finite difference matrix operators
		self.Dx_2d, self.Dy_2d, self.D2x_2d, self.D2y_2d = Diff_mat_2D(self.Nx,self.Ny)

		# Initiate empty solution matrix
		self.u = 0

	def solve(self, lmbda):
		''' Constructs matrix problem and solves Poisson equation
		# Args:
			lmbda : sigma_xy / sigma_xx. Must be finite
		# Returns:
			self.u : electric potential
		'''

		self.lmbda = lmbda

		# Construct system matrix without boundary conditions
		start_time = time.time()
		I_sp = sp.eye(self.Nx*self.Ny).tocsr()
		L_sys = self.D2x_2d/self.dx**2 + self.D2y_2d/self.dy**2

		# Boundary operators
		BD = I_sp       # Dirichlet boundary operator
		BNx = self.Dx_2d / (2 * self.dx)     # Neumann boundary operator for x component
		BNy = self.Dy_2d / (2 * self.dy)     # Neumann boundary operator for y component

		# CURRENT THROUGH TOP/BOTTOM EDGES
		L_sys[self.ind_unravel_T,:] = BNy[self.ind_unravel_T,:] - lmbda * BNx[self.ind_unravel_T,:]    # Boundaries at the top layer
		L_sys[self.ind_unravel_B,:] = BNy[self.ind_unravel_B,:] - lmbda * BNx[self.ind_unravel_B,:]    # Boundaries at the bottom layer
		# CURRENT THROUGH LEFT/RIGHT EDGES
		L_sys[self.ind_unravel_L,:] = BNx[self.ind_unravel_L,:] + lmbda * BNy[self.ind_unravel_L,:]
		L_sys[self.ind_unravel_R,:] = BNx[self.ind_unravel_R,:] + lmbda * BNy[self.ind_unravel_R,:]

		# REPLACE WITH DIRICHLET BOUNDARY CONDITIONS FOR SOURCE/DRAIN
		L_sys[self.ind_unravel_source,:] = BD[self.ind_unravel_source,:]
		L_sys[self.ind_unravel_drain,:] = BD[self.ind_unravel_drain,:]

		# Source function (right hand side vector)
		g = np.zeros(self.Nx*self.Ny)
		# No current boundary conditions
		g[self.ind_unravel_L] = 0
		g[self.ind_unravel_R] = 0
		g[self.ind_unravel_T] = 0
		g[self.ind_unravel_B] = 0
		# Replace source with potential
		g[self.ind_unravel_source] = 1

		print("System matrix and right hand vector computation time = %1.6s" % (time.time()-start_time)) 
		
		start_time = time.time()
		self.u = spsolve(L_sys,g).reshape(self.Ny,self.Nx).T
		print("spsolve() time = %1.6s" % (time.time()-start_time))

	def voltage_measurement(self, x1, x2, side='top'):
		# Args: 	x1 : point of V_A
		#			x2 : point of V_B
		#			side ('top' or 'bottom') : which side of Hall bar to measure
		# Returns:	V_A - V_B

		if np.all(self.u==0):
			raise Exception('System has not been solved')
		if x1 > self.Lx or x1 < 0 or x2 > self.Lx or x2 < 0:
			raise Exception('Points out of bounds')

		if side=='top':
			y = self.Ny-1
		elif side=='bottom':
			y = 0
		else:
			raise Exception('Side must be top or bottom')

		# Find nearest index value to input coordinates
		xa = np.searchsorted(self.x, x1, side='left')
		xb = np.searchsorted(self.x, x2, side='left')

		return self.u[xa, y] - self.u[xb, y]

	def plot_potential(self):

		if np.all(self.u==0):
			raise Exception('System has not been solved')

		fig = plt.figure(figsize = [8,5])
		# plt.contour(self.x,self.y,self.u.T,41,cmap = 'viridis', vmin=0, vmax=1)
		plt.pcolormesh(self.X, self.Y, self.u.T, cmap='inferno', vmin=0, vmax=1)
		cbar = plt.colorbar(ticks = np.arange(0, 1.01, 0.2), label = r'$\phi / \phi_s$')
		plt.xlabel(r'x ($\mu$m)');
		plt.ylabel(r'y ($\mu$m)');
		plt.show()

class corbino():
	"""The class for a Corbino disk
	Args:
		ro : outer radius
		ri : inner radius
		Nx : number of points in grid along x
		Ny : number of points in grid along y
	"""

	def __init__(self, ro, ri, Nx = 301, Ny = 201):

		# Initiate with no contacts
		self.contacts = []

		# Define coordinate variables
		self.Nx = Nx
		self.Ny = Ny
		self.ro = ro
		self.ri = ri
		self.x = np.linspace(-self.ro, self.ro, self.Nx)
		self.y = np.linspace(-self.ro, self.ro, self.Ny) 

		self.dx = self.x[1] - self.x[0]                # grid spacing along x direction
		self.dy = self.y[1] - self.y[0]                # grid spacing along y direction

		self.X,self.Y = np.meshgrid(self.x,self.y)          # 2D meshgrid

		# 1D indexing
		self.Xu = self.X.ravel()                  # Unravel 2D meshgrid to 1D array
		self.Yu = self.Y.ravel()

		# Search for boundary indices
		start_time = time.time()
		self.ind_unravel_outer = np.squeeze(np.where(self.Xu**2 + self.Yu**2 >= self.ro**2))	# outer boundary
		self.ind_unravel_inner = np.squeeze(np.where(self.Xu**2 + self.Yu**2 <= self.ri**2))	# inner boundary

		self.ind_boundary_unravel = np.squeeze(np.where((self.Xu**2 + self.Yu**2 >= self.ro**2) | (self.Xu**2 + self.Yu**2 <= self.ri**2)))  # boundary 1D unravel indices
		self.ind_boundary = np.where((self.Xu**2 + self.Yu**2 >= self.ro**2) | (self.Xu**2 + self.Yu**2 <= self.ri**2))    # boundary

		print("Boundary search time = %1.4s" % (time.time()-start_time))

		# Load finite difference matrix operators
		self.Dx_2d, self.Dy_2d, self.D2x_2d, self.D2y_2d = Diff_mat_2D(self.Nx,self.Ny)

		# Initiate empty solution matrix
		self.u = 0

	def solve(self, lmbda):
		# constructs matrix problem and solves Poisson equation
		# Args: lmbda : sigma_xy / sigma_xx. Must be finite
		# Returns: self.u : electric potential

		self.lmbda = lmbda

		# Construct system matrix without boundary conditions
		start_time = time.time()
		I_sp = sp.eye(self.Nx*self.Ny).tocsr()
		L_sys = self.D2x_2d/self.dx**2 + self.D2y_2d/self.dy**2

		# Boundary operators
		BD = I_sp       # Dirichlet boundary operator

		# DIRICHLET BOUNDARY CONDITIONS FOR CONTACTS
		L_sys[self.ind_boundary_unravel,:] = BD[self.ind_boundary_unravel,:]

		# Source function (right hand side vector)
		g = np.zeros(self.Nx*self.Ny)
		# Insert boundary values at the boundary points
		g[self.ind_unravel_outer] = 1 # Dirichlet boundary condition at source
		g[self.ind_unravel_inner] = 0 # Dirichlet boundary condition at drain

		print("System matrix and right hand vector computation time = %1.6s" % (time.time()-start_time)) 
		
		start_time = time.time()
		self.u = spsolve(L_sys,g).reshape(self.Ny,self.Nx).T
		print("spsolve() time = %1.6s" % (time.time()-start_time))

	def plot_potential(self):

		if np.all(self.u==0):
			raise Exception('System has not been solved')

		fig = plt.figure(figsize = [8,5])
		plt.contourf(self.x,self.y,self.u.T,41,cmap = 'inferno')
		cbar = plt.colorbar(ticks = np.arange(0, 1.01, 0.2), label = r'$\phi / \phi_s$')
		plt.xlabel(r'x ($\mu$m)');
		plt.ylabel(r'y ($\mu$m)');
		plt.show()