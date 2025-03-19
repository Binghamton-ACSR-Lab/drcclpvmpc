""" Params for ORCA 1:43 scale car from ETH Zurich
"""

import numpy as np
__author__ = 'Shiming Fang'
__email__ = 'sfang10@binghamton.edu'

def ORCA(control='acc'):

	lf = 0.029
	lr = 0.033
	mass = 0.041
	dr = lr/(lf + lr)
	Iz = 27.8e-6
	Bf = 2.579
	Cf = 1.2
	Df = 0.192
	Br = 3.3852
	Cr = 1.2691
	Dr = 0.1737

	Caf = 1.78
	Car = 2.24

	g = 9.81
	d = 0.046
	w = 0.029

	max_vx = 1.5		# max velocity cycle
	min_vx = 1.2

	max_vy = 0.2
	min_vy = -0.2

	max_omega = np.pi/(3*0.05)
	min_omega = -np.pi/(3*0.05)

	max_acc = 2/5			# max acceleration [m/s^2]
	min_acc = -2/5	# max deceleration [m/s^2]
	
	max_steer = 34*np.pi/180 		# max steering angle [rad]
	min_steer = -34*np.pi/180 		# min steering angle [rad]

	
	max_inputs = [max_acc, max_steer]
	min_inputs = [min_acc, min_steer]

	max_rate = [25*np.pi/180,1.5]
	min_rate = [-25*np.pi/180,-1.5]
	
	if control == 'acc':
		approx = True
	else:
		approx = False
  
	Cm1 = 0.287
	Cm2 = 0.0545
	Cr0 = 0.0518
	Cd = 0.00035

	params = {
		'lf': lf,
		'lr': lr,
		'mass': mass,
		'Iz': Iz,
		'Cf': Cf,
		'Cr': Cr,
		'Bf': Bf,
		'Br': Br,
		'Df': Df,
		'Dr': Dr,
		'Caf': Caf,
		'Car': Car,
		'dr': dr,
		'max_vx': max_vx,
		'min_vx': min_vx,
		'max_vy': max_vy,
		'min_vy': min_vy,
		'g': g,
		'd': d,
		'w': w,
		'max_acc': max_acc,
		'min_acc': min_acc,		
		'max_omega': max_omega,
		'min_omega': min_omega,
		'max_steer': max_steer,
		'min_steer': min_steer,
		'max_inputs': max_inputs,
		'min_inputs': min_inputs,
		'max_omega': max_omega,
		'min_omega': min_omega,
		'max_rate' : max_rate,
		'min_rate' : min_rate,
		'Cm1': Cm1,
		'Cm2': Cm2,
		'Cr0': Cr0,
		'Cd': Cd,
		'approx':approx
		}
	return params