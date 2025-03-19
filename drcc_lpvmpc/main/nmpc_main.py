import numpy as np
import matplotlib.pyplot as plt
from drcc_lpvmpc.vehicle.orca import ORCA
from drcc_lpvmpc.dynamics.lpvdynamics import BicycleDynamics
from drcc_lpvmpc.tracks.ethz import ETHZ
from drcc_lpvmpc.raceline_formulation.racecar_path import RacecarPath
from drcc_lpvmpc.obstacles.obstacle_shape import Rectangle_obs

import os
from casadi import DM
from casadi import Slice
import casadi as ca
import yaml


import drcc_lpvmpc.raceline_formulation.utils.utils as utils
from drcc_lpvmpc.mpc.dynamics_nmpc import BicycleDynamicsNMPC

SAMPLING_TIME = 0.02
HORIZON = 6
SIM_TIME = 40
script_dir = os.path.dirname(__file__)
config_file = os.path.join(script_dir,"config","config.yaml")

# Load the yaml file
with open(config_file, 'r') as file:
	config = yaml.safe_load(file)

# load configuration

xylabel_fontsize = config['xylabel_fontsize']
legend_fontsize = config['legend_fontsize']
xytick_size = config['xytick_size']
save_figure = config['save_figure']
save_noise = config['save_noise']
use_fixed_noise = config['use_fixed_noise']
plot_error = config['plot_error']
draw_safe_region = config['draw_safe_region']
test_debug = config['test_debug']
useDRCC = config['useDRCC']
add_disturbance = config['add_disturbance']

#########################################################################################################################
##################################################### ENVIRONMENT SETUP #################################################
#########################################################################################################################

# load vehicle parameters
params = ORCA()
model = BicycleDynamics(**params)

#####################################################################
# load track
TRACK_NAME = 'ETHZ'
track = ETHZ(reference='optimal', longer=True)

####################################################################
# generate reference raceline
track_file = os.path.join(script_dir,"data","smoothed_optimized_track.txt")
track_arr = np.genfromtxt(track_file,delimiter=' ')
track_dm = DM(track_arr)
a=DM()
track_ptr = RacecarPath(track_dm[Slice(0,2),Slice()],a)
start_tau = 5
start_pt = track_ptr.f_taun_to_xy(start_tau,0.00)
start_phi = track_ptr.f_phi(start_tau)[0]
current_pos = DM([start_pt[0],start_pt[1],float(start_phi),params['min_vx'],0.0,0.0])

###################################  generate centerline pointer ##################################
# check the boundary, set boundary limit to 0.185m upper and lower distance
centerline = DM([track.x_center,track.y_center])
center_ptr = RacecarPath(centerline[Slice(0,2),Slice()],a)

####################################################################
# extract data

Ts = SAMPLING_TIME
n_steps = int(SIM_TIME/Ts)
n_states = model.n_states
n_inputs = model.n_inputs
horizon = HORIZON

# initialize
states = np.zeros([n_states, n_steps+1])
dstates = np.zeros([n_states, n_steps+1])
inputs = np.zeros([n_inputs, n_steps])
time = np.linspace(0, n_steps, n_steps+1)*Ts
hstates = np.zeros([n_states,horizon+1])
hstates2 = np.zeros([n_states,horizon+1])

z_states = np.zeros([n_states+horizon*n_inputs,10])
error_states = np.zeros([10*n_states,horizon+1])

ex = np.zeros([n_steps+1])
ey = np.zeros([n_steps+1])
ephi = np.zeros([n_steps+1])
evx = np.zeros([n_steps+1])
evy = np.zeros([n_steps+1])
eomega = np.zeros([n_steps+1])

vxt = np.zeros([horizon+1])
vyt = np.zeros([horizon+1])
omegat = np.zeros([horizon+1])

x_init = np.zeros(n_states)
x_init[0], x_init[1] = float(current_pos[0]), float(current_pos[1])
x_init[2] = float(current_pos[2])
x_init[3] = current_pos[3]
x_init[4] = 0
x_init[5] = 0

dstates[0,0] = x_init[3]
dstates[3,0] = params['min_inputs'][0]
dstates[4,0] = 0

states[:,0] = x_init

dstatex = np.cos(states[2,0])
dstatey = np.sin(states[2,0])

print('starting at ({:.3f},{:.3f})'.format(x_init[0], x_init[1]))

# dynamic plot
fig = track.plot(color='black', grid=False)
plt.plot(track.x_center, track.y_center, '--k', alpha=0.5, lw=0.5)
utils.plot_path(track_ptr,type=1,labels="reference track")
# utils.plot_path(centerline_ptr,type=1,labels="reference line")
ax = plt.gca()
LnS, = ax.plot(states[0,0], states[1,0], 'r', alpha=1,lw=2,label="Trajectory")
LnR, = ax.plot(states[0,0], states[1,0], '-b', marker='o', markersize=1, lw=1,label="Local Reference")
LnP, = ax.plot(float(current_pos[0]), float(current_pos[1]), 'g', marker='o', alpha=0.5, markersize=5,label="current position")
# LnH, = ax.plot(hstates[0], hstates[1], '-g', marker='o', markersize=1, lw=0.5)
LnH2, = ax.plot(hstates2[0], hstates2[1], '-r', marker='o', markersize=1, lw=0.5,label="Ground Truth Path")

# LnS, = ax.plot(states[0,0], states[1,0], 'r', alpha=1,lw=2,label="Trajectory")
# LnR, = ax.plot(states[0,0], states[1,0], '-b', marker='o', markersize=1, lw=1)
# LnP, = ax.plot(float(current_pos[0]), float(current_pos[1]), 'g', marker='o', alpha=0.5, markersize=5)
# LnH, = ax.plot(hstates[0], hstates[1], '-g', marker='o', markersize=1, lw=0.5)
# LnH2, = ax.plot(hstates2[0], hstates2[1], '-r', marker='o', markersize=1, lw=0.5)
# LnPd = ax.quiver(states[0,0], states[1,0],dstatex,dstatey,angles='xy', scale_units='xy', scale=10, color='r',width = 0.002)
ax.figure.canvas.manager.window.wm_geometry("+1000+1")
ax.figure.set_size_inches(15, 15)
if plot_error:

	plt.figure()
	plt.grid(True)
	ax2 = plt.gca()
	Lnx, = ax2.plot(0, 0, label='ex')
	Lny, = ax2.plot(0, 0, label='ey')
	ax2.figure.canvas.manager.window.wm_geometry("+800+100")
	plt.xlim([0, SIM_TIME])
	plt.ylim([-0.6794623848648096,0.562250260331467])
	plt.xlabel('time [s]')
	plt.ylabel('difference [m]')
	plt.legend()

	plt.figure()
	plt.grid(True)
	ax3 = plt.gca()
	ax3.figure.canvas.manager.window.wm_geometry("+1500+100")
	Lnphi, = ax3.plot(0, 0, label='ephi')
	plt.xlim([0, SIM_TIME])
	plt.ylim([-2.1810990665737062,7.906553232250088])
	plt.xlabel('time [s]')
	plt.ylabel('difference [rad]')
	plt.legend()

	plt.figure()
	plt.grid(True)
	ax4 = plt.gca()
	ax4.figure.canvas.manager.window.wm_geometry("+2200+100")
	Lnvx, = ax4.plot(0, 0, label='evx')
	Lnvy, = ax4.plot(0, 0, label='evy')
	plt.xlim([0, SIM_TIME])
	plt.ylim([-0.5879810835856902,0.12376013325951957])
	plt.xlabel('time [s]')
	plt.ylabel('difference [m/s]')
	plt.legend()

	plt.figure()
	plt.grid(True)
	ax5 = plt.gca()
	ax5.figure.canvas.manager.window.wm_geometry("+800+700")
	Lnomega, = ax5.plot(0, 0, label='eomega')
	plt.xlim([0, SIM_TIME])
	plt.ylim([-1.8560492707591574,14.455822217841625])
	plt.xlabel('time [s]')
	plt.ylabel('difference [rad/s]')
	plt.legend()

######################## define the obstacle center position##########################
track_tau_max = track_ptr.get_max_tau()
n_obs = 4
ob_center = ca.DM([track_tau_max*8/20,track_tau_max*13/20,track_tau_max*16/20,track_tau_max*19/20]).T
# ob_center = ca.DM([track_tau_max*8/20,track_tau_max*16/20,track_tau_max*19/20]).T
side_avoid = np.array([-1,-1,-1,-1]) # 1:left avoid, -1:right avoid
ob_centern = ca.DM.ones(1,n_obs) # smaller to get bigger height
ob_centern[0,0] = 0.05
ob_centern[0,1] = 0.08
ob_centern[0,2] = 0.08
ob_centern[0,3] = 0.08
ob_centerxy = track_ptr.f_taun_to_xy(ob_center,ob_centern)
ob_centerxy = np.array(ob_centerxy)
############ Form rectangle obs ##########
ob_phi = track_ptr.f_phi(ob_center) * 180/ca.pi
# print("ob phi:",ob_phi)
rec_obsxy = ob_centerxy.transpose().tolist()
# print("rec obsxy:",rec_obsxy)
length = 0.09
width = 0.24
angles = np.array(ob_phi).squeeze().tolist()
Rec_obs = Rectangle_obs(rec_obsxy,width,length,angles,side_avoid)
#############################################################################################################################
######################################## END OF ENVIRONMENT SETUP ###########################################################
#############################################################################################################################

nmpc = BicycleDynamicsNMPC(track_ptr,center_ptr,Rec_obs,current_pos,SAMPLING_TIME,horizon,track.track_width/2)

#####################################################################################################################################
############################################################# test section ##########################################################
#####################################################################################################################################

if test_debug:
	x0_test = np.array(current_pos)
	print("x0 :",x0_test)
	# dro_success,dro_z0,dro_control = drompc.get_Updated_local_path(current_pos)
	debug_result, opz, opu = nmpc.get_Updated_local_path(current_pos)
	op_x = np.array(opz[0,:]).squeeze().tolist()
	op_y = np.array(opz[1,:]).squeeze().tolist()
	op_phi = np.array(opz[2,:]).squeeze().tolist()
	op_vx = np.array(opz[3,:]).squeeze().tolist()
	op_vy = np.array(opz[4,:]).squeeze().tolist()
	op_omega = np.array(opz[5,:]).squeeze().tolist()

	op_delta = np.array(opu[0,:]).squeeze().tolist()
	op_acc = np.array(opu[1,:]).squeeze().tolist()

	ax.plot(op_x,op_y,lw = 1,ls='-',color = 'red')

############# draw the safe range #############
if draw_safe_region:
	test_a_tau = nmpc.get_obs_atau()
	test_a_n = nmpc.get_obs_an()

	test_b_tau = nmpc.get_obs_btau()
	test_b_n = nmpc.get_obs_bn()

	test_tau_0 = nmpc.get_obs_tau0()
	test_tau_1 = nmpc.get_obs_tau1()


	print("test a tau is:",test_a_tau)
	print("test tau0 is:",test_tau_0)

	print("test b tau is:",test_b_tau)
	print("test tau1 is:",test_tau_1)

	print("test an is :",test_a_n)
	print("test bn is :",test_b_n)

	for i in range(test_a_tau.shape[0]):
		if test_a_n[i] > 0:
			safe_ataus = ca.linspace(test_tau_0[i],test_a_tau[i],100).T

			safe_ans = ca.linspace(0,test_a_n[i],100).T

			safe_axys = track_ptr.f_taun_to_xy(safe_ataus,safe_ans)

			safe_btaus = ca.linspace(test_b_tau[i],test_tau_1[i],100).T

			safe_bns = ca.linspace(test_b_n[i],0,100).T

			safe_bxys = track_ptr.f_taun_to_xy(safe_btaus,safe_bns)

		else:
			safe_ataus = ca.linspace(test_a_tau[i],test_tau_0[i],10).T

			safe_ans = ca.linspace(test_a_n[i],0,10).T

			safe_axys = track_ptr.f_taun_to_xy(safe_ataus,safe_ans)

			safe_btaus = ca.linspace(test_tau_1[i],test_b_tau[i],10).T

			safe_bns = ca.linspace(0,test_b_n[i],10).T

			safe_bxys = track_ptr.f_taun_to_xy(safe_btaus,safe_bns)

		print("safe a :",safe_axys)
		print("safe b :",safe_bxys)

		safe_axys = np.array(safe_axys)
		safe_bxys = np.array(safe_bxys)

		ax.plot(safe_axys[0,:],safe_axys[1,:],lw = 1,ls='--',color = 'black')
		ax.plot(safe_bxys[0,:],safe_bxys[1,:],lw = 1,ls='--',color = 'black')
##############################################################################################################################
##############################################################################################################################

######################################### Plot rectangle ############################################
Rec_obs.plot_rectangle(ax)
ax.set_xlabel('x [m]',fontsize = xylabel_fontsize)
ax.set_ylabel('y [m]',fontsize = xylabel_fontsize)
ax.legend(fontsize=legend_fontsize,borderpad=0.1,labelspacing=0.2, handlelength=1.4, handletextpad=0.37,loc='lower right')
ax.tick_params(axis='both',which='major',labelsize = xytick_size)

save_z_idx = 0
obs_dis = [[] for _ in range(n_obs)]
LB_dis = [[] for _ in range(n_obs)]


###############################################################################################################################
########################################## PLANNER START ######################################################################
###############################################################################################################################

if not test_debug:
	disturbance_ranges = [
    (-0.005, 0.005),  # Range for x
    (-0.005, 0.005),      # Range for y
    (-0.00005, 0.00005),      # Range for phi
    (-0.00001, 0.00001),      # Range for vx
    (-0.000001, 0.000001),  # Range for vy
    (-0.00001, 0.00001)   # Range for omega
	]
	plt.ion()
	np.set_printoptions(threshold=np.inf)  # To print the entire array
	np.set_printoptions(precision=17)
	for idt in range(n_steps-horizon):

		x0 = states[:,idt]
		# print("x0 :",x0)

		current_xy = ca.DM(x0).T
		nmpc_success,nmpc_z,nmpc_control = nmpc.get_Updated_local_path(current_xy)

		nmpc_z = np.array(nmpc_z)
		nmpc_control = np.array(nmpc_control)

		# op_x = np.array(nmpc_z[0,:]).squeeze().tolist()
		# op_y = np.array(nmpc_z[1,:]).squeeze().tolist()
		# op_phi = np.array(nmpc_z[2,:]).squeeze().tolist()
		# op_vx = np.array(nmpc_z[3,:]).squeeze().tolist()
		# op_vy = np.array(nmpc_z[4,:]).squeeze().tolist()
		# op_omega = np.array(nmpc_z[5,:]).squeeze().tolist()

		# op_delta = np.array(nmpc_control[0,:]).squeeze().tolist()
		# op_acc = np.array(nmpc_control[1,:]).squeeze().tolist()
		if not nmpc_success:
			break

		op_xy = nmpc_z[:2,:]
		ref_xy = nmpc.get_reference_path()
		ref_phi = nmpc.get_reference_phi()
		ref_x = np.array(ref_xy[0,:]).transpose()
		ref_y = np.array(ref_xy[1,:]).transpose()

		obs_detect = nmpc.get_obs_detect()
		if obs_detect is not False:
			obs_dis[obs_detect].append(np.linalg.norm(ref_xy[:2,1] - op_xy[:2,1]))

		ref_phi = np.array(ref_phi).transpose()
		drex = np.cos(ref_phi)
		drey = np.sin(ref_phi)

		states[:,idt+1] = nmpc_z[:,1]
		if add_disturbance:
			disturbances = np.array([[np.random.uniform(low, high)] for low, high in disturbance_ranges]).squeeze()
			states[:,idt+1] += disturbances

		hstates2[:,0] = x0
		for idh in range(nmpc.horizon):
			hstates2[:,idh+1] = nmpc_z[:,idh+1]

		#################################################################################################

		# update plot
		LnS.set_xdata(states[0,:idt+1])
		LnS.set_ydata(states[1,:idt+1])

		LnR.set_xdata(ref_x)
		LnR.set_ydata(ref_y)

		LnP.set_xdata(states[0,idt])
		LnP.set_ydata(states[1,idt])

		# LnH.set_xdata(nmpc_z[0,:])
		# LnH.set_ydata(nmpc_z[1,:])

		LnH2.set_xdata(hstates2[0,:])
		LnH2.set_ydata(hstates2[1,:])
		ax.figure.canvas.draw()
		plt.pause(Ts/100)

		####################################################################################################
		# visualize model error
		if plot_error:
			Lnx.set_xdata(time[:idt+1])
			Lnx.set_ydata(ex[:idt+1])

			Lny.set_xdata(time[:idt+1])
			Lny.set_ydata(ey[:idt+1])

			Lnphi.set_xdata(time[:idt+1])
			Lnphi.set_ydata(ephi[:idt+1])

			Lnvx.set_xdata(time[:idt+1])
			Lnvx.set_ydata(evx[:idt+1])

			Lnvy.set_xdata(time[:idt+1])
			Lnvy.set_ydata(evy[:idt+1])

			Lnomega.set_xdata(time[:idt+1])
			Lnomega.set_ydata(eomega[:idt+1])

	total_error = np.stack((ex, ey, ephi, evx, evy, eomega))
	print("total error :",total_error.shape)

	print("x min max error:",np.amax(ex),np.amin(ex))
	print("y min max error:",np.amax(ey),np.amin(ey))
	print("phi min max error:",np.amax(ephi),np.amin(ephi))
	print("vx min max error:",np.amax(evx),np.amin(evx))
	print("vy min max error:",np.amax(evy),np.amin(evy))
	print("omega min max error:",np.amax(eomega),np.amin(eomega))

	if not nmpc_success:
		for i in range(n_obs):

			obs_disMean = np.mean(obs_dis[i])
			obs_disMax = np.amax(obs_dis[i])
			obs_disMin = np.amin(obs_dis[i])

			print("obstacle {} Mean distance : {:.5f}".format(i,obs_disMean))
			print("obstacle {} Max distance : {:.5f}".format(i,obs_disMax))
			print("obstacle {} Min distance : {:.5f}".format(i,obs_disMin))

	plt.ioff()

if save_figure:
	# # List of formats to save the figure in
	formats = ['png', 'pdf', 'svg']

	# Save the figure in each format
	for fmt in formats:
		if useDRCC:
			filename = os.path.join(script_dir,"outputFigure",f"nmpc.{fmt}")
		else:
			filename = os.path.join(script_dir,"outputFigure",f"nmpc.{fmt}")

		plt.savefig(filename,transparent=True, format=fmt)
		print(f"Figure saved as {filename}")

plt.show(block=True)

