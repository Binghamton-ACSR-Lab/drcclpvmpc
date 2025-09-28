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
from drcc_lpvmpc.mpc.dynamics_drccmpc_rhunc import BicycleDynamicsDRCC

SAMPLING_TIME = 0.02
HORIZON = 6
SIM_TIME = 15
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
control_type = config['control_type']
#########################################################################################################################
##################################################### ENVIRONMENT SETUP #################################################
#########################################################################################################################

# load vehicle parameters
params = ORCA(control=control_type)
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
approx = model.approx

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
plt.ion()

# dynamic plot
fig = track.plot(color='black', grid=False)
plt.plot(track.x_center, track.y_center, '--k', alpha=0.5, lw=0.5)
utils.plot_path(track_ptr,type=1,labels="reference track")
# utils.plot_path(centerline_ptr,type=1,labels="reference line")
ax = plt.gca()
LnS, = ax.plot(states[0,0], states[1,0], 'r', alpha=1,lw=2,label="Trajectory")
LnR, = ax.plot(states[0,0], states[1,0], '-b', marker='o', markersize=1, lw=1,label="Local Reference")
LnP, = ax.plot(float(current_pos[0]), float(current_pos[1]), 'g', marker='o', alpha=0.5, markersize=5,label="current position")
LnH, = ax.plot(hstates[0], hstates[1], '-g', marker='o', markersize=1, lw=0.5)
LnH2, = ax.plot(hstates2[0], hstates2[1], '-r', marker='o', markersize=1, lw=0.5,label="Ground Truth Path")

safeA, = ax.plot([],[],lw = 1,ls='--',color = 'black')
safeB, = ax.plot([],[],lw = 1,ls='--',color = 'black')


# LnS, = ax.plot(states[0,0], states[1,0], 'r', alpha=1,lw=2,label="Trajectory")
# LnR, = ax.plot(states[0,0], states[1,0], '-b', marker='o', markersize=1, lw=1)
# LnP, = ax.plot(float(current_pos[0]), float(current_pos[1]), 'g', marker='o', alpha=0.5, markersize=5)
# LnH, = ax.plot(hstates[0], hstates[1], '-g', marker='o', markersize=1, lw=0.5)
# LnH2, = ax.plot(hstates2[0], hstates2[1], '-r', marker='o', markersize=1, lw=0.5)
# LnPd = ax.quiver(states[0,0], states[1,0],dstatex,dstatey,angles='xy', scale_units='xy', scale=10, color='r',width = 0.002)
# ax.figure.canvas.manager.window.wm_geometry("+1000+1")
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

######################## forming numerical obstacle ##########################
n_obs = 4
n_pos = ca.DM(n_obs,n_steps + 1)
n_pos[0,180:245] = ca.linspace(0.19, 0.05, 65).T
n_pos[1,350:410] = ca.linspace(0.19, 0.08, 60).T
n_pos[2,430:501] = ca.linspace(0.19, 0.08, 71).T
n_pos[3,530:601] = ca.linspace(0.19, 0.08, 71).T

n_pos[0,:180] = 0.19*ca.DM.ones(1,180)
n_pos[1,:350] = 0.19*ca.DM.ones(1,350)
n_pos[2,:430] = 0.19*ca.DM.ones(1,430)
n_pos[3,:530] = 0.19*ca.DM.ones(1,530)

n_pos[0,245:] = 0.05*ca.DM.ones(1,n_steps+1-245)
n_pos[1,410:] = 0.08*ca.DM.ones(1,n_steps+1-410)
n_pos[2,501:] = 0.08*ca.DM.ones(1,n_steps+1-501)
n_pos[3,601:] = 0.08*ca.DM.ones(1,n_steps+1-601)

ob_center = ca.DM([track_tau_max*8/20,track_tau_max*13/20,track_tau_max*16/20,track_tau_max*19/20]).T
# ob_center = ca.DM([track_tau_max*8/20,track_tau_max*16/20,track_tau_max*19/20]).T
side_avoid = np.array([-1,-1,-1,-1]) # 1:left avoid, -1:right avoid
ob_centern = ca.DM.ones(1,n_obs) # smaller to get bigger height
ob_centern[0,0] = 0.19
ob_centern[0,1] = 0.19
ob_centern[0,2] = 0.19
ob_centern[0,3] = 0.19
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
pre_Rec_obs = Rectangle_obs(rec_obsxy,width,length,angles,side_avoid)
end_ls = pre_Rec_obs.end_ls

tau_ls = ca.DM(n_obs,2) # taua, taub
s01_ls = ca.DM(n_obs,2) # s0, s1
sab_ls = ca.DM(n_obs,2) # sab, sba
# nab_ls = ca.DM(n_obs,2) # na, nb
tau01_ls = ca.DM(n_obs,2) # tau0, tau1	
s_max = track_ptr.get_max_length()

for i,end in enumerate(end_ls):
    
	taua = track_ptr.xy_to_tau(end[0])
	taub = track_ptr.xy_to_tau(end[1])
	if taua > taub:
		tau_ls[i,0] = taub
		tau_ls[i,1] = taua
		# nab_ls[i,0] = track_ptr.f_xy_to_taun(end[1],taub)
		# nab_ls[i,1] = track_ptr.f_xy_to_taun(end[0],taua)
	else:
		tau_ls[i,0] = taua
		tau_ls[i,1] = taub
		# nab_ls[i,0] = track_ptr.f_xy_to_taun(end[0],taua)
		# nab_ls[i,1] = track_ptr.f_xy_to_taun(end[1],taub)

	sa = track_ptr.tau_to_s_lookup(tau_ls[i,0])
	sb = track_ptr.tau_to_s_lookup(tau_ls[i,1])
 
	sab_ls[i,0] = sa
	sab_ls[i,1] = sb
 
	s0 = sa - params['max_vx']*Ts*horizon*2
	s1 = sb + params['max_vx']*Ts*horizon*2
 
	s1 = min(s1,s_max-0.3)
	s0 = max(s0,0.3)
 
	tau01_ls[i,0] = track_ptr.s_to_tau_lookup(s0)
	tau01_ls[i,1] = track_ptr.s_to_tau_lookup(s1)
 
	s01_ls[i,0] = s0
	s01_ls[i,1] = s1
 
print("tau ls:",tau_ls)
print("s01 ls:",s01_ls)
print("sab ls:",sab_ls)
print("tau01 ls:",tau01_ls)

 
	

#############################################################################################################################
######################################## END OF ENVIRONMENT SETUP ###########################################################
#############################################################################################################################

drompc = BicycleDynamicsDRCC(track_ptr,center_ptr,current_pos,SAMPLING_TIME,horizon,track.track_width/2,use_fixed_noise,control_type,useDRCC)

#####################################################################################################################################
############################################################# test section ##########################################################
#####################################################################################################################################

if test_debug:
	x0_test = np.array(current_pos)
	print("x0 :",x0_test)
	# dro_success,dro_z0,dro_control = drompc.get_Updated_local_path(current_pos)
	drompc.make_plan(current_pos)
	# dro_z0 = np.array(dro_z0).squeeze()
	# print("dro z0 :",dro_z0)
	# dro_control = np.array(dro_control)
	# print("dro control :",dro_control)
	# lpv_pvx, lpv_pvy, lpv_pphi, lpv_pdelta = drompc.get_old_p_param()
	# lpv_pred_x = model.LPV_states(dro_z0,dro_control,lpv_pvx,lpv_pvy,lpv_pphi,lpv_pdelta,SAMPLING_TIME)
	# new_pvx = lpv_pred_x[3,1:]
	# new_pvy = lpv_pred_x[4,1:]
	# new_pphi = lpv_pred_x[2,1:]
	# drompc.update_new_p_param(new_pvx,new_pvy,new_pphi)
	# print("lpv states :",lpv_pred_x)
	# ax.plot(lpv_pred_x[0,:],lpv_pred_x[1,:],label="lpv linear path")
	# de_lpv_x = [-5.1130751320563850e-01,-4.6341522574699395e-01, -4.1357302194238971e-01, -3.7236311774105491e-01, -3.2012574718937004e-01, -3.0673791356108887e-01, 1.3498735490743303e+02]
	# de_lpv_y = [ 8.2102891808320744e-01,  7.8245268183290939e-01 , 7.4473973276307437e-01 ,6.9632674041619214e-01 , 7.3322066879491954e-01,  7.2313047450243850e-01, 2.1921902380423745e+02]
	# de_lpv_x = np.array(de_lpv_x).squeeze()
	# de_lpv_y = np.array(de_lpv_y).squeeze()
	# ax.plot(de_lpv_x,de_lpv_y,label = "test lpv track")
	# print(dro_path)
	# dro_x = np.array(dro_path[0]).squeeze()
	# dro_y = np.array(dro_path[1]).squeeze()
	# dro_phi = np.array(dro_path[2]).squeeze()
	# dro_vx = np.array(dro_path[3]).squeeze()
	# dro_vy = np.array(dro_path[4]).squeeze()
	# dro_omega = np.array(dro_path[5]).squeeze()
	# dro_delta = np.array(dro_control[0]).squeeze()
	# dro_acc = np.array(dro_control[1]).squeeze()
	# dx = np.cos(dro_phi)
	# dy = np.sin(dro_phi)
	# ref_xy = drompc.get_reference_path()
	# ref_phi = drompc.get_reference_phi()
	# ref_x = np.array(ref_xy[0,:]).transpose()
	# ref_y = np.array(ref_xy[1,:]).transpose()
	# ref_phi = np.array(ref_phi).transpose()
	# drex = np.cos(ref_phi)
	# drey = np.sin(ref_phi)
	# dro_control = np.stack((dro_delta,dro_acc))
	# print("dro control shape :",dro_control.shape)
	# dro_states = np.stack((dro_x,dro_y,dro_phi,dro_vx,dro_vy,dro_omega))
	# print("dro states :",dro_states.shape)
	# sim_x = model.sim_states(x0_test,dro_control,SAMPLING_TIME)
	# model_diff = sim_x - dro_states
	# model_diff = model_diff[:,:-1]
	# drompc.update_model_noise(model_diff,x0_test,dro_delta,dro_acc)
	# ax.plot(dro_x,dro_y,label="dro path")
	# ax.plot(sim_x[0,:],sim_x[1,:],label="non linear path")

############# draw the safe range #############
# if draw_safe_region:
# 	test_a_tau = drompc.get_obs_atau()
# 	test_a_n = drompc.get_obs_an()

# 	test_b_tau = drompc.get_obs_btau()
# 	test_b_n = drompc.get_obs_bn()

# 	test_tau_0 = drompc.get_obs_tau0()
# 	test_tau_1 = drompc.get_obs_tau1()


# 	print("test a tau is:",test_a_tau)
# 	print("test tau0 is:",test_tau_0)

# 	print("test b tau is:",test_b_tau)
# 	print("test tau1 is:",test_tau_1)

# 	print("test an is :",test_a_n)
# 	print("test bn is :",test_b_n)

# 	for i in range(test_a_tau.shape[0]):
# 		if test_a_n[i] > 0:
# 			safe_ataus = ca.linspace(test_tau_0[i],test_a_tau[i],100).T

# 			safe_ans = ca.linspace(0,test_a_n[i],100).T

# 			safe_axys = track_ptr.f_taun_to_xy(safe_ataus,safe_ans)

# 			safe_btaus = ca.linspace(test_b_tau[i],test_tau_1[i],100).T

# 			safe_bns = ca.linspace(test_b_n[i],0,100).T

# 			safe_bxys = track_ptr.f_taun_to_xy(safe_btaus,safe_bns)

# 		else:
# 			safe_ataus = ca.linspace(test_a_tau[i],test_tau_0[i],100).T

# 			safe_ans = ca.linspace(test_a_n[i],0,100).T

# 			safe_axys = track_ptr.f_taun_to_xy(safe_ataus,safe_ans)

# 			safe_btaus = ca.linspace(test_tau_1[i],test_b_tau[i],100).T

# 			safe_bns = ca.linspace(0,test_b_n[i],100).T

# 			safe_bxys = track_ptr.f_taun_to_xy(safe_btaus,safe_bns)

# 		print("safe a :",safe_axys)
# 		print("safe b :",safe_bxys)

# 		safe_axys = np.array(safe_axys)
# 		safe_bxys = np.array(safe_bxys)

# 		ax.plot(safe_axys[0,:],safe_axys[1,:],lw = 1,ls='--',color = 'black')
# 		ax.plot(safe_bxys[0,:],safe_bxys[1,:],lw = 1,ls='--',color = 'black')
##############################################################################################################################
##############################################################################################################################

######################################### Plot rectangle ############################################
pre_Rec_obs.plot_rectangle(ax)
ax.set_xlabel('x [m]',fontsize = xylabel_fontsize)
ax.set_ylabel('y [m]',fontsize = xylabel_fontsize)
ax.legend(fontsize=legend_fontsize,borderpad=0.1,labelspacing=0.2, handlelength=1.4, handletextpad=0.37,loc='lower right')
ax.tick_params(axis='both',which='major',labelsize = xytick_size)

save_z_idx = 0
obs_dis = [[] for _ in range(n_obs)]
LB_dis = [[] for _ in range(n_obs)]
obs_detect = 0
step_idx = 0

###############################################################################################################################
########################################## PLANNER START ######################################################################
###############################################################################################################################

if not test_debug:
	disturbance_ranges = [
    (-0.005, 0.005),  # Range for x
    (-0.005, 0.005),      # Range for y
    (-0.0005, 0.0005),      # Range for phi
    (-0.00001, 0.00001),      # Range for vx
    (-0.000001, 0.000001),  # Range for vy
    (-0.00001, 0.00001)   # Range for omega
	]
	np.set_printoptions(threshold=np.inf)  # To print the entire array
	np.set_printoptions(precision=17)
	for idt in range(n_steps-horizon):
		obs_passing = None

		x0 = states[:,idt]
		# print("x0 :",x0)

		current_xy = ca.DM(x0)
		print("current xy:",current_xy)
  
		############################## update obstacle position #####################################
		current_tau = track_ptr.xy_to_tau(current_xy[:2])
		current_s = track_ptr.tau_to_s_lookup(current_tau)
		print("current_tau:", current_tau)
		print("current_s:", current_s)
  
		ob_centern = ca.DM.ones(1,n_obs) # smaller to get bigger height
		ob_centern[0,0] = n_pos[0,idt]
		ob_centern[0,1] = n_pos[1,idt]
		ob_centern[0,2] = n_pos[2,idt]
		ob_centern[0,3] = n_pos[3,idt]
		ob_centerxy = track_ptr.f_taun_to_xy(ob_center,ob_centern)
		ob_centerxy = np.array(ob_centerxy)
		rec_obsxy = ob_centerxy.transpose().tolist()
		Rec_obs = Rectangle_obs(rec_obsxy,width,length,angles,side_avoid)
		ab_ls = Rec_obs.end_ls

		side_avoid_i = 1
		rec_data = ca.DM.zeros(3,4) # rows: tau, n, s; cols: 0, a, b, 1
		for i in range(n_obs):
			if s01_ls[i,1] >= current_s:
				obs_detect = i
				
				rec_data[0,0] = tau01_ls[i,0]
				rec_data[0,1] = tau_ls[i,0]
				rec_data[0,2] = tau_ls[i,1]
				rec_data[0,3] = tau01_ls[i,1]
				rec_data[1,0] = 0
				axy = ab_ls[i][0]
				bxy = ab_ls[i][1]
				a_tau = track_ptr.xy_to_tau(axy)
				b_tau = track_ptr.xy_to_tau(bxy)
				if a_tau > b_tau:
					rec_data[1,1] = track_ptr.f_xy_to_taun(bxy,b_tau)
					rec_data[1,2] = track_ptr.f_xy_to_taun(axy,a_tau)
				else:
					rec_data[1,1] = track_ptr.f_xy_to_taun(axy,a_tau)
					rec_data[1,2] = track_ptr.f_xy_to_taun(bxy,b_tau)

				if current_tau >= tau_ls[i,0] and current_tau <= tau_ls[i,1]:
					obs_passing = i
				rec_data[1,3] = 0
				rec_data[2,0] = s01_ls[i,0]
				rec_data[2,1] = sab_ls[i,0]
				rec_data[2,2] = sab_ls[i,1]
				rec_data[2,3] = s01_ls[i,1]
				side_avoid_i = side_avoid[i]
				break
		# print("rec data:",rec_data)
		if side_avoid_i == -1 and rec_data[1,1] > 0 and rec_data[1,2] > 0:
			obs_detect = n_obs
		if side_avoid_i == 1 and rec_data[1,1] < 0 and rec_data[1,2] < 0:
			obs_detect = n_obs
		# print("obs detect:",obs_detect)

		Rec_obs.plot_rectangle(ax)
		pre_Rec_obs.remove_plot()
		pre_Rec_obs = Rec_obs
		#################################################################################################
		#################################### update safety region #########################################
		if obs_detect != n_obs and draw_safe_region:
			if rec_data[1,1] > 0: # rows: tau, n, s; cols: 0, a, b, 1
				safe_ataus = ca.linspace(rec_data[0,0],rec_data[0,1],10).T

				safe_ans = ca.linspace(0,rec_data[1,1],10).T

				safe_axys = track_ptr.f_taun_to_xy(safe_ataus,safe_ans)

				safe_btaus = ca.linspace(rec_data[0,2],rec_data[0,3],10).T

				safe_bns = ca.linspace(rec_data[1,2],0,10).T

				safe_bxys = track_ptr.f_taun_to_xy(safe_btaus,safe_bns)

			else:
				safe_ataus = ca.linspace(rec_data[0,1],rec_data[0,0],10).T

				safe_ans = ca.linspace(rec_data[1,1],0,10).T

				safe_axys = track_ptr.f_taun_to_xy(safe_ataus,safe_ans)

				safe_btaus = ca.linspace(rec_data[0,3],rec_data[0,2],10).T

				safe_bns = ca.linspace(0,rec_data[1,2],10).T

				safe_bxys = track_ptr.f_taun_to_xy(safe_btaus,safe_bns)

			# print("safe a :",safe_axys)
			# print("safe b :",safe_bxys)
			safe_axys = np.array(safe_axys)
			safe_bxys = np.array(safe_bxys)
			safeA.set_xdata(safe_axys[0,:])
			safeA.set_ydata(safe_axys[1,:])
			safeB.set_xdata(safe_bxys[0,:])
			safeB.set_ydata(safe_bxys[1,:])
		else:
			safeA.set_xdata([])
			safeA.set_ydata([])
			safeB.set_xdata([])
			safeB.set_ydata([])
		##########################################################################################################
		if obs_detect != n_obs:
			dro_success,dro_z0,dro_control = drompc.get_Updated_local_path(current_xy,rec_data,side_avoid=side_avoid_i,usedro=useDRCC)
		else:
			dro_success,dro_z0,dro_control = drompc.get_Updated_local_path(current_xy,usedro=False)
		if not dro_success:
			break


		dro_z0 = np.array(dro_z0).squeeze()
		dro_control = np.array(dro_control)

		lpv_pvx, lpv_pvy, lpv_pphi, lpv_pdelta = drompc.get_old_p_param()
		lpv_pred_x = model.LPV_states(dro_z0,dro_control,lpv_pvx,lpv_pvy,lpv_pphi,lpv_pdelta,SAMPLING_TIME)

		dro_phi = np.array(lpv_pred_x[2,:]).squeeze()

		dx = np.cos(dro_phi)
		dy = np.sin(dro_phi)
		ref_xy = drompc.get_reference_path()
		ref_phi = drompc.get_reference_phi()
		ref_x = np.array(ref_xy[0,:]).transpose()
		ref_y = np.array(ref_xy[1,:]).transpose()

		if obs_passing is not None and obs_passing < n_obs:
			LB_value = drompc.get_LB()
			obs_dis[obs_passing].append(np.linalg.norm(ref_xy[:2,1] - lpv_pred_x[:2,1]))
			LB_dis[obs_passing].append(LB_value)
			print("obs detected at: {} step",idt)


		ref_phi = np.array(ref_phi).transpose()
		drex = np.cos(ref_phi)
		drey = np.sin(ref_phi)

		sim_x = model.sim_states(x0,dro_control,SAMPLING_TIME)

		new_pvx = lpv_pred_x[3,1:]
		new_pvy = lpv_pred_x[4,1:]
		new_pphi = lpv_pred_x[2,1:]

		drompc.update_new_p_param(new_pvx,new_pvy,new_pphi)

		states[:,idt+1] = lpv_pred_x[:,1]
		if add_disturbance:
			disturbances = np.array([[np.random.uniform(low, high)] for low, high in disturbance_ranges]).squeeze()
			states[:,idt+1] += disturbances

		model_diff = sim_x - lpv_pred_x # save at t=10 as model error
		model_diff = model_diff[:,:-1]
		if not use_fixed_noise:
			drompc.update_model_noise(model_diff)

		tau0 = drompc.get_tau0_value()
		if tau0 >= ob_center[0,1] and save_noise:
			drompc.save_fixed_noise()
			save_noise = False
		model_diff = model_diff.mean(axis=1)

		hstates2[:,0] = x0
		for idh in range(drompc.horizon):
			hstates2[:,idh+1] = lpv_pred_x[:,idh+1]

		ex[idt+1] = model_diff[0]
		ey[idt+1] = model_diff[1]
		ephi[idt+1] = model_diff[2]
		evx[idt+1] = model_diff[3]
		evy[idt+1] = model_diff[4]
		eomega[idt+1] = model_diff[5]

		#################################################################################################

		# update plot
		LnS.set_xdata(states[0,:idt+1])
		LnS.set_ydata(states[1,:idt+1])

		LnR.set_xdata(ref_x)
		LnR.set_ydata(ref_y)

		LnP.set_xdata([states[0,idt]])
		LnP.set_ydata([states[1,idt]])

		LnH.set_xdata(sim_x[0,:])
		LnH.set_ydata(sim_x[1,:])

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

	if dro_success:
		for i in range(n_obs):

			obs_disMean = np.mean(obs_dis[i])
			obs_disMax = np.amax(obs_dis[i])
			obs_disMin = np.amin(obs_dis[i])

			lb_disMean = np.mean(LB_dis[i])
			lb_disMax = np.amax(LB_dis[i])
			lb_disMin = np.amin(LB_dis[i])

			print("obstacle {} Mean distance : {:.5f}".format(i,obs_disMean))
			print("obstacle {} Max distance : {:.5f}".format(i,obs_disMax))
			print("obstacle {} Min distance : {:.5f}".format(i,obs_disMin))

			print("obstacle {} Mean lb : {:.5f}".format(i,lb_disMean))
			print("obstacle {} Max lb : {:.5f}".format(i,lb_disMax))
			print("obstacle {} Min lb : {:.5f}".format(i,lb_disMin))

	plt.ioff()

if save_figure:
	# # List of formats to save the figure in
	formats = ['png', 'pdf', 'svg']

	# Save the figure in each format
	for fmt in formats:
		if useDRCC:
			filename = os.path.join(script_dir,"outputFigure",f"dro_mpc.{fmt}")
		else:
			filename = os.path.join(script_dir,"outputFigure",f"lpv_mpc.{fmt}")

		plt.savefig(filename,transparent=True, format=fmt)
		print(f"Figure saved as {filename}")

plt.show(block=True)