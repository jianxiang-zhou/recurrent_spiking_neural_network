"""
*recurrent spiking neural network (RSNN) simulation*
Traning RSNN to learn sensorimotor task like in mouse exp.
RSNN is built using brian2 module (https://brian2.readthedocs.io/en/stable/index.html)

Each net constitutes 200 LIF neurons, each having 20 input and 20 output excitatory synapses
The weights of excitatory synapses are modified using reward-modulated STDP mechanism (Izhikevich 2007, Fremaux and Gerstner 2016)

Activity balance is acheived with lateral inhibition through a hand-tuned fixed inhibitory circuity

Before running, modify root_path to store the simulation data

Author: Jianxiang Zhou
Date: 2025/7/1

"""

from brian2 import *
import numpy as np
import csv
import random
import os
from multiprocessing import Pool
import matplotlib.pyplot as plt


def net_stru_generate(Nsen, Nmtr, in_degree):
    # generate network structure, exluding duplicated connections
    Nhid = Nsen+Nmtr
    to_distri = [in_degree]*Nhid
    timer = np.zeros(Nhid,)
    upstream = []
    avail = np.zeros(Nhid,)
    total_edge = in_degree*Nhid
    for cc in range(total_edge):
        tmp_max = max(to_distri)
        avail_nodes = [i for i in range(Nhid) if to_distri[i]==tmp_max]
        xx = 0
        while xx == 0:
            tmp_node_i = np.random.randint(0,len(avail_nodes))
            tmp_node = avail_nodes[tmp_node_i]
            if timer[tmp_node] == 0:
                upstream.append(tmp_node)
                timer = np.clip(timer-1,0,in_degree)
                timer[tmp_node] = in_degree
                xx = 1
    upstream_pool_sen = upstream[0:Nsen*in_degree]
    upstream_pool_mtr = upstream[Nsen*in_degree:Nhid*in_degree]
    downstream_pool_sen = []
    for i in range(Nsen):
        downstream_pool_sen.extend([i]*in_degree)
    downstream_pool_mtr = []
    for i in range(Nsen,Nhid):
        downstream_pool_mtr.extend([i]*in_degree)
    connection_cl = [upstream_pool_sen, downstream_pool_sen]
    connection_re = [upstream_pool_mtr, downstream_pool_mtr]
    return [connection_cl, connection_re]



def simulation(tmp_para):
    
    path, sym_coef = tmp_para
    show_trial_info =  False # True #
    start_scope()
    Nsensor = 2
    Nmotor = 2
    reward_status = 0
    punish_status = 0
    Narea_sen = 1
    Narea_mtr = 1
    
    Nsize = 100
    in_degree = 20 #40 #20
    motor_unit_len = 1
    Nsen = Nsize*Narea_sen # to simulate posterior half of cortex, receiving sensory inputs
    Nmtr = Nsize*Narea_mtr # to simulate anterior half of cortex, generating motor outputs
    Nhid = Nsen+Nmtr
    
    tau = 5*ms
    taupre = taupost = 20*ms
    tauelig = 50*ms   # adapt to network size
    
    tau_I = 3*second
    tau_rest = 2*ms
    IP_rest_real = 0.8  # inverse potential for inactive neurons
    IP_rest = IP_rest_real*(1+tau_rest/tau)
    
    Apre = 0.005  # adusting learning rate here
    Apost = sym_coef*Apre 
    
    
    ReStrength = 2 # reward after correct responses
    PuStrength = 2 # punishment after error responses
    
    homeoinout = 2.5 # total synaptic input and output of individual neurons are kept in constant level
    mean_w = homeoinout/in_degree
    
    max_w = 1.2
    t0_delay = 5*ms
    
    continue_learning = 0 # 0: start a new network, 1: continue learning
    
    eqs = '''
    dv/dt = -v/tau : 1 (unless refractory)
    '''
    
    eqs_hid = '''
    dv/dt = (IP_rest-v)*I/tau_rest - v/tau: 1 (unless refractory)
    wtotinCL : 1
    wtotoutCL : 1
    wtotinRE : 1
    wtotoutRE : 1
    act_lvl : 1
    tau_I : second
    dI/dt = (1-int(I>1))/tau_I : 1
    '''
    # I: indicator of activity history
    # act_lvl: indicator of current active neuron ratio, for modulating noise input
    
    tau_m = 20*ms
    eqs_m = '''
    dv/dt = (I-v)/tau_m : 1
    I : 1
    Mark : 1
    Act : 1
    '''
    
    Sen = NeuronGroup(Nsensor, eqs, threshold='v>1', reset='v=0',
                      refractory=0*ms, method='exact')
    
    process_reset = '''
                v = 0
                I = clip(I-0.1,0,1)
                '''

    Hid = NeuronGroup(Nhid, eqs_hid, threshold='v>1',
                      refractory=10*t0_delay, reset=process_reset, method='euler') # each neuron only fires once in a trial
    Hid.tau_I[:] = tau_I
    Hid.I = rand(Nhid,)/3
    
    # spontaneous activity mechanism: add noise to hidden neurons
    # spontaneous activity is stronger when the activity level of network is low
    # in low active state, such spontaneous activity is enough to drive inactive neurons to fire
    Hid.run_regularly(
        'v += clip((1-act_lvl/k),0,1)*((1-IP_rest_real+0.1)*int(rand()<firing_ratio))', dt=t0_delay)


    # inhibitory circuitry for lateral inhibition, hand-tuned to acheive activity level balance
    IP_inh = -1  # inverse potential of inhibitory synapse
    firing_num = 40
    firing_ratio = firing_num/Nhid
    k = firing_ratio/2
    b = firing_ratio*4
    Inh = NeuronGroup(1, '''v:1
                            act_level:1
                            Winh:1''', threshold='v>0', reset='''Winh = clip((v-k)/b,0,1)**0.9
                                                                act_level = v
                                                                v=0''', method='exact')
                    # act_level and Winh (nonlinear version of act_level) indicate how active the network is
    S_hi = Synapses(Hid, Inh, 'w : 1', on_pre='v_post += w')
    S_hi.connect(True)
    S_hi.w = 1/Nhid
    
    S_ih = Synapses(
        Inh, Hid, on_pre='''v_post = clip(v_post-(1+I_post*9)*Winh_pre*(1-IP_inh),IP_inh,1)
                            act_lvl_post = act_level_pre
                            ''', delay=t0_delay-1*ms)
       # connection from inhibitory neurons to excitatory neurons
       # when network is active, the inactive neurons are inhibited more to suppress spontaneous firing from these neurons
    S_ih.connect(True)
    
    
    
    
    
    # recording trial info
    Recorder = NeuronGroup(1, '''  
                             Nhid : 1
                             stimulus1 : 1
                             stimulus2 : 1
                             choice:1
                             ifreward:1
                             action_time : 1
                             count:1
                             resp_delay:1
                             trial_len:1
                             answer_window_start:1
                             last_t_cue:1
                             start_cue:1
                             state_cue:1
                             reward_cue:1
                             reward_status:1
                             last_trial_info:1
                             trial_start_time : 1
                             lasttrial_start_time : 1
                             first_action_time : 1
                             ''')
    Recorder.Nhid = Nhid
    
    
    # motor neurons as integrator of hidden layer information
    motor_reset = '''
                    v = 0
                    Mark = 1'''
    Motor = NeuronGroup(Nmotor, eqs_m, threshold='v>1',
                        reset=motor_reset, method='exact')
    
    # sensory input
    S_sh = Synapses(Sen, Hid, 'w : 1', on_pre='v_post += w')
    sen_unit_len = 20
    for idx in range(Nsensor):
        S_sh.connect(i=idx, j=range(idx*sen_unit_len, (idx+1)*sen_unit_len))
    S_sh.w = 5
    
    tau_synHom = 10*ms # time constant of synaptic homeostasis
    eqs_para_CL = '''
    wtotinCL_post = w : 1 (summed)
    wtotoutCL_pre = w : 1 (summed)
    dw/dt = int(w>0)*int(w<max_w)*(2*homeoinout-wtotoutCL_pre-wtotoutRE_pre-wtotinCL_post-wtotinRE_post)/tau_synHom: 1 (clock-driven)
    dapre/dt = -apre/taupre : 1 (event-driven)
    dapost/dt = -apost/taupost : 1 (event-driven)
    delig/dt = -elig/tauelig : 1(clock-driven)
    '''
    eqs_para_RE = '''
    wtotinRE_post = w : 1 (summed)
    wtotoutRE_pre = w : 1 (summed)
    dw/dt = int(w>0)*int(w<max_w)*(2*homeoinout-wtotoutCL_pre-wtotoutRE_pre-wtotinCL_post-wtotinRE_post)/tau_synHom: 1 (clock-driven)
    dapre/dt = -apre/taupre : 1 (event-driven)
    dapost/dt = -apost/taupost : 1 (event-driven)
    delig/dt = -elig/tauelig : 1(clock-driven)
    '''
    eqs_onpre = '''
    v_post += w
    apre += Apre
    elig += apost
    '''
    eqs_onpost = '''
    apost += Apost
    elig += apre
    '''
    
    S_clSTDP = Synapses(Hid, Hid, eqs_para_CL, on_pre=eqs_onpre,
                        on_post=eqs_onpost, method='euler')
    S_reSTDP = Synapses(Hid, Hid, eqs_para_RE, on_pre=eqs_onpre,
                        on_post=eqs_onpost, method='euler')
    
    
    
    
    if continue_learning == 0:
        connection_cl, connection_re =  net_stru_generate(Nsen, Nmtr, in_degree)
        S_clSTDP.connect(i=connection_cl[0], j=connection_cl[1])
        S_reSTDP.connect(i=connection_re[0], j=connection_re[1])
    
        if not os.path.exists(path):
            os.mkdir(path)
    
        with open(path+'\connection.csv', 'a+', encoding='utf-8', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(S_clSTDP.i[:])
            csv_writer.writerow(S_clSTDP.j[:])
            csv_writer.writerow(S_reSTDP.i[:])
            csv_writer.writerow(S_reSTDP.j[:])
        S_clSTDP.w = 'rand()*homeoinout/in_degree*2'
        S_reSTDP.w = 'rand()*homeoinout/in_degree*2'
    
        with open(path+'\weights.csv', 'a+', encoding='utf-8', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(S_clSTDP.w[:])
            csv_writer.writerow(S_reSTDP.w[:])
        print('weights saved')
    
    else:
        with open(path+'\connection.csv') as f:
            connection_C = csv.reader(f)
            connection_raw = list(connection_C)
    
        for ii in range(4):
            while '' in connection_raw[ii]:
                connection_raw[ii].remove('')
        S_clSTDP.connect(i=np.array(connection_raw[0], dtype='int'), j=np.array(connection_raw[1], dtype='int'))
        S_reSTDP.connect(i=np.array(connection_raw[2], dtype='int'), j=np.array(connection_raw[3], dtype='int'))
        
        with open(path+'\weights.csv') as f:
            weight_C = csv.reader(f)
            weight_raw = list(weight_C)
        for ii in range(2):
            while '' in weight_raw[-ii-1]:
                weight_raw[-ii-1].remove('')
        S_clSTDP.w[:] = np.array(weight_raw[-2], dtype='float')
        S_reSTDP.w[:] = np.array(weight_raw[-1], dtype='float')
    
    
    
    S_clSTDP.pre.delay = t0_delay-0.1*ms
    S_clSTDP.post.delay = t0_delay-0.3*ms
    S_reSTDP.pre.delay = t0_delay-0.1*ms
    S_reSTDP.post.delay = t0_delay-0.3*ms
    
    
    # reward mechanism
    eqs_DA_neuron = ''' 
                        DAlvl : 1
                        '''
    RPE = NeuronGroup(1, eqs_DA_neuron, threshold='DAlvl!=0', reset='DAlvl=0',refractory=1*ms, method='euler')
    
    
    # adjusting weights according to RPE
    S_rpe_hid = Synapses(RPE, S_reSTDP, on_pre='''
                 w_post = clip(w_post+elig_post*DAlvl_pre,0,max_w)
                 elig_post = 0
                 ''', method='exact')
    S_rpe_hid.connect(True)

    
    # background STDP, set to be very slow
    BackGround = NeuronGroup(1, 'v:1', threshold='v>0',refractory=t0_delay, method='exact')
    BackGround.v = 1
    S_b_sen = Synapses(BackGround, S_clSTDP,on_pre='w_post = clip(w_post+0.001*elig_post,0,max_w)')
    S_b_sen.connect(True)
    S_b_mtr = Synapses(BackGround, S_reSTDP,on_pre='w_post = clip(w_post+0.001*elig_post,0,max_w)')
    S_b_mtr.connect(True)
    
    
    # Motor groups
    S_hm = Synapses(Hid, Motor, 'w : 1', on_pre='v_post += w', delay=t0_delay-4*ms)
    for idx in range(Nmotor):
        S_hm.connect(i=range(Nhid-(idx+1)*motor_unit_len,Nhid-idx*motor_unit_len), j=1-idx)
    S_hm.w = 2
    
    
    
    
    @network_operation
    def sensory_input(t):
        t_trial = t/ms-Recorder.trial_start_time
        if Recorder.last_t_cue == 0:
            if t_trial >= 5:
                Recorder.last_t_cue = 1
                if Recorder.last_trial_info[0]>=0:   
                    idx = int(Recorder.last_trial_info[0])
                else:
                    idx = int(rand()<0.5)
                print('Last info = ', int(Recorder.last_trial_info[0]))
                Sen.v[idx] = 2
                Recorder.stimulus1 = idx
    
    
    @network_operation
    def motor_output(t):
        t_trial = t/ms - Recorder.trial_start_time
        if t_trial<10:   # reset action immediately after stimulus
            Motor.Mark = 0
            Motor.Act = 0
            
        if Motor.Mark[0] == 1 and Motor.Mark[1] == 0:
            Motor.Mark = 0
            if sum(Motor.Act)==0:
                Recorder.first_action_time = t_trial
            else:
                RPE.DAlvl = -PuStrength/2
            Motor.Act[0] = 1
            Motor.Act[1] = 0
        elif Motor.Mark[1] == 1 and Motor.Mark[0] == 0:
            Motor.Mark = 0
            if sum(Motor.Act)==0:
                Recorder.first_action_time = t_trial
            else:
                RPE.DAlvl = -PuStrength/2
            Motor.Act[1] = 1
            Motor.Act[0] = 0
        elif Motor.Mark[1] == 1 and Motor.Mark[0] == 1:
            Motor.Mark = 0
            RPE.DAlvl = -PuStrength/2  # punishment of choosing both sides
        else:
            Motor.Mark = 0
            
        if Recorder.count == 0:
            if t_trial > Recorder.answer_window_start:
                # Hid.v[298] = 2
                if Motor.Act[0] == 1:
                    Recorder.choice = 0
                    Recorder.action_time = t_trial
                    Recorder.count = 1
                    print('Left choice at time: ', t_trial, 'ms')
                    Recorder.trial_len = 10*(int(t_trial/10)+5)
                elif Motor.Act[1] == 1:
                    Recorder.choice = 1
                    Recorder.action_time = t_trial
                    Recorder.count = 1
                    print('Right choice at time: ', t_trial, 'ms')
                    Recorder.trial_len = 10*(int(t_trial/10)+5)

    
    @network_operation
    def reward_delivery(t):
        if Recorder.count == 1 and Recorder.reward_status == 0:
            Recorder.reward_status = 1
            if Recorder.stimulus1 == Recorder.choice:
                # Reward_DA_neuron.v = 2
                Recorder.ifreward = 1
                RPE.DAlvl = ReStrength
                print('Reward!')
            else:
                # Punish_neuron.v = 2
                RPE.DAlvl = -PuStrength
                print('Punishment!')
                
    # @network_operation
    # def synaptic_volatility(t):
    #     if t/ms % 4000 == 0:
    #         idx_chg_cl = random.sample(range(len(S_clSTDP)),10)
    #         S_clSTDP.w[idx_chg_cl] = clip(S_clSTDP.w[idx_chg_cl]+np.random.randn(10)/2,0,max_w)
    #         idx_chg_dm = random.sample(range(len(S_reSTDP)),10)
    #         S_reSTDP.w[idx_chg_dm] = clip(S_reSTDP.w[idx_chg_dm]+np.random.randn(10)/2,0,max_w)
            
    
    
    @network_operation
    def stop_func(t):
        if t/ms-Recorder.trial_start_time >= Recorder.trial_len-Recorder.trial_start_time[0] % 10 - 15 and t/ms-Recorder.trial_start_time < Recorder.trial_len-Recorder.trial_start_time[0] % 10 - 10:
            Hid.v = 0
            Hid.act_lvl = k
            Inh.act_level = k
        
        if t/ms-Recorder.trial_start_time >= Recorder.trial_len-Recorder.trial_start_time[0] % 10:
            Recorder.lasttrial_start_time = Recorder.trial_start_time
            Recorder.trial_start_time = t/ms
            Recorder.last_t_cue = 0
            Recorder.start_cue = 0
            stop()
    
    
    
    # Recorder.resp_delay = 20 # delay response
    Recorder.answer_window_start = 20 # answer window start time

    
    runningtime = 300*ms
    
    plt.figure()
    plt.plot(S_clSTDP.i, S_clSTDP.j, '.')
    plt.plot(S_reSTDP.i, S_reSTDP.j, '.')
    
    plt.xlim([0, Nhid])
    plt.ylim([0, Nhid])
    
    
    Recorder.last_trial_info = -1
    Twindow = 5;t_len = 10
    trial_len = 100
    show_len = 100
    
    # running 2000 trials in total
    trialN = 20
    sessionN = 100
    for idx_s in range(sessionN):
        for idx_t in range(trialN):

            
            Recorder.stimulus1 = -1
            Recorder.stimulus2 = -1
            Recorder.choice = -1
            Recorder.ifreward = 0
            Recorder.action_time = -100
            Recorder.reward_status = 0
            Recorder.count = 0
            Recorder.last_t_cue = 0
            Recorder.trial_len = 100
    
            spikemon_hid = SpikeMonitor(Hid)
            statemon_RPE = StateMonitor(RPE, 'DAlvl', record=0)
            statemon_hid = StateMonitor(Hid, 'v', record=range(Nhid-Nmtr, Nhid))
            statemon_hid_CLout = StateMonitor(Hid, ('wtotoutCL','wtotoutRE',), record=range(Nsen-100, Nsen))
            statemon_hid_sen_out = StateMonitor(Hid, 'wtotoutRE', record=[0,20])
            statemon_hid_mtr_in = StateMonitor(Hid, 'wtotinRE', record=range(Nhid-2,Nhid))
            statemon_motor = StateMonitor(Motor, 'v', record = range(2))
            statemon_inh = StateMonitor(Inh,('Winh','act_level'),record=0)
    
            print('\nSession #', idx_s, '  Trial #', idx_t)
            print('Trial start time :', Recorder.trial_start_time[0])
               
            run(runningtime)
    
            # record behavior and neuronal activity
            with open(path+r'\behavior.csv', 'a+', encoding='utf-8', newline='') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow((idx_s, idx_t, Recorder.stimulus1[0], Recorder.stimulus2[0], Recorder.choice[0],
                                     Recorder.ifreward[0], Recorder.action_time[0],Recorder.first_action_time[0]))
            with open(path+r'\spikes.csv', 'a+', encoding='utf-8', newline='') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(spikemon_hid.i)
                spk_time = spikemon_hid.t/ms - Recorder.lasttrial_start_time
                csv_writer.writerow(spk_time)
            
            
            if show_trial_info:
                figure()
                subplot(421)
                plot(spikemon_hid.t/ms-Recorder.lasttrial_start_time,
                     spikemon_hid.i, '.k', markersize=0.2)
                xlim(0, show_len)
                ylim(0, Nhid)
                ylabel('Neurons')
                subplot(422)
                plot(statemon_RPE.t/ms-Recorder.lasttrial_start_time,statemon_RPE.DAlvl[0])
                xlim(0, show_len)
                ylim(-50,50)
                ylabel('R')
                # xlabel('Time (ms)')
                subplot(423)
                plot(Hid.I[:], '.', markersize=0.5)
                subplot(424)
                for plt_i in range(Nmtr):
                    if plt_i<motor_unit_len:
                        clr = (0.8,0.4,0)
                    else:
                        clr = (0,0.3,0.6)
                            
                    plot(statemon_hid.t/ms-Recorder.lasttrial_start_time,
                         statemon_hid.v[plt_i], linewidth=0.1,color=clr)
                xlim(0, show_len)
                ylabel('v')
                # xlabel('Time (ms)')
                ylim(-1.1, 1.1)
                subplot(425)
                for plt_i in range(2):
                    plot(statemon_hid_sen_out.t/ms-Recorder.lasttrial_start_time, statemon_hid_sen_out.wtotoutRE[plt_i], linewidth=0.7)
                    
                ylabel('wtotoutRE')
                xlabel('Time (ms)')
                xlim(0, show_len)
                
                subplot(427)
                for plt_i in range(2):
                    plot(statemon_hid_mtr_in.t/ms-Recorder.lasttrial_start_time, statemon_hid_mtr_in.wtotinRE[plt_i], linewidth=0.7)
                ylabel('wtotin_mtr_neu')
                

                xlabel('Time (ms)')
                xlim(0, show_len)
                
                subplot(426)
                plot(statemon_motor.t/ms-Recorder.lasttrial_start_time, statemon_motor.v[0],color=(0,0.2,0.8))
                plot(statemon_motor.t/ms-Recorder.lasttrial_start_time, -statemon_motor.v[1],color=(0.8,0.2,0))
                ylabel('v')
                xlim(0, show_len)
                
                firing_rate = np.zeros(t_len)
                for win_i in range(t_len):
                    idx = (spk_time>=win_i*Twindow+2)&(spk_time<(win_i+1)*Twindow+2)
                    firing_rate[win_i] = sum(idx)
                subplot(428)
                plot(np.linspace(Twindow,Twindow*t_len,t_len), firing_rate)
                plot(statemon_inh.t/ms-Recorder.lasttrial_start_time, Nhid*statemon_inh.Winh[0])
                plot(statemon_inh.t/ms-Recorder.lasttrial_start_time, Nhid*statemon_inh.act_level[0])
                xlim(0, show_len)
                xlabel('Time (ms)')
                
                trial_info = ['S1=', Recorder.stimulus1[0], ' ES=', Recorder.last_trial_info[0], ' Choice=', Recorder.choice[0],
                          ' ActionTime=', Recorder.action_time[0], '  Reward=', Recorder.ifreward[0],
                          ' FAT=',Recorder.first_action_time[0]]
                suptitle(trial_info)
                
                pause(0.01)
                print(trial_info)
                
            if Recorder.ifreward[0]==0:  # error stay mechanism
                Recorder.last_trial_info = Recorder.stimulus1[0]
            else:
                Recorder.last_trial_info = -1
                
        print('--------------End of session-----------------')
    
        with open(path+r'\weights.csv', 'a+', encoding='utf-8', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(S_clSTDP.w[:])
            csv_writer.writerow(S_reSTDP.w[:])
        

if __name__ == '__main__':

    iteration = 5
    sym_list = [-1,1] # asymmetrical or symmetrical STDP
    symN = len(sym_list)
    
    root_path = r'K:\MODEL\data\simu_data\symasym_simu0704'

    
    para = []
    for it in range(iteration):
        for sym_i in range(symN):
            path = root_path + r'\2AFC_'+str(it)+\
                '_sym'+str(np.round(sym_list[sym_i],2))
            # path = r'K:\MODEL\data\simu_data\symasym_simu0701\2AFC_'+str(it)+\
            #     '_sym'+str(np.round(sym_list[sym_i],2))
            para.append([path,sym_list[sym_i]])
    with Pool(10) as p:
        p.map(simulation,para)
        
    # tmp_para = [r'K:\MODEL\data\simu_data\test', -1 ]
    # simulation(tmp_para)
    
    

