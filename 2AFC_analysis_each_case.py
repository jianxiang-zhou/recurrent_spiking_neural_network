"""
analyze for each simulation:
1. task neuron fraction over learning
2. temporal shift of neuronal activity
3. behavioral performance

Author: Jianxiang Zhou
Date: 2025/7/1

"""

import numpy as np
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from scipy.stats import ttest_1samp
import glob
import pickle

def show_pattern_dist(pattern,trial_type):
    fig = plt.figure(figsize=(20,20))
    ax = plt.subplot(131,projection='3d')
    ax.elve = 30
    ax.azim = 15
    ax.plot(pattern[trial_type[0],0],pattern[trial_type[0],1],pattern[trial_type[0],2],'.')
    ax.plot(pattern[trial_type[1],0],pattern[trial_type[1],1],pattern[trial_type[1],2],'.')
    ax.plot(pattern[trial_type[2],0],pattern[trial_type[2],1],pattern[trial_type[2],2],'.')
    ax.plot(pattern[trial_type[3],0],pattern[trial_type[3],1],pattern[trial_type[3],2],'.')
    ax2 = plt.subplot(132,projection='3d')
    ax2.elve = 30
    ax2.azim = 45
    ax2.plot(pattern[trial_type[0],0],pattern[trial_type[0],1],pattern[trial_type[0],2],'.')
    ax2.plot(pattern[trial_type[1],0],pattern[trial_type[1],1],pattern[trial_type[1],2],'.')
    ax2.plot(pattern[trial_type[2],0],pattern[trial_type[2],1],pattern[trial_type[2],2],'.')
    ax2.plot(pattern[trial_type[3],0],pattern[trial_type[3],1],pattern[trial_type[3],2],'.')
    ax3 = plt.subplot(133,projection='3d')
    ax3.elve = 30
    ax3.azim = 75
    ax3.plot(pattern[trial_type[0],0],pattern[trial_type[0],1],pattern[trial_type[0],2],'.')
    ax3.plot(pattern[trial_type[1],0],pattern[trial_type[1],1],pattern[trial_type[1],2],'.')
    ax3.plot(pattern[trial_type[2],0],pattern[trial_type[2],1],pattern[trial_type[2],2],'.')
    ax3.plot(pattern[trial_type[3],0],pattern[trial_type[3],1],pattern[trial_type[3],2],'.')
    
def cal_center_of_mass(trace):
    t_len = len(trace)
    COM = sum(np.array([i for i in range(t_len)], dtype=int)*trace)/sum(trace)
    return COM

def cal_delta_c(trace1,trace2):
    # trace1 = mean_FP_AS[0,:,ses_i-1,neu_i]
    # trace2 = mean_FP_AS[0,:,ses_i,neu_i]
    delta_c = cal_center_of_mass(trace2)-cal_center_of_mass(trace1)
    return delta_c

def reoranization_dynamics(auc_neurons_AS,mean_FP_AS):
    thr = 0.2
    sesN = np.size(auc_neurons_AS,0)
    neuN = np.size(auc_neurons_AS,1)
    time_len = np.size(mean_FP_AS,1)
    active_low = auc_neurons_AS<-thr
    active_high = auc_neurons_AS>thr
    low_neuron = np.sum(active_low,0)>0
    high_neuron = np.sum(active_high,0)>0
    low_neuron_sel = low_neuron & np.logical_not(high_neuron)
    high_neuron_sel = np.logical_not(low_neuron) & high_neuron
    neuron_dyn_info = []
    for neu_i in range(40,neuN):
        if low_neuron_sel[neu_i] or high_neuron_sel[neu_i]:
            if low_neuron_sel[neu_i]:
                tmp_arr = list(active_low[:,neu_i])
                tmp_act = mean_FP_AS[0,:,:,neu_i]
            elif high_neuron_sel[neu_i]:
                tmp_arr = list(active_high[:,neu_i])
                tmp_act = mean_FP_AS[-1,:,:,neu_i]
                
            ses_idx = []
            com_change = []
            for ses_i in range(sesN):
                if tmp_arr[ses_i]:
                    ses_idx.append(ses_i)
                    com_change.append(cal_center_of_mass(tmp_act[:,ses_i]))
            idx = extract_conti_seq(ses_idx)   # ******************************************
            
            dyn_info = [ses_idx[idx[0]:(idx[-1]+1)], com_change[idx[0]:(idx[-1]+1)]]
            neuron_dyn_info.append(dyn_info)
    return neuron_dyn_info

def extract_conti_seq(sequence):
    seq_len = len(sequence)
    tmp_seq = sequence + np.linspace(seq_len,1,seq_len)
    tmp_ele = np.unique(tmp_seq)
    tmp_seq = list(tmp_seq)
    most_ele = 0
    most_ele_count = 0
    for ele in tmp_ele:
        cc = tmp_seq.count(ele)
        if cc>most_ele_count:
            most_ele_count = cc
            most_ele = ele
    idx = []
    for i in range(seq_len):
        if tmp_seq[i]==most_ele:
            idx.append(i)
    return idx

    

def deltac_to_deltases(auc_neurons_AS, neu_sel, mean_FP_AS, delta_ses):
    thr = 0.5
    sesN = np.size(auc_neurons_AS,0)
    delta_c_store = []
    for neu_i in neu_sel:
        active_idx_low = np.array(auc_neurons_AS[:,neu_i]<-thr,dtype=int)
        active_idx_high = np.array(auc_neurons_AS[:,neu_i]>thr,dtype=int)
        
        cumu_active_idx_low = np.zeros(np.shape(active_idx_low))
        cumu_active_idx_high = np.zeros(np.shape(active_idx_high))
        cumu_active_idx_low[delta_ses:] += active_idx_low[delta_ses:]
        cumu_active_idx_high[delta_ses:] += active_idx_high[delta_ses:]
        
        for delta_i in range(1,delta_ses+1):
            cumu_active_idx_low[delta_ses:] += active_idx_low[delta_ses-delta_i:-delta_i]
            cumu_active_idx_high[delta_ses:] += active_idx_high[delta_ses-delta_i:-delta_i]
        
        sel_ses_low = cumu_active_idx_low==(delta_ses+1)
        sel_ses_high = cumu_active_idx_high==(delta_ses+1)
        
        for ses_i in range(delta_ses,sesN):
            if sel_ses_low[ses_i]:
                delta_c = cal_delta_c(mean_FP_AS[0,:,ses_i-delta_ses,neu_i],mean_FP_AS[0,:,ses_i,neu_i])
                if delta_c<0 or delta_c>0:
                    # print(delta_c)
                    delta_c_store.append(delta_c)
            elif sel_ses_high[ses_i]:
                delta_c = cal_delta_c(mean_FP_AS[-1,:,ses_i-delta_ses,neu_i],mean_FP_AS[-1,:,ses_i,neu_i])
                if delta_c<0 or delta_c>0:
                    # print(delta_c)
                    delta_c_store.append(delta_c)
                
    return delta_c_store
                
        
        


def plot_result(path):
    with open(path + r'\behavior.csv') as f:    #STDP only  NoPlasticity
        beh = np.loadtxt(f,delimiter = ",")
    with open(path + r'\spikes.csv') as f:
        spk_C = csv.reader(f)
        spk_raw = list(spk_C)
        
    print(len(beh[:,0]),len(spk_raw))
    trialN = 20
    sessionN = min(len(beh[:,0])//trialN,100)
    neuronN = 100 # 500 #
    print('trialN=',trialN,'  sessionN=',sessionN)
    combine_rate = 2 #2 #10
    init_session = 0
    FR_all_neuron_curve = []
    sesN_comb = int(sessionN/combine_rate)
    mean_FP_AS = np.zeros((4,20,sesN_comb,neuronN))
    mean_FP_AS_2type = np.zeros((2,20,sesN_comb,neuronN))
    mean_ft_per_neuron_AS = np.zeros((20,sesN_comb,neuronN))
    mean_fr_per_neuron_AS = []
    auc_neurons_AS = np.zeros((sesN_comb,neuronN))
    
    
    # plot learning curve
    learning_curve = [[],[],[],[]]
    learning_curve_miss_all = [[],[]]
    learning_curve_miss_left = [[],[]]
    learning_curve_miss_right = [[],[]]
    learning_curve_left = [[],[]]
    learning_curve_right = [[],[]]
    low_tone_ratio = [[],[]]
    for ses_i in range(init_session,int(sessionN/combine_rate)):
        stimulus = beh[ses_i*trialN*combine_rate:(ses_i+1)*trialN*combine_rate,2]
        choice = beh[ses_i*trialN*combine_rate:(ses_i+1)*trialN*combine_rate,4]
        ifopto = np.zeros(trialN*combine_rate) #beh[ses_i*trialN*combine_rate:(ses_i+1)*trialN*combine_rate,7]
        stimulus_ctrl = stimulus[ifopto==0]
        choice_ctrl = choice[ifopto==0]
        stimulus_opto = stimulus[ifopto==1]
        choice_opto = choice[ifopto==1]
        crt_rate_ctrl = sum(stimulus_ctrl==choice_ctrl)/(sum(choice_ctrl>=0)+1e-5) #(sum(choice_ctrl>=0)+1e-5)
        crt_rate_opto = sum(stimulus_opto==choice_opto)/(sum(choice_opto>=0)+1e-5) #(sum(choice_opto>=0)+1e-5)
        learning_curve[0].append(crt_rate_ctrl)
        learning_curve[1].append(crt_rate_opto)
        learning_curve[2].append(sum(stimulus_ctrl==choice_ctrl)/(sum(choice_ctrl>=0)+1e-5))
        learning_curve[3].append(sum(stimulus_opto==choice_opto)/(sum(choice_opto>=0)+1e-5))
        learning_curve_miss_all[0].append(sum(choice_ctrl<0)/(len(choice_ctrl)+1e-5))
        learning_curve_miss_all[1].append(sum(choice_opto<0)/(len(choice_opto)+1e-5))
        
        learning_curve_miss_left[0].append(sum(choice_ctrl[stimulus_ctrl==0]<0)/(len(choice_ctrl[stimulus_ctrl==0])+1e-5))
        learning_curve_miss_right[0].append(sum(choice_ctrl[stimulus_ctrl==1]<0)/(len(choice_ctrl[stimulus_ctrl==1])+1e-5))
        learning_curve_left[0].append(sum((stimulus_ctrl==0) & (choice_ctrl==0))/(sum(choice_ctrl[stimulus_ctrl==0]>=0)+1e-5))
        learning_curve_right[0].append(sum((stimulus_ctrl==1) & (choice_ctrl==1))/(sum(choice_ctrl[stimulus_ctrl==1]>=0)+1e-5))
        low_tone_ratio[0].append(sum(stimulus_ctrl==0)/len(stimulus_ctrl))
        learning_curve_miss_left[1].append(sum(choice_opto[stimulus_opto==0]<0)/(len(choice_opto[stimulus_opto==0])+1e-5))
        learning_curve_miss_right[1].append(sum(choice_opto[stimulus_opto==1]<0)/(len(choice_opto[stimulus_opto==1])+1e-5))
        learning_curve_left[1].append(sum((stimulus_opto==0) & (choice_opto==0))/(sum(choice_opto[stimulus_opto==0]>=0)+1e-5))
        learning_curve_right[1].append(sum((stimulus_opto==1) & (choice_opto==1))/(sum(choice_opto[stimulus_opto==1]>=0)+1e-5))
        low_tone_ratio[1].append(sum(stimulus_opto==0)/(len(stimulus_opto)+1e-5))
        
    lc_all = learning_curve[0]
    lc_left = learning_curve_left[0]
    lc_right = learning_curve_right[0]
    lc_miss_left = learning_curve_miss_left[0]
    lc_miss_right = learning_curve_miss_right[0]
    perf_all = [lc_all, lc_left, lc_right, lc_miss_left, lc_miss_right]
    save_name = path+r'\perf_AS.npy'
    np.save(save_name, perf_all)
    
    plt.figure()
    plt.subplot(211)
    plt.plot(learning_curve_left[0],label='left')
    plt.plot(learning_curve_right[0],label='right')
    plt.plot(learning_curve[0],label='all')
    plt.legend()
    plt.subplot(212)
    plt.plot(learning_curve_miss_left[0],label='miss_left')
    plt.plot(learning_curve_miss_right[0],label='miss_right')
    plt.legend()
    
    t_len = 20
    for ses_i in range(init_session,sesN_comb):
        spkN = []
        spkT = []
        Twindow = 5
        firing_rate = np.zeros((trialN*combine_rate,20))
        mean_firing_pattern = np.zeros((4,20,neuronN))
        mean_firing_pattern_2type = np.zeros((2,20,neuronN))
        mean_firing_time = np.zeros((20,neuronN))
        act_amp = np.zeros((neuronN))
        auc_neurons = np.zeros((neuronN))
        
        for i in range(trialN*combine_rate):
            tmp_spkN = spk_raw[2*i+2*ses_i*trialN*combine_rate]
            tmp_spkT = spk_raw[2*i+1+2*ses_i*trialN*combine_rate]
            while '' in tmp_spkN:
                tmp_spkN.remove('')
            while '' in tmp_spkT:
                tmp_spkT.remove('')
    
            spkN.append(tmp_spkN)  
            spkT.append(tmp_spkT)
        activity_2stim = np.zeros((trialN*combine_rate,neuronN))
        activity_2stim_label = np.zeros((trialN*combine_rate,))

        trial_count = np.zeros((4))
        crt_trial = []
        
        for i in range(trialN*combine_rate):
            s1 = int(beh[i+ses_i*trialN*combine_rate,2])
            tmp_spkT = np.array(spkT[i],dtype=float)
            tmp_spkN = np.array(spkN[i],dtype=int)
            
            # extracting neuronal activity under two stimuli
            activity_2stim_label[i] = s1
            for neu_i in tmp_spkN:
                if neu_i >= 100:
                    activity_2stim[i,neu_i-100] += 1
            
            ch = int(beh[i+ses_i*trialN*combine_rate,4])
            if s1==ch:
                crt_trial.append(i)
            tmp_trialtype = 2*s1+ch
            if s1>=0 and ch>=0:
 
                trial_count[tmp_trialtype]+=1
                
                for win_i in range(t_len):
                    idx = (tmp_spkT>=win_i*Twindow+2)&(tmp_spkT<(win_i+1)*Twindow+2)
                    firing_rate[i,win_i] = sum(idx)
                    tmp_neuron = tmp_spkN[idx]
                    for neu_i in tmp_neuron:
                        if neu_i >= 100:
                            mean_firing_pattern[tmp_trialtype,win_i,neu_i-100] += 1
                        
        for neu_i in range(neuronN):
            mean_firing_time[:,neu_i] = np.sum(mean_firing_pattern[:,:,neu_i], 0)/sum(trial_count)
            act_amp[neu_i] = np.sum(mean_firing_pattern[:,:,neu_i])/sum(trial_count)
            # discrimination ability of each neuron
            fpr, tpr, thresholds = roc_curve(activity_2stim_label, activity_2stim[:,neu_i])
            auc_neurons[neu_i] = (auc(fpr,tpr)-0.5)*2
            
        for tt_i in range(2):
            if (trial_count[2*tt_i]+trial_count[2*tt_i+1])>0:
                mean_firing_pattern_2type[tt_i,:,:] = (mean_firing_pattern[2*tt_i,:,:]+\
                                                       mean_firing_pattern[2*tt_i+1,:,:])/\
                    (trial_count[2*tt_i]+trial_count[2*tt_i+1])
        
        for tt_i in range(4):
            if trial_count[tt_i]>0:
                mean_firing_pattern[tt_i,:,:] = mean_firing_pattern[tt_i,:,:]/trial_count[tt_i]
                
        

        mean_ft_per_neuron_AS[:,ses_i,:] = mean_firing_time
        mean_FP_AS[:,:,ses_i,:] = mean_firing_pattern
        mean_FP_AS_2type[:,:,ses_i,:] = mean_firing_pattern_2type
        

        FR_all_neuron_curve.append(np.mean(firing_rate,0))
        mean_fr_per_neuron_AS.append(act_amp)
        auc_neurons_AS[ses_i,:] = auc_neurons
        
        
    # show example neuron
    # for neu_i in range(100):
    #     plt.figure()
    #     plt.subplot(121)
    #     sns.heatmap(mean_FP_AS_2type[0,:,:,neu_i],vmin=0,vmax=1)
    #     plt.subplot(122)
    #     sns.heatmap(mean_FP_AS_2type[1,:,:,neu_i],vmin=0,vmax=1)
    #     # plt.title(auc_neurons_AS[:,neu_i])
    #     plt.pause(0.01)
        
        
    neuron_dyn_info = reoranization_dynamics(auc_neurons_AS,mean_FP_AS_2type)  # ***********
    # save_name3 = path+r'\neuron_dyn_info.pkl'
    # with open(save_name3, 'wb') as f:
    #     pickle.dump(neuron_dyn_info, f)
    
    
    # show number of sound selective neurons 
    thr=0.5
    plt.figure()
    plt.plot(np.sum(abs(auc_neurons_AS)>thr,1))
    save_name1 = path+r'\auc_neurons_AS_2type.npy'
    np.save(save_name1, auc_neurons_AS)
    
    
    # analyze temporal progression of neuronal activity
    neu_sel = [i for i in range(neuronN) if np.sum(abs(auc_neurons_AS)>thr,0)[i]>0]
    deltac_to_deltases_store = []
    delta_sesall = 5
    for delta_i in range(delta_sesall):
        # delta_c_store = deltac_to_deltases(auc_neurons_AS, neu_sel, mean_FP_AS, delta_i+1)
        delta_c_store = deltac_to_deltases(auc_neurons_AS, neu_sel, mean_FP_AS_2type, delta_i+1)
        
        deltac_to_deltases_store.append(delta_c_store)

    t,p = ttest_1samp(deltac_to_deltases_store[delta_sesall-1],0)
    plt.figure()
    n,bins,patches = plt.hist(deltac_to_deltases_store[delta_sesall-1],bins=40)
    plt.xlabel('delta center')
    plt.ylabel('count')
    plt.title('Mean = '+str(np.mean(deltac_to_deltases_store[delta_sesall-1]))+' (arbitraty time unit); P = '+str(p))
    plt.pause(0.01)
    
    save_name2 = path+r'\delta_center_to_deltases_2type.pkl'
    with open(save_name2, 'wb') as f:
        pickle.dump(deltac_to_deltases_store, f)
    




    plt.pause(0.01)
    
if __name__=='__main__':
    
    path = r'K:\MODEL\data\simu_data\symasym_simu0704\2AFC*sym1*'
    
    path_all_mice = glob.glob(path)
    miceN = len(path_all_mice)
    
    for i in range(miceN):
        print('mouse#',i)
        tmp_path = path_all_mice[i]
        plot_result(tmp_path)
        
    
        
        
        
