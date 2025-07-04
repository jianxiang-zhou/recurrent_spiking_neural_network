"""
summary the simulation results  for all cases

Author: Jianxiang Zhou
Date: 2025/7/1


"""


import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp
from matplotlib import rcParams
import pickle

para = 'sym-1'
save_path = r'K:\MODEL\data\simu_data\figures'
folder_name = r'symasym_simu0704'
# folder_name = r'symasym'
path = r'K:\MODEL\data\simu_data/' + folder_name + r'\2AFC*'+para+'*'
path_delta_center = path+r'\delta_center_to_deltases_2type.pkl'
file_delta_center = glob.glob(path_delta_center)

miceN = len(file_delta_center)


rcParams['font.size'] = 14

delta_sesmax = 5
delta_COM_store = []
for i in range(delta_sesmax):
    delta_COM_store.append([])
for i in range(miceN):
    with open(file_delta_center[i],'rb') as f:
        tmp_deltac_to_deltases_store = pickle.load(f)
    for k in range(delta_sesmax):
        sublist = delta_COM_store[k]
        sublist.extend(tmp_deltac_to_deltases_store[k])


meansem = np.zeros((3,delta_sesmax))
for i in range(delta_sesmax):
    t,p = ttest_1samp(delta_COM_store[i],0)
    plt.figure(figsize=(8,6))
    plt.subplot(324)
    n,bins,patches = plt.hist(delta_COM_store[i],bins=40)
    # print(patches)
    plt.xlabel('Temporal progression (arbitrary time unit)')
    plt.ylabel('Count')
    plt.title('Mean = ' + str(np.mean(delta_COM_store[i]))+'; P = '+f"{p:.2g}")
    plt.xlim([-5, 5])
    # plt.savefig(save_path+r'\\'+para+'_temporal_progression'+str(i)+'.svg',format='svg')
    
    meansem[0,i] = np.mean(delta_COM_store[i])
    meansem[1,i] = np.std(delta_COM_store[i]) /np.sqrt(np.size(delta_COM_store[i]))
    meansem[2,i] = p
np.save(save_path+'/'+folder_name+ r'_meansem_'+para+'.npy',meansem)



path_task_neuron = path+r'\auc_neurons_AS_2type.npy'
file_task_neuron = glob.glob(path_task_neuron)

thr = 0.5
min_sessionN = 100
org_min_sesN = min_sessionN
task_neuron_store = np.zeros((org_min_sesN, miceN))
delta_max = 20
remain_fraction_store = []
for i in range(delta_max):
    remain_fraction_store.append([])
    
for i in range(miceN):
    tmp_task_neuron = abs(np.load(file_task_neuron[i]))
    neuronN = np.size(tmp_task_neuron,1)
    min_sessionN = np.min([np.size(tmp_task_neuron,0),min_sessionN])
    tmp_min_sesN = np.min([np.size(tmp_task_neuron,0),org_min_sesN])
    task_neuron_store[:tmp_min_sesN, i] = np.sum(tmp_task_neuron>thr,1)[:tmp_min_sesN]
    for delta_i in range(1,21):
        for ses_i in range(min_sessionN):
            if ses_i + delta_i < min_sessionN:
                tmp_tsk_neuron_idx_pre = tmp_task_neuron[ses_i,:]>thr
                tmp_tsk_neuron_idx_post = tmp_task_neuron[ses_i+delta_i,:]>thr
                tmp_tsk_neuron_idx_pre = tmp_tsk_neuron_idx_pre[40:-1]
                tmp_tsk_neuron_idx_post = tmp_tsk_neuron_idx_post[40:-1]
                if sum(tmp_tsk_neuron_idx_pre)>10:
                    tmp_fraction = sum(tmp_tsk_neuron_idx_post[tmp_tsk_neuron_idx_pre])/sum(tmp_tsk_neuron_idx_pre)
                    remain_fraction_store[delta_i-1].extend([tmp_fraction])

drifting_curve_meansem = np.zeros((2,delta_max))
for i in range(delta_max):
    drifting_curve_meansem[0,i] = np.mean(remain_fraction_store[i])
    drifting_curve_meansem[1,i] = np.std(remain_fraction_store[i],ddof=1)/np.sqrt(len(remain_fraction_store[i]))
plt.figure(figsize=(8,6))
plt.subplot(324)
plt.plot(range(1,delta_max+1),drifting_curve_meansem[0,:])
plt.ylabel('Fraction remaining')
np.save(save_path+'/'+folder_name+r'_drifting_curve_meansem_'+para+'.npy',drifting_curve_meansem)


sem = np.std(task_neuron_store[:min_sessionN,:]/neuronN,1)/np.sqrt(miceN)
mn = np.mean(task_neuron_store[:min_sessionN,:]/neuronN,1)


plt.figure(figsize=(8,6))
plt.subplot(324)

plt.plot(range(1,min_sessionN+1),mn)
plt.xlabel('Session number')
plt.ylabel('Task neuron\nfraction (SI>'+str(thr)+')')
plt.savefig(save_path+r'\\'+folder_name +'_'+para+'_task_neuron.svg',format='svg')


path_perf = path+r'\perf_AS.npy'
file_perf = glob.glob(path_perf)
miceN_perf = len(file_perf)
perf_all_mice = []
for i in range(miceN_perf):
    tmp_perf = np.load(file_perf[i])
    perf_all_mice.append(tmp_perf[:,:min_sessionN])
sesN_perf = np.size(perf_all_mice, 2)
perf_all_mice = np.array(perf_all_mice)
perf_all_mice_mean = np.mean(perf_all_mice,axis=0)
perf_all_mice_std = np.std(perf_all_mice,axis=0)/np.sqrt(miceN_perf)


plt.figure(figsize=(8,6))
plt.subplot(324)
plt.plot(range(1,sesN_perf+1), perf_all_mice_mean[0],label='Accuracy (w/o miss)')
plt.plot(range(1,sesN_perf+1), np.mean(perf_all_mice_mean[3:4],axis=0),label='Miss rate')
plt.ylabel('Probability')
plt.xlabel('Session number')
plt.ylim([0,1])
plt.legend(bbox_to_anchor=(0.3,0.5))
plt.savefig(save_path+r'\\'+folder_name +'_'+para+'_performance.svg',format='svg')





# compare sym_asym


