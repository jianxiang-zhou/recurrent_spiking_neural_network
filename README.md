# Recurrent Spiking Neural Network
**A recurrent spiking neural network (RSNN) model trained with 3 factor rules to learn sensorimotor associations**

To explore the synaptic plasticity rules underlying the observed neuronal activity reorganization during learning, we built a biological recurrent spiking neural network (RSNN) using an open-source simulator brian2 (https://github.com/brian-team/brian2).

The RSNN model is composed of 200 leaky integrate-and-fire (LIF) neurons, each having 20 input and 20 output excitatory synapses with other LIF neurons. The dynamics of neuron $i$ is modeled as

$${dv_i (t) \over dt}={v_{rest,i}-v_i (t) \over τ_m}+\sum_j w_{ij}·s_j (t-t_{delay})$$

where $v$ is the membrane potential, $v_{rest}$ is the resting membrane potential, $τ_m$ is the time constant of membrane potential, $w_{ij}$ is the synaptic weight from neuron $j$ to neuron $i$, $t_{delay}$ is the synaptic delay of spikes, and $s_j$ is the spike train of neuron $j$ modelled as

$$s_j (t)= \sum_k δ(t-t_j^{(k)})$$

where the $t^{(k)}$ is the time of $k_{th}$ spike, ${δ(·)}$ is the Dirac delta function. When membrane potential reaches a threshold $v_{thr}$, a spike is generated, and the membrane potential $v$ is reset to $v_{reset}$. To simulate the spontaneous activity of neurons, random currents are injected into the LIF neurons. All LIF neurons are excitatory, and the network activity is balanced by an inhibitory neuron group receiving inputs from and projecting back to all LIF neurons equally, mediating lateral inhibition between LIF neurons.
The RSNN interacts with the environment using defined sensory and motor neurons. Forty LIF neurons are selected as sensory neurons that are directly excited by stimuli (twenty for stimulus A and twenty for stimulus B), and two other LIF neurons are selected as motor neurons whose firings determine the actions of the neural network (i.e., left and right choices). A reward with amplitude $R_{amp}$ is given when RSNN makes left choice after stimulus A or right choice after stimulus B. The reward drives learning through three-factor learning rule (Izhikevich 2007, Fremaux and Gerstner 2016) modeled as

$${de_{ij} \over dt}=-{e_{ij} \over τ_e} +STDP(s_j,s_i)$$

$${dw_{ij} \over dt}=R·e_{ij}$$

where $e_{ij}$ is the eligibility trace of synapse from neuron $j$ to neuron $i$, $τ_e$ is the time constant of the eligibility trace, $s_j$ and $s_i$ are spike trains of neuron $j$ and $i$, respectively, and ${STDP(·)}$ is the function mapping pre- and post-synaptic spike trains to the change of the eligibility trace. The reward is represented as a global signal $R$ which interacts with local eligibility traces $e$ to cause permanent changes of synaptic weights $w$. In a nutshell, if some activity pattern was generated before the reward (e.g., through spontaneous activity), the three-factor learning rule would selectively strengthen the synapses supporting that activity pattern. Therefore, the behavior represented by that activity pattern would appear more frequently in the future, helping the animal to get more rewards.
The RSNNs were trained for 2000 trials. The spike trains of LIF neurons were analyzed after training. In this network, neuronal activity was mainly driven by stimulus. We quantified the stimulus selectivity of neurons using receiver operator characteristic (ROC) analysis, and defined selectivity index (${SI}$) of neurons as the $2*|AUC-0.5|$ (AUC, area under ROC curve). Neurons with ${SI}$ larger than 0.5 were considered as task related.

Table-1 Values of main parameters in the RSNN model 

|Parameters|Values|Parameters|Values|Parameters|Values|
|---|---|---|---|---|---|
|$V_{rest}$|0|$τ_e$|50|$τ_{post}$|20|
|$v_{thr}$|1|$A_{pre}$|0.005|$W_{all}$|2.5|
|$τ_m$|5|$A_{post}$|±0.005|$W_{max}$|1.2|
|$t_{delay}$|5|$τ_{pre}$|20|$R_{amp}$|2|

 
## Running
Step 1: generate simulation data (2AFCnet_para_spc.py)

Step 2: analyse the behavior and the spike train over learning for each case (2AFC_analysis_each_case.py)

Step 3: summary for all cases (result_summary.py)

## Details
1. A half of neurons' input synapses are not modulated by reward (i.e., their plasticity is only dependent on pre- and post-synaptic neuronal spikes), the other half of neurons' input synapses are modulated by reward (i.e., 3 facter plasticity rules). The first half neurons including all sensory neurons help the network to learn stable sensory representations which are important for learning sensorimotor associations. The other half of neurons including all motor neurons help connnecting sensation to action according to reward feedback.
2. Spontaneous activity is essential for learning. It activates inactive neurons, having them recruited in the sensorimotor pathways through activity-dependent plasticity rules.
3. The acitvity of the network is reset before each trial (i.e., the network doesn't store trial history information) as we only want to study the learning of sensorimotor associations.
4. For each trial, stimulus is presented at time 5 A.U.. Actions after the stimulus determine the reward release. Reward is released not earlier than time 20 A.U., simulating the mouse behavior paradigm.
