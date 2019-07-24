
# Benchmark Dataset
import numpy as np
import os
from scipy.signal import cheby2,sosfiltfilt
import scipy.io as sio
from CCA import cca
import re

n_channels=64 # all channels
n_points=1500
n_targets=40
n_blocks=6
n_subjects=35
fs=250 # sampling frequency
subject_latency=0.14*np.ones(n_subjects) # latencies for each subject
time_Steps=np.linspace(0.25,5,num=20) # time windows length

channels = {'FP1':1, 'FPZ':2, 'FP2':3, 'AF3':4, 'AF4':5,
'F7':6, 'F5':7, 'F3':8, 'F1':9, 'FZ':10,
'F2':11, 'F4':12, 'F6':13, 'F8':14, 'FT7':15,
'FC5':16, 'FC3':17, 'FC1':18, 'FCz':19,'FC2':20,
'FC4':21, 'FC6':22, 'FT8':23, 'T7':24, 'C5':25,
'C3':26, 'C1':27, 'Cz':28, 'C2':29, 'C4':30,
'C6':31, 'T8':32, 'M1':33, 'TP7':34, 'CP5':35,
'CP3':36, 'CP1':37, 'CPZ':38,'CP2':39,'CP4':40,
'CP6':41, 'TP8':42, 'M2':43, 'P7':44, 'P5':45,
'P3':46, 'P1':47, 'PZ':48, 'P2':49, 'P4':50,
'P6':51, 'P8':52, 'PO7':53, 'PO5':54, 'PO3':55,
'POz':56,'PO4':57, 'PO6':58,'PO8':59,'CB1':60,
'O1':61, 'Oz':62, 'O2':63, 'CB2':64}

# Only channel_idx is selected
channel_idx= [channels[items] for items in ['PZ', 'PO3', 'PO4', 'PO5', 'PO6', 'POz', 'O1', 'O2', 'Oz']]
channel_idx=np.array(channel_idx)-1
# bandpass filter 
fmax=fs/2
sos = cheby2(6,20,[6/fmax,60/fmax],btype='bandpass',output='sos')

# data path
path=r'C:\Users\Hadi2\Desktop\SSVEP dataset\ssvep_data'
dir_list=os.listdir(path)
dir_list.sort(key=lambda a:int(re.findall('\d+',a)[0])) # sort list by numbers

for ii,f_name in enumerate(dir_list):
    
    path1 = os.path.join(path,f_name)
    temp_file = sio.loadmat(path1)
    data = temp_file['data']    
    
    # select [0.5+0.14 s : 5.5+0.14 s] time window
    time_index = np.arange((fs/2)+round(subject_latency[ii]*fs),(11*fs/2)+round(subject_latency[ii]*fs))
    data = data[channel_idx,:,:,:][:,time_index.astype(dtype=int),:,:]
    data = data.reshape(data.shape[0],data.shape[1],-1,order='F')

    # concatenation
    if ii==0:
        All_data=data
    else:
        All_data=np.concatenate((All_data,data),axis=2)

# define labels
labels=[]    
temp_lablels=list(range(n_targets))
for i in range(n_subjects*n_blocks):
    labels.extend(temp_lablels)
labels=np.array(labels)

# preprocessing 
def preprocess(data,sos,time_window):
    data = sosfiltfilt(sos,data[:,time_window,:],axis=1,padtype='constant') # bandpass filtering
    # normalization
#    mean1=data.mean(axis=1).reshape(data.shape[0],1,-1)
#    std1=data.std(axis=1).reshape(data.shape[0],1,-1)
#    data=(data-mean1)/std1              
    return data

time_window = np.arange(fs).astype(dtype=int)
CCA_input = preprocess(All_data,sos,time_window)

## CCA benchmark
Freq_phase = sio.loadmat(r'C:\Users\Hadi2\Desktop\SSVEP dataset\NN_training\Freq_Phase')
freqs=Freq_phase['freqs']
Nh=3 # number of harmonics
sin_ref=[]
t = np.arange(1,fs+1) # time samples

for i in range(n_targets):
    sin_signal=np.zeros(shape=(fs,2*Nh))
    for h in range(Nh):
        sin_signal[:,2*h]=np.sin(2*np.pi*t*(h+1)*freqs[0,i]/fs)
        sin_signal[:,2*h+1]=np.cos(2*np.pi*t*(h+1)*freqs[0,i]/fs)
    sin_ref.append(sin_signal)
    
# compute canonica correlation coefficients
ro = np.zeros(shape=(n_subjects,n_targets*n_blocks,n_targets))
correctness_flag = np.zeros(shape=(n_subjects,n_targets*n_blocks)).astype(dtype=bool)
Acc=np.zeros(shape=(n_subjects,))

for ii in range(n_subjects):
    k = ii*n_targets*n_blocks
    for trial in range(n_targets*n_blocks):
        input_data = CCA_input[:,:,k+trial].transpose()
        # apply CCA for each target
        for targ in range(n_targets):
            Wy,Wx,r = cca(sin_ref[targ],input_data)
            ro[ii,trial,targ] = r[0]
            
    # find the predicted labels for subject ii
    predicted_labels = np.argmax(ro[ii,:,:],axis=1)
    for trial in range(n_targets*n_blocks):
        if predicted_labels[trial] == labels[k+trial]:
            correctness_flag[ii,trial] = True
    # calculate accuracy
    Acc[ii] = np.sum(correctness_flag[ii,:])/(n_targets*n_blocks)
    print('Subject ' + str(ii+1) + ':' + str(Acc[ii]))
        
np.mean(Acc)