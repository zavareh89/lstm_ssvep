
# Benchmark Dataset
import numpy as np
import os
import re
from scipy.signal import cheby2,sosfiltfilt
import scipy.io as sio
from keras.models import Sequential
from keras.layers import Flatten, Dense,LSTM

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


### LSTM neural network
def one_hot_encode(x, n_classes):
    """
    One hot encodes a list of sample labels. Return a one-hot encoded vector for each label.
    : x: List of sample Labels
    : return: Numpy array of one-hot encoded labels
     """
    return np.eye(n_classes)[x]

# define labels
labels=[]    
temp_lablels=list(range(n_targets))
for i in range(n_subjects*n_blocks):
    labels.extend(temp_lablels)
labels=np.array(labels)

OH_labels=one_hot_encode(labels,n_targets) # one hot encoding

# splitting
train_index = np.arange(n_targets*n_blocks*30).astype(dtype=int)
test_index = np.arange(n_targets*n_blocks*30,n_targets*n_blocks*n_subjects).astype(dtype=int)

# preprocessing 
def preprocess(data,sos,time_window):
    data = sosfiltfilt(sos,data[:,time_window,:],axis=1,padtype='constant') # bandpass filtering
    # normalization
    mean1=data.mean(axis=1).reshape(data.shape[0],1,-1)
    std1=data.std(axis=1).reshape(data.shape[0],1,-1)
    data=(data-mean1)/std1              
    return data

time_window = np.arange(fs).astype(dtype=int)
train_data = All_data[:,:,train_index]
train_data = preprocess(train_data,sos,time_window)
train_data=np.transpose(train_data,(2,1,0))
train_labels=OH_labels[train_index]
test_data = All_data[:,:,test_index]
test_data = preprocess(test_data,sos,time_window)
test_data=np.transpose(test_data,(2,1,0))
test_labels=OH_labels[test_index]

# define lstm network
model = Sequential()
model.add(LSTM(128,dropout=0.1, recurrent_dropout=0.5, return_sequences=False, 
               input_shape=[len(time_window),len(channel_idx)]))

model.add(Dense(40, activation='softmax'))
model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['acc'])
history = model.fit(train_data, train_labels,epochs=50,batch_size=256)

predicted_labels = model.predict(test_data)
predicted_labels = np.argmax(predicted_labels,axis=1)

Acc=0
for i in range(len(predicted_labels)):
    if predicted_labels[i]==np.argmax(test_labels[i,:]):
        Acc+=1

Acc/=len(predicted_labels)
print(Acc)