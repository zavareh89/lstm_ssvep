

import numpy as np
import os
import gc
import re
from scipy.signal import cheby2,sosfiltfilt
import scipy.io as sio
from keras import backend as K
from keras import Input,layers
from keras.layers import Flatten, Dense,LSTM,GRU,Dropout,Bidirectional,CuDNNGRU,SimpleRNN
from keras.optimizers import Adam
from keras.models import load_model
from keras.models import Sequential, Model
import keras
#---------------------------------------------------- FUNCS

n_channels=64 # all channels
n_points=1500
n_targets=40
n_blocks=6
n_subjects=35
fs=250 # sampling frequency
TrainSubjects=20
# NN
DropOut = 0.1
R_DropOut = 0.2
N_Neurons_GRU = [128]
N_Neurons_Classify = 1
LearningRate = 0.001
workspace = r'F:\papers\lstm_ssvep'
Batch_size = [128] 
Epochs = 25
#windows
latency = 0.14
duration = 11.0/2

def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)
        
def one_hot_encode(x, n_classes):
    """
    One hot encodes a list of sample labels. Return a one-hot encoded vector for each label.
    : x: List of sample Labels
    : return: Numpy array of one-hot encoded labels
     """
    return np.eye(n_classes)[x]
 
# preprocessing 
def preprocess(data,sos,time_window):
    data = sosfiltfilt(sos,data[:,time_window,:],axis=1,padtype='constant') # bandpass filtering
    return data

# normalization
def normalization(data):
    mean1=data.mean(axis=1).reshape(data.shape[0],1,-1)
    std1=data.std(axis=1).reshape(data.shape[0],1,-1)
    data=(data-mean1)/std1              
    return data
#---------------------------------------------------- PREPROCESS

# load CCA_features
temp_data = np.load('CCA_benchmark_materials.npz')
Acc_benchmark = temp_data['Acc']
CCA_features = temp_data['CCA_features']

# define labels
labels=[]    
temp_lablels=list(range(n_targets))
for i in range(n_subjects*n_blocks):
    labels.extend(temp_lablels)
labels=np.array(labels)

OH_labels=one_hot_encode(labels,n_targets) # one hot encoding

#---------------------------------------------------- splitting

train_index = np.arange(n_targets*n_blocks*TrainSubjects).astype(dtype=int)
test_index = np.arange(n_targets*n_blocks*TrainSubjects,n_targets*n_blocks*n_subjects).astype(dtype=int)
train_labels=OH_labels[train_index].reshape(-1) # zero one labels (two class)
test_labels=OH_labels[test_index].reshape(-1) # zero one labels (two class)
train_data=CCA_features[:,:,:,train_index]
test_data=CCA_features[:,:,:,test_index]
del CCA_features
gc.collect() # free memory

# transpose dimensions
train_data=np.transpose(train_data,(1,3,2,0))
test_data=np.transpose(test_data,(1,3,2,0))

# reshape data (for two class problem)
train_data = train_data.reshape((-1,train_data.shape[2],train_data.shape[3]),order='F')
test_data = test_data.reshape((-1,test_data.shape[2],test_data.shape[3]),order='F')

#---------------------------------------------------- Neural nets

for N_Neuron in N_Neurons_GRU:
    for bs in Batch_size:

        print('===============start of neurons =',N_Neuron,', batch = ',bs,"===================")

        ##structure
        x_shape_eeg,y_shape_eeg = 250,1
        EEG = Input (shape = (x_shape_eeg,y_shape_eeg), name = 'EEG')
        X1 = CuDNNGRU(N_Neuron, return_sequences=True, activation=K.tanh)(EEG)
        X1 = CuDNNGRU(2*N_Neuron,dropout=DropOut, recurrent_dropout=R_DropOut, return_sequences=False)(X1)
        Y1 = Dense(N_Neurons_Classify, activation='sigmoid')(X1)
        ###compile
        model=Model(EEG,Y1)
        model.summary()
        model.compile(loss='binary_crossentropy',metrics=['acc'],
                      optimizer='adam')
        ###train

        model_dir = os.path.join(workspace, "models", "%ddb Neuron_CCA_features_idea_2" % int(N_Neuron))
        create_folder(model_dir)

        History = model.fit(train_data[:,:,0:y_shape_eeg],train_labels,batch_size=bs,
                            epochs=Epochs,validation_data=(test_data[:,:,0:y_shape_eeg],test_labels))


final_prediction = np.zeros((test_data.shape[0],n_targets))
final_argmax = np.zeros((test_data.shape[0],n_targets))

for targ in range(n_targets):
    predicted_prob = model.predict(test_data[:,targ,:,0:y_shape_eeg])
    max_prob = np.max(predicted_prob,axis=1)
    argmax_prob = np.argmax(predicted_prob,axis=1)
    final_prediction[:,targ] = max_prob
    final_argmax[:,targ] = argmax_prob

# get labels
label_arg = np.argmax(final_prediction,axis=1)
pred_labels = np.zeros((test_data.shape[0],))
Acc = 0
for i in range(test_data.shape[0]):
    pred_labels[i] = final_argmax[i,label_arg[i]]
    if pred_labels[i]==np.argmax(test_labels[i]):
        Acc += 1

Acc /= test_data.shape[0]
print(Acc)
