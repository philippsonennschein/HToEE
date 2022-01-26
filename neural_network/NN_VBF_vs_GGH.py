import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
import pickle
import ROOT as r
r.gROOT.SetBatch(True)
import sys
from os import path, system
from array import array
from root_numpy import tree2array, fill_hist
from math import pi
import h5py
from itertools import product
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential 
from keras.initializers import RandomNormal 
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Nadam, adam, Adam
from keras.regularizers import l2 
from keras.callbacks import EarlyStopping 
from keras.utils import np_utils 
from keras.metrics import categorical_crossentropy, binary_crossentropy

#Define key quantities, use to tune NN
num_epochs = 4
batch_size = 400
val_split = 0.3
learning_rate = 0.001

epochs = np.linspace(1,num_epochs,num_epochs,endpoint=True).astype(int) #For plotting
binNames = ['ggH','VBF'] 
bins = 50

#Directories
modelDir = 'neural_networks/models/'
plotDir  = 'neural_networks/plots/'

#Define the input features
train_vars = ['diphotonMass', 'diphotonPt', 'diphotonEta',
'diphotonPhi', 'diphotonCosPhi', 'diphotonSigmaMoM',
'leadPhotonIDMVA', 'leadPhotonPtOvM', 'leadPhotonEta',
'leadPhotonEn', 'leadPhotonMass', 'leadPhotonPt', 'leadPhotonPhi',
'subleadPhotonIDMVA', 'subleadPhotonPtOvM', 'subleadPhotonEta',
'subleadPhotonEn', 'subleadPhotonMass', 'subleadPhotonPt',
'subleadPhotonPhi', 'dijetMass', 'dijetPt', 'dijetEta', 'dijetPhi',
'dijetDPhi', 'dijetAbsDEta', 'dijetCentrality', 'dijetMinDRJetPho',
'dijetDiphoAbsDEta', 'leadJetPUJID', 'leadJetPt', 'leadJetEn',
'leadJetEta', 'leadJetPhi', 'leadJetMass', 'leadJetBTagScore',
'leadJetDiphoDEta', 'leadJetDiphoDPhi', 'subleadJetPUJID',
'subleadJetPt', 'subleadJetEn', 'subleadJetEta', 'subleadJetPhi',
'subleadJetMass', 'subleadJetBTagScore', 'subleadJetDiphoDPhi',
'subleadJetDiphoDEta', 'subsubleadJetPUJID', 'subsubleadJetPt',
'subsubleadJetEn', 'subsubleadJetEta', 'subsubleadJetPhi',
'subsubleadJetMass', 'subsubleadJetBTagScore','nSoftJets']

#Add proc and weight to shuffle with data
train_vars.append('proc')
train_vars.append('weight')

#Load the dataframe
dataframes = []
dataframes.append(pd.read_csv('2017/MC/DataFrames/VBF_VBF_BDT_df_2017.csv'))
dataframes.append(pd.read_csv('2017/MC/DataFrames/ggH_VBF_BDT_df_2017.csv'))
df = pd.concat(dataframes, sort=False, axis=0 )

#Data dataframe
#dataframes = []
#dataframes.append(pd.read_csv('2017/Data/DataFrames/Data_VBF_ggH_BDT_df_2017.csv'))
#x_data = pd.concat(dataframes, sort=False, axis=0 )

#dataframe of train_vars
data = df[train_vars]

#Preselection cuts
data = data[data.diphotonMass>100.]
data = data[data.diphotonMass<180.]
data = data[data.leadPhotonPtOvM>0.333]
data = data[data.subleadPhotonPtOvM>0.25]

#Shuffle dataframe
data = data.sample(frac=1)

#Define the procs as the labels
y_train_labels = np.array(data['proc'])
y_train_labels_num = np.where(y_train_labels=='VBF',1,0)
#y_train_labels_hot = np_utils.to_categorical(y_train_labels_num, num_classes=2)
weights = np.array(data['weight'])

#Remove proc after shuffle
data = data.drop(columns=['proc'])
data = data.drop(columns=['weight'])

#Set -999.0 values to -10.0 to decrease effect on scaling 
data = data.replace(-999.0,-10.0) 

#Scaling the variables to a range from 0-1
scaler = MinMaxScaler()
data_scaled = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
#weights_scaled = pd.DataFrame(scaler.fit_transform(weights_df), columns=weights_df.columns)

#Input shape for the first hidden layer
num_inputs  = data_scaled.shape[1]

#Splitting the dataframe into training and test
x_train, x_test, y_train, y_test, train_w, test_w, proc_arr_train, proc_arr_test = train_test_split(data_scaled, y_train_labels_num, weights, y_train_labels, test_size = val_split, shuffle = True)

#Initialize the model
model=Sequential([Dense(units=100,input_shape=(num_inputs,),activation='relu'),
                Dense(units=100,activation='relu'),
                #Dense(units=100,activation='relu'),
                Dense(units=1,activation='sigmoid')]) #For multiclass NN use softmax, Sigmoid is better for binary

#Compile the model
model.compile(optimizer=Adam(lr=learning_rate),loss='binary_crossentropy',metrics=['accuracy'])

model.summary()

# Normalizing training weights 
train_w_df = pd.DataFrame()
train_w_df['weight'] = train_w 
train_w_norm = train_w_df['weight'] / train_w_df['weight'].sum()
train_w_scaled = pd.DataFrame(scaler.fit_transform(train_w_df), columns=train_w_df.columns)
train_w_scaled = np.array(train_w_scaled)
condition = np.ones(len(train_w_scaled))
train_w_scaled = np.compress(condition=condition, a=np.array(train_w_scaled))

#Training the model
history = model.fit(x=x_train,y=y_train,batch_size=batch_size,epochs=num_epochs,shuffle=True,sample_weight=train_w_scaled,verbose=2)

# Output Score
# Output Score
y_pred_test = model.predict_proba(x=x_test)
x_test['proc'] = proc_arr_test #.tolist()
x_test['weight'] = test_w #.to_numpy()
x_test['output_score'] = y_pred_test
#_test['output_score_vbf'] = y_pred_test
#x_test['output_score_ggh'] = 1 - x_test['output_score_vbf']


x_test_vbf = x_test[x_test['proc'] == 'VBF']
x_test_ggh = x_test[x_test['proc'] == 'ggH']

# Weights
vbf_w = x_test_vbf['weight'] / x_test_vbf['weight'].sum()
ggh_w = x_test_ggh['weight'] / x_test_ggh['weight'].sum()

output_vbf = np.array(x_test_vbf['output_score'])
output_ggh = np.array(x_test_ggh['output_score'])

#Accuracy Score
output_score_vbf = np.array(x_test['output_score'],ndmin=2)
output_score_ggh = np.array(x_test['output_score'],ndmin=2)
output_score = np.concatenate((output_score_ggh,output_score_vbf),axis=0)
y_pred = output_score.argmax(axis=0)
y_true = y_test
print 'Accuracy score: '
NNaccuracy = accuracy_score(y_true, y_pred)
print(NNaccuracy)

#Confusion Matrix
cm = confusion_matrix(y_true=y_true,y_pred=y_pred)

# weights Plot
train_w_scaled = pd.DataFrame(scaler.fit_transform(train_w_df), columns=train_w_df.columns)
train_w_scaled['proc'] = proc_arr_train 
train_w_scaled_vbf = train_w_scaled[train_w_scaled['proc']=='VBF']
train_w_scaled_ggh = train_w_scaled[train_w_scaled['proc']=='ggH']

train_w_scaled_vbf = np.array(train_w_scaled_vbf['weight'])
train_w_scaled_ggh = np.array(train_w_scaled_ggh['weight'])
condition_vbf = np.ones(len(train_w_scaled_vbf))
condition_ggh = np.ones(len(train_w_scaled_ggh))
train_w_scaled_vbf = np.compress(condition=condition_vbf, a=train_w_scaled_vbf)
train_w_scaled_ggh = np.compress(condition=condition_ggh, a=train_w_scaled_ggh)

fig, ax = plt.subplots()
ax.hist(train_w_scaled_vbf, bins=50, label='VBF Weight', density = True, histtype='step') #density = density,
ax.hist(train_w_scaled_ggh, bins=50, label='ggH Weight', density = True, histtype='step') #density = density,
plt.legend()
plt.title('Weights')
plt.ylabel('Number of Weights')
plt.xlabel('Weights')
plt.savefig('plotting/NN_plots/Weights_Plot', dpi = 200)

'''
# ----
# ROC CURVE
# testing
#mask_vbf = (y_test[:] == 1)
#mask_ggh = (y_test[:] == 0)
#y_test = np.concatenate((y_test[mask_vbf], y_test[mask_ggh]), axis = None)
#y_pred_test = np.concatenate((output_vbf, output_ggh), axis = None)
#y_pred_test = model.predict_proba(x = x_test_old)
fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred_test)
auc_keras_test = roc_auc_score(y_test, y_pred_test)
#np.savetxt('neural_networks/models/nn_roc_fpr.csv', fpr_keras, delimiter=',')
#np.savetxt('neural_networks/models/nn_roc_tpr.csv', tpr_keras, delimiter=',')
print("Area under ROC curve for testing: ", auc_keras_test)

# training
y_pred_train = model.predict_proba(x = x_train)
fpr_keras_tr, tpr_keras_tr, thresholds_keras = roc_curve(y_train, y_pred_train)
auc_keras_train = roc_auc_score(y_train, y_pred_train)
print("Area under ROC curve for training: ", auc_keras_train)

# TRAIN VS TEST ON OUTPUT SCORE
def train_vs_test_analysis(x_train = x_train, proc_arr_train = proc_arr_train, train_w = train_w):
     x_train['proc'] = proc_arr_train.tolist()
     x_train['weight'] = train_w.to_numpy()
     x_train_vbf = x_train[x_train['proc'] == 'VBF']
     # now weights
     vbf_w_tr = x_train_vbf['weight'] / x_train_vbf['weight'].sum()

     x_train_vbf = x_train_vbf.drop(columns=['proc'])
     x_train_vbf = x_train_vbf.drop(columns=['weight'])
     output_vbf_train = model.predict_proba(x=x_train_vbf)
     return output_vbf_train, vbf_w_tr

'''

#Plotting:
#Plot output score
def plot_output_score(signal=output_vbf,bkg=output_ggh,name='plotting/NN_plots/NN_Output_Score',signal_label='VBF',bkg_label='ggH',bins=50,density=True,histtype='step',sig_weight = vbf_w,bkg_weight=ggh_w):
    print("Plotting Output Score")
    fig, ax = plt.subplots()
    ax.hist(signal, bins=bins, label=signal_label, histtype=histtype, weights=sig_weight) #density = density,
    ax.hist(bkg, bins=bins, label=bkg_label, histtype=histtype, weights=bkg_weight) #density = density,
    plt.legend()
    plt.title('Output Score')
    plt.ylabel('Fraction of Events')
    plt.xlabel('NN Score')
    plt.savefig('plotting/NN_plots/NN_Output_Score', dpi = 200)

#Plot accuracy
def plot_accuracy():
    val_accuracy = history.history['val_acc']
    accuracy = history.history['acc']
    fig, ax = plt.subplots(1)
    plt.plot(epochs,val_accuracy,label='Validation')
    plt.plot(epochs,accuracy,label='Train')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.xticks(epochs_int)
    plt.legend()
    name = 'plotting/NN_plots/NN_Accuracy'
    fig.savefig(name)

#Plot loss
def plot_loss():
    val_loss = history.history['val_loss']
    loss = history.history['loss']
    fig, ax = plt.subplots(1)
    plt.plot(epochs,val_loss,label='Validation')
    plt.plot(epochs,loss,label='Train')
    plt.title('Loss function')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.xticks(epochs)
    plt.legend()
    name = 'plotting/NN_plots/NN_Loss'
    fig.savefig(name)


#Confusion Matrix
def plot_confusion_matrix(cm,classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
    fig, ax = plt.subplots(1)
    plt.imshow(cm,interpolation='nearest',cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks,classes,rotation=45)
    plt.yticks(tick_marks,classes)
    if normalize:
        cm = cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
        for i in range(len(cm[0])):
            for j in range(len(cm[1])):
                cm[i][j] = float("{:.2f}".format(cm[i][j]))
    thresh = cm.max()/2.
    print(cm)
    for i, j in product(range(cm.shape[0]),range(cm.shape[1])):
        plt.text(j,i,cm[i,j],horizontalalignment='center',color='white' if cm[i,j]>thresh else 'black')
        plt.tight_layout()
        plt.ylabel('True Label')
        plt.xlabel('Predicted label')
    name = 'plotting/NN_plots/NN_Confusion_Matrix'
    fig.savefig(name)


#plot_output_score()
#plot_accuracy()
#plot_loss()
#plot_confusion_matrix(cm,binNames,normalize=False)

#save as a pickle file
#trainTotal.to_pickle('%s/nClassNNTotal.pkl'%frameDir)
#print 'frame saved as %s/nClassNNTotal.pkl'%frameDir
#Read in pickle file
#trainTotal = pd.read_pickle(opts.dataFrame)
#print 'Successfully loaded the dataframe'
