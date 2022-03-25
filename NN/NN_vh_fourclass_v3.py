from __future__ import division
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
import math
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
from keras.callbacks import LearningRateScheduler
from keras.utils import np_utils 
from keras.metrics import categorical_crossentropy, binary_crossentropy
import xgboost as xgb

#Define key quantities

#HPs
#Original
num_epochs = 5
batch_size = 32
#val_split = 0.05
test_split = 0.4
learning_rate = 0.0001
num_estimators = 200

#Optimized according to 4class
#num_epochs = 50
#batch_size = 40
#test_split = 0.2
#val_split = 0.1
#learning_rate = 0.0001

#STXS mapping
map_def_3 = [
['qqH',200,201,202,203,204,205,206,207,208,209,210], #qqH
['QQ2HLNU_FWDH',300], #WH
['QQ2HLNU_PTV_0_75',301],
['QQ2HLNU_PTV_75_150',302],
['WH_Rest',303,304,305],
['QQ2HLL_FWDH',400], #ZH
['ZH_Rest',401,402,403,404,405]
]

binNames = ['qqH',
            'QQ2HLNU_FWDH',
            'QQ2HLNU_PTV_0_75',
            'QQ2HLNU_PTV_75_150',
            'WH_Rest',
            'QQ2HLL_FWDH',
            'ZH_Rest']

labelNames = ['WH $p^H_T$<75',
            'WH 75<$p^H_T$<150',
            'WH Rest',
            'ZH Rest']

#color = ['#f0700c', '#e03b04', '#eef522', '#8cad05', '#f5df87', '#6e0903', '#8c4503']
color  = ['silver','indianred','salmon','lightgreen','seagreen','mediumturquoise','darkslategrey','skyblue','steelblue','lightsteelblue','mediumslateblue']


#Define the input features
train_vars = ['diphotonPt', 'diphotonMass', 'diphotonCosPhi', 'diphotonEta','diphotonPhi', 'diphotonSigmaMoM',
     'dijetMass', 'dijetAbsDEta', 'dijetDPhi', 'dijetCentrality',
     'dijetPt','dijetEta','dijetPhi','dijetMinDRJetPho','dijetDiphoAbsDEta',
     'leadPhotonEta', 'leadPhotonIDMVA', 'leadPhotonEn', 'leadPhotonPt', 'leadPhotonPhi', 'leadPhotonPtOvM',
     'leadJetPt', 'leadJetPUJID', 'leadJetBTagScore', 'leadJetMass',
     'leadJetDiphoDEta','leadJetDiphoDPhi','leadJetEn','leadJetEta','leadJetPhi',
     'subleadPhotonEta', 'subleadPhotonIDMVA', 'subleadPhotonPhi',
     'subleadPhotonEn','subleadPhotonPt', 'subleadPhotonPtOvM',
     'subleadJetDiphoDPhi','subleadJetDiphoDEta',
     'subleadJetPt', 'subleadJetPUJID', 'subleadJetBTagScore', 'subleadJetMass',
     'subleadJetEn','subleadJetEta','subleadJetPhi',
     'subsubleadJetEn','subsubleadJetPt','subsubleadJetEta','subsubleadJetPhi', 'subsubleadJetBTagScore', 
     'subsubleadJetMass',
     'metPt','metPhi','metSumET',
     'nSoftJets',
     'leadElectronEn', 'leadElectronMass', 'leadElectronPt', 'leadElectronEta', 'leadElectronPhi', 'leadElectronCharge',
     'leadMuonEn', 'leadMuonMass', 'leadMuonPt', 'leadMuonEta', 'leadMuonPhi', 'leadMuonCharge',
     'subleadElectronEn', 'subleadElectronMass', 'subleadElectronPt', 'subleadElectronEta', 'subleadElectronPhi', 'subleadElectronCharge', 
     'subleadMuonEn', 'subleadMuonMass', 'subleadMuonPt', 'subleadMuonEta', 'subleadMuonPhi', 'subleadMuonCharge'
     ]


#Add proc and weight to shuffle with data
train_vars.append('proc')
train_vars.append('weight')
train_vars.append('HTXS_stage_0')
train_vars.append('HTXS_stage1_2_cat_pTjet30GeV')

#Load the dataframe
dataframes = []
dataframes.append(pd.read_csv('2017/MC/DataFrames/VH_VBF_BDT_df_2017.csv'))
df = pd.concat(dataframes, sort=False, axis=0 )  

#dataframe of train_vars
data = df[train_vars]

# pTHjj and njets variable construction

leadJetPt = np.array(data['leadJetPt'])
subleadJetPt = np.array(data['subleadJetPt'])

# creating n-jets variable
njets = []
num_jet = 0
for i in range(data.shape[0]):
    if leadJetPt[i] != -999.0:
        if subleadJetPt[i] != -999.0:
            num_jet = 2
        else:
            num_jet = 1
    else:
        num_jet = 0
    njets.append(num_jet)
data['njets'] = njets

# creating n-leptons variable
leadElectronPt = np.array(data['leadElectronPt'])
leadMuonPt = np.array(data['leadMuonPt'])
subleadElectronPt = np.array(data['subleadElectronPt'])
subleadMuonPt = np.array(data['subleadMuonPt'])

nleptons = []
num_leptons = 0
for i in range(data.shape[0]):
    if leadElectronPt[i] != -999.0 or leadMuonPt[i] != -999.0:
        if subleadElectronPt[i] != -999.0 or subleadMuonPt[i] != -999:
            num_leptons = 2
        else:
            num_leptons = 1
    else:
        num_leptons = 0
    nleptons.append(num_leptons)
data['nleptons'] = nleptons


#Preselection cuts
data = data[data.diphotonMass>100.]
data = data[data.diphotonMass<180.]
data = data[data.leadPhotonPtOvM>0.333]
data = data[data.subleadPhotonPtOvM>0.25]

def mapping(map_list,stage):
    proc_list = []
    num_list = []
    proc = []
    for i in range(len(map_list)):
        proc_list.append(map_list[i][0])
        temp = []
        for j in range(len(map_list[i])-1):
            temp.append(map_list[i][j+1])
        num_list.append(temp)
    for i in stage:
        for j in range(len(num_list)):
            if i in num_list[j]:
                proc.append(proc_list[j])
    return proc

data['proc_new'] = mapping(map_list=map_def_3,stage=data['HTXS_stage1_2_cat_pTjet30GeV'])

# now I only want to keep the qqH - 7class
data = data[data.proc_new != 'qqH']
data = data[data.proc_new != 'QQ2HLNU_FWDH']
data = data[data.proc_new != 'QQ2HLL_FWDH']


#Define the procs as the labels
num_categories = data['proc_new'].nunique()

proc_new = np.array(data['proc_new'])
#Assign the numbers in the same order as the binNames above
y_train_labels_num = []
for i in proc_new:
    if i == 'QQ2HLNU_PTV_0_75':
        y_train_labels_num.append(0)
    if i == 'QQ2HLNU_PTV_75_150':
        y_train_labels_num.append(1)
    if i == 'WH_Rest':
        y_train_labels_num.append(2)
    if i == 'ZH_Rest':
        y_train_labels_num.append(3)
data['proc_num'] = y_train_labels_num


y_train_labels = np.array(data['proc_new'])
y_train_labels_num = np.array(data['proc_num'])
y_train_labels_hot = np_utils.to_categorical(y_train_labels_num, num_classes=num_categories)
weights = np.array(data['weight'])

#Remove proc after shuffle
data = data.drop(columns=['proc','weight','proc_num','HTXS_stage_0','HTXS_stage1_2_cat_pTjet30GeV','proc_new'])

#Set -999.0 values to -10.0 to decrease effect on scaling 
data = data.replace(-999.0,-10.0) 

#Scaling the variables to a range from 0-1
scaler = MinMaxScaler()
data_scaled = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

#Input shape for the first hidden layer
num_inputs  = data_scaled.shape[1]

#Splitting the dataframe into training and test
x_train, x_test, y_train, y_test, train_w, test_w, proc_arr_train, proc_arr_test = train_test_split(data_scaled, y_train_labels_hot, weights, y_train_labels, test_size = test_split, shuffle = True)

#Initialize the model
model=Sequential([Dense(units=500,input_shape=(num_inputs,),activation='relu'),
                Dense(units=500,activation='relu'),
                Dense(units=500,activation='relu'),
                Dense(units=num_categories,activation='softmax')]) #For multiclass NN use softmax

#Compile the model
model.compile(optimizer=Adam(lr=learning_rate),loss='categorical_crossentropy',metrics=['accuracy'])

model.summary()

# callbacks
def scheduler(epoch, lr):
    print("epoch: ", epoch+1)
    if epoch < 10:
        print("lr: ", lr)
        return lr
    else:
        lr *= math.exp(-0.1)
        print("lr: ", lr)
        return lr
callback_lr = LearningRateScheduler(scheduler)
callback_earlystop = EarlyStopping(monitor='val_loss', min_delta = 0.001, patience=10)

#Equalizing weights
train_w_df = pd.DataFrame()
train_w = 300 * train_w # to make loss function O(1)
train_w_df['weight'] = train_w
train_w_df['proc'] = proc_arr_train
vh1_sum_w = train_w_df[train_w_df['proc'] == 'QQ2HLNU_PTV_0_75']['weight'].sum()
vh2_sum_w = train_w_df[train_w_df['proc'] == 'QQ2HLNU_PTV_75_150']['weight'].sum()
vh3_sum_w = train_w_df[train_w_df['proc'] == 'WH_Rest']['weight'].sum()
vh4_sum_w = train_w_df[train_w_df['proc'] == 'ZH_Rest']['weight'].sum()

train_w_df.loc[train_w_df.proc == 'QQ2HLNU_PTV_75_150','weight'] = (train_w_df[train_w_df['proc'] == 'QQ2HLNU_PTV_75_150']['weight'] * vh1_sum_w / vh2_sum_w)
train_w_df.loc[train_w_df.proc == 'WH_Rest','weight'] = (train_w_df[train_w_df['proc'] == 'WH_Rest']['weight'] * vh1_sum_w / vh3_sum_w)
train_w_df.loc[train_w_df.proc == 'ZH_Rest','weight'] = (train_w_df[train_w_df['proc'] == 'ZH_Rest']['weight'] * vh1_sum_w / vh4_sum_w)
train_w = np.array(train_w_df['weight'])

#Training the model
history = model.fit(x=x_train,y=y_train,batch_size=batch_size,epochs=num_epochs,sample_weight=train_w,shuffle=True,verbose=2, callbacks=[callback_lr,callback_earlystop]) #validation_split = val_split,

# Output Score
#y_pred_0 = model.predict(x_test)
y_pred_test = model.predict_proba(x=x_test)
y_pred_0 = y_pred_test.argmax(axis=1)
x_test['proc'] = proc_arr_test
x_test['weight'] = test_w

x_test['output_score_vh1'] = y_pred_test[:,0]
x_test['output_score_vh2'] = y_pred_test[:,1]
x_test['output_score_vh3'] = y_pred_test[:,2]
x_test['output_score_vh4'] = y_pred_test[:,3]

output_score_vh1 = np.array(y_pred_test[:,0])
output_score_vh2 = np.array(y_pred_test[:,1])
output_score_vh3 = np.array(y_pred_test[:,2])
output_score_vh4 = np.array(y_pred_test[:,3])

x_test_vh1 = x_test[x_test['proc'] == 'QQ2HLNU_PTV_0_75']
x_test_vh2 = x_test[x_test['proc'] == 'QQ2HLNU_PTV_75_150']
x_test_vh3 = x_test[x_test['proc'] == 'WH_Rest']
x_test_vh4 = x_test[x_test['proc'] == 'ZH_Rest']

vh1_w = x_test_vh1['weight'] / x_test_vh1['weight'].sum()
vh2_w = x_test_vh2['weight'] / x_test_vh2['weight'].sum()
vh3_w = x_test_vh3['weight'] / x_test_vh3['weight'].sum()
vh4_w = x_test_vh4['weight'] / x_test_vh4['weight'].sum()
total_w = x_test['weight'] / x_test['weight'].sum()

#Accuracy Score
y_pred = y_pred_test.argmax(axis=1)
y_true = y_test.argmax(axis=1)
print 'Accuracy score: '
NNaccuracy = accuracy_score(y_true, y_pred, sample_weight = test_w)
acc = accuracy_score(y_true, y_pred)
#print(NNaccuracy)
#print(acc)

#Confusion Matrix
cm_old = confusion_matrix(y_true=y_true,y_pred=y_pred)
cm = confusion_matrix(y_true=y_true,y_pred=y_pred,sample_weight=test_w)

#Generatin own confusion & weights matrix to calculate the accuracy scores

cm_new = np.zeros((len(labelNames),len(labelNames)),dtype=int)
cm_weights = np.zeros((len(labelNames),len(labelNames)),dtype=float)
cm_weights_squared = np.zeros((len(labelNames),len(labelNames)),dtype=float)
for i in range(len(y_true)):
    cm_new[y_true[i]][y_pred[i]] += 1
    cm_weights[y_true[i]][y_pred[i]] += test_w[i]
    cm_weights_squared[y_true[i]][y_pred[i]] += test_w[i]**2

num_correct = 0
num_correct_w = 0
num_correct_w_squared = 0
num_all = 0
num_all_w = 0
num_all_w_squared = 0
for i in range(cm_new.shape[0]):
    for j in range(cm_new.shape[1]):
        num_all += cm_new[i][j]
        num_all_w += cm_weights[i][j]
        num_all_w_squared += cm_weights_squared[i][j]
        if i == j: # so diagonal
            num_correct += cm_new[i][j]
            num_correct_w += cm_weights[i][j]
            num_correct_w_squared += cm_weights_squared[i][j]
accuracy = num_correct/num_all
sigma_e = np.sqrt(num_correct_w_squared)
sigma_f = np.sqrt(num_all_w_squared)
e = num_correct_w
f = num_all_w

def error_function(num_correct, num_all, sigma_correct, sigma_all): 
    error = (((1/num_all)**2) * (sigma_correct**2) + ((num_correct / (num_all**2))**2) * (sigma_all**2))**0.5
    return error

accuracy_error = error_function(num_correct=e, num_all=f, sigma_correct=sigma_e, sigma_all=sigma_f)
print(accuracy)
print(accuracy_error)

s_in = []
s_in_w = []
s_in_w_squared = []
s_tot = []
s_tot_w = []
s_tot_w_squared = []
e_s = []
signal_error_list = []
b_in = []
b_in_w = []
b_in_w_squared = []
b_tot = []
b_tot_w = []
b_tot_w_squared = []
e_b = []
bckg_error_list = []

for i in range(len(labelNames)):
    s_in.append(cm_new[i][i])
    s_in_w.append(cm_weights[i][i])
    s_in_w_squared.append(cm_weights_squared[i][i])
    s_tot.append(np.sum(cm_new[i,:]))
    s_tot_w.append(np.sum(cm_weights[i,:]))
    s_tot_w_squared.append(np.sum(cm_weights_squared[i,:]))
    e_s.append(s_in[i]/s_tot[i])

    b_in.append(np.sum(cm_new[:,i]) - s_in[i])
    b_in_w.append(np.sum(cm_weights[:,i]) - s_in_w[i])
    b_in_w_squared.append(np.sum(cm_weights_squared[:,i]) - s_in_w_squared[i])
    b_tot.append(np.sum(cm_new) - s_tot[i])
    b_tot_w.append(np.sum(cm_weights) - s_tot_w[i])
    b_tot_w_squared.append(np.sum(cm_weights_squared) - s_tot_w_squared[i])
    e_b.append(b_in[i]/b_tot[i])

    print(labelNames[i])
    signal_error = error_function(s_in_w[i], s_tot_w[i], np.sqrt(s_in_w_squared[i]), np.sqrt(s_tot_w_squared[i]))
    print('Final Signal Efficiency: ', e_s[i])
    print('with error: ', signal_error)
    signal_error_list.append(signal_error)

    bckg_error = error_function(b_in_w[i], b_tot_w[i], np.sqrt(b_in_w_squared[i]), np.sqrt(b_tot_w_squared[i]))
    print('Final Background Efficiency: ', e_b[i])
    print('with error: ', bckg_error)
    bckg_error_list.append(bckg_error)

accuracy_error = error_function(num_correct=e, num_all=f, sigma_correct=sigma_e, sigma_all=sigma_f)
print(accuracy)
print(accuracy_error)

name_original_cm = 'csv_files/VH_fourclass_NN_cm'
np.savetxt(name_original_cm, cm, delimiter = ',')


def plot_output_score(data='output_score_qqh', density=False,):
    #Can then change it to plotting proc
    print('Plotting',data)
    output_score_vh1 = np.array(x_test_vh1[data])
    output_score_vh2 = np.array(x_test_vh2[data])
    output_score_vh3 = np.array(x_test_vh3[data])
    output_score_vh4 = np.array(x_test_vh4[data])
    output_score_vh5 = np.array(x_test_vh5[data])
    output_score_vh6 = np.array(x_test_vh6[data])
    output_score_vh7 = np.array(x_test_vh7[data])
    output_score_vh8 = np.array(x_test_vh8[data])
    output_score_vh9 = np.array(x_test_vh9[data])
    output_score_vh10 = np.array(x_test_vh10[data])

    fig, ax = plt.subplots()
    #ax.hist(output_score_ggh, bins=50, label='ggH', histtype='step',weights=ggh_w)#,density=True) 
    #ax.hist(output_score_qqh0, bins=50, label='FWDH', histtype='step',weights=qqh0_w)
    ax.hist(output_score_vh1, bins=50, label='QQ2HLNU_PTV_0_75', histtype='step',weights=vh1_w)
    ax.hist(output_score_vh2, bins=50, label='QQ2HLNU_PTV_75_150', histtype='step',weights=vh2_w)
    ax.hist(output_score_vh3, bins=50, label='QQ2HLNU_PTV_150_250_0J', histtype='step',weights=qqh3_w)
    ax.hist(output_score_vh4, bins=50, label='QQ2HLNU_PTV_150_250_GE1J', histtype='step',weights=vh4_w)
    ax.hist(output_score_vh5, bins=50, label='QQ2HLNU_PTV_GT250', histtype='step',weights=vh5_w)
    ax.hist(output_score_vh6, bins=50, label='QQ2HLL_PTV_0_75', histtype='step',weights=vh6_w)
    ax.hist(output_score_vh7, bins=50, label='QQ2HLL_PTV_75_150', histtype='step',weights=vh7_w)
    ax.hist(output_score_vh8, bins=50, label='QQ2HLL_PTV_150_250_0J', histtype='step',weights=vh8_w)
    ax.hist(output_score_vh9, bins=50, label='QQ2HLL_PTV_150_250_GE1J', histtype='step',weights=vh9_w)
    ax.hist(output_score_vh10, bins=50, label='QQ2HLL_PTV_GT250', histtype='step',weights=vh10_w)
    plt.legend()
    plt.title('Output Score')
    plt.ylabel('Fraction of Events')
    plt.xlabel('NN Score')
    name = 'plotting/NN_plots/NN_VH_tnclass_'+data
    plt.savefig(name, dpi = 1200)

#Confusion Matrix
def plot_confusion_matrix(cm,classes,normalize=True,title='Confusion matrix',cmap=plt.cm.Blues):
    fig, ax = plt.subplots(figsize = (10,10))
    #plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.rcParams.update({
    'font.size': 10})
    plt.xticks(tick_marks,classes,rotation=45,horizontalalignment='right')
    plt.yticks(tick_marks,classes)
    if normalize:
        cm = cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
        for i in range(len(cm[0])):
            for j in range(len(cm[1])):
                cm[i][j] = float("{:.2f}".format(cm[i][j]))
    thresh = cm.max()/2.
    print(cm)
    plt.imshow(cm,interpolation='nearest',cmap=cmap)
    plt.title(title)
    for i, j in product(range(cm.shape[0]),range(cm.shape[1])):
        plt.text(j,i,cm[i,j],horizontalalignment='center',color='white' if cm[i,j]>thresh else 'black')
    plt.tight_layout()
    plt.colorbar()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    name = 'plotting/NN_plots/NN_vh_fourclass_Confusion_Matrix'
    fig.savefig(name, dpi = 1200)

def plot_performance_plot(cm=cm,labels=labelNames):
    #cm = cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
    cm = cm.astype('float')/cm.sum(axis=0)[np.newaxis,:]
    for i in range(len(cm[0])):
        for j in range(len(cm[1])):
            cm[i][j] = float("{:.3f}".format(cm[i][j]))
    print(cm)
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize = (10,10))
    #fig, ax = plt.subplots()
    #plt.rcParams.update({
    #'font.size': 14})
    tick_marks = np.arange(len(labels))
    #plt.xticks(tick_marks,labels,rotation=90)
    plt.xticks(tick_marks,labels,rotation=45,horizontalalignment='right')
    #color = ['#24b1c9','#e36b1e','#1eb037','#c21bcf','#dbb104']
    bottom = np.zeros(len(labels))
    for i in range(len(cm)):
        #ax.bar(labels, cm[:,i],label=labels[i],bottom=bottom)
        #bottom += np.array(cm[:,i])
        ax.bar(labels, cm[i,:],label=labels[i],bottom=bottom)#,color=color[i])
        bottom += np.array(cm[i,:])
    plt.legend(loc='upper right')
    current_bottom, current_top = ax.get_ylim()
    ax.set_ylim(bottom=0, top=current_top*1.3)
    #plt.title('Performance Plot')
    #plt.ylabel('Fraction of events')
    ax.set_ylabel('Events', ha='center',size=14) #y=0.5,
    ax.set_xlabel('Predicted Production Modes', ha='center',size=14) #, x=1, size=13)
    name = 'plotting/NN_plots/NN_vh_fourclass_Performance_Plot'
    plt.savefig(name, dpi = 1200)
    plt.show()

def plot_roc_curve(binNames = binNames, y_test = y_test, y_pred_test = y_pred_test, x_test = x_test, color = color):
    # sample weights
    # find weighted average 
    fig, ax = plt.subplots()
    #y_pred_test  = clf.predict_proba(x_test)
    for k in range(len(binNames)):
        signal = binNames[k]
        for i in range(num_categories):
            if binNames[i] == signal:
                #sig_y_test  = np.where(y_test==i, 1, 0)
                sig_y_test = y_test[:,i]
                print('sig_y_test', sig_y_test)
                y_pred_test_array = y_pred_test[:,i]
                print('y_pred_test_array', y_pred_test_array)
                print('Here')
                #test_w = test_w.reshape(1, -1)
                print('test_w', test_w)
                #auc = roc_auc_score(sig_y_test, y_pred_test_array, sample_weight = test_w)
                fpr_keras, tpr_keras, thresholds_keras = roc_curve(sig_y_test, y_pred_test_array, sample_weight = test_w)
                #print('auc: ', auc)
                print('Here')
                fpr_keras.sort()
                tpr_keras.sort()
                auc_test = auc(fpr_keras, tpr_keras)
                ax.plot(fpr_keras, tpr_keras, label = 'AUC = {0}, {1}'.format(round(auc_test, 3), binNames[i]), color = color[i])
    ax.legend(loc = 'lower right', fontsize = 'x-small')
    ax.set_xlabel('Background Efficiency', ha='right', x=1, size=9)
    ax.set_ylabel('Signal Efficiency',ha='right', y=1, size=9)
    ax.grid(True, 'major', linestyle='dotted', color='grey', alpha=0.5)
    name = 'plotting/NN_plots/NN_stage_0_ROC_curve'
    plt.savefig(name, dpi = 1200)
    print("Plotting ROC Curve")
    plt.close()

#plot_roc_curve()

plot_performance_plot()
plot_confusion_matrix(cm,labelNames,normalize=True)

#plot_output_score(data='output_score_qqh1')
#plot_output_score(data='output_score_qqh2')
#plot_output_score(data='output_score_qqh3')
#plot_output_score(data='output_score_qqh4')
#plot_output_score(data='output_score_qqh5')
#plot_output_score(data='output_score_qqh6')
#plot_output_score(data='output_score_qqh7')

s_in_2 = []
s_in_w_2 = []
s_in_w_squared_2 = []
s_tot_2 = []
s_tot_w_2 = []
s_tot_w_squared_2 = []
e_s_2 = []
signal_error_list_2 = []
b_in_2 = []
b_in_w_2 = []
b_in_w_squared_2 = []
b_tot_2 = []
b_tot_w_2 = []
b_tot_w_squared_2 = []
e_b_2 = []
bckg_error_list_2 = []

error_final_array = []


signal = ['QQ2HLNU_PTV_0_75',
        'QQ2HLNU_PTV_75_150',
        'WH_Rest',
        'ZH_Rest']

y_label = ['WH $p^H_T$<75',
            'WH 75<$p^H_T$<150',
            'WH Rest',
            'ZH Rest',
            'No category']

conf_matrix_w = np.zeros((2,len(signal)))
conf_matrix_no_w = np.zeros((2,len(signal)))

conf_matrix_w2 = np.zeros((1,len(signal)))
conf_matrix_no_w2 = np.zeros((1,len(signal)))

fig, ax = plt.subplots()
plt.rcParams.update({'font.size': 9})

for i in range(len(signal)):

    clf_2 = xgb.XGBClassifier(objective='binary:logistic', n_estimators=num_estimators, 
                            eta=0.0001, maxDepth=6, min_child_weight=0.01, 
                            subsample=0.6, colsample_bytree=0.6, gamma=4)
    
    data_new = x_test.copy()  
    data_new = data_new.drop(columns = ['output_score_vh1','output_score_vh2', 'output_score_vh3', 'output_score_vh4'])
    # now i want to get the predicted labels
    proc_pred = []      
    for j in range(len(y_pred_0)):
        if(y_pred_0[j] == i): # so that the predicted label is the signal
            proc_pred.append(signal[i])
        else:
            proc_pred.append('background')
    data_new['proc_pred'] = proc_pred       # Problem might be here, they don't seem to line up

    # now cut down the dataframe to the predicted ones -  this is the split for the different dataframes
    data_new = data_new[data_new.proc_pred == signal[i]] 

    # now from proc make signal against background (binary classifier)

    proc_true = np.array(data_new['proc'])
    y_train_labels_num = []
    y_train_labels = []
    for j in range(len(proc_true)):
        if proc_true[j] == signal[i]:
            y_train_labels.append(signal[i])
            y_train_labels_num.append(1)
        else: 
            y_train_labels.append('background')
            y_train_labels_num.append(0)
    y_train_labels = np.array(y_train_labels)
    y_train_labels_num = np.array(y_train_labels_num)
    
    weights_new = np.array(data_new['weight'])

    
    data_new = data_new.drop(columns=['weight'])
    data_new = data_new.drop(columns=['proc'])
    data_new = data_new.drop(columns=['proc_pred'])

    # the new split
    x_train_2, x_test_2, y_train_2, y_test_2, train_w_2, test_w_2, proc_arr_train_2, proc_arr_test_2 = train_test_split(data_new, y_train_labels_num, weights_new, y_train_labels, test_size = test_split, shuffle = True)

    train_w_df = pd.DataFrame()
    train_w = 300 * train_w_2 # to make loss function O(1)
    train_w_df['weight'] = train_w
    train_w_df['proc'] = proc_arr_train_2
    signal_sum_w = train_w_df[train_w_df['proc'] == signal[i]]['weight'].sum()
    background_sum_w = train_w_df[train_w_df['proc'] == 'background']['weight'].sum()

    train_w_df.loc[train_w_df.proc == 'background','weight'] = (train_w_df[train_w_df['proc'] == 'background']['weight'] * signal_sum_w / background_sum_w)
    train_w_new = np.array(train_w_df['weight'])

    print (' Training classifier with Signal = ', signal[i])
    clf_2 = clf_2.fit(x_train_2, y_train_2, sample_weight=train_w_new)
    print (' Finished classifier with Signal = ', signal[i])

    y_pred_test_2 = clf_2.predict_proba(x_test_2) 
    y_pred_2 = y_pred_test_2.argmax(axis=1)

    cm_2 = confusion_matrix(y_true = y_test_2, y_pred = y_pred_2, sample_weight = test_w_2)  #weights result in decimal values <1 so not sure if right
    cm_2_no_weights = confusion_matrix(y_true = y_test_2, y_pred = y_pred_2)

    #print('cm_2:')
    #print(cm_2)
    # ERROR STUFF -----------

    # dear philipp, here is your christmas present:

    threshold = 0.05 # bckg efficiency threshold (manually set)
    # get output score
    x_test_2['proc'] = proc_arr_test_2
    x_test_2['weight'] = test_w_2
    x_test_2['output_score_background'] = y_pred_test_2[:,0]
    x_test_2[signal[i]] = y_pred_test_2[:,1]

    x_test_qqh1 = x_test_2[x_test_2['proc'] == signal[i]]
    x_test_qqh2 = x_test_2[x_test_2['proc'] == 'background']

    qqh1_w = x_test_qqh1['weight'] / x_test_qqh1['weight'].sum()
    qqh2_w = x_test_qqh2['weight'] / x_test_qqh2['weight'].sum()

    output_score_qqh2 = np.array(x_test_qqh2[signal[i]])
    counts, bins, _ = plt.hist(output_score_qqh2, bins=100, label='Background', histtype='step',weights=qqh2_w,density=True)
    plt.savefig('plotting/TESTING', dpi = 1200)
    for j in range(len(bins)):
        bins_2 = bins[:j+1]
        counts_2 = counts[:j]
        area = sum(np.diff(bins_2)*counts_2)
        if area <= (1-threshold):
            bdt_score = bins_2[j]
    print('bdt_score: ', bdt_score)
    
    thresh = bdt_score
    #thresh = 0.3

    y_pred_errors = []
    for k in range(len(y_test_2)):
        if y_pred_test_2[:,1][k]>thresh:
            y_pred_errors.append(1)
        else:
            y_pred_errors.append(0)
    y_pred_errors = np.array(y_pred_errors)

    cm_errors = np.zeros((2,2),dtype=int)
    cm_errors_weights = np.zeros((2,2),dtype=float)
    cm_errors_weights_squared = np.zeros((2,2),dtype=float)
    for l in range(len(y_test_2)):
        cm_errors[y_test_2[l]][y_pred_errors[l]] += 1
        cm_errors_weights[y_test_2[l]][y_pred_errors[l]] += test_w_2[l]
        cm_errors_weights_squared[y_test_2[l]][y_pred_errors[l]] += test_w_2[l]**2
    
    print(cm_errors)

    s_in_2.append(cm_errors[1][1])
    s_in_w_2.append(cm_errors_weights[1][1])
    s_in_w_squared_2.append(cm_errors_weights_squared[1][1])
    s_tot_2.append(np.sum(cm_errors[1,:]))
    s_tot_w_2.append(np.sum(cm_errors_weights[1,:]))
    s_tot_w_squared_2.append(np.sum(cm_errors_weights_squared[1,:]))
    e_s_2.append(s_in_2[i]/s_tot_2[i])

    b_in_2.append(cm_errors[0][1])
    b_in_w_2.append(cm_errors_weights[0][1])
    b_in_w_squared_2.append(cm_errors_weights_squared[0][1])
    b_tot_2.append(np.sum(cm_errors[0,:]))
    b_tot_w_2.append(np.sum(cm_errors_weights[0,:]))
    b_tot_w_squared_2.append(np.sum(cm_errors_weights_squared[0,:]))
    e_b_2.append(b_in_2[i]/b_tot_2[i])

    print(signal[i])
    signal_error = error_function(s_in_w_2[i], s_tot_w_2[i], np.sqrt(s_in_w_squared_2[i]), np.sqrt(s_tot_w_squared_2[i]))
    print('Final Signal Efficiency: ', e_s_2[i])
    print('with error: ', signal_error)
    signal_error_list_2.append(signal_error)

    bckg_error = error_function(b_in_w_2[i], b_tot_w_2[i], np.sqrt(b_in_w_squared_2[i]), np.sqrt(b_tot_w_squared_2[i]))
    print('Final Background Efficiency: ', e_b_2[i])
    print('with error: ', bckg_error)
    bckg_error_list_2.append(bckg_error)

    error_final_array.append(np.sqrt(signal_error_list_2[i]**2 + bckg_error_list_2[i]**2 + signal_error_list[i]**2 + bckg_error_list[i]**2))
    print('Error final: ', error_final_array[i])

    # grabbing predicted label column
    #norm = cm_2[0][1] + cm_2[1][1]
    #conf_matrix[0][i] = (cm_2[0][1])/norm
    #conf_matrix[1][i] = (cm_2[1][1])/norm

    conf_matrix_w[0][i] = cm_2[0][1]
    conf_matrix_w[1][i] = cm_2[1][1]
    conf_matrix_no_w[0][i] = cm_2_no_weights[0][1]
    conf_matrix_no_w[1][i] = cm_2_no_weights[1][1]

    conf_matrix_w2[0][i] = (cm_2[0][0] + cm_2[1][0]) / np.sum(np.array(cm_2))
    conf_matrix_no_w2[0][i] = (cm_2_no_weights[0][0] + cm_2_no_weights[1][0])/ np.sum(np.array(cm_2_no_weights))

    # ROC Curve
    sig_y_test  = np.where(y_test_2==1, 1, 0)
    #sig_y_test  = y_test_2
    y_pred_test_array = y_pred_test_2[:,1] # to grab the signal
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(sig_y_test, y_pred_test_array, sample_weight = test_w_2)
    fpr_keras.sort()
    tpr_keras.sort()
    name_fpr = 'csv_files/VH_fourclass_NN_binary_fpr_' + signal[i]
    name_tpr = 'csv_files/VH_fourclass_NN_binary_tpr_' + signal[i]
    np.savetxt(name_fpr, fpr_keras, delimiter = ',')
    np.savetxt(name_tpr, tpr_keras, delimiter = ',')
    auc_test = auc(fpr_keras, tpr_keras)
    ax.plot(fpr_keras, tpr_keras, label = 'AUC = {0}, {1}'.format(round(auc_test, 3), labelNames[i]), color = color[i])

ax.legend(loc = 'lower right', fontsize = 'small')
ax.set_xlabel('Background Efficiency', ha='right', x=1, size=9)
ax.set_ylabel('Signal Efficiency',ha='right', y=1, size=9)
ax.grid(True, 'major', linestyle='dotted', color='grey', alpha=0.5)
plt.tight_layout()
name = 'plotting/NN_plots/NN_VH_Fourclass_binary_Multi_ROC_curve'
plt.savefig(name, dpi = 1200)
print("Plotting ROC Curve")
plt.close()

print('Final conf_matrix:')
print(conf_matrix_w)

#final final confusion matrix for plotting
confusion_matrix = np.concatenate((cm,conf_matrix_w2),axis=0)

#Exporting final confusion matrix
name_cm = 'csv_files/VH_fourclass_NN_binary_cm'
np.savetxt(name_cm, conf_matrix_w, delimiter = ',')

#Need a new function beause the cm structure is different
def plot_performance_plot_final(cm=conf_matrix_w,labels=labelNames, color = color, name = 'plotting/NN_plots/NN_VH_Fourclass_Performance_Plot'):
    cm = cm.astype('float')/cm.sum(axis=0)[np.newaxis,:]
    #for i in range(len(cm[0])):
    #    for j in range(len(cm[:,1])):
    #        cm[j][i] = float("{:.3f}".format(cm[j][i]))
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize = (10,10))
    plt.rcParams.update({
    'font.size': 9})
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks,labels,rotation=45, horizontalalignment = 'right')
    bottom = np.zeros(len(labels))
    ax.bar(labels, cm[1,:],label='Signal',bottom=bottom,color=color[1])
    bottom += np.array(cm[1,:])
    ax.bar(labels, cm[0,:],label='Background',bottom=bottom,color=color[0])
    plt.legend()
    current_bottom, current_top = ax.get_ylim()
    ax.set_ylim(bottom=0, top=current_top*1.3)
    plt.ylabel('Fraction of events', size = 12)
    ax.set_xlabel('Events',size=12)
    plt.tight_layout()
    plt.savefig(name, dpi = 1200)
    plt.show()

def plot_final_confusion_matrix(cm,classes,labels = labelNames,y_labels = y_label, normalize=True,title='Confusion matrix',cmap=plt.cm.Blues, name = 'plotting/NN_plots/NN_VH_Fourclass_final_Confusion_Matrix'):
    fig, ax = plt.subplots(figsize = (10,10))
    #plt.colorbar()
    tick_marks_x = np.arange(len(classes))
    tick_marks_y = np.arange(len(y_label))
    plt.xticks(tick_marks_x,labels,rotation=45, horizontalalignment = 'right')
    plt.yticks(tick_marks_y,y_labels)
    if normalize:
        #cm = cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
        division = cm.sum(axis=1)[0:4,np.newaxis]
        division = np.append(division,1)[:,np.newaxis]
        cm = cm.astype('float')/division
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                cm[i][j] = float("{:.2f}".format(cm[i][j]))
    thresh = cm.max()/2.
    print(cm)
    plt.imshow(cm,interpolation='nearest',cmap=cmap)
    #plt.title(title)
    for i, j in product(range(cm.shape[0]),range(cm.shape[1])):
        plt.text(j,i,cm[i,j],horizontalalignment='center',color='white' if cm[i,j]>thresh else 'black')
    plt.tight_layout()
    plt.colorbar()
    plt.ylabel('True Label', size = 12)
    plt.xlabel('Predicted label', size = 12)
    plt.tight_layout()
    fig.savefig(name, dpi = 1200)
# now to make our final plot of performance
plot_performance_plot_final(cm = conf_matrix_w,labels = labelNames, name = 'plotting/NN_plots/NN_VH_Fourclass_Performance_Plot_final')

plot_final_confusion_matrix(cm=confusion_matrix,classes=binNames,labels = labelNames,y_labels = y_label,normalize=True)

#num_false = np.sum(conf_matrix_w[0,:])
#num_correct = np.sum(conf_matrix_w[1,:])
#accuracy = num_correct / (num_correct + num_false)
#print('Final Accuracy Score:')
#print(accuracy)

print(error_final_array)