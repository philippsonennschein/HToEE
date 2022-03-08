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
from keras.utils import np_utils 
from keras.metrics import categorical_crossentropy, binary_crossentropy

map_def_2 = [
['QQ2HQQ_FWDH',200],
['qqH_Rest', 201, 202, 203, 205],
['QQ2HQQ_GE2J_MJJ_60_120',204],
['QQ2HQQ_GE2J_MJJ_GT350_PTH_GT200',206],
['QQ2HQQ_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_0_25',207],
['QQ2HQQ_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_GT25',208],
['QQ2HQQ_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_0_25',209],
['QQ2HQQ_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_GT25',210],
['WH',300,301,302,303,304,305],
['ZH',400,401,402,403,404,405],
]

map_def_3 = [
['qqH',200,201,202,203,204,205,206,207,208,209,210], #qqH
['QQ2HLNU_FWDH',300], #WH
['QQ2HLNU_PTV_0_75',301],
['QQ2HLNU_PTV_75_150',302],
['QQ2HLNU_PTV_150_250_0J',303],
['QQ2HLNU_PTV_150_250_GE1J',304],
['QQ2HLNU_PTV_GT250',305],
['QQ2HLL_FWDH',400], #ZH
['QQ2HLL_PTV_0_75',401],
['QQ2HLL_PTV_75_150',402],
['QQ2HLL_PTV_150_250_0J',403],
['QQ2HLL_PTV_150_250_GE1J',404],
['QQ2HLL_PTV_GT250',405]
]

binNames = ['QQ2HLNU_PTV_0_75',
            'QQ2HLNU_PTV_75_150',
            'QQ2HLNU_PTV_150_250_0J',
            'QQ2HLNU_PTV_150_250_GE1J',
            'QQ2HLNU_PTV_GT250',
            'QQ2HLL_PTV_0_75',
            'QQ2HLL_PTV_75_150',
            'QQ2HLL_PTV_150_250_0J',
            'QQ2HLL_PTV_150_250_GE1J',
            'QQ2HLL_PTV_GT250']

binNames = ['WH $p^H_T$<75',
            'WH 75<$p^H_T$<150',
            'WH 150<$p^H_T$<250 0 Jets',
            'WH 150<$p^H_T$<250 1 Jet',
            'WH $p^H_T$>250',
            'ZH $p^H_T$<75',
            'ZH 75<$p^H_T$<150',
            'ZH 150<$p^H_T$<250 0 Jets',
            'ZH 150<$p^H_T$<250 1 Jet',
            'ZH $p^H_T$>250']

color  = ['silver','indianred','yellowgreen','lightgreen','green','mediumturquoise','darkslategrey','skyblue','steelblue','lightsteelblue','mediumslateblue']

bins = 50

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

train_vars.append('proc')
train_vars.append('weight')
train_vars.append('HTXS_stage_0')
train_vars.append('HTXS_stage1_2_cat_pTjet30GeV')

#Load the dataframe
dataframes = []
#dataframes.append(pd.read_csv('2017/MC/DataFrames/ggH_VBF_BDT_df_2017.csv'))
#dataframes.append(pd.read_csv('2017/MC/DataFrames/VBF_VBF_BDT_df_2017.csv'))
dataframes.append(pd.read_csv('2017/MC/DataFrames/VH_VBF_BDT_df_2017.csv'))
#dataframes.append(pd.read_csv('2017/MC/DataFrames/ttH_VBF_BDT_df_2017.csv'))
#dataframes.append(pd.read_csv('2017/MC/DataFrames/tHq_VBF_BDT_df_2017.csv'))
#dataframes.append(pd.read_csv('2017/MC/DataFrames/tHW_VBF_BDT_df_2017.csv'))
data = pd.concat(dataframes, sort=False, axis=0 )


# Pre-selection cuts
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

data['proc_original'] = mapping(map_list=map_def_3,stage=data['HTXS_stage1_2_cat_pTjet30GeV'])


# now I only want to keep the qqH - 7class
#data = data.drop(data[(data.proc_original == 'QQ2HQQ_FWDH') & (data.proc_original == 'WH') & (data.proc_original == 'ZH')].index)
data = data[data.proc_original != 'qqH']
data = data[data.proc_original != 'QQ2HLNU_FWDH']
data = data[data.proc_original != 'QQ2HLL_FWDH']

#Define the procs as the labels
#ggh: 0, VBF:1, VH: 2, ttH: 3
#num_categories = data['proc'].nunique()
#y_train_labels_num, y_train_labels_def = pd.factorize(data['proc'])

num_categories = data['proc_original'].nunique()

proc_original = np.array(data['proc_original'])
#Assign the numbers in the same order as the binNames above
y_train_labels_num = []
for i in proc_original:
    if i == 'QQ2HLNU_PTV_0_75':
        y_train_labels_num.append(0)
    if i == 'QQ2HLNU_PTV_75_150':
        y_train_labels_num.append(1)
    if i == 'QQ2HLNU_PTV_150_250_0J':
        y_train_labels_num.append(2)
    if i == 'QQ2HLNU_PTV_150_250_GE1J':
        y_train_labels_num.append(3)
    if i == 'QQ2HLNU_PTV_GT250':
        y_train_labels_num.append(4)
    if i == 'QQ2HLL_PTV_0_75':
        y_train_labels_num.append(5)
    if i == 'QQ2HLL_PTV_75_150':
        y_train_labels_num.append(6)
    if i == 'QQ2HLL_PTV_150_250_0J':
        y_train_labels_num.append(7)
    if i == 'QQ2HLL_PTV_150_250_GE1J':
        y_train_labels_num.append(8)
    if i == 'QQ2HLL_PTV_GT250':
        y_train_labels_num.append(9)

data['proc_num'] = y_train_labels_num

y_train_labels = np.array(data['proc_original'])
#y_train_labels = np.array(data['proc'])
y_train_labels_num = np.array(data['proc_num'])
y_train_labels_hot = np_utils.to_categorical(y_train_labels_num, num_classes=num_categories)
weights = np.array(data['weight'])

stage_1_2 = np.array(data['HTXS_stage1_2_cat_pTjet30GeV'])
leadJetPt = np.array(data['leadJetPt'])

data = data.drop(columns=['proc'])
data = data.drop(columns=['proc_num'])
data = data.drop(columns=['HTXS_stage_0'])
data = data.drop(columns=['HTXS_stage1_2_cat_pTjet30GeV'])

# well also need to only keep the qqH 7-class btw


# creating n-jets variable
njets = []
num_jet = 0
for i in range(data.shape[0]):
    if leadJetPt[i] != -999.0:
        num_jet = 1
    else:
        num_jet = 0
    njets.append(num_jet)
data['njets'] = njets

# manually setting cuts
njets = np.array(data['njets'])
diphotonpt = np.array(data['diphotonPt'])


proc = []
y_train_labels_num_pred = []
for i in range(data.shape[0]):
    if stage_1_2[i] == 300 or 301 or 302 or 303 or 304 or 305:
        if diphotonpt[i] < 75:
            proc_value = 'QQ2HLNU_PTV_0_75'
            proc_value_num = 0
        elif diphotonpt[i] >= 75 and diphotonpt[i] < 150:
            proc_value = 'QQ2HLNU_PTV_75_150'
            proc_value_num = 1
        elif diphotonpt[i] >= 150 and diphotonpt[i] < 250:
            if njets[i] == 2:
                proc_value = 'QQ2HLNU_PTV_150_250_0J'
                proc_value_num = 2
            elif njets[i] == 1:
                proc_value = 'QQ2HLNU_PTV_150_250_GE1J'
                proc_value_num = 3
        elif diphotonpt[i] >= 250:
            proc_value = 'QQ2HLNU_PTV_75_150'
            proc_value_num = 4
    #else:
    elif stage_1_2[i] == 400 or 401 or 402 or 403 or 404 or 405:
        if diphotonpt[i] < 75:
            proc_value = 'QQ2HLL_PTV_0_75'
            proc_value_num = 5
        elif diphotonpt[i] >= 75 and diphotonpt[i] < 150:
            proc_value = 'QQ2HLL_PTV_75_150'
            proc_value_num = 6
        elif diphotonpt[i] >= 150 and diphotonpt[i] < 250:
            if njets[i] == 0:
                proc_value = 'QQ2HLL_PTV_150_250_0J'
                proc_value_num = 7
            elif njets[i] == 1:
                proc_value = 'QQ2HLL_PTV_150_250_GE1J'
                proc_value_num = 8
        elif diphotonpt[i] >= 250:
            proc_value = 'QQ2HLL_PTV_GT250'
            proc_value_num = 9

    proc.append(proc_value)
    y_train_labels_num_pred.append(proc_value_num)
y_train_labels_num_pred = np.array(y_train_labels_num_pred)
data['proc_new'] = proc

# Confusion Matrix

y_true_old = data['proc_original']
y_pred_old = data['proc_new']
y_true = y_train_labels_num
y_pred = y_train_labels_num_pred

print 'Accuracy score: '
NNaccuracy = accuracy_score(y_true, y_pred)
print(NNaccuracy)

cm_old = confusion_matrix(y_true=y_true,y_pred=y_pred)
cm = confusion_matrix(y_true=y_true,y_pred=y_pred,sample_weight=weights)
cm_new = np.zeros((len(binNames),len(binNames)),dtype=int)
for i in range(len(y_true)):
    cm_new[y_true[i]][y_pred[i]] += 1


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
    name = 'plotting/Cuts/Cut_VH_Confusion_Matrix'
    fig.savefig(name, dpi = 1200)

def plot_performance_plot(cm=cm,labels=binNames):
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
    name = 'plotting/Cuts/Cut_VH_Performance_Plot'
    plt.savefig(name, dpi = 1200)
    plt.show()
'''
def plot_roc_curve(binNames = binNames, y_test = y_train_labels_num, y_pred_test = y_pred, x_test = data, color = color):
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
    name = 'plotting/Cuts/Cut_stage_0_ROC_curve'
    plt.savefig(name, dpi = 1200)
    print("Plotting ROC Curve")
    plt.close()
'''
#y_test not correct yet, need to check this again
#plot_roc_curve(y_test=y_train_labels_num,y_pred_test=y_pred,x_test=data)

plot_performance_plot()

plot_confusion_matrix(cm,binNames,normalize=True)