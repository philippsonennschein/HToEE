import argparse
import pandas as pd
import numpy as np
import matplotlib
import xgboost as xgb
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
from sklearn.model_selection import train_test_split
import pickle
from itertools import product
from keras.utils import np_utils 
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, auc

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

color  = ['silver','indianred','salmon','lightgreen','seagreen','mediumturquoise','darkslategrey','skyblue','steelblue','lightsteelblue','mediumslateblue']

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
    if i == 'WH_Rest':
        y_train_labels_num.append(2)
    if i == 'ZH_Rest':
        y_train_labels_num.append(3)
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
leadElectronPt = np.array(data['leadElectronPt'])
leadMuonPt = np.array(data['leadMuonPt'])
subleadElectronPt = np.array(data['subleadElectronPt'])
subleadMuonPt = np.array(data['subleadMuonPt'])

# creating n-leptons variable
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
print('Done 2')

proc = []
y_train_labels_num_pred = []
for i in range(data.shape[0]):
    #if stage_1_2[i] == 300 or stage_1_2[i] == 301 or stage_1_2[i] == 302 or stage_1_2[i] == 303 or stage_1_2[i] == 304 or stage_1_2[i] == 305: # FIXME NLEPTONS
    if nleptons[i] == 1:                        # WH
        if diphotonpt[i] < 75:
            proc_value = 'QQ2HLNU_PTV_0_75'
            proc_value_num = 0
        elif diphotonpt[i] >= 75 and diphotonpt[i] < 150:
            proc_value = 'QQ2HLNU_PTV_75_150'
            proc_value_num = 1
        else:
            proc_value = 'WH_Rest'
            proc_value_num = 2
    if nleptons[i] == 0 or nleptons[i] == 2:  # ZH
        proc_value = 'ZH_Rest'
        proc_value_num = 3

    proc.append(proc_value)
    y_train_labels_num_pred.append(proc_value_num)
y_train_labels_num_pred = np.array(y_train_labels_num_pred)
data['proc_new'] = proc


# Confusion Matrix
y_true_old = data['proc_original']
y_pred_old = data['proc_new']
y_true = y_train_labels_num
y_pred = y_train_labels_num_pred

#cm_old = confusion_matrix(y_true=y_true,y_pred=y_pred)
cm = confusion_matrix(y_true=y_true,y_pred=y_pred,sample_weight=weights)
#cm_new = np.zeros((len(binNames),len(binNames)),dtype=int)
#for i in range(len(y_true)):
#    cm_new[y_true[i]][y_pred[i]] += 1

# Accuracy Score
print 'Accuracy score: '
NNaccuracy = accuracy_score(y_true, y_pred, sample_weight = weights)
print(NNaccuracy)

num_correct = 0
num_all = 0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        num_all += cm[i][j]
        if i == j:     # so diagonal
            num_correct += cm[i][j]

accuracy = num_correct / num_all
print('Final Accuracy Score: ', accuracy)


#Confusion Matrix
def plot_confusion_matrix(cm,classes,labels = labelNames, normalize=True,title='Confusion matrix',cmap=plt.cm.Blues):
    fig, ax = plt.subplots(figsize = (10,10))
    #plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks,labels,rotation=45, horizontalalignment = 'right')
    plt.yticks(tick_marks,labels)
    if normalize:
        cm = cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
        for i in range(len(cm[0])):
            for j in range(len(cm[1])):
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
    name = 'plotting/Cuts/Cuts_VH_tenclass_Confusion_Matrix'
    plt.tight_layout()
    fig.savefig(name, dpi = 1200)

def plot_performance_plot(cm=cm,labels=labelNames, normalize = True, color = color):
    #cm = cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
    cm = cm.astype('float')/cm.sum(axis=0)[np.newaxis,:]
    for i in range(len(cm[0])):
        for j in range(len(cm[1])):
            cm[i][j] = float("{:.3f}".format(cm[i][j]))
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize = (10,10))
    plt.rcParams.update({
    'font.size': 9})
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks,labels,rotation=45, horizontalalignment = 'right')
    bottom = np.zeros(len(labels))
    #color = ['#24b1c9','#e36b1e','#1eb037','#c21bcf','#dbb104']
    for i in range(len(cm)):
        #ax.bar(labels, cm[i,:],label=labels[i],bottom=bottom)
        #bottom += np.array(cm[i,:])
        ax.bar(labels, cm[i,:],label=labels[i],bottom=bottom,color=color[i])
        bottom += np.array(cm[i,:])
    plt.legend()
    current_bottom, current_top = ax.get_ylim()
    ax.set_ylim(bottom=0, top=current_top*1.3)
    #plt.title('Performance Plot')
    plt.ylabel('Fraction of events', size = 12)
    ax.set_xlabel('Events',size=12) #, x=1, size=13)
    name = 'plotting/Cuts/Cuts_VH_tenclass_Performance_Plot'
    plt.tight_layout()
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

#plot_performance_plot()

#plot_confusion_matrix(cm,binNames,normalize=True)


# ------------------------ 
# Binary BDT for signal purity
# okayy lessgooo

# data_new['proc']  # are the true labels
# data_new['weight'] are the weights

num_estimators = 200
test_split = 0.4

clf_2 = xgb.XGBClassifier(objective='binary:logistic', n_estimators=num_estimators, 
                            eta=0.1, maxDepth=6, min_child_weight=0.01, 
                            subsample=0.6, colsample_bytree=0.6, gamma=4)

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
    data_new = data.copy()  
    # now i want to get the predicted labels
    proc_pred = []      
    for j in range(len(y_pred)):
        if(y_pred[j] == i): # so that the predicted label is the signal
            proc_pred.append(signal[i])
        else:
            proc_pred.append('background')
    data_new['proc_pred'] = proc_pred       # Problem might be here, they don't seem to line up

    rest, data_new = train_test_split(data_new, test_size = test_split, shuffle = True)

    # now cut down the dataframe to the predicted ones -  this is the split for the different dataframes
    data_new = data_new[data_new.proc_pred == signal[i]] 

    # now from proc make signal against background (binary classifier)

    proc_true = np.array(data_new['proc_original'])
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
    data_new = data_new.drop(columns=['proc_original'])
    data_new = data_new.drop(columns=['proc_pred'])
    data_new = data_new.drop(columns=['proc_new'])

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
    name_fpr = 'csv_files/VH_fourclass_Cuts_binary_fpr_' + signal[i]
    name_tpr = 'csv_files/VH_fourclass_Cuts_binary_tpr_' + signal[i]
    np.savetxt(name_fpr, fpr_keras, delimiter = ',')
    np.savetxt(name_tpr, tpr_keras, delimiter = ',')
    auc_test = auc(fpr_keras, tpr_keras)
    ax.plot(fpr_keras, tpr_keras, label = 'AUC = {0}, {1}'.format(round(auc_test, 3), labelNames[i]), color = color[i])

ax.legend(loc = 'lower right', fontsize = 'small')
ax.set_xlabel('Background Efficiency', ha='right', x=1, size=9)
ax.set_ylabel('Signal Efficiency',ha='right', y=1, size=9)
ax.grid(True, 'major', linestyle='dotted', color='grey', alpha=0.5)
plt.tight_layout()
name = 'plotting/Cuts/Cuts_VH_Fourclass_binary_Multi_ROC_curve'
plt.savefig(name, dpi = 1200)
print("Plotting ROC Curve")
plt.close()

print('Final conf_matrix:')
print(conf_matrix_w)

#final final confusion matrix for plotting
confusion_matrix = np.concatenate((cm,conf_matrix_w2),axis=0)

#Exporting final confusion matrix
name_cm = 'csv_files/VH_fourclass_Cuts_binary_cm'
np.savetxt(name_cm, conf_matrix_w, delimiter = ',')

#Need a new function beause the cm structure is different
def plot_performance_plot_final(cm=conf_matrix_w,labels=labelNames, color = color, name = 'plotting/Cuts/Cuts_VH_Fourclass_Performance_Plot'):
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

def plot_final_confusion_matrix(cm,classes,labels = labelNames,y_labels = y_label, normalize=True,title='Confusion matrix',cmap=plt.cm.Blues, name = 'plotting/Cuts/Cuts_VH_Fourclass_final_Confusion_Matrix'):
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
plot_performance_plot_final(cm = conf_matrix_w,labels = labelNames, name = 'plotting/Cuts/Cuts_VH_Fourclass_Performance_Plot_final')

plot_final_confusion_matrix(cm=confusion_matrix,classes=binNames,labels = labelNames,y_labels = y_label,normalize=True)

num_false = np.sum(conf_matrix_w[0,:])
num_correct = np.sum(conf_matrix_w[1,:])
accuracy = num_correct / (num_correct + num_false)
print('BDT Final Accuracy Score:')
print(accuracy)