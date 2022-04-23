from __future__ import division
import argparse
import pandas as pd
import numpy as np
import matplotlib
#import matplotlib as mpl
import xgboost as xgb
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
from sklearn.model_selection import train_test_split
import pickle
from itertools import product
from keras.utils import np_utils 
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score

#Define key quantities, use to tune BDT
num_estimators = 300
test_split = 0.30
learning_rate = 0.0001

map_def = [['ggH',10,11],['qqH',20,21,22,23],['VH',30,31,40,41],['ttH',60,61],['tH',80,81]]

binNames = ['ggH','qqH','VH','ttH','tH'] 
labelNames = binNames
bins = 50

color  = ['silver','indianred','salmon','lightgreen','seagreen','mediumturquoise','darkslategrey','skyblue','steelblue','lightsteelblue','mediumslateblue']


train_vars = ['diphotonMass', 'diphotonPt', 'diphotonEta',
'diphotonPhi', 'diphotonCosPhi', 'diphotonSigmaMoM',
'leadPhotonIDMVA', 'leadPhotonPtOvM', 'leadPhotonEta',
'leadPhotonEn', 'leadPhotonMass', 'leadPhotonPt', 'leadPhotonPhi',
'subleadPhotonIDMVA', 
'subleadPhotonPtOvM', 
'subleadPhotonEta',
'subleadPhotonEn', 'subleadPhotonMass', 'subleadPhotonPt',
'subleadPhotonPhi', 'dijetMass', 'dijetPt', 'dijetEta', 'dijetPhi',
'dijetDPhi', 'dijetAbsDEta', 'dijetCentrality', 
'dijetMinDRJetPho',
'dijetDiphoAbsDEta', 'leadJetPUJID', 
'leadJetPt', 'leadJetEn',
'leadJetEta', 'leadJetPhi', 'leadJetMass', 'leadJetBTagScore',
'leadJetDiphoDEta', 'leadJetDiphoDPhi', 'subleadJetPUJID',
'subleadJetPt', 'subleadJetEn', 'subleadJetEta', 'subleadJetPhi','subleadJetBTagScore','subsubleadJetBTagScore',
#'subleadJetMass', 'subleadJetBTagScore', 'subleadJetDiphoDPhi',
#'subleadJetDiphoDEta',
#'subsubleadJetPUJID', 'subsubleadJetPt',
#'subsubleadJetEn', 'subsubleadJetEta', 'subsubleadJetPhi',
#'subsubleadJetMass', 'subsubleadJetBTagScore',
'metPt','metPhi','metSumET','nSoftJets',
'leadElectronEn',
#'leadElectronMass',
#'leadElectronPt','leadElectronEta','leadElectronPhi','leadElectronCharge',
'leadMuonEn'
#,'leadMuonMass','
#leadMuonPt','leadMuonEta','leadMuonPhi','leadMuonCharge',
#'subleadElectronEn','subleadElectronMass','subleadElectronPt','subleadElectronEta','subleadElectronPhi','subleadElectronCharge',
#'subleadMuonEn','subleadMuonMass','subleadMuonPt','subleadMuonEta','subleadMuonPhi','subleadMuonCharge'
]

train_vars.append('proc')
train_vars.append('weight')
train_vars.append('HTXS_stage_0')
#train_vars.append('HTXS_stage1_2_cat_pTjet30GeV')

dataframes = []
dataframes.append(pd.read_csv('2017/MC/DataFrames/ggH_VBF_BDT_df_2017.csv', nrows = 494025))
dataframes.append(pd.read_csv('2017/MC/DataFrames/VBF_VBF_BDT_df_2017.csv', nrows = 533739))
dataframes.append(pd.read_csv('2017/MC/DataFrames/VH_VBF_BDT_df_2017.csv'))
dataframes.append(pd.read_csv('2017/MC/DataFrames/ttH_VBF_BDT_df_2017.csv'))
dataframes.append(pd.read_csv('2017/MC/DataFrames/tHq_VBF_BDT_df_2017.csv', nrows = 254039))
dataframes.append(pd.read_csv('2017/MC/DataFrames/tHW_VBF_BDT_df_2017.csv', nrows = 130900))
df = pd.concat(dataframes, sort=False, axis=0 )

data = df[train_vars]

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

data['proc_new'] = mapping(map_list=map_def,stage=data['HTXS_stage_0'])

data.loc[data.proc_new == 'ggH','weight'] = data[data['proc_new'] == 'ggH']['weight'] * 2
data.loc[data.proc_new == 'qqH','weight'] = data[data['proc_new'] == 'qqH']['weight'] * 2
data.loc[data.proc_new == 'tH','weight'] = data[data['proc_new'] == 'tH']['weight'] * 4

#Define the procs as the labels
#ggh: 0, VBF:1, VH: 2, ttH: 3
#num_categories = data['proc'].nunique()
#y_train_labels_num, y_train_labels_def = pd.factorize(data['proc'])

num_categories = data['proc_new'].nunique()
proc_new = np.array(data['proc_new'])
#Assign the numbers in the same order as the binNames above
y_train_labels_num = []
for i in proc_new:
    if i == 'ggH':
        y_train_labels_num.append(0)
    if i == 'qqH':
        y_train_labels_num.append(1)
    if i == 'VH':
        y_train_labels_num.append(2)
    if i == 'ttH':
        y_train_labels_num.append(3)
    if i == 'tH':
        y_train_labels_num.append(4)
data['proc_num'] = y_train_labels_num


y_train_labels = np.array(data['proc_new'])
y_train_labels_num = np.array(data['proc_num'])
y_train_labels_hot = np_utils.to_categorical(y_train_labels_num, num_classes=num_categories)
weights = np.array(data['weight'])

data = data.drop(columns=['proc','weight','proc_num','HTXS_stage_0','proc_new'])

#With num
x_train, x_test, y_train, y_test, train_w, test_w, proc_arr_train, proc_arr_test = train_test_split(data, y_train_labels_num, weights, y_train_labels, test_size = test_split, shuffle = True)
#With hot
#x_train, x_test, y_train, y_test, train_w, test_w, proc_arr_train, proc_arr_test = train_test_split(data, y_train_labels_hot, weights, y_train_labels, test_size = val_split, shuffle = True)

#Before n_estimators = 100, maxdepth=4, gamma = 1
#Improved n_estimators = 300, maxdepth = 7, gamme = 4
# Current best compbination of HP values:, 300, 0.01, 6)

clf = xgb.XGBClassifier(objective='multi:softprob', n_estimators=300, 
                            eta=0.0001, maxDepth=6, min_child_weight=0.01, 
                            subsample=0.6, colsample_bytree=0.6, gamma=4,
                            num_class=5)

#Equalizing weights
train_w_df = pd.DataFrame()
train_w = 300 * train_w # to make loss function O(1)
train_w_df['weight'] = train_w
train_w_df['proc'] = proc_arr_train
qqh_sum_w = train_w_df[train_w_df['proc'] == 'qqH']['weight'].sum()
ggh_sum_w = train_w_df[train_w_df['proc'] == 'ggH']['weight'].sum()
vh_sum_w = train_w_df[train_w_df['proc'] == 'VH']['weight'].sum()
tth_sum_w = train_w_df[train_w_df['proc'] == 'ttH']['weight'].sum()
th_sum_w = train_w_df[train_w_df['proc'] == 'tH']['weight'].sum()
train_w_df.loc[train_w_df.proc == 'qqH','weight'] = (train_w_df[train_w_df['proc'] == 'qqH']['weight'] * ggh_sum_w / qqh_sum_w)
train_w_df.loc[train_w_df.proc == 'VH','weight'] = (train_w_df[train_w_df['proc'] == 'VH']['weight'] * ggh_sum_w / vh_sum_w)
train_w_df.loc[train_w_df.proc == 'ttH','weight'] = (train_w_df[train_w_df['proc'] == 'ttH']['weight'] * ggh_sum_w / tth_sum_w)
train_w_df.loc[train_w_df.proc == 'tH','weight'] = (train_w_df[train_w_df['proc'] == 'tH']['weight'] * ggh_sum_w / th_sum_w)
train_w = np.array(train_w_df['weight'])

print (' Training classifier...')
clf = clf.fit(x_train, y_train, sample_weight=train_w)
print ('Finished Training classifier!')

y_pred_test = clf.predict_proba(x_test)

#Accuracy Score
y_pred = y_pred_test.argmax(axis=1)
#y_true = y_test.argmax(axis=1)
y_true = y_test
print 'Accuracy score: '
NNaccuracy = accuracy_score(y_true, y_pred,sample_weight=test_w)
print(NNaccuracy)

#Confusion Matrix
cm = confusion_matrix(y_true=y_true,y_pred=y_pred,sample_weight=test_w)

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
    name = 'plotting/BDT_plots/BDT_Stage0_Confusion_Matrix'
    fig.savefig(name,dpi = 1200)

def plot_performance_plot(cm=cm,labels=binNames):
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
    color = ['#24b1c9','#e36b1e','#1eb037','#c21bcf','#dbb104']
    bottom = np.zeros(len(labels))
    for i in range(len(cm)):
        #ax.bar(labels, cm[:,i],label=labels[i],bottom=bottom)
        #bottom += np.array(cm[:,i])
        ax.bar(labels, cm[i,:],label=labels[i],bottom=bottom,color=color[i])
        bottom += np.array(cm[i,:])
    plt.legend(loc='upper right')
    current_bottom, current_top = ax.get_ylim()
    ax.set_ylim(bottom=0, top=current_top*1.3)
    #plt.title('Performance Plot')
    #plt.ylabel('Fraction of events')
    ax.set_ylabel('Events', ha='center',size=14) #y=0.5,
    ax.set_xlabel('Predicted Production Modes', ha='center',size=14) #, x=1, size=13)
    name = 'plotting/BDT_plots/BDT_Stage0_Performance_Plot'
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
    name = 'plotting/BDT_plots/BDT_stage_0_ROC_curve'
    plt.savefig(name, dpi = 1200)
    print("Plotting ROC Curve")
    plt.close()

def feature_importance(num_plots='single',num_feature=20,imp_type='gain',values = False):
    if num_plots == 'single':
        plt.rcParams["figure.figsize"] = (12,7)
        xgb.plot_importance(clf, max_num_features=num_feature, grid = False, height = 0.4, importance_type = imp_type, title = 'Feature importance ({})'.format(imp_type), show_values = values, color ='blue')
        plt.tight_layout()
        plt.savefig('plotting/BDT_plots/BDT_Stage0_feature_importance_{0}'.format(imp_type), dpi = 1200)
        print('saving: /plotting/BDT_plots/BDT_Stage0_feature_importance_{0}'.format(imp_type))
        
    else:
        imp_types = ['weight','gain','cover']
        for i in imp_types:
            xgb.plot_importance(clf, max_num_features=num_feature, grid = False, height = 0.4, importance_type = imp_type, title = 'Feature importance ({})'.format(i), show_values = values, color ='blue')
            plt.tight_layout()
            plt.savefig('plotting/BDT_plots/BDT_qqH_sevenclass_feature_importance_{0}'.format(i), dpi = 1200)
            print('saving: plotting/BDT_plots/BDT_qqH_sevenclass_feature_importance_{0}'.format(i))


#feature_importance()

#plot_roc_curve()

plot_performance_plot()

plot_confusion_matrix(cm,binNames,normalize=True)
