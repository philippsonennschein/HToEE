
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
test_split = 0.15
learning_rate = 0.001

map_def = [['ggH',10,11],['qqH',20,21,22,23],['VH',30,31,40,41],['ttH',60,61],['tH',80,81]]

binNames = ['ggH','qqH','VH','ttH','tH'] 
bins = 50

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
'subleadJetDiphoDEta',
#, 'subsubleadJetPUJID', 'subsubleadJetPt',
#'subsubleadJetEn', 'subsubleadJetEta', 'subsubleadJetPhi',
#'subsubleadJetMass', 'subsubleadJetBTagScore','nSoftJets',
'metPt','metPhi','metSumET'
#'leadElectronEn','leadElectronMass','leadElectronPt','leadElectronEta','leadElectronPhi','leadElectronCharge',
#'leadMuonEn','leadMuonMass','leadMuonPt','leadMuonEta','leadMuonPhi','leadMuonCharge',
#'subleadElectronEn','subleadElectronMass','subleadElectronPt','subleadElectronEta','subleadElectronPhi','subleadElectronCharge',
#'subleadMuonEn','subleadMuonMass','subleadMuonPt','subleadMuonEta','subleadMuonPhi','subleadMuonCharge'
]

train_vars.append('proc')
train_vars.append('weight')
train_vars.append('HTXS_stage_0')
#train_vars.append('HTXS_stage1_2_cat_pTjet30GeV')

dataframes = []
dataframes.append(pd.read_csv('2017/MC/DataFrames/ggH_VBF_BDT_df_2017.csv'))
dataframes.append(pd.read_csv('2017/MC/DataFrames/VBF_VBF_BDT_df_2017.csv'))
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

data.loc[data.proc_new == 'tH','weight'] = data[data['proc_new'] == 'tH']['weight'] * 4

#Define the procs as the labels
#ggh: 0, VBF:1, VH: 2, ttH: 3
#num_categories = data['proc'].nunique()
#y_train_labels_num, y_train_labels_def = pd.factorize(data['proc'])

num_categories = data['proc_new'].nunique()
y_train_labels_num, y_train_labels_def = pd.factorize(data['proc_new'])

#Label definition:
print('Label Definition:')
label_def = []
for i in range(num_categories):
    label_def.append([i,y_train_labels_def[i]])
    print(i,y_train_labels_def[i])

data['proc_num'] = y_train_labels_num

y_train_labels = np.array(data['proc_new'])
#y_train_labels = np.array(data['proc'])
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
                            eta=0.01, maxDepth=6, min_child_weight=0.01, 
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
NNaccuracy = accuracy_score(y_true, y_pred)
print(NNaccuracy)

#Confusion Matrix
cm = confusion_matrix(y_true=y_true,y_pred=y_pred)


#Confusion Matrix
def plot_confusion_matrix(cm,classes,normalize=True,title='Confusion matrix',cmap=plt.cm.Blues):
    fig, ax = plt.subplots(1)
    #plt.colorbar()
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
    plt.imshow(cm,interpolation='nearest',cmap=cmap)
    plt.title(title)
    for i, j in product(range(cm.shape[0]),range(cm.shape[1])):
        plt.text(j,i,cm[i,j],horizontalalignment='center',color='white' if cm[i,j]>thresh else 'black')
    plt.tight_layout()
    plt.colorbar()
    plt.ylabel('True Label')
    plt.xlabel('Predicted label')
    name = 'plotting/BDT_plots/BDT_Stage0_Confusion_Matrix'
    fig.savefig(name)

def plot_performance_plot(cm=cm,labels=binNames):
    #cm = cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
    cm = cm.astype('float')/cm.sum(axis=0)[np.newaxis,:]
    for i in range(len(cm[0])):
        for j in range(len(cm[1])):
            cm[i][j] = float("{:.3f}".format(cm[i][j]))
    print(cm)
    cm = np.array(cm)
    #fig, ax = plt.subplots(figsize = (12,12))
    fig, ax = plt.subplots()
    plt.rcParams.update({
    'font.size': 12})
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks,labels,rotation=90)
    color = ['#24b1c9','#e36b1e','#1eb037','#c21bcf','#dbb104']
    bottom = np.zeros(len(labels))
    for i in range(len(cm)):
        #ax.bar(labels, cm[:,i],label=labels[i],bottom=bottom)
        #bottom += np.array(cm[:,i])
        ax.bar(labels, cm[i,:],label=labels[i],bottom=bottom,color=color[i])
        bottom += np.array(cm[i,:])
    plt.legend()
    current_bottom, current_top = ax.get_ylim()
    ax.set_ylim(bottom=0, top=current_top*1.3)
    #plt.title('Performance Plot')
    plt.ylabel('Fraction of events')
    ax.set_xlabel('Events', ha='right',x=1,size=9) #, x=1, size=13)
    name = 'plotting/NN_plots/NN_Stage0_Performance_Plot'
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


plot_roc_curve()

plot_performance_plot()

plot_confusion_matrix(cm,binNames,normalize=True)
