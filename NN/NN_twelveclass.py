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

#Define key quantities

#HPs
#Original
#num_epochs = 2
#batch_size = 400
#val_split = 0.3
#learning_rate = 0.001

#Optimized according to 4class
num_epochs = 30
batch_size = 64
val_split = 0.3
learning_rate = 0.0001


#STXS mapping
map_def_0 = [['ggH',10,11],['qqH',20,21,22,23],['WH',30,31],['ZH',40,41],['ttH',60,61],['tH',80,81]]
map_def_1 = [
['ggH',100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116],
['QQ2HQQ_FWDH',200],
['QQ2HQQ_0J',201],
['QQ2HQQ_1J',202],
['QQ2HQQ_GE2J_MJJ_0_60',203],
['QQ2HQQ_GE2J_MJJ_60_120',204],
['QQ2HQQ_GE2J_MJJ_120_350',205],
['QQ2HQQ_GE2J_MJJ_GT350_PTH_GT200',206],
['QQ2HQQ_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_0_25',207],
['QQ2HQQ_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_GT25',208],
['QQ2HQQ_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_0_25',209],
['QQ2HQQ_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_GT25',210],
['WH',300,301,302,303,304,305],
['ZH',400,401,402,403,404,405],
['ttH',600,601,602,603,604,605],
['tH',800,801]
]

epochs = np.linspace(1,num_epochs,num_epochs,endpoint=True).astype(int) #For plotting
#binNames = ['ggH','qqH','ZH','WH','ttH','tH'] 
#binNames = ['ggH','QQ2HQQ_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_GT25','QQ2HQQ_GE2J_MJJ_120_350',
#'QQ2HQQ_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_GT25','QQ2HQQ_1J','QQ2HQQ_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_0_25',
#'QQ2HQQ_0J','QQ2HQQ_GE2J_MJJ_GT350_PTH_GT200','QQ2HQQ_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_0_25',
#'QQ2HQQ_GE2J_MJJ_0_60','QQ2HQQ_GE2J_MJJ_60_120','QQ2HQQ_FWDH','ZH','WH','ttH','tH'] 
binNames = ['ggH','qqH1','qqH2','qqH3','qqH4','qqH5','qqH6','qqH7','qqH8','qqH9','qqH10','qqH0','ZH','WH','ttH','tH'] 
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
'subsubleadJetMass', 'subsubleadJetBTagScore','nSoftJets',
'metPt','metPhi','metSumET',
'leadElectronEn','leadElectronMass','leadElectronPt','leadElectronEta','leadElectronPhi','leadElectronCharge',
'leadMuonEn','leadMuonMass','leadMuonPt','leadMuonEta','leadMuonPhi','leadMuonCharge',
'subleadElectronEn','subleadElectronMass','subleadElectronPt','subleadElectronEta','subleadElectronPhi','subleadElectronCharge',
'subleadMuonEn','subleadMuonMass','subleadMuonPt','subleadMuonEta','subleadMuonPhi','subleadMuonCharge']

#Add proc and weight to shuffle with data
train_vars.append('proc')
train_vars.append('weight')
train_vars.append('HTXS_stage_0')
train_vars.append('HTXS_stage1_2_cat_pTjet30GeV')

#Load the dataframe
dataframes = []
dataframes.append(pd.read_csv('2017/MC/DataFrames/ggH_VBF_BDT_df_2017.csv'))
dataframes.append(pd.read_csv('2017/MC/DataFrames/VBF_VBF_BDT_df_2017.csv'))
dataframes.append(pd.read_csv('2017/MC/DataFrames/VH_VBF_BDT_df_2017.csv'))
dataframes.append(pd.read_csv('2017/MC/DataFrames/ttH_VBF_BDT_df_2017.csv'))
dataframes.append(pd.read_csv('2017/MC/DataFrames/tHq_VBF_BDT_df_2017.csv'))
dataframes.append(pd.read_csv('2017/MC/DataFrames/tHW_VBF_BDT_df_2017.csv'))
df = pd.concat(dataframes, sort=False, axis=0 )

#dataframe of train_vars
data = df[train_vars]

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

#data['proc_new'] = mapping(map_list=map_def,stage=data['HTXS_stage_0'])
data['proc_new'] = mapping(map_list=map_def_1,stage=data['HTXS_stage1_2_cat_pTjet30GeV'])

#Define the procs as the labels
#ggh: 0, VBF:1, VH: 2, ttH: 3
num_categories = data['proc_new'].nunique()
y_train_labels_num, y_train_labels_def = pd.factorize(data['proc_new'])
#y_train_labels_def = np.array(y_train_labels_def)

#Label definition:
print('Label Definition:')
label_def = []
binNames_auto = []
for i in range(num_categories):
    label_def.append([i,y_train_labels_def[i]])
    binNames_auto.append(y_train_labels_def[i])
    print(i,y_train_labels_def[i])

data['proc_num'] = y_train_labels_num

#Shuffle dataframe
data = data.sample(frac=1)

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
x_train, x_test, y_train, y_test, train_w, test_w, proc_arr_train, proc_arr_test = train_test_split(data_scaled, y_train_labels_hot, weights, y_train_labels, test_size = val_split, shuffle = True)

#Initialize the model
model=Sequential([Dense(units=400,input_shape=(num_inputs,),activation='relu'),
                Dense(units=400,activation='relu'),
                #Dense(units=100,activation='relu'),
                Dense(units=num_categories,activation='softmax')]) #For multiclass NN use softmax

#Compile the model
model.compile(optimizer=Adam(lr=learning_rate),loss='categorical_crossentropy',metrics=['accuracy'])

model.summary()

#Equalizing weights
train_w_df = pd.DataFrame()
train_w = 300 * train_w # to make loss function O(1)
train_w_df['weight'] = train_w
train_w_df['proc'] = proc_arr_train
ggh_sum_w = train_w_df[train_w_df['proc'] == 'ggH']['weight'].sum()
wh_sum_w = train_w_df[train_w_df['proc'] == 'WH']['weight'].sum()
zh_sum_w = train_w_df[train_w_df['proc'] == 'ZH']['weight'].sum()
tth_sum_w = train_w_df[train_w_df['proc'] == 'ttH']['weight'].sum()
th_sum_w = train_w_df[train_w_df['proc'] == 'tH']['weight'].sum()
qqh0_sum_w = train_w_df[train_w_df['proc'] == 'QQ2HQQ_FWDH']['weight'].sum()
qqh1_sum_w = train_w_df[train_w_df['proc'] == 'QQ2HQQ_0J']['weight'].sum()
qqh2_sum_w = train_w_df[train_w_df['proc'] == 'QQ2HQQ_1J']['weight'].sum()
qqh3_sum_w = train_w_df[train_w_df['proc'] == 'QQ2HQQ_GE2J_MJJ_0_60']['weight'].sum()
qqh4_sum_w = train_w_df[train_w_df['proc'] == 'QQ2HQQ_GE2J_MJJ_60_120']['weight'].sum()
qqh5_sum_w = train_w_df[train_w_df['proc'] == 'QQ2HQQ_GE2J_MJJ_120_350']['weight'].sum()
qqh6_sum_w = train_w_df[train_w_df['proc'] == 'QQ2HQQ_GE2J_MJJ_GT350_PTH_GT200']['weight'].sum()
qqh7_sum_w = train_w_df[train_w_df['proc'] == 'QQ2HQQ_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_0_25']['weight'].sum()
qqh8_sum_w = train_w_df[train_w_df['proc'] == 'QQ2HQQ_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_GT25']['weight'].sum()
qqh9_sum_w = train_w_df[train_w_df['proc'] == 'QQ2HQQ_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_0_25']['weight'].sum()
qqh10_sum_w = train_w_df[train_w_df['proc'] == 'QQ2HQQ_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_GT25']['weight'].sum()

train_w_df.loc[train_w_df.proc == 'WH','weight'] = (train_w_df[train_w_df['proc'] == 'WH']['weight'] * ggh_sum_w / wh_sum_w)
train_w_df.loc[train_w_df.proc == 'ZH','weight'] = (train_w_df[train_w_df['proc'] == 'ZH']['weight'] * ggh_sum_w / zh_sum_w)
train_w_df.loc[train_w_df.proc == 'ttH','weight'] = (train_w_df[train_w_df['proc'] == 'ttH']['weight'] * ggh_sum_w / tth_sum_w)
train_w_df.loc[train_w_df.proc == 'tH','weight'] = (train_w_df[train_w_df['proc'] == 'tH']['weight'] * ggh_sum_w / th_sum_w)
train_w_df.loc[train_w_df.proc == 'QQ2HQQ_FWDH','weight'] = (train_w_df[train_w_df['proc'] == 'QQ2HQQ_FWDH']['weight'] * ggh_sum_w / qqh0_sum_w)
train_w_df.loc[train_w_df.proc == 'QQ2HQQ_0J','weight'] = (train_w_df[train_w_df['proc'] == 'QQ2HQQ_0J']['weight'] * ggh_sum_w / qqh1_sum_w)
train_w_df.loc[train_w_df.proc == 'QQ2HQQ_1J','weight'] = (train_w_df[train_w_df['proc'] == 'QQ2HQQ_1J']['weight'] * ggh_sum_w / qqh2_sum_w)
train_w_df.loc[train_w_df.proc == 'QQ2HQQ_GE2J_MJJ_0_60','weight'] = (train_w_df[train_w_df['proc'] == 'QQ2HQQ_GE2J_MJJ_0_60']['weight'] * ggh_sum_w / qqh3_sum_w)
train_w_df.loc[train_w_df.proc == 'QQ2HQQ_GE2J_MJJ_60_120','weight'] = (train_w_df[train_w_df['proc'] == 'QQ2HQQ_GE2J_MJJ_60_120']['weight'] * ggh_sum_w / qqh4_sum_w)
train_w_df.loc[train_w_df.proc == 'QQ2HQQ_GE2J_MJJ_120_350','weight'] = (train_w_df[train_w_df['proc'] == 'QQ2HQQ_GE2J_MJJ_120_350']['weight'] * ggh_sum_w / qqh5_sum_w)
train_w_df.loc[train_w_df.proc == 'QQ2HQQ_GE2J_MJJ_GT350_PTH_GT200','weight'] = (train_w_df[train_w_df['proc'] == 'QQ2HQQ_GE2J_MJJ_GT350_PTH_GT200']['weight'] * ggh_sum_w / qqh6_sum_w)
train_w_df.loc[train_w_df.proc == 'QQ2HQQ_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_0_25','weight'] = (train_w_df[train_w_df['proc'] == 'QQ2HQQ_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_0_25']['weight'] * ggh_sum_w / qqh7_sum_w)
train_w_df.loc[train_w_df.proc == 'QQ2HQQ_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_GT25','weight'] = (train_w_df[train_w_df['proc'] == 'QQ2HQQ_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_GT25']['weight'] * ggh_sum_w / qqh8_sum_w)
train_w_df.loc[train_w_df.proc == 'QQ2HQQ_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_0_25','weight'] = (train_w_df[train_w_df['proc'] == 'QQ2HQQ_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_0_25']['weight'] * ggh_sum_w / qqh9_sum_w)
train_w_df.loc[train_w_df.proc == 'QQ2HQQ_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_GT25','weight'] = (train_w_df[train_w_df['proc'] == 'QQ2HQQ_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_GT25']['weight'] * ggh_sum_w / qqh10_sum_w)
train_w = np.array(train_w_df['weight'])

#Training the model
history = model.fit(x=x_train,y=y_train,batch_size=batch_size,epochs=num_epochs,sample_weight=train_w,shuffle=True,verbose=2)

# Output Score
y_pred_test = model.predict_proba(x=x_test)
x_test['proc'] = proc_arr_test
x_test['weight'] = test_w
x_test['output_score_ggh'] = y_pred_test[:,0]

x_test['output_score_qqh0'] = y_pred_test[:,1]
x_test['output_score_qqh1'] = y_pred_test[:,2]
x_test['output_score_qqh2'] = y_pred_test[:,3]
x_test['output_score_qqh3'] = y_pred_test[:,4]
x_test['output_score_qqh4'] = y_pred_test[:,5]
x_test['output_score_qqh5'] = y_pred_test[:,6]
x_test['output_score_qqh6'] = y_pred_test[:,7]
x_test['output_score_qqh7'] = y_pred_test[:,8]
x_test['output_score_qqh8'] = y_pred_test[:,9]
x_test['output_score_qqh9'] = y_pred_test[:,10]
x_test['output_score_qqh10'] = y_pred_test[:,11]
x_test['output_score_wh'] = y_pred_test[:,12]
x_test['output_score_zh'] = y_pred_test[:,13]
x_test['output_score_tth'] = y_pred_test[:,14]
x_test['output_score_th'] = y_pred_test[:,15]

output_score_ggh = np.array(y_pred_test[:,0])
output_score_qqh0 = np.array(y_pred_test[:,1])
output_score_qqh1 = np.array(y_pred_test[:,2])
output_score_qqh2 = np.array(y_pred_test[:,3])
output_score_qqh3 = np.array(y_pred_test[:,4])
output_score_qqh4 = np.array(y_pred_test[:,5])
output_score_qqh5 = np.array(y_pred_test[:,6])
output_score_qqh6 = np.array(y_pred_test[:,7])
output_score_qqh7 = np.array(y_pred_test[:,8])
output_score_qqh8 = np.array(y_pred_test[:,9])
output_score_qqh9 = np.array(y_pred_test[:,10])
output_score_qqh10 = np.array(y_pred_test[:,11])
output_score_wh = np.array(y_pred_test[:,12])
output_score_zh = np.array(y_pred_test[:,13])
output_score_tth = np.array(y_pred_test[:,14])
output_score_th = np.array(y_pred_test[:,15])

x_test_ggh = x_test[x_test['proc'] == 'ggH']
x_test_qqh0 = x_test[x_test['proc'] == 'QQ2HQQ_FWDH']
x_test_qqh1 = x_test[x_test['proc'] == 'QQ2HQQ_0J']
x_test_qqh2 = x_test[x_test['proc'] == 'QQ2HQQ_1J']
x_test_qqh3 = x_test[x_test['proc'] == 'QQ2HQQ_GE2J_MJJ_0_60']
x_test_qqh4 = x_test[x_test['proc'] == 'QQ2HQQ_GE2J_MJJ_60_120']
x_test_qqh5 = x_test[x_test['proc'] == 'QQ2HQQ_GE2J_MJJ_120_350']
x_test_qqh6 = x_test[x_test['proc'] == 'QQ2HQQ_GE2J_MJJ_GT350_PTH_GT200']
x_test_qqh7 = x_test[x_test['proc'] == 'QQ2HQQ_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_0_25']
x_test_qqh8 = x_test[x_test['proc'] == 'QQ2HQQ_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_GT25']
x_test_qqh9 = x_test[x_test['proc'] == 'QQ2HQQ_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_0_25']
x_test_qqh10 = x_test[x_test['proc'] == 'QQ2HQQ_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_GT25']
x_test_wh = x_test[x_test['proc'] == 'WH']
x_test_zh = x_test[x_test['proc'] == 'ZH']
x_test_tth = x_test[x_test['proc'] == 'ttH']
x_test_th = x_test[x_test['proc'] == 'tH']

ggh_w = x_test_ggh['weight'] / x_test_ggh['weight'].sum()
qqh0_w = x_test_qqh0['weight'] / x_test_qqh0['weight'].sum()
qqh1_w = x_test_qqh1['weight'] / x_test_qqh1['weight'].sum()
qqh2_w = x_test_qqh2['weight'] / x_test_qqh2['weight'].sum()
qqh3_w = x_test_qqh3['weight'] / x_test_qqh3['weight'].sum()
qqh4_w = x_test_qqh4['weight'] / x_test_qqh4['weight'].sum()
qqh5_w = x_test_qqh5['weight'] / x_test_qqh5['weight'].sum()
qqh6_w = x_test_qqh6['weight'] / x_test_qqh6['weight'].sum()
qqh7_w = x_test_qqh7['weight'] / x_test_qqh7['weight'].sum()
qqh8_w = x_test_qqh8['weight'] / x_test_qqh8['weight'].sum()
qqh9_w = x_test_qqh9['weight'] / x_test_qqh9['weight'].sum()
qqh10_w = x_test_qqh10['weight'] / x_test_qqh10['weight'].sum()
wh_w = x_test_wh['weight'] / x_test_wh['weight'].sum()
zh_w = x_test_zh['weight'] / x_test_zh['weight'].sum()
tth_w = x_test_tth['weight'] / x_test_tth['weight'].sum()
th_w = x_test_th['weight'] / x_test_th['weight'].sum()
total_w = x_test['weight'] / x_test['weight'].sum()

#Accuracy Score
y_pred = y_pred_test.argmax(axis=1)
y_true = y_test.argmax(axis=1)
print 'Accuracy score: '
NNaccuracy = accuracy_score(y_true, y_pred)
print(NNaccuracy)

#Confusion Matrix
cm = confusion_matrix(y_true=y_true,y_pred=y_pred)


def plot_output_score(data='output_score_qqh', density=False,):
    #Can then change it to plotting proc
    print('Plotting',data)
    output_score_ggh = np.array(x_test_ggh[data])
    output_score_qqh0 = np.array(x_test_qqh0[data])
    output_score_qqh1 = np.array(x_test_qqh1[data])
    output_score_qqh2 = np.array(x_test_qqh2[data])
    output_score_qqh3 = np.array(x_test_qqh3[data])
    output_score_qqh4 = np.array(x_test_qqh4[data])
    output_score_qqh5 = np.array(x_test_qqh5[data])
    output_score_qqh6 = np.array(x_test_qqh6[data])
    output_score_qqh7 = np.array(x_test_qqh7[data])
    output_score_qqh8 = np.array(x_test_qqh8[data])
    output_score_qqh9 = np.array(x_test_qqh9[data])
    output_score_qqh10 = np.array(x_test_qqh10[data])
    output_score_wh = np.array(x_test_wh[data])
    output_score_zh = np.array(x_test_zh[data])
    output_score_tth = np.array(x_test_tth[data])
    output_score_th = np.array(x_test_th[data])

    fig, ax = plt.subplots()
    ax.hist(output_score_ggh, bins=50, label='ggH', histtype='step',weights=ggh_w)#,density=True) 
    #ax.hist(output_score_qqh0, bins=50, label='FWDH', histtype='step',weights=qqh0_w)
    ax.hist(output_score_qqh1, bins=50, label='0J', histtype='step',weights=qqh1_w)
    ax.hist(output_score_qqh2, bins=50, label='1J', histtype='step',weights=qqh2_w)
    ax.hist(output_score_qqh3, bins=50, label='MJJ_0_60', histtype='step',weights=qqh3_w)
    ax.hist(output_score_qqh4, bins=50, label='MJJ_60_120', histtype='step',weights=qqh4_w)
    ax.hist(output_score_qqh5, bins=50, label='MJJ_120_350', histtype='step',weights=qqh5_w)
    ax.hist(output_score_qqh6, bins=50, label='MJJ_GT350_PTH_GT200', histtype='step',weights=qqh6_w)
    ax.hist(output_score_qqh7, bins=50, label='MJJ_350_700_PTH_0_200_PTHJJ_0_25', histtype='step',weights=qqh7_w)
    ax.hist(output_score_qqh8, bins=50, label='MJJ_350_700_PTH_0_200_PTHJJ_GT25', histtype='step',weights=qqh8_w)
    ax.hist(output_score_qqh9, bins=50, label='MJJ_GT700_PTH_0_200_PTHJJ_0_25', histtype='step',weights=qqh9_w)
    ax.hist(output_score_qqh10, bins=50, label='MJJ_GT700_PTH_0_200_PTHJJ_GT25', histtype='step',weights=qqh10_w)
    ax.hist(output_score_wh, bins=50, label='WH', histtype='step',weights=wh_w) #density=True) 
    ax.hist(output_score_zh, bins=50, label='ZH', histtype='step',weights=zh_w) #density=True) 
    ax.hist(output_score_tth, bins=50, label='ttH', histtype='step',weights=tth_w) #density=True)
    ax.hist(output_score_th, bins=50, label='tH', histtype='step',weights=th_w) #density=True)
    plt.legend()
    plt.title('Output Score')
    plt.ylabel('Fraction of Events')
    plt.xlabel('NN Score')
    name = 'plotting/NN_plots/NN_Fifteenclass_'+data
    plt.savefig(name, dpi = 1200)


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
    name = 'plotting/NN_plots/NN_Fifteenclass_Confusion_Matrix'
    fig.savefig(name)

plot_output_score(data='output_score_ggh')
#plot_output_score(data='output_score_qqh0')
plot_output_score(data='output_score_qqh1')
plot_output_score(data='output_score_qqh2')
plot_output_score(data='output_score_qqh3')
plot_output_score(data='output_score_qqh4')
plot_output_score(data='output_score_qqh5')
plot_output_score(data='output_score_qqh6')
plot_output_score(data='output_score_qqh7')
plot_output_score(data='output_score_qqh8')
plot_output_score(data='output_score_qqh9')
plot_output_score(data='output_score_qqh10')
plot_output_score(data='output_score_wh')
plot_output_score(data='output_score_zh')
plot_output_score(data='output_score_tth')
plot_output_score(data='output_score_th')

#plot_accuracy()
#plot_loss()
plot_confusion_matrix(cm,binNames,normalize=True)

#roc_score()


#save as a pickle file
#trainTotal.to_pickle('%s/nClassNNTotal.pkl'%frameDir)
#print 'frame saved as %s/nClassNNTotal.pkl'%frameDir
#Read in pickle file
#trainTotal = pd.read_pickle(opts.dataFrame)
#print 'Successfully loaded the dataframe'