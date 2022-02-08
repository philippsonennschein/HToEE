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
num_epochs = 12
batch_size = 400
val_split = 0.3
learning_rate = 0.001

epochs = np.linspace(1,num_epochs,num_epochs,endpoint=True).astype(int) #For plotting
binNames = ['ggH','qqH','ZH','WH','ttH'] 
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
'metPt','metPhi','metSumET']

#Add proc and weight to shuffle with data
train_vars.append('proc')
train_vars.append('weight')
train_vars.append('HTXS_stage_0')
#train_vars.append('HTXS_stage1_2_cat_pTjet30GeV')

#Load the dataframe
dataframes = []
dataframes.append(pd.read_csv('2017/MC/DataFrames/ggH_VBF_BDT_df_2017.csv'))
dataframes.append(pd.read_csv('2017/MC/DataFrames/VBF_VBF_BDT_df_2017.csv'))
dataframes.append(pd.read_csv('2017/MC/DataFrames/VH_VBF_BDT_df_2017.csv'))
dataframes.append(pd.read_csv('2017/MC/DataFrames/ttH_VBF_BDT_df_2017.csv'))
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

proc_temp = np.array(data['HTXS_stage_0'])
proc_new = []
for i in proc_temp:
    if i == 10 or i == 11:
        proc_new.append('ggH')
    elif i == 20 or i == 21 or i == 22 or i == 23:
        proc_new.append('qqH')
    elif i == 30 or i == 31:
        proc_new.append('WH')
    elif i == 40 or i == 41:
        proc_new.append('ZH')
    elif i == 60 or i == 61:
        proc_new.append('ttH')
    else:
        proc_new.append(i)
        print(i)
data['proc_new'] = proc_new

#Define the procs as the labels
#ggh: 0, VBF:1, VH: 2, ttH: 3
num_categories = data['proc_new'].nunique()
y_train_labels_num, y_train_labels_def = pd.factorize(data['proc_new'])

#Label definition:
print('Label Definition:')
label_def = []
for i in range(num_categories):
    label_def.append([i,y_train_labels_def[i]])
    print(i,y_train_labels_def[i])

data['proc_num'] = y_train_labels_num

#Shuffle dataframe
data = data.sample(frac=1)

y_train_labels = np.array(data['proc_new'])
y_train_labels_num = np.array(data['proc_num'])
y_train_labels_hot = np_utils.to_categorical(y_train_labels_num, num_classes=num_categories)
weights = np.array(data['weight'])

#Remove proc after shuffle
data = data.drop(columns=['proc'])
data = data.drop(columns=['weight'])
data = data.drop(columns=['proc_num'])
data = data.drop(columns=['HTXS_stage_0'])
data = data.drop(columns=['proc_new'])

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
model=Sequential([Dense(units=100,input_shape=(num_inputs,),activation='relu'),
                Dense(units=100,activation='relu'),
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
qqh_sum_w = train_w_df[train_w_df['proc'] == 'qqH']['weight'].sum()
ggh_sum_w = train_w_df[train_w_df['proc'] == 'ggH']['weight'].sum()
wh_sum_w = train_w_df[train_w_df['proc'] == 'WH']['weight'].sum()
zh_sum_w = train_w_df[train_w_df['proc'] == 'ZH']['weight'].sum()
tth_sum_w = train_w_df[train_w_df['proc'] == 'ttH']['weight'].sum()
train_w_df.loc[train_w_df.proc == 'qqH','weight'] = (train_w_df[train_w_df['proc'] == 'qqH']['weight'] * ggh_sum_w / qqh_sum_w)
train_w_df.loc[train_w_df.proc == 'WH','weight'] = (train_w_df[train_w_df['proc'] == 'WH']['weight'] * ggh_sum_w / wh_sum_w)
train_w_df.loc[train_w_df.proc == 'ZH','weight'] = (train_w_df[train_w_df['proc'] == 'ZH']['weight'] * ggh_sum_w / zh_sum_w)
train_w_df.loc[train_w_df.proc == 'ttH','weight'] = (train_w_df[train_w_df['proc'] == 'ttH']['weight'] * ggh_sum_w / tth_sum_w)
train_w = np.array(train_w_df['weight'])

#Training the model
history = model.fit(x=x_train,y=y_train,batch_size=batch_size,epochs=num_epochs,sample_weight=train_w,shuffle=True,verbose=2)

# Output Score
y_pred_test = model.predict_proba(x=x_test)
x_test['proc'] = proc_arr_test
x_test['weight'] = test_w
x_test['output_score_ggh'] = y_pred_test[:,0]
x_test['output_score_qqh'] = y_pred_test[:,1]
x_test['output_score_wh'] = y_pred_test[:,2]
x_test['output_score_zh'] = y_pred_test[:,3]
x_test['output_score_tth'] = y_pred_test[:,4]

output_score_ggh = np.array(y_pred_test[:,0])
output_score_qqh = np.array(y_pred_test[:,1])
output_score_wh = np.array(y_pred_test[:,2])
output_score_zh = np.array(y_pred_test[:,3])
output_score_tth = np.array(y_pred_test[:,4])

x_test_ggh = x_test[x_test['proc'] == 'ggH']
x_test_qqh = x_test[x_test['proc'] == 'qqH']
x_test_wh = x_test[x_test['proc'] == 'WH']
x_test_zh = x_test[x_test['proc'] == 'ZH']
x_test_tth = x_test[x_test['proc'] == 'ttH']

ggh_w = x_test_ggh['weight'] / x_test_ggh['weight'].sum()
qqh_w = x_test_qqh['weight'] / x_test_qqh['weight'].sum()
wh_w = x_test_wh['weight'] / x_test_wh['weight'].sum()
zh_w = x_test_zh['weight'] / x_test_zh['weight'].sum()
tth_w = x_test_tth['weight'] / x_test_tth['weight'].sum()

#Accuracy Score
y_pred = y_pred_test.argmax(axis=1)
y_true = y_test.argmax(axis=1)
print 'Accuracy score: '
NNaccuracy = accuracy_score(y_true, y_pred)
print(NNaccuracy)

#Confusion Matrix
cm = confusion_matrix(y_true=y_true,y_pred=y_pred)

#ROC computations
y_true_ggh = np.where(y_true == 0, 1, 0)
y_pred_ggh = np.where(y_pred == 0, 1, 0)
y_pred_ggh_prob = []
for i in range(len(y_pred_ggh)):
    if y_pred_ggh[i] == 0:
        y_pred_ggh_prob.append(0)
    elif y_pred_ggh[i] == 1:
        y_pred_ggh_prob.append(output_score_ggh[i])
y_true_qqh = np.where(y_true == 1, 1, 0)
y_pred_qqh = np.where(y_pred == 1, 1, 0)
y_pred_qqh_prob = []
for i in range(len(y_pred_qqh)):
    if y_pred_qqh[i] == 0:
        y_pred_qqh_prob.append(0)
    elif y_pred_qqh[i] == 1:
        y_pred_qqh_prob.append(output_score_qqh[i])
y_true_wh = np.where(y_true == 2, 1, 0)
y_pred_wh = np.where(y_pred == 2, 1, 0)
y_pred_wh_prob = []
for i in range(len(y_pred_wh)):
    if y_pred_wh[i] == 0:
        y_pred_wh_prob.append(0)
    elif y_pred_wh[i] == 1:
        y_pred_wh_prob.append(output_score_wh[i])
y_true_zh = np.where(y_true == 2, 1, 0)
y_pred_zh = np.where(y_pred == 2, 1, 0)
y_pred_zh_prob = []
for i in range(len(y_pred_zh)):
    if y_pred_zh[i] == 0:
        y_pred_zh_prob.append(0)
    elif y_pred_zh[i] == 1:
        y_pred_zh_prob.append(output_score_zh[i])
y_true_tth = np.where(y_true == 3, 1, 0)
y_pred_tth = np.where(y_pred == 3, 1, 0)
y_pred_tth_prob = []
for i in range(len(y_pred_tth)):
    if y_pred_tth[i] == 0:
        y_pred_tth_prob.append(0)
    elif y_pred_tth[i] == 1:
        y_pred_tth_prob.append(output_score_tth[i])

def roc_score(y_true = y_true, y_pred = y_pred_test):

    fpr_keras_ggh, tpr_keras_ggh, thresholds_keras_ggh = roc_curve(y_true_ggh, y_pred_ggh_prob)
    auc_keras_test_ggh = roc_auc_score(y_true_ggh, y_pred_ggh_prob)
    print("Area under ROC curve for ggH (test): ", auc_keras_test_ggh)

    fpr_keras_qqh, tpr_keras_qqh, thresholds_keras_qqh = roc_curve(y_true_qqh, y_pred_qqh_prob)
    auc_keras_test_qqh = roc_auc_score(y_true_qqh, y_pred_qqh)
    print("Area under ROC curve for qqH (test): ", auc_keras_test_qqh)

    fpr_keras_wh, tpr_keras_wh, thresholds_keras_wh = roc_curve(y_true_wh, y_pred_wh_prob)
    auc_keras_test_wh = roc_auc_score(y_true_wh, y_pred_wh)
    print("Area under ROC curve for WH (test): ", auc_keras_test_wh)

    fpr_keras_zh, tpr_keras_zh, thresholds_keras_zh = roc_curve(y_true_zh, y_pred_zh_prob)
    auc_keras_test_zh = roc_auc_score(y_true_zh, y_pred_zh)
    print("Area under ROC curve for ZH (test): ", auc_keras_test_zh)

    fpr_keras_tth, tpr_keras_tth, thresholds_keras_tth = roc_curve(y_true_tth, y_pred_tth_prob)
    auc_keras_test_tth = roc_auc_score(y_true_tth, y_pred_tth)
    print("Area under ROC curve for ttH (test): ", auc_keras_test_tth)

    print("Plotting ROC Score")
    fig, ax = plt.subplots()
    ax.plot(fpr_keras_ggh, tpr_keras_ggh, label = 'ggH (area = %0.2f)'%auc_keras_test_ggh)
    ax.plot(fpr_keras_qqh, tpr_keras_qqh, label = 'qqH (area = %0.2f)'%auc_keras_test_qqh)
    ax.plot(fpr_keras_wh, tpr_keras_wh, label = 'WH (area = %0.2f)'%auc_keras_test_wh)
    ax.plot(fpr_keras_zh, tpr_keras_zh, label = 'ZH (area = %0.2f)'%auc_keras_test_zh)
    ax.plot(fpr_keras_tth, tpr_keras_tth, label = 'ttH (area = %0.2f)'%auc_keras_test_tth)
    ax.legend()
    ax.set_xlabel('Background Efficiency', ha='right', x=1, size=9)
    ax.set_ylabel('Signal Efficiency',ha='right', y=1, size=9)
    ax.grid(True, 'major', linestyle='solid', color='grey', alpha=0.5)
    name = 'plotting/NN_plots/NN_Multi_ROC_curve'
    plt.savefig(name, dpi = 200)

roc_score()

#Need to do other 3 plots too and include the MC weights!
#Change it to be proc = 'VBF' and then do 'output_score_%'.format(proc)
#Can then loop through the y_train_labels_def and set data = i to plot all possible production modes

# VBF
def plot_output_score(data='output_score_qqh', density=False,):
    #Can then change it to plotting proc
    print('Plotting',data)
    output_score_ggh = np.array(x_test_ggh[data])
    output_score_qqh = np.array(x_test_qqh[data])
    output_score_wh = np.array(x_test_wh[data])
    output_score_zh = np.array(x_test_zh[data])
    output_score_tth = np.array(x_test_tth[data])

    fig, ax = plt.subplots()
    ax.hist(output_score_ggh, bins=50, label='ggH', histtype='step',weights=ggh_w)#,density=True) 
    ax.hist(output_score_qqh, bins=50, label='qqH', histtype='step',weights=qqh_w) #density=True)
    ax.hist(output_score_wh, bins=50, label='WH', histtype='step',weights=wh_w) #density=True) 
    ax.hist(output_score_zh, bins=50, label='ZH', histtype='step',weights=zh_w) #density=True) 
    ax.hist(output_score_tth, bins=50, label='ttH', histtype='step',weights=tth_w) #density=True)
    plt.legend()
    plt.title('Output Score')
    plt.ylabel('Fraction of Events')
    plt.xlabel('NN Score')
    name = 'plotting/NN_plots/NN_Fiveclass_'+data
    plt.savefig(name, dpi = 200)

#Plotting:
#Plot accuracy
def plot_accuracy():
    #val_accuracy = history.history['val_acc']
    accuracy = history.history['acc']
    fig, ax = plt.subplots(1)
    #plt.plot(epochs,val_accuracy,label='Validation')
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
    #val_loss = history.history['val_loss']
    loss = history.history['loss']
    fig, ax = plt.subplots(1)
    #plt.plot(epochs,val_loss,label='Validation')
    plt.plot(epochs,loss,label='Train')
    plt.title('Loss function')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.xticks(epochs)
    plt.legend()
    name = 'plotting/NN_plots/NN_Loss'
    fig.savefig(name)


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
    name = 'plotting/NN_plots/NN_Fiveclass_Confusion_Matrix'
    fig.savefig(name)

plot_output_score(data='output_score_qqh')
plot_output_score(data='output_score_ggh')
plot_output_score(data='output_score_wh')
plot_output_score(data='output_score_zh')
plot_output_score(data='output_score_tth')

#plot_accuracy()
#plot_loss()
plot_confusion_matrix(cm,binNames,normalize=True)


#save as a pickle file
#trainTotal.to_pickle('%s/nClassNNTotal.pkl'%frameDir)
#print 'frame saved as %s/nClassNNTotal.pkl'%frameDir
#Read in pickle file
#trainTotal = pd.read_pickle(opts.dataFrame)
#print 'Successfully loaded the dataframe'