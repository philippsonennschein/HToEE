
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
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score

#Define key quantities, use to tune BDT
num_estimators = 300
val_split = 0.3
learning_rate = 0.001

binNames = ['ggH','VBF'] 
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
'subleadJetDiphoDEta', 'subsubleadJetPUJID', 'subsubleadJetPt',
'subsubleadJetEn', 'subsubleadJetEta', 'subsubleadJetPhi',
'subsubleadJetMass', 'subsubleadJetBTagScore','nSoftJets','metPt','metPhi','metSumET']

train_vars.append('proc')
train_vars.append('weight')

dataframes = []
dataframes.append(pd.read_csv('2017/MC/DataFrames/ggH_VBF_BDT_df_2017.csv'))
dataframes.append(pd.read_csv('2017/MC/DataFrames/VBF_VBF_BDT_df_2017.csv'))
df = pd.concat(dataframes, sort=False, axis=0 )

data = df[train_vars]

data = data[data.diphotonMass>100.]
data = data[data.diphotonMass<180.]
data = data[data.leadPhotonPtOvM>0.333]
data = data[data.subleadPhotonPtOvM>0.25]


#Define the procs as the labels
#ggh: 0, VBF:1, VH: 2, ttH: 3
num_categories = data['proc'].nunique()
y_train_labels_num, y_train_labels_def = pd.factorize(data['proc'])

#Label definition:
print('Label Definition:')
label_def = []
for i in range(num_categories):
    label_def.append([i,y_train_labels_def[i]])
    print(i,y_train_labels_def[i])

data['proc_num'] = y_train_labels_num

y_train_labels = np.array(data['proc'])
y_train_labels_num = np.array(data['proc_num'])
y_train_labels_hot = np_utils.to_categorical(y_train_labels_num, num_classes=num_categories)
weights = np.array(data['weight'])

data = data.drop(columns=['proc'])
data = data.drop(columns=['proc_num'])
data = data.drop(columns=['weight'])


#With num
x_train, x_test, y_train, y_test, train_w, test_w, proc_arr_train, proc_arr_test = train_test_split(data, y_train_labels_num, weights, y_train_labels, test_size = val_split, shuffle = True)
#With hot
#x_train, x_test, y_train, y_test, train_w, test_w, proc_arr_train, proc_arr_test = train_test_split(data, y_train_labels_hot, weights, y_train_labels, test_size = val_split, shuffle = True)

#Before n_estimators = 100, maxdepth=4, gamma = 1
#Improved n_estimators = 300, maxdepth = 7, gamme = 4
clf = xgb.XGBClassifier(objective='binary:logistic', n_estimators=100, 
                            eta=0.1, maxDepth=4, min_child_weight=0.01, 
                            subsample=0.6, colsample_bytree=0.6, gamma=1)

#Equalizing weights
train_w_df = pd.DataFrame()
train_w = 300 * train_w # to make loss function O(1)
train_w_df['weight'] = train_w
train_w_df['proc'] = proc_arr_train
vbf_sum_w = train_w_df[train_w_df['proc'] == 'VBF']['weight'].sum()
ggh_sum_w = train_w_df[train_w_df['proc'] == 'ggH']['weight'].sum()
train_w_df.loc[train_w_df.proc == 'VBF','weight'] = (train_w_df[train_w_df['proc'] == 'VBF']['weight'] * ggh_sum_w / vbf_sum_w)
train_w = np.array(train_w_df['weight'])

print (' Training classifier...')
clf = clf.fit(x_train, y_train, sample_weight=train_w)
print ('Finished Training classifier!')

#print('Saving Classifier...')
#pickle.dump(clf, open("models/Multi_BDT_clf.pickle.dat"))
#print('Finished Saving classifier!')

#print('loading classifier:')
#clf = pickle.load(open("models/Multi_BDT_clf.pickle.dat", "rb"))

y_pred_test = clf.predict_proba(x_test)

x_test['proc'] = proc_arr_test
x_test['weight'] = test_w
x_test['output_score_ggh'] = y_pred_test[:,0]
x_test['output_score_vbf'] = y_pred_test[:,1]

output_score_ggh = np.array(y_pred_test[:,0])
output_score_vbf = np.array(y_pred_test[:,1])

x_test_ggh = x_test[x_test['proc'] == 'ggH']
x_test_vbf = x_test[x_test['proc'] == 'VBF']

ggh_w = x_test_ggh['weight'] / x_test_ggh['weight'].sum()
vbf_w = x_test_vbf['weight'] / x_test_vbf['weight'].sum()

#Accuracy Score
y_pred = y_pred_test.argmax(axis=1)
#y_true = y_test.argmax(axis=1)
y_true = y_test
print('Accuracy score: ')
NNaccuracy = accuracy_score(y_true, y_pred)
print(NNaccuracy)

#Confusion Matrix
#m = confusion_matrix(y_true=y_true,y_pred=y_pred)

#Calculations for the ROC curve
y_true_ggh = np.where(y_true == 0, 1, 0)
y_pred_ggh = np.where(y_pred == 0, 1, 0)
y_pred_ggh_prob = []
for i in range(len(y_pred_ggh)):
    if y_pred_ggh[i] == 0:
        y_pred_ggh_prob.append(0)
    elif y_pred_ggh[i] == 1:
        y_pred_ggh_prob.append(output_score_ggh[i])
y_true_vbf = np.where(y_true == 1, 1, 0)
y_pred_vbf = np.where(y_pred == 1, 1, 0)
y_pred_vbf_prob = []
for i in range(len(y_pred_vbf)):
    if y_pred_vbf[i] == 0:
        y_pred_vbf_prob.append(0)
    elif y_pred_vbf[i] == 1:
        y_pred_vbf_prob.append(output_score_vbf[i])


#Plotting:
def roc_score(y_true = y_true, y_pred = y_pred_test):

    fpr_keras_ggh, tpr_keras_ggh, thresholds_keras_ggh = roc_curve(y_true_ggh, y_pred_ggh_prob)
    auc_keras_test_ggh = roc_auc_score(y_true_ggh, y_pred_ggh_prob)
    print("Area under ROC curve for ggH (test): ", auc_keras_test_ggh)

    fpr_keras_vbf, tpr_keras_vbf, thresholds_keras_vbf = roc_curve(y_true_vbf, y_pred_vbf_prob)
    auc_keras_test_vbf = roc_auc_score(y_true_vbf, y_pred_vbf)
    print("Area under ROC curve for VBF (test): ", auc_keras_test_vbf)

    print("Plotting ROC Score")
    fig, ax = plt.subplots()
    ax.plot(fpr_keras_ggh, tpr_keras_ggh, label = 'ggH (area = %0.2f)'%auc_keras_test_ggh)
    ax.plot(fpr_keras_vbf, tpr_keras_vbf, label = 'VBF (area = %0.2f)'%auc_keras_test_vbf)
    ax.legend()
    ax.set_xlabel('Background Efficiency', ha='right', x=1, size=9)
    ax.set_ylabel('Signal Efficiency',ha='right', y=1, size=9)
    ax.grid(True, 'major', linestyle='solid', color='grey', alpha=0.5)
    name = 'plotting/BDT_plots/BDT_Multi_ROC_curve'
    plt.savefig(name, dpi = 200)

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
    name = 'plotting/BDT_plots/BDT_Multi_Confusion_Matrix'
    fig.savefig(name)


def plot_output_score(data='output_score_vbf', density=False,):
    #Can then change it to plotting proc
    print('Plotting',data)
    output_score_ggh = np.array(x_test_ggh[data])
    output_score_vbf = np.array(x_test_vbf[data])

    fig, ax = plt.subplots()
    ax.hist(output_score_ggh, bins=50, label='ggH', histtype='step',weights=ggh_w)#,density=True) 
    ax.hist(output_score_vbf, bins=50, label='VBF', histtype='step',weights=vbf_w) #density=True)
    plt.legend()
    plt.title('Output Score')
    plt.ylabel('Fraction of Events')
    plt.xlabel('BDT Score')
    name = 'plotting/BDT_plots/BDT_Binary_'+data
    plt.savefig(name, dpi = 200)

plot_output_score(data='output_score_vbf')
plot_output_score(data='output_score_ggh')

#roc_score()

#plot_confusion_matrix(cm,binNames,normalize=True)
