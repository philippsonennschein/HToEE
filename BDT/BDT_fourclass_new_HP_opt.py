
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

#def roc_score(y_true = y_true, y_pred = y_pred_test):
def roc_score():

    fpr_keras_ggh, tpr_keras_ggh, thresholds_keras_ggh = roc_curve(y_true_ggh, y_pred_ggh_prob,sample_weight=ggh_w)
    fpr_keras_ggh.sort()
    tpr_keras_ggh.sort()
    auc_keras_test_ggh = auc(fpr_keras_ggh,tpr_keras_ggh)
    print("Area under ROC curve for ggH (test): ", auc_keras_test_ggh)

    fpr_keras_qqh, tpr_keras_qqh, thresholds_keras_qqh = roc_curve(y_true_qqh, y_pred_qqh_prob,sample_weight=qqh_w)
    fpr_keras_qqh.sort()
    tpr_keras_qqh.sort()
    auc_keras_test_qqh = auc(fpr_keras_qqh,tpr_keras_qqh)
    print("Area under ROC curve for qqH (test): ", auc_keras_test_qqh)

    fpr_keras_vh, tpr_keras_vh, thresholds_keras_vh = roc_curve(y_true_vh, y_pred_vh_prob,sample_weight=vh_w)
    fpr_keras_vh.sort()
    tpr_keras_vh.sort()
    auc_keras_test_vh = auc(fpr_keras_vh,tpr_keras_vh)
    print("Area under ROC curve for VH (test): ", auc_keras_test_vh)

    fpr_keras_tth, tpr_keras_tth, thresholds_keras_tth = roc_curve(y_true_tth, y_pred_tth_prob,sample_weight=tth_w)
    fpr_keras_tth.sort()
    tpr_keras_tth.sort()
    auc_keras_test_tth = auc(fpr_keras_tth,tpr_keras_tth)
    print("Area under ROC curve for ttH (test): ", auc_keras_test_tth)

    return auc_keras_test_ggh, auc_keras_test_qqh, auc_keras_test_vh, auc_keras_test_tth

#Define key quantities, use to tune BDT
test_split = 0.2
#learning_rate = 0.001

binNames = ['ggH','qqH','VH','ttH'] 
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
train_vars.append('HTXS_stage_0')
#train_vars.append('HTXS_stage1_2_cat_pTjet30GeV')

dataframes = []
dataframes.append(pd.read_csv('2017/MC/DataFrames/ggH_VBF_BDT_df_2017.csv'))
dataframes.append(pd.read_csv('2017/MC/DataFrames/VBF_VBF_BDT_df_2017.csv'))
dataframes.append(pd.read_csv('2017/MC/DataFrames/VH_VBF_BDT_df_2017.csv'))
dataframes.append(pd.read_csv('2017/MC/DataFrames/ttH_VBF_BDT_df_2017.csv'))
df = pd.concat(dataframes, sort=False, axis=0 )

data = df[train_vars]

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
    elif i == 30 or i == 31 or i == 40 or i == 41:
        proc_new.append('VH')
    elif i == 60 or i == 61:
        proc_new.append('ttH')
    else:
        proc_new.append(i)
        print(i)
data['proc_new'] = proc_new

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

data = data.drop(columns=['proc'])
data = data.drop(columns=['proc_num'])
data = data.drop(columns=['weight'])
data = data.drop(columns=['HTXS_stage_0'])
data = data.drop(columns=['proc_new'])
#data = data.drop(columns=['HTXS_stage1_2_cat_pTjet30GeV'])

num_estimators = [200,300,400]
eta = [0.1,0.05,0.01]
maxDepth = [5,6,7]

#num_estimators = [200]
#eta = [0.1]
#maxDepth = [5]

num_combinations = len(num_estimators) * len(eta) * len(maxDepth)

scores = []
accuracy = []
count = 0.0

for num_estimators_value in num_estimators:
    for eta_value in eta:
        for maxDepth_value in maxDepth:

            count += 1.0
            print('Progress:',count,'/',num_combinations)

            x_train, x_test, y_train, y_test, train_w, test_w, proc_arr_train, proc_arr_test = train_test_split(data, y_train_labels_num, weights, y_train_labels, test_size = test_split, shuffle = True)

            clf = xgb.XGBClassifier(objective='multi:softprob', n_estimators=num_estimators_value, 
                                        eta=eta_value, maxDepth=maxDepth, min_child_weight=0.01, 
                                        subsample=0.6, colsample_bytree=0.6, gamma=4,
                                        num_class=4)

            #Equalizing weights
            train_w_df = pd.DataFrame()
            train_w = 300 * train_w # to make loss function O(1)
            train_w_df['weight'] = train_w
            train_w_df['proc'] = proc_arr_train
            qqh_sum_w = train_w_df[train_w_df['proc'] == 'qqH']['weight'].sum()
            ggh_sum_w = train_w_df[train_w_df['proc'] == 'ggH']['weight'].sum()
            vh_sum_w = train_w_df[train_w_df['proc'] == 'VH']['weight'].sum()
            tth_sum_w = train_w_df[train_w_df['proc'] == 'ttH']['weight'].sum()
            train_w_df.loc[train_w_df.proc == 'qqH','weight'] = (train_w_df[train_w_df['proc'] == 'qqH']['weight'] * ggh_sum_w / qqh_sum_w)
            train_w_df.loc[train_w_df.proc == 'VH','weight'] = (train_w_df[train_w_df['proc'] == 'VH']['weight'] * ggh_sum_w / vh_sum_w)
            train_w_df.loc[train_w_df.proc == 'ttH','weight'] = (train_w_df[train_w_df['proc'] == 'ttH']['weight'] * ggh_sum_w / tth_sum_w)
            train_w = np.array(train_w_df['weight'])

            print (' Training classifier...')
            clf = clf.fit(x_train, y_train, sample_weight=train_w)
            print ('Finished Training classifier!')

            y_pred_test = clf.predict_proba(x_test)

            x_test['proc'] = proc_arr_test
            x_test['weight'] = test_w
            x_test['output_score_ggh'] = y_pred_test[:,0]
            x_test['output_score_qqh'] = y_pred_test[:,1]
            x_test['output_score_vh'] = y_pred_test[:,2]
            x_test['output_score_tth'] = y_pred_test[:,3]

            output_score_ggh = np.array(y_pred_test[:,0])
            output_score_qqh = np.array(y_pred_test[:,1])
            output_score_vh = np.array(y_pred_test[:,2])
            output_score_tth = np.array(y_pred_test[:,3])

            x_test_ggh = x_test[x_test['proc'] == 'ggH']
            x_test_qqh = x_test[x_test['proc'] == 'qqH']
            x_test_vh = x_test[x_test['proc'] == 'VH']
            x_test_tth = x_test[x_test['proc'] == 'ttH']

            ggh_w = x_test_ggh['weight'] / x_test_ggh['weight'].sum()
            qqh_w = x_test_qqh['weight'] / x_test_qqh['weight'].sum()
            vh_w = x_test_vh['weight'] / x_test_vh['weight'].sum()
            tth_w = x_test_tth['weight'] / x_test_tth['weight'].sum()

            #Accuracy Score
            y_pred = y_pred_test.argmax(axis=1)
            y_true = y_test
            print 'Accuracy score: '
            NNaccuracy = accuracy_score(y_true, y_pred)
            print('HP values:',num_estimators_value, eta_value, maxDepth_value)
            print(NNaccuracy)

            accuracy.append(NNaccuracy)
            scores.append([count, NNaccuracy, num_estimators_value, eta_value, maxDepth_value])

index = np.argmax(accuracy)
print('Best HP Combination:')
print(scores[index])

a_file = open("BDT/HP_opt.txt", "w")
for row in scores:
    np.savetxt(a_file, row)
a_file.close()