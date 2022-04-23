 
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
import seaborn as sns
import scipy.stats as sp
from more_itertools import sort_together
 

map_def = [['ggH',10,11],['qqH',20,21,22,23],['VH',30,31,40,41],['ttH',60,61],['tH',80,81]]

binNames = ['ggH','qqH','VH','ttH','tH'] 

dataframes = []
dataframes.append(pd.read_csv('2017/MC/DataFrames/ggH_VBF_BDT_df_2017.csv'))
dataframes.append(pd.read_csv('2017/MC/DataFrames/VBF_VBF_BDT_df_2017.csv'))
dataframes.append(pd.read_csv('2017/MC/DataFrames/VH_VBF_BDT_df_2017.csv'))
dataframes.append(pd.read_csv('2017/MC/DataFrames/ttH_VBF_BDT_df_2017.csv'))
#dataframes.append(pd.read_csv('2017/MC/DataFrames/tHq_VBF_BDT_df_2017.csv', nrows = 100000))
#dataframes.append(pd.read_csv('2017/MC/DataFrames/tHW_VBF_BDT_df_2017.csv', nrows = 100000))
df = pd.concat(dataframes, sort=False, axis=0 )

train_vars = ['diphotonPt','diphotonMass', 'diphotonEta','diphotonPhi',
     'dijetMass', 'dijetCentrality',
     'dijetPt','dijetEta','dijetPhi',
     'leadPhotonEta', 'leadPhotonEn', 'leadPhotonPt', 'leadPhotonPhi', 
     'leadJetPt', 'leadJetBTagScore', 'leadJetMass',
     'leadJetEn','leadJetEta','leadJetPhi',
     'metPt','metPhi','metSumET',
     'nSoftJets',
     'leadElectronEn', 'leadElectronPt', 'leadElectronEta', 'leadElectronPhi',
     'leadMuonEn', 'leadMuonPt', 'leadMuonEta', 'leadMuonPhi',
     'leadPhotonPtOvM','subleadPhotonPtOvM',
     'dijetAbsDEta', 'dijetDPhi'
     ]
train_vars.append('weight')
train_vars.append('proc')
train_vars.append('HTXS_stage_0')
#train_vars.append('HTXS_stage1_2_cat_pTjet30GeV')

data = df[train_vars]

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

#data['proc_new'] = mapping(map_list=map_def,stage=data['HTXS_stage_0'])

#Creating dR variable
data['DR'] = np.sqrt(data['diphotonEta']**2 + data['diphotonPhi']**2)


data = data.drop(columns=['HTXS_stage_0'])

list_minus1_default = ['leadJetBTagScore'] # ,'subleadJetBTagScore','subsubleadJetBTagScore']
for variable in list_minus1_default:
  data[variable] = np.where(data[variable] == -1,-999,data[variable])

# -------------------------------
list_variables1 = [
     'DR',
     'dijetCentrality',
     'dijetAbsDEta', 'leadJetBTagScore',
     'metPt','metPhi','metSumET',
     'nSoftJets',
     'diphotonPt', 'dijetPt', 'leadElectronPt', 'leadMuonPt',
     'diphotonEta',  'dijetEta','leadElectronEta',
     'diphotonPhi', 'dijetPhi','leadElectronPhi'
]

variables_names = [
     'D R',
     'Dijet Centrality',
     'Dijet Abs Delta $\eta$', 
     'Lead Jet B-Tag Score',
     'Met $P_T$',
     'Met $\phi$',
     'Met Sum $E_T$',
     'Number of Soft Jets',
     'Diphoton $P_T$', 
     'Dijet $P_T$', 
     'Lead Electron $P_T$', 
     'Lead Muon $P_T$',
     'Diphoton $\eta$', 
     'Dijet $\eta$',
     'Lead Electron $\eta$',
     'Diphoton $\phi$', 
     'Dijet $\phi$',
     'Lead Electron $\phi$'
]

list_variables2 = list_variables1
  
'''  
  ['diphotonPt', 'diphotonEta','diphotonPhi',
  'dijetMass', 'dijetCentrality',
  'dijetPt','dijetEta','dijetPhi',
  'leadPhotonEta', 'leadPhotonEn', 'leadPhotonPt', 'leadPhotonPhi', 
  'leadJetPt', 'leadJetBTagScore', 
  'leadJetEn','leadJetEta','leadJetPhi',
  #'metPt','metPhi','metSumET',
  #'nSoftJets',
  'leadElectronPt', 'leadElectronEta', 'leadElectronPhi',
  'leadMuonPt', 'leadMuonEta', 'leadMuonPhi','DR']
'''

 
#list_plots is a list that contains all the variable combinations that have been previously plotted
#Variable is used to avoid the same pair to be plotted twice
list_plots = []
lower_lim = -1000
upper_lim = 50000
 
y_len = len(list_variables1)
x_len = len(list_variables2)

pearson_matrix_VBF = np.zeros((y_len,x_len))
pearson_matrix_VH = np.zeros((y_len,x_len))
pearson_matrix_ggH = np.zeros((y_len,x_len))
pearson_matrix_ttH = np.zeros((y_len,x_len))
pearson_matrix_combined = np.zeros((y_len,x_len))

pearson_matrix_variables = np.zeros((y_len,x_len),dtype=object)

i = 0
j = 0
 
for variable1 in list_variables1:
  j = 0

  vbf_sig_00 = np.array(data[data['proc'] == 'VBF'][variable1])
  vbf_sig_1_w = np.array(data[data['proc'] == 'VBF']['weight'])[(vbf_sig_00 > lower_lim) & (vbf_sig_00 <upper_lim)]
  vbf_sig_1_final = vbf_sig_00[(vbf_sig_00 > lower_lim) & (vbf_sig_00 < upper_lim)]
  vh_sig_00 = np.array(data[data['proc'] == 'VH'][variable1])
  vh_sig_1_w = np.array(data[data['proc'] == 'VH']['weight'])[(vh_sig_00 > lower_lim) & (vh_sig_00 <upper_lim)]
  vh_sig_1_final = vh_sig_00[(vh_sig_00 > lower_lim) & (vh_sig_00 <upper_lim)]
  ggh_sig_00 = np.array(data[data['proc'] == 'ggH'][variable1])
  ggh_sig_1_w = np.array(data[data['proc'] == 'ggH']['weight'])[(ggh_sig_00 > lower_lim) & (ggh_sig_00 <upper_lim)]
  ggh_sig_1_final = ggh_sig_00[(ggh_sig_00 > lower_lim) & (ggh_sig_00 <upper_lim)]
  tth_sig_00 = np.array(data[data['proc'] == 'ttH'][variable1])
  tth_sig_1_w = np.array(data[data['proc'] == 'ttH']['weight'])[(tth_sig_00 > lower_lim) & (tth_sig_00 <upper_lim)]
  tth_sig_1_final = tth_sig_00[(tth_sig_00 > lower_lim) & (tth_sig_00 <upper_lim)]
 
  for variable2 in list_variables1:
    vbf_sig_01 = np.array(data[data['proc'] == 'VBF'][variable2])
    vbf_sig_2_w = np.array(data[data['proc'] == 'VBF']['weight'])[(vbf_sig_01 > lower_lim) & (vbf_sig_01 <upper_lim)]
    vbf_sig_2_final = vbf_sig_01[(vbf_sig_01 > lower_lim) & (vbf_sig_01 < upper_lim)]
    vh_sig_01 = np.array(data[data['proc'] == 'VH'][variable2])
    vh_sig_2_w = np.array(data[data['proc'] == 'VH']['weight'])[(vh_sig_01 > lower_lim) & (vh_sig_01 <upper_lim)]
    vh_sig_2_final = vh_sig_01[(vh_sig_01 > lower_lim) & (vh_sig_01 <upper_lim)]
    ggh_sig_01 = np.array(data[data['proc'] == 'ggH'][variable2])
    ggh_sig_2_w = np.array(data[data['proc'] == 'ggH']['weight'])[(ggh_sig_01 > lower_lim) & (ggh_sig_01 <upper_lim)]
    ggh_sig_2_final = ggh_sig_01[(ggh_sig_01 > lower_lim) & (ggh_sig_01 <upper_lim)]
    tth_sig_01 = np.array(data[data['proc'] == 'ttH'][variable2])
    tth_sig_2_w = np.array(data[data['proc'] == 'ttH']['weight'])[(tth_sig_01 > lower_lim) & (tth_sig_01 <upper_lim)]
    tth_sig_2_final = tth_sig_01[(tth_sig_01 > lower_lim) & (tth_sig_01 <upper_lim)]
 
    # VBF
    # removing the points where vbf_sig_1 is -999
    vbf_sig_1_a = vbf_sig_1_final[vbf_sig_1_final != -999]
    vbf_sig_2_a = vbf_sig_2_final[vbf_sig_1_final != -999]
    # removing the points where vbf_sig_2 is -999
    vbf_sig_1 = vbf_sig_1_a[vbf_sig_2_a != -999]
    vbf_sig_2 = vbf_sig_2_a[vbf_sig_2_a != -999]
    #VH
    # removing the points where vh_sig_1 is -999
    vh_sig_1_a = vh_sig_1_final[vh_sig_1_final != -999]
    vh_sig_2_a = vh_sig_2_final[vh_sig_1_final != -999]
    # removing the points where vh_sig_2 is -999
    vh_sig_1 = vh_sig_1_a[vh_sig_2_a != -999]
    vh_sig_2 = vh_sig_2_a[vh_sig_2_a != -999]
    # ggH
    # removing the points where ggh_sig_1 is -999
    ggh_sig_1_a = ggh_sig_1_final[ggh_sig_1_final != -999]
    ggh_sig_2_a = ggh_sig_2_final[ggh_sig_1_final != -999]
    # removing the points where ggh_sig_2 is -999
    ggh_sig_1 = ggh_sig_1_a[ggh_sig_2_a != -999]
    ggh_sig_2 = ggh_sig_2_a[ggh_sig_2_a != -999]
    # ttH
    # removing the points where tth_sig_1 is -999
    tth_sig_1_a = tth_sig_1_final[tth_sig_1_final != -999]
    tth_sig_2_a = tth_sig_2_final[tth_sig_1_final != -999]
    # removing the points where tth_sig_2 is -999
    tth_sig_1 = tth_sig_1_a[tth_sig_2_a != -999]
    tth_sig_2 = tth_sig_2_a[tth_sig_2_a != -999]

    combined_sig_1 = np.concatenate((vbf_sig_1,vh_sig_1,ggh_sig_1,tth_sig_1),axis=0)
    combined_sig_1_w = np.concatenate((vbf_sig_1_w,vh_sig_1_w,ggh_sig_1_w,tth_sig_1_w),axis=0)
    combined_sig_2 = np.concatenate((vbf_sig_2,vh_sig_2,ggh_sig_2,tth_sig_2),axis=0)
    combined_sig_2_w = np.concatenate((vbf_sig_2_w,vh_sig_2_w,ggh_sig_2_w,tth_sig_2_w),axis=0)
   
    #r = correlation coefficient
    #p = p-value of correlation
    r_VBF,p_VBF = sp.pearsonr(vbf_sig_1,vbf_sig_2)
    r_VH,p_VH = sp.pearsonr(vh_sig_1,vh_sig_2)
    r_ggH,p_ggH = sp.pearsonr(ggh_sig_1,ggh_sig_2)
    r_ttH,p_ttH = sp.pearsonr(tth_sig_1,tth_sig_2)
    r_combined,p_combined = sp.pearsonr(combined_sig_1,combined_sig_2)

    pearson_matrix_VBF[i][j] = r_VBF
    pearson_matrix_VH[i][j] = r_VH
    pearson_matrix_ggH[i][j] = r_ggH
    pearson_matrix_ttH[i][j] = r_ttH
    pearson_matrix_combined[i][j] = r_combined

    pearson_matrix_variables[i][j] = variable1 + '_' + variable2

    j = j+1
  i = i+1
  print('Progress:',i,'/',len(list_variables1))

#print(pearson_matrix_combined)   
fig, ax = plt.subplots(1,figsize=(15,15))
ax = sns.heatmap(pearson_matrix_combined, vmin=-1.0, vmax=1.0, linewidths=0.3, cmap='coolwarm',cbar=True,xticklabels=variables_names,yticklabels=variables_names,square=True) #.set_title("Signal")
name = 'plotting/plots/Pearson_Own_signal'
fig.savefig(name)
'''
fig, ax = plt.subplots(1,figsize=(15,15))
ax = sns.heatmap(pearson_matrix_VBF, vmin=-1.0, vmax=1.0, linewidths=0.3, cmap='coolwarm',cbar=True,xticklabels=list_variables1,yticklabels=list_variables1,square=True).set_title("VBF")
name = 'plotting/plots/Pearson_Own_VBF'
fig.savefig(name)

fig, ax = plt.subplots(1,figsize=(15,15))
ax = sns.heatmap(pearson_matrix_VH, vmin=-1.0, vmax=1.0, linewidths=0.3, cmap='coolwarm',cbar=True,xticklabels=list_variables1,yticklabels=list_variables1,square=True).set_title("VH")
name = 'plotting/plots/Pearson_Own_VH'
fig.savefig(name)

fig, ax = plt.subplots(1,figsize=(15,15))
ax = sns.heatmap(pearson_matrix_ggH, vmin=-1.0, vmax=1.0, linewidths=0.3, cmap='coolwarm',cbar=True,xticklabels=list_variables1,yticklabels=list_variables1,square=True).set_title("ggH")
name = 'plotting/plots/Pearson_Own_ggH'
fig.savefig(name)

fig, ax = plt.subplots(1,figsize=(15,15))
ax = sns.heatmap(pearson_matrix_ttH, vmin=-1.0, vmax=1.0, linewidths=0.3, cmap='coolwarm',cbar=True,xticklabels=list_variables1,yticklabels=list_variables1,square=True).set_title("ttH")
name = 'plotting/plots/Pearson_Own_ttH'
fig.savefig(name)
'''

pearson_matrix_VBF_flat = pearson_matrix_VBF.flatten()
pearson_matrix_VH_flat = pearson_matrix_VH.flatten()
pearson_matrix_ggH_flat = pearson_matrix_ggH.flatten()
pearson_matrix_ttH_flat = pearson_matrix_ttH.flatten()
pearson_matrix_combined_flat = pearson_matrix_combined.flatten()
pearson_matrix_variables_flat = pearson_matrix_variables.flatten()

VBF_corr_var_list = np.flip(sort_together([pearson_matrix_VBF_flat, pearson_matrix_variables_flat])[1])
VBF_corr_num_list = np.flip(sorted(pearson_matrix_VBF_flat))
VH_corr_var_list = np.flip(sort_together([pearson_matrix_VH_flat, pearson_matrix_variables_flat])[1])
VH_corr_num_list = np.flip(sorted(pearson_matrix_VH_flat))
ggH_corr_var_list = np.flip(sort_together([pearson_matrix_ggH_flat, pearson_matrix_variables_flat])[1])
ggH_corr_num_list = np.flip(sorted(pearson_matrix_ggH_flat))
ttH_corr_var_list = np.flip(sort_together([pearson_matrix_ttH_flat, pearson_matrix_variables_flat])[1])
ttH_corr_num_list = np.flip(sorted(pearson_matrix_ttH_flat))
signal_corr_var_list = np.flip(sort_together([pearson_matrix_combined_flat, pearson_matrix_variables_flat])[1])
signal_corr_num_list = np.flip(sorted(pearson_matrix_combined_flat))
'''
print('VBF:')
for i in range(100):
  if VBF_corr_num_list[i] != 1.0:
    print(VBF_corr_var_list[i],':',VBF_corr_num_list[i])
print('VH:')
for i in range(100):
  if VH_corr_num_list[i] != 1.0:
    print(VH_corr_var_list[i],':',VH_corr_num_list[i])
print('ggH:')
for i in range(100):
  if ggH_corr_num_list[i] != 1.0:
    print(ggH_corr_var_list[i],':',ggH_corr_num_list[i])
print('ttH:')
for i in range(100):
  if ttH_corr_num_list[i] != 1.0:
    print(ttH_corr_var_list[i],':',ttH_corr_num_list[i])
'''
print('Total Signal:')
for i in range(100):
  if signal_corr_num_list[i] != 1.0:
    print(signal_corr_var_list[i],':',signal_corr_num_list[i])