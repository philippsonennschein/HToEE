 
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
 
parser = argparse.ArgumentParser()
required_args = parser.add_argument_group('Required Arguments')
required_args.add_argument('-m','--mc', action='store', help='Input MC dataframe dir', required=True)
required_args.add_argument('-d','--data', action='store', help='Input data dataframe dir', required=True)
options=parser.parse_args()
 
files_mc_csv = glob.glob("%s/*.csv"%options.mc)
files_mc_data = glob.glob("%s/*.csv"%options.data)
 
dataframes = []
for f in files_mc_csv:
  dataframes.append( pd.read_csv(f) )
  print " --> Read: %s"%f
for f in files_mc_data:
  dataframes.append( pd.read_csv(f) )
  print " --> Read: %s"%f
 
df = pd.concat( dataframes, sort=False, axis=0 )
print " --> Successfully read dataframes"
 
# -------------------------------
 
#Adapted code to plot a 2D matrix of Pearson Correlation coefficients
 
# Background Grouping
# QCD
df['proc'] = np.where(df['proc'] == 'QCD30toinf', 'QCD', df['proc'])
df['proc'] = np.where(df['proc'] == 'QCD40toinf', 'QCD', df['proc'])
df['proc'] = np.where(df['proc'] == 'QCD30to40', 'QCD', df['proc'])
 
# Gjet
df['proc'] = np.where(df['proc'] == 'GJet20to40', 'Gjet', df['proc'])
df['proc'] = np.where(df['proc'] == 'GJet40toinf', 'Gjet', df['proc'])
df['proc'] = np.where(df['proc'] == 'GJet20toinf', 'Gjet', df['proc'])
 
# Diphoton
df['proc'] = np.where(df['proc'] == 'Diphoton40to80', 'Diphoton', df['proc'])
df['proc'] = np.where(df['proc'] == 'Diphoton80toinf', 'Diphoton', df['proc'])
 
#Creating the min and max_IDMVA variables
df['min_IDMVA'] = df[['leadPhotonIDMVA', 'subleadPhotonIDMVA']].min(axis=1)
df['max_IDMVA'] = df[['leadPhotonIDMVA', 'subleadPhotonIDMVA']].max(axis=1)

#Creating dR variable
df['DR'] = np.sqrt(df['diphotonEta']**2 + df['diphotonPhi']**2)

list_minus1_default = ['leadJetBTagScore','subleadJetBTagScore','subsubleadJetBTagScore']
for variable in list_minus1_default:
  df[variable] = np.where(df[variable] == -1,-999,df[variable])

# -------------------------------
list_variables1 = [
  'diphotonMass','diphotonPt','diphotonEta','diphotonPhi','diphotonCosPhi',
  'leadPhotonIDMVA','leadPhotonPtOvM','leadPhotonEta','leadPhotonEn','leadPhotonMass','leadPhotonPt','leadPhotonPhi',
  'subleadPhotonIDMVA','subleadPhotonPtOvM','subleadPhotonEta','subleadPhotonEn','subleadPhotonMass','subleadPhotonPt','subleadPhotonPhi',
  'dijetMass','dijetPt','dijetEta','dijetPhi','dijetDPhi','dijetAbsDEta','dijetCentrality','dijetMinDRJetPho','dijetDiphoAbsDEta',
  'leadJetPUJID','leadJetPt','leadJetEn','leadJetEta','leadJetPhi','leadJetMass','leadJetBTagScore','leadJetDiphoDEta','leadJetDiphoDPhi',
  'subleadJetPUJID','subleadJetPt','subleadJetEn','subleadJetEta','subleadJetPhi','subleadJetMass','subleadJetBTagScore','subleadJetDiphoDPhi','subleadJetDiphoDEta',
  'subsubleadJetPUJID','subsubleadJetPt','subsubleadJetEn','subsubleadJetEta','subsubleadJetPhi','subsubleadJetMass','subsubleadJetBTagScore',
  'nSoftJets','DR'
]
 
list_variables2 = [
  'diphotonMass','diphotonPt','diphotonEta','diphotonPhi','diphotonCosPhi',
  'leadPhotonIDMVA','leadPhotonPtOvM','leadPhotonEta','leadPhotonEn','leadPhotonMass','leadPhotonPt','leadPhotonPhi',
  'subleadPhotonIDMVA','subleadPhotonPtOvM','subleadPhotonEta','subleadPhotonEn','subleadPhotonMass','subleadPhotonPt','subleadPhotonPhi',
  'dijetMass','dijetPt','dijetEta','dijetPhi','dijetDPhi','dijetAbsDEta','dijetCentrality','dijetMinDRJetPho','dijetDiphoAbsDEta',
  'leadJetPUJID','leadJetPt','leadJetEn','leadJetEta','leadJetPhi','leadJetMass','leadJetBTagScore','leadJetDiphoDEta','leadJetDiphoDPhi',
  'subleadJetPUJID','subleadJetPt','subleadJetEn','subleadJetEta','subleadJetPhi','subleadJetMass','subleadJetBTagScore','subleadJetDiphoDPhi','subleadJetDiphoDEta',
  'subsubleadJetPUJID','subsubleadJetPt','subsubleadJetEn','subsubleadJetEta','subsubleadJetPhi','subsubleadJetMass','subsubleadJetBTagScore',
  'nSoftJets','DR'
]
 
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

  vbf_sig_00 = np.array(df[df['proc'] == 'VBF'][variable1])
  vbf_sig_1_w = np.array(df[df['proc'] == 'VBF']['weight'])[(vbf_sig_00 > lower_lim) & (vbf_sig_00 <upper_lim)]
  vbf_sig_1_final = vbf_sig_00[(vbf_sig_00 > lower_lim) & (vbf_sig_00 < upper_lim)]
  vh_sig_00 = np.array(df[df['proc'] == 'VH'][variable1])
  vh_sig_1_w = np.array(df[df['proc'] == 'VH']['weight'])[(vh_sig_00 > lower_lim) & (vh_sig_00 <upper_lim)]
  vh_sig_1_final = vh_sig_00[(vh_sig_00 > lower_lim) & (vh_sig_00 <upper_lim)]
  ggh_sig_00 = np.array(df[df['proc'] == 'ggH'][variable1])
  ggh_sig_1_w = np.array(df[df['proc'] == 'ggH']['weight'])[(ggh_sig_00 > lower_lim) & (ggh_sig_00 <upper_lim)]
  ggh_sig_1_final = ggh_sig_00[(ggh_sig_00 > lower_lim) & (ggh_sig_00 <upper_lim)]
  tth_sig_00 = np.array(df[df['proc'] == 'ttH'][variable1])
  tth_sig_1_w = np.array(df[df['proc'] == 'ttH']['weight'])[(tth_sig_00 > lower_lim) & (tth_sig_00 <upper_lim)]
  tth_sig_1_final = tth_sig_00[(tth_sig_00 > lower_lim) & (tth_sig_00 <upper_lim)]
 
  for variable2 in list_variables1:
    vbf_sig_01 = np.array(df[df['proc'] == 'VBF'][variable2])
    vbf_sig_2_w = np.array(df[df['proc'] == 'VBF']['weight'])[(vbf_sig_01 > lower_lim) & (vbf_sig_01 <upper_lim)]
    vbf_sig_2_final = vbf_sig_01[(vbf_sig_01 > lower_lim) & (vbf_sig_01 < upper_lim)]
    vh_sig_01 = np.array(df[df['proc'] == 'VH'][variable2])
    vh_sig_2_w = np.array(df[df['proc'] == 'VH']['weight'])[(vh_sig_01 > lower_lim) & (vh_sig_01 <upper_lim)]
    vh_sig_2_final = vh_sig_01[(vh_sig_01 > lower_lim) & (vh_sig_01 <upper_lim)]
    ggh_sig_01 = np.array(df[df['proc'] == 'ggH'][variable2])
    ggh_sig_2_w = np.array(df[df['proc'] == 'ggH']['weight'])[(ggh_sig_01 > lower_lim) & (ggh_sig_01 <upper_lim)]
    ggh_sig_2_final = ggh_sig_01[(ggh_sig_01 > lower_lim) & (ggh_sig_01 <upper_lim)]
    tth_sig_01 = np.array(df[df['proc'] == 'ttH'][variable2])
    tth_sig_2_w = np.array(df[df['proc'] == 'ttH']['weight'])[(tth_sig_01 > lower_lim) & (tth_sig_01 <upper_lim)]
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
ax = sns.heatmap(pearson_matrix_combined, vmin=-1.0, vmax=1.0, linewidths=0.3, cmap='coolwarm',cbar=True,xticklabels=list_variables1,yticklabels=list_variables1,square=True).set_title("Signal")
name = 'plotting/plots/Pearson_Own_signal'
fig.savefig(name)

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
print('Total Signal:')
for i in range(100):
  if signal_corr_num_list[i] != 1.0:
    print(signal_corr_var_list[i],':',signal_corr_num_list[i])