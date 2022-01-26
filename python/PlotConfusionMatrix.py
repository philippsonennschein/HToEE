
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
import pickle

parser = argparse.ArgumentParser()
required_args = parser.add_argument_group('Required Arguments')
required_args.add_argument('-m','--mc', action='store', help='Input MC dataframe dir', required=True)
required_args.add_argument('-d','--data', action='store', help='Input data dataframe dir', required=True)
options=parser.parse_args()

files_mc_csv = glob.glob("%s/*.csv"%options.mc)
files_mc_data = glob.glob("%s/*.csv"%options.data)

dataframes = []
dataframes.append(pd.read_csv('2017/MC/DataFrames/VBF_VBF_BDT_df_2017.csv'))
dataframes.append(pd.read_csv('2017/MC/DataFrames/ggH_VBF_BDT_df_2017.csv'))
df = pd.concat(dataframes, sort=False, axis=0 )

train_vars = ['leadPhotonIDMVA','subleadPhotonIDMVA',
    'diphotonMass','diphotonPt',
    'leadPhotonPtOvM','subleadPhotonPtOvM',
    'leadPhotonEta','subleadPhotonEta',
    'dijetMass','dijetAbsDEta','dijetDPhi','diphotonCosPhi',
    'leadJetPUJID','subleadJetPUJID','subsubleadJetPUJID',
    'leadJetPt','leadJetEn','leadJetEta','leadJetPhi',
    'subleadJetPt','subleadJetEn','subleadJetEta','subleadJetPhi',
    'subsubleadJetPt','subsubleadJetEn','subsubleadJetEta','subsubleadJetPhi'
    ] 

print('loading classifier:')
clf = pickle.load(open("models/VBF_BDT_clf.pickle.dat", "rb"))
df['bdt_score'] = clf.predict_proba(df[train_vars].values)[:,1:].ravel()

#Removing variables that we don't need
df = df.drop(labels='Unnamed: 0',axis=1)
#df = df.drop(labels='proc',axis=1)
df = df.drop(labels='year',axis=1)
df = df.drop(labels='weight',axis=1)
df = df.drop(labels='centralObjectWeight',axis=1)
df = df.drop(labels='genWeight',axis=1)

#Introducing a split between the 4 different production modes
df_ggh = df[df['proc']=='ggH'] 
df_vbf = df[df['proc']=='VBF']
df_tot = pd.concat(df_ggh,df_vbf)

df['pred']

df = df[df['proc']=='VBF']


'''
#Plotting

fig, ax = plt.subplots(1,figsize=(15,15))
ax = sns.heatmap(corr, cmap='coolwarm',cbar=True,xticklabels=corr.columns.values,yticklabels=corr.columns.values,square=True).set_title("Combined Signal")
name = 'plotting/plots/Pearson_Signal'
fig.savefig(name)

fig, ax = plt.subplots(1,figsize=(15,15))
ax = sns.heatmap(corr_ggh, cmap='coolwarm',cbar=True,xticklabels=corr_ggh.columns.values,yticklabels=corr_ggh.columns.values,square=True).set_title("ggH")
name = 'plotting/plots/Pearson_ggh'
fig.savefig(name)

fig, ax = plt.subplots(1,figsize=(15,15))
ax = sns.heatmap(corr_vbf, cmap='coolwarm',cbar=True,xticklabels=corr_vbf.columns.values,yticklabels=corr_vbf.columns.values,square=True).set_title("VBF")
name = 'plotting/plots/Pearson_vbf'
fig.savefig(name)

fig, ax = plt.subplots(1,figsize=(15,15))
ax = sns.heatmap(corr_vh, cmap='coolwarm',cbar=True,xticklabels=corr_vh.columns.values,yticklabels=corr_vh.columns.values,square=True).set_title("VH")
name = 'plotting/plots/Pearson_vh'
fig.savefig(name)

fig, ax = plt.subplots(1,figsize=(15,15))
ax = sns.heatmap(corr_tth, cmap='coolwarm',cbar=True,xticklabels=corr_tth.columns.values,yticklabels=corr_tth.columns.values,square=True).set_title("ttH")
name = 'plotting/plots/Pearson_tth'
fig.savefig(name)
'''