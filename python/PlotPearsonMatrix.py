
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob

import seaborn as sns

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
df_vh = df[df['proc']=='VH']  
df_tth = df[df['proc']=='ttH'] 

corr = df.corr()
corr_ggh = df_ggh.corr()
corr_vbf = df_vbf.corr()
corr_vh = df_vh.corr()
corr_tth = df_tth.corr()

#Need to run the following command to display all rows
pd.set_option('display.max_rows', 1000)

corr_matrix = df.corr().abs()
corr_matrix_ggh = df_ggh.corr().abs()
corr_matrix_vbf = df_vbf.corr().abs()
corr_matrix_vh = df_vh.corr().abs()
corr_matrix_tth = df_tth.corr().abs()

corr_list = (corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool)).stack().sort_values(ascending=False))
corr_list_ggh = (corr_matrix_ggh.where(np.triu(np.ones(corr_matrix_ggh.shape), k=1).astype(np.bool)).stack().sort_values(ascending=False))
corr_list_vbf = (corr_matrix_vbf.where(np.triu(np.ones(corr_matrix_vbf.shape), k=1).astype(np.bool)).stack().sort_values(ascending=False))
corr_list_vh = (corr_matrix_vh.where(np.triu(np.ones(corr_matrix_vh.shape), k=1).astype(np.bool)).stack().sort_values(ascending=False))
corr_list_tth = (corr_matrix_tth.where(np.triu(np.ones(corr_matrix_tth.shape), k=1).astype(np.bool)).stack().sort_values(ascending=False))

print('Total Signal')
print(corr_list)
print('ggH')
print(corr_list_ggh)
print('VBF')
print(corr_list_vbf)
print('VH')
print(corr_list_vh)
print('ttH')
print(corr_list_tth)

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