#import pandas
#df = pandas.read_pickle('models/VBF_BDT_clf.pickle.dat')


import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob

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

Output_BDT = np.loadtxt('models/Output_BDT.csv', delimiter=',')

# new columns aye
df['Output_BDT'] = Output_BDT
df['weight_vbf'] = df[df['proc']=='VBF']['weight']
df['weight_ggh'] = df[df['proc']=='ggH']['weight']
df['weight_vbf_norm'] = df['weight_vbf'] / df['weight_vbf'].sum()
df['weight_ggh_norm'] = df['weight_ggh'] / df['weight_ggh'].sum()
vbf_output = np.array(df[df['proc']=='VBF']['Output_BDT'])
ggh_output = np.array(df[df['proc']=='ggH']['Output_BDT'])

# weights aye**2
vbf_w = np.array(df[df['proc']=='VBF']['weight_vbf_norm'])
ggh_w = np.array(df[df['proc']=='ggH']['weight_ggh_norm'])

bins = 40
normalize = True
fig, ax = plt.subplots(1)

if normalize:
    plt.hist(vbf_output, bins = bins, histtype = 'step', label='VBF', color = 'blue', weights = vbf_w)
    plt.hist(ggh_output, bins = bins, histtype = 'step', label='ggH', color = 'red', weights = ggh_w)

else:
    plt.hist(vbf_output, bins = bins, histtype = 'step', label='VBF', color = 'blue', density = True)
    plt.hist(ggh_output, bins = bins, histtype = 'step', label='ggH', color = 'red', density = True)

name = 'plotting/plots/BDT2'
plt.title('BDT Plot VBF vs. ggH')
plt.legend()
plt.xlabel('BDT_score')
plt.ylabel('Arbitrary_Units')
fig.savefig(name)
plt.close()