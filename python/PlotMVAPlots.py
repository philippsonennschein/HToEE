#import pandas
#df = pandas.read_pickle('models/VBF_BDT_clf.pickle.dat')


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

df = df[df['proc']=='VBF']
#df['weight_vbf'] = df[df['proc']=='VBF']['weight']
#df['weight_vbf_norm'] = df['weight_vbf'] / df['weight_vbf'].sum()
vbf_output = np.array(df[df['proc']=='VBF']['bdt_score'])
#vbf_w = np.array(df[df['proc']=='VBF']['weight_vbf_norm'])
vbf_w = df[df['proc']=='VBF']['weight'] / df[df['proc']=='VBF']['weight'].sum()

variables = ['leadPhotonIDMVA','subleadPhotonIDMVA',
    'diphotonMass',
    'diphotonPt',
    'leadPhotonPtOvM','subleadPhotonPtOvM',
    'leadPhotonEta','subleadPhotonEta',
    'dijetMass','dijetAbsDEta','dijetDPhi','diphotonCosPhi',
    'leadJetPUJID','subleadJetPUJID','subsubleadJetPUJID',
    'leadJetPt','leadJetEn','leadJetEta','leadJetPhi',
    'subleadJetPt','subleadJetEn','subleadJetEta','subleadJetPhi',
    'subsubleadJetPt','subsubleadJetEn','subsubleadJetEta','subsubleadJetPhi'
    ]     

var_num = 0

for variable in variables: 

    bins = 40

    var_num = var_num + 1
    print(variable)

    vbf_sig = np.array(df[df['proc'] == 'VBF'][variable])

    vbf_08_10_w = []
    vbf_06_08_w = []
    vbf_04_06_w = []
    vbf_02_04_w = []
    vbf_00_02_w = []

    vbf_08_10_w_n = []
    vbf_06_08_w_n = []
    vbf_04_06_w_n = []
    vbf_02_04_w_n = []
    vbf_00_02_w_n = []

    variable_08_10 = []
    variable_06_08 = []
    variable_04_06 = []
    variable_02_04 = []
    variable_00_02 = []

    count = 0

    for i in vbf_output:
        if i>0.95 and i<=1.0 and vbf_sig[count]!=-999.0:
            variable_08_10 = np.append(variable_08_10,vbf_sig[count])
            vbf_08_10_w = np.append(vbf_08_10_w,vbf_w[count])
        if i>0.9 and i<=0.95 and vbf_sig[count]!=-999.0:
            variable_06_08 = np.append(variable_06_08,vbf_sig[count])
            vbf_06_08_w = np.append(vbf_06_08_w,vbf_w[count])
        if i>0.85 and i<=0.9 and vbf_sig[count]!=-999.0:
            variable_04_06 = np.append(variable_04_06,vbf_sig[count])
            vbf_04_06_w = np.append(vbf_04_06_w,vbf_w[count])
        if i>0.8 and i<=0.85 and vbf_sig[count]!=-999.0:
            variable_02_04 = np.append(variable_02_04,vbf_sig[count])
            vbf_02_04_w = np.append(vbf_02_04_w,vbf_w[count])
        if i>0.5 and i<=0.8 and vbf_sig[count]!=-999.0:
            variable_00_02 = np.append(variable_00_02,vbf_sig[count])
            vbf_00_02_w = np.append(vbf_00_02_w,vbf_w[count])

        count = count + 1


    print('Plotting')
    fig, ax = plt.subplots(1)

    #Xlims for different variables
    if variable == 'dijetMass':
        plt.xlim(0,2000)
        bins = 100
    elif variable == 'diphotonPt':
        plt.xlim(0,400)
        bins = 200
    elif variable == 'leadJetEn':
        plt.xlim(0,2000)
        bins = 100
    elif variable == 'leadJetPt':
        plt.xlim(0,300)
        bins = 200

    plt.hist(variable_08_10, bins = bins, histtype = 'step', label='0.95 < MVA < 1.0', weights = vbf_08_10_w, density=True)
    plt.hist(variable_06_08, bins = bins, histtype = 'step', label='0.9 < MVA < 0.95', weights = vbf_06_08_w, density=True)
    plt.hist(variable_04_06, bins = bins, histtype = 'step', label='0.85 < MVA < 0.9', weights = vbf_04_06_w, density=True)
    plt.hist(variable_02_04, bins = bins, histtype = 'step', label='0.8 < MVA < 0.85', weights = vbf_02_04_w, density=True)
    plt.hist(variable_00_02, bins = bins, histtype = 'step', label='0.5 < MVA < 0.8', weights = vbf_00_02_w, density=True)

    

    name = 'plotting/plots/MVA_Plot_' + variable
    plt.title('VBF MVA Plot ' + variable)
    plt.legend()
    plt.xlabel(variable)
    plt.ylabel('Arbitrary Units')
    fig.savefig(name)
    print('Progress:',var_num,'/',len(variables))