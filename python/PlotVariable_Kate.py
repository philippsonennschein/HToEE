import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob

#Load the dataframe
dataframes = []
dataframes.append(pd.read_csv('2017/MC/DataFrames/ggH_VBF_BDT_df_2017.csv'))
dataframes.append(pd.read_csv('2017/MC/DataFrames/VBF_VBF_BDT_df_2017.csv'))
dataframes.append(pd.read_csv('2017/MC/DataFrames/VH_VBF_BDT_df_2017.csv'))
dataframes.append(pd.read_csv('2017/MC/DataFrames/ttH_VBF_BDT_df_2017.csv'))
dataframes.append(pd.read_csv('2017/MC/DataFrames/tHq_VBF_BDT_df_2017.csv', nrows = 100000)) #254039
dataframes.append(pd.read_csv('2017/MC/DataFrames/tHW_VBF_BDT_df_2017.csv', nrows = 100000)) #130900
df = pd.concat(dataframes, sort=False, axis=0)

print('Loaded dataframe')

train_vars = ['diphotonPt', 'diphotonMass', 'diphotonCosPhi', 'diphotonEta','diphotonPhi', 'diphotonSigmaMoM',
     #'dijetMass', 'dijetAbsDEta', 'dijetDPhi', 'dijetCentrality',
     #'dijetPt','dijetEta','dijetPhi','dijetMinDRJetPho','dijetDiphoAbsDEta',
     'dijetAbsDEta',
     'leadPhotonEta', 'leadPhotonIDMVA', 'leadPhotonEn', 'leadPhotonPt', 'leadPhotonPhi', 'leadPhotonPtOvM',
     'leadJetPt', 'leadJetPUJID', 'leadJetBTagScore', 'leadJetMass',
     'leadJetDiphoDEta','leadJetDiphoDPhi','leadJetEn','leadJetEta','leadJetPhi',
     'subleadPhotonEta', 'subleadPhotonIDMVA', 'subleadPhotonPhi',
     'subleadPhotonEn','subleadPhotonPt', 'subleadPhotonPtOvM',
     #'subleadJetDiphoDPhi','subleadJetDiphoDEta',
     #'subleadJetPt', 'subleadJetPUJID', 'subleadJetBTagScore', 'subleadJetMass',
     'subleadJetEn','subleadJetEta','subleadJetPhi',
     #'subsubleadJetEn','subsubleadJetPt','subsubleadJetEta','subsubleadJetPhi', 'subsubleadJetBTagScore', 
     #'subsubleadJetMass',
     #'metPt','metPhi','metSumET',
     #'nSoftJets',
     #'leadElectronEn', 'leadElectronMass', 'leadElectronPt', 'leadElectronEta', 'leadElectronPhi', 'leadElectronCharge',
     #'leadMuonEn', 'leadMuonMass', 'leadMuonPt', 'leadMuonEta', 'leadMuonPhi', 'leadMuonCharge'
     #'subleadElectronEn', 'subleadElectronMass', 'subleadElectronPt', 'subleadElectronEta', 'subleadElectronPhi', 'subleadElectronCharge', 
     #'subleadMuonEn', 'subleadMuonMass', 'subleadMuonPt', 'subleadMuonEta', 'subleadMuonPhi', 'subleadMuonCharge'
     ]

train_vars.append('proc')
train_vars.append('weight')
train_vars.append('HTXS_stage_0')
train_vars.append('HTXS_stage1_2_cat_pTjet30GeV')

data = df[train_vars]

map_def = [['ggH',10,11],['qqH',20,21,22,23],['VH',30,31,40,41],['ttH',60,61],['tH',80,81]]


binNames = ['ggH','qqH','VH','ttH','tH'] 

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

data['proc_new'] = mapping(map_list=map_def,stage=data['HTXS_stage_0'])


list_variables = [#'diphotonPt'
                #, 'leadPhotonEta'
                #, 'subleadPhotonEta' 
                #, 'leadPhotonIDMVA' 
                #, 'subleadPhotonIDMVA'
                #, 'diphotonMass' 
                #, 'weight' 
                #, 'centralObjectWeight' 
                # 'leadPhotonPtOvM'
                #, 'dijetMass' 
                #, 'subleadPhotonPtOvM'
                 #'leadJetPt'
                #, 'subleadJetPt' #'leadElectronIDMVA', 'subleadElectronIDMVA',
                #, 'dijetMass'
                #, 'dijetAbsDEta' 
                #, 'dijetDPhi'
                #, 'diphotonCosPhi'
                #, 'leadJetPUJID'
                #, 'subleadJetPUJID'
                #, 'subsubleadJetEn'
                #, 'subsubleadJetPt'
                #, 'subsubleadJetEta'
                #, 'subsubleadJetPhi' 
                #, 'min_IDMVA'
                #, 'max_IDMVA'
                # 'dijetCentrality'
                # 'leadJetBTagScore' 
                # 'subleadJetBTagScore', 'subsubleadJetBTagScore'
                #, 'leadJetMass' ,'leadPhotonEn', 'leadPhotonMass' ,
                # 'leadPhotonPt'
                #, 'subleadJetMass', 'subleadPhotonEn', 'subleadPhotonMass', 'subleadPhotonPt'
                #, 'DR' 
                #'leadPhotonPhi','leadJetDiphoDEta','leadJetDiphoDPhi','leadJetEn','leadJetEta','leadJetPhi'
                #'subleadPhotonPhi','subsubleadPhotonEn','subsubleadJetMass','subsubleadPhotonMass',
                #'subsubleadPhotonPt','subsubleadPhotonEta','subsubleadPhotonPhi','subleadJetDiphoDPhi',
                #'subleadJetDiphoDEta','subleadJetEn','subleadJetEta','subleadJetPhi','subsubleadPhotonIDMVA',				
                #'diphotonEta','diphotonPhi','dijetPt','dijetEta','dijetPhi','dijetMinDRJetPho','dijetDiphoAbsDEta',
                #'nSoftJets'
                #'leadElectronMass',
                #'leadMuonMass'
                'dijetAbsDEta'
]

for variable in list_variables:
    
    print('Inside loop')
    # Signal 
    qqh_sig_0 = np.array(data[data['proc_new'] == 'qqH'][variable])
    qqh_sig_w = np.array(data[data['proc_new'] == 'qqH']['weight'])[(qqh_sig_0 > -10) & (qqh_sig_0 <2000)]
    #vbf_sig_w = vbf_sig_w / np.sum(vbf_sig_w)
    qqh_sig = qqh_sig_0[(qqh_sig_0 > -10) & (qqh_sig_0 < 2000)]
    print('vbf')
    vh_sig_0 = np.array(data[data['proc_new'] == 'VH'][variable])
    vh_sig_w = np.array(data[data['proc_new'] == 'VH']['weight'])[(vh_sig_0 > -10) & (vh_sig_0 <2000)]
    #vh_sig_w = vh_sig_w / np.sum(vh_sig_w)
    vh_sig = vh_sig_0[(vh_sig_0 > -10) & (vh_sig_0 <2000)]
    print('vh')
    ggh_sig_0 = np.array(data[data['proc_new'] == 'ggH'][variable])
    ggh_sig_w = np.array(data[data['proc_new'] == 'ggH']['weight'])[(ggh_sig_0 > -10) & (ggh_sig_0 <2000)]
    #ggh_sig_w = ggh_sig_w / np.sum(ggh_sig_w)
    ggh_sig = ggh_sig_0[(ggh_sig_0 > -10) & (ggh_sig_0 <2000)]
    print('ggh')
    tth_sig_0 = np.array(data[data['proc_new'] == 'ttH'][variable])
    tth_sig_w = np.array(data[data['proc_new'] == 'ttH']['weight'])[(tth_sig_0 > -10) & (tth_sig_0 <2000)]
    #tth_sig_w = tth_sig_w / np.sum(tth_sig_w)
    tth_sig = tth_sig_0[(tth_sig_0 > -10) & (tth_sig_0 <2000)]
    print('ttH')
    th_sig_0 = np.array(data[data['proc_new'] == 'tH'][variable])
    th_sig_w = np.array(data[data['proc_new'] == 'tH']['weight'])[(th_sig_0 > -10) & (th_sig_0 <2000)]
    th_sig = th_sig_0[(th_sig_0 > -10) & (th_sig_0 <2000)]
    print('tH')


    scale = 100
    num_bins = 300
    normalize = True

    fig, ax = plt.subplots()

    # signal
    ax.hist(ggh_sig, bins = num_bins, density = normalize, color = '#24b1c9', label = 'ggH', stacked = True, histtype = 'step', weights = scale * ggh_sig_w)
    ax.hist(qqh_sig, bins = num_bins, density = normalize, color = '#e36b1e', label = 'qqH', histtype = 'step', weights = scale * qqh_sig_w, alpha = 1)
    ax.hist(vh_sig, bins = num_bins, density = normalize, color = '#1eb037', label = 'VH', stacked = True, histtype = 'step', weights = scale * vh_sig_w)
    ax.hist(tth_sig, bins = num_bins, density = normalize, color = '#c21bcf', label = 'ttH', stacked = True, histtype = 'step', weights = scale * tth_sig_w)
    ax.hist(th_sig, bins = num_bins, density = normalize, color = '#dbb104', label = 'tH', stacked = True, histtype = 'step', weights = scale * th_sig_w)

    #ax.set_xlim(0,300)
    #ax.set_ylim(0,0.04)
    #ax.set_xticks([0,50,100,150,200,250,300])
    #ax.set_yticks([0,0.005,0.01, 0.015,0.02, 0.025, 0.03, 0.035, 0.04])
    ax.legend(loc = 'upper right')
    ax.set_xlabel(variable, ha='center',x=0.5, size = 12)
    ax.set_ylabel('Fraction of events',ha='center', y=0.5, size = 12)
    #ax.grid(True, 'major', linestyle='solid', color='grey', alpha=0.5)

    name = 'plotting/Single_var_dis/' + variable 
    #print('Plotting leadJetPt plot')
    fig.savefig(name, dpi=1200)