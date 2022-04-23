
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob

map_def = [['ggH',10,11],['qqH',20,21,22,23],['VH',30,31,40,41],['ttH',60,61],['tH',80,81]]

binNames = ['ggH','qqH','VH','ttH','tH'] 

dataframes = []
#dataframes.append(pd.read_csv('2017/MC/DataFrames/ggH_VBF_BDT_df_2017.csv'))
#dataframes.append(pd.read_csv('2017/MC/DataFrames/VBF_VBF_BDT_df_2017.csv'))
#dataframes.append(pd.read_csv('2017/MC/DataFrames/VH_VBF_BDT_df_2017.csv'))
#dataframes.append(pd.read_csv('2017/MC/DataFrames/ttH_VBF_BDT_df_2017.csv'))
dataframes.append(pd.read_csv('2017/MC/DataFrames/tHq_VBF_BDT_df_2017.csv')) #, nrows = 254039))
dataframes.append(pd.read_csv('2017/MC/DataFrames/tHW_VBF_BDT_df_2017.csv')) #, nrows = 130900))
df = pd.concat(dataframes, sort=False, axis=0 )

train_vars = ['diphotonPt', 'diphotonMass', 'diphotonCosPhi', 'diphotonEta','diphotonPhi', 'diphotonSigmaMoM',
     #'dijetMass', 'dijetAbsDEta', 'dijetDPhi', 'dijetCentrality',
     #'dijetPt','dijetEta','dijetPhi','dijetMinDRJetPho','dijetDiphoAbsDEta',
     'leadPhotonEta', 'leadPhotonIDMVA', 'leadPhotonEn', 'leadPhotonPt', 'leadPhotonPhi', 'leadPhotonPtOvM',
     #'leadJetPt', 'leadJetPUJID', 'leadJetBTagScore', 'leadJetMass',
     'leadJetDiphoDEta','leadJetDiphoDPhi','leadJetEn','leadJetEta','leadJetPhi',
     'subleadPhotonEta', 'subleadPhotonIDMVA', 'subleadPhotonPhi',
     'subleadPhotonEn','subleadPhotonPt', 'subleadPhotonPtOvM',
     #'subleadJetDiphoDPhi','subleadJetDiphoDEta',
     #'subleadJetPt', 'subleadJetPUJID', 'subleadJetBTagScore', 'subleadJetMass',
     'subleadJetEn','subleadJetEta','subleadJetPhi'
     #'subsubleadJetEn','subsubleadJetPt','subsubleadJetEta','subsubleadJetPhi', 'subsubleadJetBTagScore', 
     #'subsubleadJetMass',
     #'metPt','metPhi','metSumET',
     #'nSoftJets',
     #'leadElectronEn', 'leadElectronMass', 'leadElectronPt', 'leadElectronEta', 'leadElectronPhi', 'leadElectronCharge',
     #'leadMuonEn', 'leadMuonMass', 'leadMuonPt', 'leadMuonEta', 'leadMuonPhi', 'leadMuonCharge',
     #'subleadElectronEn', 'subleadElectronMass', 'subleadElectronPt', 'subleadElectronEta', 'subleadElectronPhi', 'subleadElectronCharge', 
     #'subleadMuonEn', 'subleadMuonMass', 'subleadMuonPt', 'subleadMuonEta', 'subleadMuonPhi', 'subleadMuonCharge'
     ]

train_vars.append('proc')
train_vars.append('weight')
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

data['proc_new'] = mapping(map_list=map_def,stage=data['HTXS_stage_0'])

#data.loc[data.proc_new == 'tH','weight'] = data[data['proc_new'] == 'tH']['weight'] * 4

# -------------------------------

#Adapted code for 2D correlation plots


#For issues with plotting variables that have -999 values can implement the following code logic
#Find max and min of both variable arrays
#If none have -999 then no changes
#If variables have -999 then xlim and ylim from -1 to max of respective variable

#Creating dR variable
data['DR'] = np.sqrt(data['diphotonEta']**2 + data['diphotonPhi']**2)

# -------------------------------
list_variables1 = [
#'leadPhotonIDMVA','subleadPhotonIDMVA',
#'min_IDMVA','max_IDMVA',
#'diphotonMass',
#'DR'
'diphotonPt'
#'leadPhotonPtOvM','subleadPhotonPtOvM',
#'leadPhotonEta','subleadPhotonEta',
#'dijetMass',
#'dijetAbsDEta'
#,'dijetDPhi',
#'leadJetPt'
#,'leadJetEn',
#'leadJetEta'
#'leadJetPhi',
#'subleadJetPt','subleadJetEn','subleadJetEta','subleadJetPhi',
#'subsubleadJetPt','subsubleadJetEn','subsubleadJetEta','subsubleadJetPhi'
]

list_variables2 = [
#'leadPhotonIDMVA','subleadPhotonIDMVA',
#'min_IDMVA','max_IDMVA',
#'diphotonMass',
#'diphotonPt'
#'leadPhotonPtOvM','subleadPhotonPtOvM',
#'leadPhotonEta','subleadPhotonEta'
#'dijetMass','dijetAbsDEta','dijetDPhi',
#'leadJetPt','leadJetEn','leadJetEta',
#'leadJetPhi'
#'subleadJetPt','subleadJetEn'
#'subleadJetEta'
#'subleadJetPhi',
#'subsubleadJetPt','subsubleadJetEn','subsubleadJetEta','subsubleadJetPhi'
#'diphotonCosPhi'
'DR'
]

#list_plots is a list that contains all the variable combinations that have been previously plotted
#Variable is used to avoid the same pair to be plotted twice
list_plots = []
lower_lim = -1000
upper_lim = 50000
num_bins = 50

data_temp = data[data.DR>-10.]
data_temp = data_temp[data_temp.diphotonPt<180]


for variable1 in list_variables1:
  
  # Signal
  #data_temp1 = data[data.leadJetEta>-10.]
  '''
  qqh_sig_1 = np.array(data_temp[data_temp['proc_new'] == 'qqH'][variable1])
  qqh_sig_1_w = np.array(data_temp[data_temp['proc_new'] == 'qqH']['weight'])

  vh_sig_1 = np.array(data_temp[data_temp['proc_new'] == 'VH'][variable1])
  vh_sig_1_w = np.array(data_temp[data_temp['proc_new'] == 'VH']['weight'])

  ggh_sig_1 = np.array(data_temp[data_temp['proc_new'] == 'ggH'][variable1])
  ggh_sig_1_w = np.array(data_temp[data_temp['proc_new'] == 'ggH']['weight'])

  tth_sig_1 = np.array(data_temp[data_temp['proc_new'] == 'ttH'][variable1])
  tth_sig_1_w = np.array(data_temp[data_temp['proc_new'] == 'ttH']['weight'])
  '''
  th_sig_1 = np.array(data_temp[data_temp['proc_new'] == 'tH'][variable1])
  th_sig_1_w = np.array(data_temp[data_temp['proc_new'] == 'tH']['weight'])


  #combined_sig_1 = np.concatenate((qqh_sig_1,vh_sig_1,ggh_sig_1,tth_sig_1,th_sig_1),axis=0)
  #combined_sig_1 = np.concatenate((qqh_sig_1,vh_sig_1,ggh_sig_1,tth_sig_1),axis=0)
  #combined_sig_1_w = np.concatenate((qqh_sig_1_w,vh_sig_1_w,ggh_sig_1_w,tth_sig_1_w,th_sig_1_w),axis=0)

  for variable2 in list_variables2:

    #data_temp2 = data[data.subleadJetEta>-10.]
    '''
    qqh_sig_2 = np.array(data_temp[data_temp['proc_new'] == 'qqH'][variable2])
    qqh_sig_2_w = np.array(data_temp[data_temp['proc_new'] == 'qqH']['weight'])

    vh_sig_2 = np.array(data_temp[data_temp['proc_new'] == 'VH'][variable2])
    vh_sig_2_w = np.array(data_temp[data_temp['proc_new'] == 'VH']['weight'])

    ggh_sig_2 = np.array(data_temp[data_temp['proc_new'] == 'ggH'][variable2])
    ggh_sig_2_w = np.array(data_temp[data_temp['proc_new'] == 'ggH']['weight'])

    tth_sig_2= np.array(data_temp[data_temp['proc_new'] == 'ttH'][variable2])
    tth_sig_2_w = np.array(data_temp[data_temp['proc_new'] == 'ttH']['weight'])
    '''
    th_sig_2 = np.array(data_temp[data_temp['proc_new'] == 'tH'][variable2])
    th_sig_2_w = np.array(data_temp[data_temp['proc_new'] == 'tH']['weight'])

    #combined_sig_2 = np.concatenate((qqh_sig_2,vh_sig_2,ggh_sig_2,tth_sig_2,th_sig_2),axis=0)
    #combined_sig_2 = np.concatenate((qqh_sig_2,vh_sig_2,ggh_sig_2,tth_sig_2),axis=0)
    #combined_sig_2_w = np.concatenate((qqh_sig_2_w,vh_sig_2_w,ggh_sig_2_w,tth_sig_2_w,th_sig_2_w),axis=0)
    

    if variable1 != variable2:
      cur_var_pair = [variable1,variable2]
      if cur_var_pair not in list_plots:
        '''
        #Plot all 4 signal modes
        fig, ax = plt.subplots(2,2,sharex=True,sharey=True)
        ax[0][0].hist2d(vbf_sig_1,vbf_sig_2,bins=num_bins,cmap=plt.cm.jet,label='VBF')
        ax[0][0].set_title('VBF')
        ax[0][0].set(ylabel=variable2)
        ax[0][0].set_xlim(-5,5)
        ax[0][0].set_ylim(-5,5)
        ax[0][1].hist2d(vh_sig_1,vh_sig_2,bins=num_bins,cmap=plt.cm.jet,label='VH')
        ax[0][1].set_title('VH')
        ax[0][1].set_xlim(-5,5)
        ax[0][1].set_ylim(-5,5)
        ax[1][0].hist2d(ggh_sig_1,ggh_sig_2,bins=num_bins,cmap=plt.cm.jet,label='ggH')
        ax[1][0].set_title('ggH')
        ax[1][0].set(xlabel=variable1)
        ax[1][0].set(ylabel=variable2)
        ax[0][1].set_xlim(-5,5)
        ax[1][0].set_ylim(-5,5)
        ax[1][1].hist2d(tth_sig_1,tth_sig_2,bins=num_bins,cmap=plt.cm.jet,label='ttH')
        ax[1][1].set_title('ttH')
        ax[1][1].set(xlabel=variable1)
        ax[0][1].set_xlim(-5,5)
        ax[1][1].set_ylim(-5,5)
        plt.suptitle('Correlation Plot {}'.format(variable1 + variable2))
        
        #Plot 2 signal modes
        fig, ax = plt.subplots(2) 
        ax[0].hist2d(vbf_sig_1,vbf_sig_2,bins=num_bins,cmap=plt.cm.jet,label='VBF')
        #ax[0].set_title('VBF')
        ax[0].set(ylabel=variable2)
        ax[0].set_xlim(-5,5)
        ax[0].set_ylim(-5,5)
        ax[0].set_aspect('equal', adjustable='box')
        ax[1].hist2d(ggh_sig_1,ggh_sig_2,bins=num_bins,cmap=plt.cm.jet,label='ggH')
        #ax[1].set_title('ggH')
        ax[1].set(xlabel=variable1)
        ax[1].set(ylabel=variable2)
        ax[1].set_xlim(-5,5)
        ax[1].set_ylim(-5,5)
        ax[1].set_aspect('equal', adjustable='box')
        '''
        '''
        fig, ax = plt.subplots()
        ax.hist2d(ggh_sig_1,ggh_sig_2,bins=num_bins,cmap=plt.cm.jet)
        counts, xedges, yedges, im = ax.hist2d(ggh_sig_1,ggh_sig_2,bins=num_bins,cmap=plt.cm.jet)
        fig.colorbar(im, ax=ax)
        ax.set(xlabel=variable1)
        ax.set(ylabel=variable2)
        name = 'plotting/Correlation_plots/Correlation_' + 'ggH_' + variable1 + '_' + variable2 
        fig.savefig(name, dpi = 1200)
        print("Saved Figure with name", name)

        fig, ax = plt.subplots()
        ax.hist2d(qqh_sig_1,qqh_sig_2,bins=num_bins,cmap=plt.cm.jet)
        counts, xedges, yedges, im = ax.hist2d(qqh_sig_1,qqh_sig_2,bins=num_bins,cmap=plt.cm.jet)
        fig.colorbar(im, ax=ax)
        ax.set(xlabel=variable1)
        ax.set(ylabel=variable2)
        name = 'plotting/Correlation_plots/Correlation_' + 'qqH_' + variable1 + '_' + variable2 
        fig.savefig(name, dpi = 1200)
        print("Saved Figure with name", name)

        fig, ax = plt.subplots()
        ax.hist2d(vh_sig_1,vh_sig_2,bins=num_bins,cmap=plt.cm.jet)
        counts, xedges, yedges, im = ax.hist2d(vh_sig_1,vh_sig_2,bins=num_bins,cmap=plt.cm.jet)
        fig.colorbar(im, ax=ax)
        ax.set(xlabel=variable1)
        ax.set(ylabel=variable2)
        name = 'plotting/Correlation_plots/Correlation_' + 'VH_' + variable1 + '_' + variable2 
        fig.savefig(name, dpi = 1200)
        print("Saved Figure with name", name)

        fig, ax = plt.subplots()
        ax.hist2d(tth_sig_1,tth_sig_2,bins=num_bins,cmap=plt.cm.jet)
        counts, xedges, yedges, im = ax.hist2d(tth_sig_1,tth_sig_2,bins=num_bins,cmap=plt.cm.jet)
        fig.colorbar(im, ax=ax)
        ax.set(xlabel=variable1)
        ax.set(ylabel=variable2)
        name = 'plotting/Correlation_plots/Correlation_' + 'ttH_' + variable1 + '_' + variable2 
        fig.savefig(name, dpi = 1200)
        print("Saved Figure with name", name)

        '''
        fig, ax = plt.subplots()
        ax.hist2d(th_sig_1,th_sig_2,bins=num_bins,cmap=plt.cm.jet)
        counts, xedges, yedges, im = ax.hist2d(th_sig_1,th_sig_2,bins=num_bins,cmap=plt.cm.jet)
        fig.colorbar(im, ax=ax)
        ax.set(xlabel=variable1)
        ax.set(ylabel=variable2)
        name = 'plotting/Correlation_plots/Correlation_' + 'tH_' + variable1 + '_' + variable2 
        fig.savefig(name, dpi = 1200)
        print("Saved Figure with name", name)
        
        
      list_plots.append([variable1,variable2])
      list_plots.append([variable2,variable1])