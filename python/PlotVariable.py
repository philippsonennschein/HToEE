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
for f in files_mc_csv:
  dataframes.append( pd.read_csv(f) )
  print " --> Read: %s"%f
for f in files_mc_data:
  dataframes.append( pd.read_csv(f) )
  print " --> Read: %s"%f

df = pd.concat( dataframes, sort=False, axis=0 )
print " --> Successfully read dataframes"

#print df.shape
#df = df.dropna()
#print df.shape


# -------------------------------
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

#df = df.drop(columns=['genWeight'])
#print df.shape
#df = df.dropna()
#print df.shape

# -------------------------------
list_variables = [
'leadPhotonIDMVA','subleadPhotonIDMVA',
#'min_IDMVA','max_IDMVA',
#'diphotonMass','diphotonPt',
#'leadPhotonPtOvM','subleadPhotonPtOvM',
#'leadPhotonEta','subleadPhotonEta',
#'dijetMass','dijetAbsDEta','dijetDPhi',
#'leadJetPt','leadJetEn','leadJetEta','leadJetPhi',
#'subleadJetPt','subleadJetEn','subleadJetEta','subleadJetPhi',
#'subsubleadJetPt','subsubleadJetEn','subsubleadJetEta','subsubleadJetPhi'
]

for variable in list_variables:

  #df[variable].fillna(1.1, inplace=True)

  # Splitting into the different parts
  # Signal
  vbf_sig = np.array(df[df['proc'] == 'VBF'][variable])
  vbf_sig_w = np.array(df[df['proc'] == 'VBF']['weight'])
  vh_sig = np.array(df[df['proc'] == 'VH'][variable])
  vh_sig_w = np.array(df[df['proc'] == 'VH']['weight'])
  ggh_sig = np.array(df[df['proc'] == 'ggH'][variable])
  ggh_sig_w = np.array(df[df['proc'] == 'ggH']['weight'])
  tth_sig = np.array(df[df['proc'] == 'ttH'][variable])
  tth_sig_w = np.array(df[df['proc'] == 'ttH']['weight'])
  combined_sig = np.concatenate((vbf_sig,vh_sig,ggh_sig,tth_sig),axis=0)
  combined_sig_w = np.concatenate((vbf_sig_w,vh_sig_w,ggh_sig_w,tth_sig_w),axis=0)

  # Background
  qcd = np.array(df[df['proc'] == 'QCD'][variable])
  gjet = np.array(df[df['proc'] == 'Gjet'][variable])
  diphoton = np.array(df[df['proc'] == 'Diphoton'][variable])
  qcd_w = np.array(df[df['proc'] == 'QCD']['weight'])
  gjet_w = np.array(df[df['proc'] == 'Gjet']['weight'])
  diphoton_w = np.array(df[df['proc'] == 'Diphoton']['weight'])
  combined_bkg = np.concatenate((qcd,gjet,diphoton),axis=0)
  combined_bkg_w = np.concatenate((qcd_w,gjet_w,diphoton_w),axis=0)

  # Data
  data = np.array(df[df['proc'] == 'Data'][variable])
  data_w = np.array(df[df['proc'] == 'Data']['weight'])

  #count = 0
  #for i in data:
  #  if i == -999.0:
  #    count = count + 1
  #print variable
  #print count

  # Now let's plot the histogram
  fig, ax = plt.subplots(1)

  num_bins = 1000
  scale = 10**2
  normalise = True

  #background
  #plt.hist(qcd, bins = num_bins, density = normalise, color = 'lightgrey', label = 'QCD background', histtype = 'step', weights = qcd_w)
  #plt.hist(gjet, bins = num_bins, density = normalise, color = 'lightgreen', label = 'Gjet background',  histtype = 'step', weights = gjet_w)
  #plt.hist(diphoton, bins = num_bins, density = normalise, color = 'lightskyblue', label = 'Diphoton background',  histtype = 'step', weights = diphoton_w)
  #plt.hist(combined_bkg, bins = num_bins, density = normalise, color = 'lightskyblue', label = 'Background',  histtype = 'step', weights = combined_bkg_w)

  #signal
  #plt.hist(vbf_sig, bins = num_bins, density = normalise, color = 'firebrick', label = 'VBF', histtype = 'step', weights = vbf_sig_w*scale)
  #plt.hist(vh_sig, bins = num_bins, density = normalise, color = 'cyan', label = 'VH', stacked = True, histtype = 'step', weights = vh_sig_w*scale)
  #plt.hist(ggh_sig, bins = num_bins, density = normalise, color = 'green', label = 'ggH', stacked = True, histtype = 'step', weights = ggh_sig_w*scale)
  #plt.hist(tth_sig, bins = num_bins, density = normalise, color = 'coral', label = 'ttH', stacked = True, histtype = 'step', weights = tth_sig_w*scale)
  #plt.hist(combined_sig, bins = num_bins, density = normalise, color = 'coral', label = 'Signal', stacked = True, histtype = 'step', weights = combined_sig_w*scale)

  # data
  #plt.hist(data, bins = num_bins, color = 'blue', label = 'Data', stacked = True, histtype = 'step', weights = data_w)


  plt.legend()
  plt.xlabel(variable)
  plt.ylabel('Events')

  #Xlim for the diphoton Mass
  if variable == 'diphotonMass':
    plt.xlim(120,130)
  elif variable == 'diphotonPt':
    plt.xlim(0,140)
  elif variable == 'leadPhotonIDMVA':
    plt.xlim(-1,1)
    #plt.legend(loc='upper left')
  elif variable == 'subleadPhotonIDMVA':
    plt.xlim(-1,1)
    #plt.legend(loc='upper left')
  elif variable == 'leadJetPt':
    plt.xlim(0,400)
    #plt.ylim(0,0.03)
  elif variable == 'subleadJetPt':
    plt.xlim(0,250)
    #plt.ylim(0,0.03)
  elif variable == 'leadPhotonEta':
    plt.xlim(-3,3)
  elif variable == 'subleadPhotonEta':
    plt.xlim(-3,3)
  elif variable == 'dijetMass':
    plt.xlim(0,300)
    plt.ylim(0,0.06)
  elif variable == 'leadPhotonPtOvM':
    plt.xlim(0,3)
  elif variable == 'subleadPhotonPtOvM':
    plt.xlim(0,3)
  elif variable == 'max_IDMVA':
    plt.xlim(-1,1)
  elif variable == 'min_IDMVA':
    plt.xlim(-1,1)

  name = 'plotting/plots/' + variable
  fig.savefig(name)
  plt.show()