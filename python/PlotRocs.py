import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
from numpy import trapz
from scipy.integrate import simps
import scipy as sp

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

list_variables = ['min_IDMVA'
]

class ROC:
    """
    """

    def __init__(self, df, list_variables):
        self.df = df
        self.list_variables = list_variables
        self.values_bkg = 0
        self.bins_bkg = 0
        self.values_sig = 0
        self.bins_sig = 0
        self.values_data = 0
        self.bins_data = 0

   
    def plotting(self):
        df = self.df
        list_variables = self.list_variables
       
        for variable in list_variables:
 
            # Signal
 
            vbf_sig_0 = np.array(df[df['proc'] == 'VBF'][variable])
            vbf_sig_w = np.array(df[df['proc'] == 'VBF']['weight'])[(vbf_sig_0 > -10) & (vbf_sig_0 <300)]
            vbf_sig = vbf_sig_0[(vbf_sig_0 > -10) & (vbf_sig_0 < 300)]
            vh_sig_0 = np.array(df[df['proc'] == 'VH'][variable])
            vh_sig_w = np.array(df[df['proc'] == 'VH']['weight'])[(vh_sig_0 > -10) & (vh_sig_0 <300)]
            vh_sig = vh_sig_0[(vh_sig_0 > -10) & (vh_sig_0 <300)]
            ggh_sig_0 = np.array(df[df['proc'] == 'ggH'][variable])
            ggh_sig_w = np.array(df[df['proc'] == 'ggH']['weight'])[(ggh_sig_0 > -10) & (ggh_sig_0 <300)]
            ggh_sig = ggh_sig_0[(ggh_sig_0 > -10) & (ggh_sig_0 <300)]
            tth_sig_0 = np.array(df[df['proc'] == 'ttH'][variable])
            tth_sig_w = np.array(df[df['proc'] == 'ttH']['weight'])[(tth_sig_0 > -10) & (tth_sig_0 <300)]
            tth_sig = tth_sig_0[(tth_sig_0 > -10) & (tth_sig_0 <300)]
            combined_sig = np.concatenate((vbf_sig,vh_sig,ggh_sig,tth_sig),axis=0)
            combined_sig_w = np.concatenate((vbf_sig_w,vh_sig_w,ggh_sig_w,tth_sig_w),axis=0)

            # Background
            qcd_0 = np.array(df[df['proc'] == 'QCD'][variable])
            qcd_w = np.array(df[df['proc'] == 'QCD']['weight'])[(qcd_0 > -10) & (qcd_0 <300)]
            qcd = qcd_0[(qcd_0 > -10) & (qcd_0 <300)]
            gjet_0 = np.array(df[df['proc'] == 'Gjet'][variable])
            gjet_w = np.array(df[df['proc'] == 'Gjet']['weight'])[(gjet_0 > -10) & (gjet_0 <300)]
            gjet = gjet_0[(gjet_0 > -10) & (gjet_0 <300)]
            diphoton_0 = np.array(df[df['proc'] == 'Diphoton'][variable])
            diphoton_w = np.array(df[df['proc'] == 'Diphoton']['weight'])[(diphoton_0 > -10) & (diphoton_0 <300)]
            diphoton = diphoton_0[(diphoton_0 > -10) & (diphoton_0 <300)]
            combined_bkg = np.concatenate((qcd,gjet,diphoton),axis=0)
            combined_bkg_w = np.concatenate((qcd_w,gjet_w,diphoton_w),axis=0)

            # Data
            data = np.array(df[df['proc'] == 'Data'][variable])
            data_w = np.array(df[df['proc'] == 'Data']['weight'])

            # Now let's plot the histogram

            scale = 100
            num_bins = 100
            normalize = True

            fig, ax = plt.subplots(1)

   
            self.values_bkg, self.bins_bkg, _ = plt.hist(combined_bkg, bins = num_bins, density = normalize, color = 'lightskyblue', label = 'Background',  histtype = 'step', weights = combined_bkg_w)
            self.values_sig, self.bins_sig, _ = plt.hist(combined_sig, bins = num_bins, density = normalize, color = 'blue', label = 'Signal',  histtype = 'step', weights = scale * combined_sig_w)
            self.values_data, self.bins_data, _ = plt.hist(data, bins = num_bins, density = normalize, color = 'black', label = 'Data', histtype = 'step', weights = data_w)

            plt.legend()
            plt.xlabel(variable)
            plt.ylabel('Events')
 
            if variable == 'min_IDMVA':
                plt.xlim(-1,1)

            name = 'plotting/plots/ROC_' + variable
            #fig.savefig(name)

            return self.values_bkg, self.bins_bkg, self.values_sig, self.bins_sig, self.values_data, self.bins_data
       
    def roc_curve(self):

      self.plotting()   # call the plotting to get the bins and values of the histograms

      values_bkg = self.values_bkg
      values_sig = self.values_sig
      bins_bkg = self.bins_bkg
      bins_sig = self.bins_sig

      sum_bkg_all = np.sum(values_bkg)
      sum_sig_all = np.sum(values_sig)

      epsilon_b = []
      epsilon_s = []

      for i in range(len(bins_bkg)):    
        # bkg
        values_bkg_new = values_bkg[i:] # to keep the ones that pass the cut
        eps_b = np.sum(values_bkg_new) / sum_bkg_all
        epsilon_b.append(eps_b)

        # sig
        values_sig_new = values_sig[i:] # to keep the ones that pass the cut
        eps_s = np.sum(values_sig_new) / sum_sig_all
        epsilon_s.append(eps_s)
      #print(epsilon_b)
      #print(epsilon_s)
      # now let's plot the ROC curve
         
      fig, ax = plt.subplots(1)
      plt.plot(epsilon_b, epsilon_s, 'o')
      plt.xlabel('$\epsilon_b$')
      plt.ylabel('$\epsilon_s$')
     

      # finding AUC (area under curve)
      epsilon_b.sort() # need to put them in ascending order
      #print(epsilon_b)
      epsilon_s.sort()
      #print(epsilon_s)
      # composite trapezoidal rule
      #area_trapz = trapz(epsilon_s, epsilon_b)
      #print('Area using trapz: ', area_trapz)
      # composite Simpson's rule
      area_simps = simps(epsilon_s, epsilon_b)
      #print('Area using simps: ', area_simps)

      variable = self.list_variables[0]   # this will have to change - will add a loop once we incorporate more variables in etc
      title = variable + ' with Area = {}'.format(round(area_simps,3))
      plt.title(title)
      name = 'plotting/plots/ROC_min_IDMVA_curve'
      fig.savefig(name)  

# Running commands
roc = ROC(df, list_variables)
roc.roc_curve()