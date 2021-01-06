#data handling imports
import uproot as upr
import numpy as np
import pandas as pd
import os
from ROOT import TLorentzVector as lv
from numpy import pi
from os import path, system
from variables import nominal_vars, gen_vars, gev_vars
import yaml


#BDT imports
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import pickle
import warnings

#NN imports. Will eventually migrate NN to separate file
import keras
from keras.models import Sequential
from keras.initializers import RandomNormal
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import Nadam
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from pickle import load, dump

#plotting imports
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
try:
     plt.style.use("cms10_6_HP")
except IOError:
     warnings.warn('Could not import user defined matplot style file. Using default style settings...')
import scipy.stats



class SampleObject(object):
    '''
    Class to store attributes of each sample. One object to be used per year, per sample -
    essentially per root file
    ''' 

    def __init__(self, proc_tag, year, file_name, tree_path):
        self.proc_tag  = proc_tag
        self.year      = year
        self.file_name = file_name
        self.tree_name = tree_path

class ROOTHelpers(object):
    '''
    Class produce dataframes from any number of signal, background, or data processes 
    for multiple years of data taking

    :mc_dir: directory where root files for simulation are held. Files for all years should be in this directory
    :data_dir: directory where root files for data are held. Files for all years should be in this directory
    '''
  
    def __init__(self, out_tag, mc_dir, mc_fnames, data_dir, data_fnames, proc_to_tree_name, train_vars, vars_to_add, presel_str=''):
        self.years              = set()
        self.lumi_map           = {'2016':35.9, '2017':41.5, '2018':59.7}
        self.lumi_scale         = True
        self.XS_map             = {'ggH':48.58*5E-9, 'VBF':3.782*5E-9, 'DYMC': 6225.4, 'TT2L2Nu':86.61, 'TTSemiL':358.57} #all in pb. also have BR for signals
        self.eff_acc            = {'ggH':0.4515728, 'VBF':0.4670169, 'DYMC':0.0748512, 'TT2L2Nu':0.0405483, 'TTSemiL':0.0003810} #from dumper. update if selection changes

        self.out_tag            = out_tag
        self.mc_dir             = mc_dir #FIXME: remove '\' using if_ends_with()
        self.data_dir           = data_dir
       
        self.sig_procs          = []
        self.sig_objects        = []
        for proc, year_to_file in mc_fnames['sig'].items():
            if proc not in self.sig_procs: self.sig_procs.append(proc) 
            else: raise IOError('Multiple versions of same signal proc trying to be read')
            for year, file_name in year_to_file.iteritems():
                self.years.add(year)
                self.sig_objects.append( SampleObject(proc, year, file_name, proc_to_tree_name[proc]) )
 
        self.bkg_procs          = []
        self.bkg_objects        = []
        for proc, year_to_file in mc_fnames['bkg'].items():
            if proc not in self.bkg_procs: self.bkg_procs.append(proc) 
            else: raise IOError('Multiple versions of same background proc trying to be read')
            for year, file_name in year_to_file.iteritems():
                if year not in self.years:  raise IOError('Incompatible sample years')
                self.bkg_objects.append( SampleObject(proc, year, file_name, proc_to_tree_name[proc]) )

        self.data_objects       = []
        for proc, year_to_file in data_fnames.items():
            for year, file_name in year_to_file.iteritems():
                if year not in self.years:  raise IOError('Incompatible sample years')
                self.data_objects.append( SampleObject(proc, year, file_name, proc_to_tree_name[proc]) )

        self.mc_df_sig          = []
        self.mc_df_bkg          = []
        self.data_df            = []

        if vars_to_add is None: vars_to_add = {}
        self.vars_to_add        = vars_to_add
        missing_vars = [x for x in train_vars if x not in (nominal_vars+list(vars_to_add.keys()))]
        if len(missing_vars)!=0: raise IOError('Missing variables: {}'.format(missing_vars))
        self.nominal_vars       = nominal_vars
        self.train_vars         = train_vars

        self.cut_string         = presel_str

    def no_lumi_scale(self):
        ''' 
        bool for lumi scale
        '''
        self.lumi_scale=False

    def load_mc(self, sample_obj, bkg=False, reload_samples=False):
        '''
        Try to load mc dataframe. If it doesn't exist, read in the root file.
        This should be used once per year, if reading in multiple years.
        '''
        try: 
            if reload_samples: raise IOError
            elif not bkg: self.mc_df_sig.append( self.load_df(self.mc_dir+'DataFrames/', sample_obj.proc_tag, sample_obj.year) )
            else: self.mc_df_bkg.append( self.load_df(self.mc_dir+'DataFrames/', sample_obj.proc_tag, sample_obj.year) )
        except IOError: 
            if not bkg: self.mc_df_sig.append( self.root_to_df(self.mc_dir, 
                                                               sample_obj.proc_tag,
                                                               sample_obj.file_name,
                                                               sample_obj.tree_name,
                                                               'sig', sample_obj.year
                                                              )
                                             )
            else: self.mc_df_bkg.append( self.root_to_df(self.mc_dir,
                                                         sample_obj.proc_tag,
                                                         sample_obj.file_name, 
                                                         sample_obj.tree_name,
                                                         'bkg', sample_obj.year
                                                        )
                                       )

    def load_data(self, sample_obj, reload_samples=False):
        '''
        Try to load Data dataframe. If it doesn't exist, read in the root file.
        This should be used once per year, if reading in multiple years.
        '''
        try: 
            if reload_samples: raise IOError
            else: self.data_df.append( self.load_df(self.data_dir+'DataFrames/', 'Data', sample_obj.year) )
        except IOError: 
            self.data_df.append( self.root_to_df(self.data_dir, sample_obj.proc_tag, sample_obj.file_name, sample_obj.tree_name, 'Data', sample_obj.year) )

    def load_df(self, df_dir, proc, year):
        print 'loading {}{}_{}_df_{}.h5'.format(df_dir, proc, self.out_tag, year)
        df = pd.read_hdf('{}{}_{}_df_{}.h5'.format(df_dir, proc, self.out_tag, year))

        missing_vars = [x for x in self.train_vars if x not in df.columns]
        if len(missing_vars)!=0: raise IOError('Missing variables in dataframe: {}. Reload with option -r and try again'.format(missing_vars))

        else: print('Sucessfully loaded DataFrame: {}{}_{}_df_{}.h5'.format(df_dir, proc, self.out_tag, year))
        return df    

    def root_to_df(self, file_dir, proc_tag, file_name, tree_name, flag, year):
        '''
        Load a single .root dataset for simulation. Apply any preselection and lumi scaling
        If reading in simulated samples, apply lumi scaling and read in gen-level variables too
        '''
        print('Reading {} file: {}, for year: {}'.format(proc_tag, file_dir+file_name, year))
        df_file = upr.open(file_dir+file_name)
        df_tree = df_file[tree_name]
        del df_file

        if flag == 'Data':
            #can cut on data now as dont need to run MC_norm
            data_vars = self.nominal_vars
            #needed for preselection and training
            #df = df_tree.pandas.df(data_vars.remove('genWeight')).query('dielectronMass>110 and dielectronMass<150 and dijetMass>250 and leadJetPt>40 and subleadJetPt>30')
            #FIXME: temp fix until ptOm in samples. Then can just do normal query string again (which is set up to to only read wider mass range if pT reweighting)
            #df = df_tree.pandas.df(data_vars.remove('genWeight')).query('dielectronMass>80 and dielectronMass<150')
            df = df_tree.pandas.df(data_vars.remove('genWeight')).query('dielectronMass>80 and dielectronMass<150')
            df['leadElectronPToM'] = df['leadElectronPt']/df['dielectronMass'] 
            df['subleadElectronPToM'] = df['leadElectronPt']/df['dielectronMass']
            df = df.query(self.cut_string)
            df['weight'] = np.ones_like(df.shape[0])
        else:
            #cant cut on sim now as need to run MC_norm and need sumW before selection!
            df = df_tree.pandas.df(self.nominal_vars)
            #needed for preselection and training
            df['leadElectronPToM'] = df['leadElectronPt']/df['dielectronMass']
            df['subleadElectronPToM'] = df['leadElectronPt']/df['dielectronMass']
            df['weight'] = df['genWeight']
            #dont apply cuts yet as need to do MC norm!


        if len(self.cut_string)>0:
            if flag != 'Data':
                df = self.MC_norm(df, proc_tag, year)
                df = df.query(self.cut_string)
        else:
            if flag != 'Data':
                df = self.MC_norm(df, proc_tag, year)

        df = df.sample(frac=1).reset_index(drop=True)
        df = df.dropna()
        df['proc'] = proc_tag
        df['year'] = year

        print('Number of events in final dataframe: {}'.format(np.sum(df['weight'].values)))
        #save everything
        Utils.check_dir(file_dir+'DataFrames/') 
        df.to_hdf('{}/{}_{}_df_{}.h5'.format(file_dir+'DataFrames', proc_tag, self.out_tag, year), 'df', mode='w', format='t')
        print('Saved dataframe: {}/{}_{}_df_{}.h5'.format(file_dir+'DataFrames', proc_tag, self.out_tag, year))

        return df

    def MC_norm(self, df, proc_tag, year):
        '''
        normalisation to perform before prelection
        '''
        #Do scaling that used to happen in flashgg: XS * BR(for sig only) eff * acc
        sum_w_initial = np.sum(df['weight'].values)
        print 'scaling by {} by XS: {}'.format(proc_tag, self.XS_map[proc_tag])
        df['weight'] *= (self.XS_map[proc_tag]) 
        if self.lumi_scale: #should not be doing this in the final Tag producer
            print 'scaling by {} by Lumi: {} * 1000 /pb'.format(proc_tag, self.lumi_map[year])
            df['weight'] *= self.lumi_map[year]*1000 #lumi is added earlier but XS is in pb, so need * 1000
        print 'scaling by {} by eff*acc: {}'.format(proc_tag, self.eff_acc[proc_tag])
        df['weight'] *= (self.eff_acc[proc_tag])
        df['weight'] /= sum_w_initial
        print 'sumW for proc {}: {}'.format(proc_tag, np.sum(df['weight'].values))
        return df

    def apply_more_cuts(self, cut_string):
        '''
        Apply some additional cut, after nominal preselection when file is read in
        '''
        self.mc_df_sig          = self.mc_df_sig.query(cut_string)
        self.mc_df_bkg          = self.mc_df_bkg.query(cut_string)
        self.data_df            = self.data_df.query(cut_string)

    def concat(self):
        '''
        Concat sample types (sig, bkg, data) together, if more than one df in the associated sample type list.
        Years will also be automatically concatennated over. Could split this up into another function if desired
        but year info is only needed for lumi scaling.
        If the list is empty (not reading anything), leave it empty
        '''
        if len(self.mc_df_sig) == 1: self.mc_df_sig = self.mc_df_sig[0]
        elif len(self.mc_df_sig) == 0: pass
        else: self.mc_df_sig = pd.concat(self.mc_df_sig)

        if len(self.mc_df_bkg) == 1: self.mc_df_bkg = self.mc_df_bkg[0] 
        elif len(self.mc_df_bkg) == 0: pass
        else: self.mc_df_bkg = pd.concat(self.mc_df_bkg)

        if len(self.data_df) == 1: self.data_df = self.data_df[0] 
        elif len(self.data_df) == 0 : pass
        else: self.data_df = pd.concat(self.data_df)
   
    def pt_reweight(self, bkg_proc, year, presel):
        '''
        Derive a reweighting for a single bkg process in a m(ee) control region around the Z-peak, in bins on pT(ee),
        to map bkg process to Data. Then apply this in the signal region
        '''
        pt_bins = np.linspace(0,250,51)
        scaled_list = []

        bkg_df = self.mc_df_bkg.query('proc=="{}" and year=="{}" and dielectronMass>70 and dielectronMass<110'.format(bkg_proc,year))
        bkg_pt_binned, _ = np.histogram(bkg_df['dielectronPt'], bins=pt_bins, weights=bkg_df['weight'])

        data_df = self.data_df.query('year=="{}" and dielectronMass>70 and dielectronMass<110'.format(year))       
        data_pt_binned, bin_edges = np.histogram(data_df['dielectronPt'], bins=pt_bins)
        scale_factors = data_pt_binned/bkg_pt_binned

        #now apply the proc targeting selection on all dfs, and re-save 
        self.apply_more_cuts(presel)
        self.mc_df_bkg['weight'] = self.mc_df_bkg.apply(self.pt_njet_reweight_helper, axis=1, args=[bkg_proc, year, bin_edges, scale_factors, False])
        self.save_modified_dfs(year)


    def pt_njet_reweight(self, bkg_proc, year, presel):
        '''
        Derive a reweighting for a single bkg process in a m(ee) control region around the Z-peak, in bins on pT(ee) and nJets,
        to map bkg process to Data. Then apply this in the signal region
        '''

        #can remove this once nJets is put in ntuples from dumper
        outcomes_mc_bkg = [ self.mc_df_bkg['leadJetPt'].lt(0),
                            self.mc_df_bkg['leadJetPt'].gt(0) & self.mc_df_bkg['subleadJetPt'].lt(0), 
                            self.mc_df_bkg['leadJetPt'].gt(0) & self.mc_df_bkg['subleadJetPt'].gt(0)
                          ]

        outcomes_data   = [ self.data_df['leadJetPt'].lt(0),
                            self.data_df['leadJetPt'].gt(0) & self.data_df['subleadJetPt'].lt(0), 
                            self.data_df['leadJetPt'].gt(0) & self.data_df['subleadJetPt'].gt(0)
                          ]
        jets    = [0, 1, 2] # 2 really means nJet >= 2

        self.mc_df_bkg['nJets'] = np.select(outcomes_mc_bkg, jets) 
        self.data_df['nJets'] = np.select(outcomes_data, jets) 

        #apply re-weighting
        pt_bins = np.linspace(0,200,101)
        jet_bins = [0,1,2]
        n_jets_to_sfs_map = {}

        #derive pt and njet based SFs
        for n_jets in jet_bins:
            if not n_jets==jet_bins[-1]: 
                bkg_df = self.mc_df_bkg.query('proc=="{}" and year=="{}" and dielectronMass>70 and dielectronMass<110 and nJets=={}'.format(bkg_proc,year, n_jets))
                data_df = self.data_df.query('year=="{}" and dielectronMass>70 and dielectronMass<110 and nJets=={}'.format(year,n_jets))       
            else: 
                bkg_df = self.mc_df_bkg.query('proc=="{}" and year=="{}" and dielectronMass>70 and dielectronMass<110 and nJets>={}'.format(bkg_proc,year, n_jets))
                data_df = self.data_df.query('year=="{}" and dielectronMass>70 and dielectronMass<110 and nJets>={}'.format(year,n_jets))       

            bkg_pt_binned, _ = np.histogram(bkg_df['dielectronPt'], bins=pt_bins, weights=bkg_df['weight'])
            data_pt_binned, bin_edges = np.histogram(data_df['dielectronPt'], bins=pt_bins)
            n_jets_to_sfs_map[n_jets] = data_pt_binned/bkg_pt_binned

        #now apply the proc targeting selection on all dfs, and re-save. Then apply derived SFs
        self.apply_more_cuts(presel)
        self.mc_df_bkg['weight'] = self.mc_df_bkg.apply(self.pt_njet_reweight_helper, axis=1, args=[bkg_proc, year, bin_edges, n_jets_to_sfs_map, True])
        self.save_modified_dfs(year)
         
    def pt_njet_reweight_helper(self, row, bkg_proc, year, bin_edges, scale_factors, do_jets):
        '''
        Tests which pT a bkg proc is, and if it is the proc to reweight, before
        applying a pT dependent scale factor to apply (derived from CR)
        
        If dielectron pT is above the max pT bin, just return the nominal weight (very small num of events)
        '''
        if row['proc']==bkg_proc and row['year']==year and row['dielectronPt']<bin_edges[-1]:
            if do_jets: rew_factors = scale_factors[row['nJets']]
            else: rew_factors = scale_factors
            for i_bin in range(len(bin_edges)):
                if (row['dielectronPt'] > bin_edges[i_bin]) and (row['dielectronPt'] < bin_edges[i_bin+1]):
                    return row['weight'] * rew_factors[i_bin]
        else:
            return row['weight']


    def save_modified_dfs(self,year):
        '''
        Save dataframes again. Useful if modifications were made since reading in and saving e.g. pT reweighting or applying more selection
        (or both).
        '''

        print 'saving modified dataframes...'
        for sig_proc in self.sig_procs:
            sig_df = self.mc_df_sig[np.logical_and(self.mc_df_sig.proc==sig_proc, self.mc_df_sig.year==year)]
            sig_df.to_hdf('{}/{}_{}_df_{}.h5'.format(self.mc_dir+'DataFrames', sig_proc, self.out_tag, year), 'df', mode='w', format='t')
            print('saved dataframe: {}/{}_{}_df_{}.h5'.format(self.mc_dir+'DataFrames', sig_proc, self.out_tag, year))

        for bkg_proc in self.bkg_procs:
            bkg_df = self.mc_df_bkg[np.logical_and(self.mc_df_bkg.proc==bkg_proc,self.mc_df_bkg.year==year)]
            bkg_df.to_hdf('{}/{}_{}_df_{}.h5'.format(self.mc_dir+'DataFrames', bkg_proc, self.out_tag, year), 'df', mode='w', format='t')
            print('saved dataframe: {}/{}_{}_df_{}.h5'.format(self.mc_dir+'DataFrames', bkg_proc, self.out_tag, year))

        data_df = self.data_df[self.data_df.year==year]
        data_df.to_hdf('{}/{}_{}_df_{}.h5'.format(self.data_dir+'DataFrames', 'Data', self.out_tag, year), 'df', mode='w', format='t')
        print('saved dataframe: {}/{}_{}_df_{}.h5'.format(self.data_dir+'DataFrames', 'Data', self.out_tag, year))



class BDTHelpers(object):

    def __init__(self, data_obj, train_vars, train_frac, eq_train=True):
        #attributes train/test X and y datasets
        self.data_obj         = data_obj
        self.train_vars       = train_vars
        self.train_frac       = train_frac
        self.eq_train         = eq_train

        self.X_train          = None
        self.y_train          = None
        self.train_weights    = None
        self.train_weights_eq = None
        self.y_pred_train     = None
        self.proc_arr_train   = None

        self.X_test           = None
        self.y_test           = None
        self.test_weights     = None
        self.y_pred_test      = None
        self.proc_arr_test    = None

        #attributes for the hp optmisation and cross validation
        self.clf              = xgb.XGBClassifier(objective='binary:logistic', n_estimators=100, 
                                                  eta=0.05, maxDepth=4, min_child_weight=0.01, 
                                                  subsample=0.6, colsample_bytree=0.6, gamma=1)

        self.hp_grid_rnge     = {'learning_rate': [0.01, 0.05, 0.1, 0.3],
                                 'max_depth':[x for x in range(3,10)],
                                 #'min_child_weight':[x for x in range(0,3)], #FIXME: probs not appropriate range for a small sumw!
                                 'gamma': np.linspace(0,5,6).tolist(),
                                 'subsample': [0.5, 0.8, 1.0],
                                 'n_estimators':[200,300,400,500]
                                }

        self.X_folds_train    = []
        self.y_folds_train    = []
        self.X_folds_validate = []
        self.y_folds_validate = []
        self.w_folds_train    = []
        self.w_folds_train_eq = []
        self.w_folds_validate = []
        self.validation_rocs  = []

        #attributes for plotting. 
        self.plotter          = Plotter(data_obj, train_vars)
        self.sig_procs        = np.unique(data_obj.mc_df_sig['proc']).tolist()
        self.bkg_procs        = np.unique(data_obj.mc_df_bkg['proc']).tolist()
        del data_obj
        
    def create_X_and_y(self, mass_res_reweight=True):
        '''
        Create X train/test and y train/test

        mass_res_reweight scales the signal by its mass resolution. This is done before self.eq_train, which scales the signal by its mass resolution
        '''
        mc_df_sig = self.data_obj.mc_df_sig
        mc_df_bkg = self.data_obj.mc_df_bkg

        #add y_target label (1 for signal, 0 for background)
        mc_df_sig['y'] = np.ones(self.data_obj.mc_df_sig.shape[0]).tolist()
        mc_df_bkg['y'] = np.zeros(self.data_obj.mc_df_bkg.shape[0]).tolist()


        if self.eq_train: 
            if mass_res_reweight:
                #be careful not to change the real weight variable, or test scores will be invalid
                mc_df_sig['MoM_weight'] = (mc_df_sig['weight']) * (1./mc_df_sig['dielectronSigmaMoM'])
                b_to_s_ratio = np.sum(mc_df_bkg['weight'].values)/np.sum(mc_df_sig['MoM_weight'].values)
                mc_df_sig['eq_weight'] = (mc_df_sig['MoM_weight']) * (b_to_s_ratio)
            else: 
                 b_to_s_ratio = np.sum(mc_df_bkg['weight'].values)/np.sum(mc_df_sig['weight'].values)
                 mc_df_sig['eq_weight'] = (mc_df_sig['weight']) * (b_to_s_ratio) 
            mc_df_bkg['eq_weight'] = mc_df_bkg['weight']

            Z_tot = pd.concat([mc_df_sig, mc_df_bkg], ignore_index=True)
            X_train, X_test, train_w, test_w, train_w_eqw, test_w_eqw, y_train, y_test, proc_arr_train, proc_arr_test = train_test_split(Z_tot[self.train_vars], Z_tot['weight'], 
                                                                                                                                         Z_tot['eq_weight'], Z_tot['y'], Z_tot['proc'],
                                                                                                                                         train_size=self.train_frac, 
                                                                                                                                         test_size=1-self.train_frac,
                                                                                                                                         shuffle=True, 
                                                                                                                                         random_state=1357
                                                                                                                                         )
            #NB: will never test/evaluate with equalised weights. This is explicitly why we set another train weight attribute, 
            #    because for overtraining we need to evaluate on the train set (and hence need nominal MC train weights)
            self.train_weights_eq = train_w_eqw.values
        elif mass_res_reweight:
           mc_df_sig['MoM_weight'] = (mc_df_sig['weight']) * (1./mc_df_sig['dielectronSigmaMoM'])
           Z_tot = pd.concat([mc_df_sig, mc_df_bkg], ignore_index=True)
           X_train, X_test, train_w, test_w, train_w_eqw, test_w_eqw, y_train, y_test, proc_arr_train, proc_arr_test = train_test_split(Z_tot[self.train_vars], Z_tot['weight'], 
                                                                                                                                        Z_tot['MoM_weight'], Z_tot['y'], Z_tot['proc'],
                                                                                                                                        train_size=self.train_frac, 
                                                                                                                                        test_size=1-self.train_frac,
                                                                                                                                        shuffle=True, 
                                                                                                                                        random_state=1357
                                                                                                                                        )
           self.train_weights_eq = train_w_eqw.values
           self.eq_train = True #use alternate weight in training. could probs rename this to something better
        else:
           print 'not applying any reweighting...'
           Z_tot = pd.concat([mc_df_sig, mc_df_bkg], ignore_index=True)
           X_train, X_test, train_w, test_w, y_train, y_test, proc_arr_train, proc_arr_test = train_test_split(Z_tot[self.train_vars], 
                                                                                                               Z_tot['weight'],
                                                                                                               Z_tot['y'], Z_tot['proc'],
                                                                                                               train_size=self.train_frac, 
                                                                                                               test_size=1-self.train_frac,
                                                                                                               shuffle=True, random_state=1357
                                                                                                               )
        self.X_train          = X_train.values
        self.y_train          = y_train.values
        self.train_weights    = train_w.values
        self.proc_arr_train   = proc_arr_train

        self.X_test           = X_test.values
        self.y_test           = y_test.values
        self.test_weights     = test_w.values
        self.proc_arr_test    = proc_arr_test

        self.X_data_train, self.X_data_test = train_test_split(self.data_obj.data_df[self.train_vars], train_size=self.train_frac, test_size=1-self.train_frac, shuffle=True, random_state=1357)

    def train_classifier(self, file_path, save=False, model_name='my_model'):
        if self.eq_train: train_weights = self.train_weights_eq
        else: train_weights = self.train_weights

        print 'Training classifier... '
        clf = self.clf.fit(self.X_train, self.y_train, sample_weight=train_weights)
        print 'Finished Training classifier!'
        self.clf = clf

        Utils.check_dir(os.getcwd() + '/models')
        if save:
            pickle.dump(clf, open("{}/models/{}.pickle.dat".format(os.getcwd(), model_name), "wb"))
            print ("Saved classifier as: {}/models/{}.pickle.dat".format(os.getcwd(), model_name))

    def batch_gs_cv(self, k_folds=3):
        '''
        Submit a sets of hyperparameters permutations (based on attribute hp_grid_rnge) to the IC batch.
        Perform k-fold cross validation; take care to separate training weights, which
        may be modified w.r.t nominal weights, and the weights used when evaluating on the
        validation set which should be the nominal weights
        '''
        #get all possible HP sets from permutations of the above dict
        hp_perms = self.get_hp_perms()
        #submit job to the batch for the given HP range:
        for hp_string in hp_perms:
            Utils.sub_hp_script(self.eq_train, hp_string, k_folds)
            
    def get_hp_perms(self):
        from itertools import product
        ''''
        returns list of all possible hyper parameter combinations in format 'hp1:val1,hp2:val2, ...'
        '''
        hp_perms  = [perm for perm in apply(product, self.hp_grid_rnge.values())]
        final_hps = []
        counter   = 0
        for hp_perm in hp_perms:
            l_entry = ''
            for hp_name, hp_value in zip(self.hp_grid_rnge.keys(), hp_perm):
                l_entry+='{}:{},'.format(hp_name,hp_value)
                counter+=1
                if (counter % len(self.hp_grid_rnge.keys())) == 0: final_hps.append(l_entry[:-1])
        return final_hps

    def set_hyper_parameters(self, hp_string):
        hp_dict = {}
        for params in hp_string.split(','):
            hp_name = params.split(':')[0]
            hp_value =params.split(':')[1]
            try: hp_value = int(hp_value)
            except ValueError: hp_value = float(hp_value)
            hp_dict[hp_name] = hp_value
        self.clf = xgb.XGBClassifier(**hp_dict)
 
    def set_k_folds(self, k_folds):
        '''
        Partition the X and Y matrix into folds = k_folds, and append to list (X and y separate) attribute for the class, from the training samples (i.e. X_train -> X_train + X_validate, and same for y and w)
        Used in conjunction with the get_i_fold function to pull one fold out for training+validating
        Note that validation weights should always be the nominal MC weights
        '''
        kf = KFold(n_splits=k_folds)
        for train_index, valid_index in kf.split(self.X_train):
            self.X_folds_train.append(self.X_train[train_index])
            self.X_folds_validate.append(self.X_train[valid_index])

            self.y_folds_train.append(self.y_train[train_index])
            self.y_folds_validate.append(self.y_train[valid_index])

            #deal with two possible train weight scenarios
            self.w_folds_train.append(self.train_weights[train_index])
            if self.eq_train:
                self.w_folds_train_eq.append(self.train_weights_eq[train_index])

            self.w_folds_validate.append(self.train_weights[valid_index])

       
    def set_i_fold(self, i_fold):
        '''
        Gets the training and validation fold for a given CV iteration from class attribute,
        and overwrites the self.X_train, self.y_train and self.X_train, self.y_train respectively, and the weights, to train
        Note that for these purposes, our "test" sets are really the "validation" sets
        '''
        self.X_train          = self.X_folds_train[i_fold]
        self.train_weights    = self.w_folds_train[i_fold] #nominal MC weights needed for computing roc on train set (overtraining test)
        if self.eq_train:
            self.train_weights_eq = self.w_folds_train_eq[i_fold] 
        self.y_train          = self.y_folds_train[i_fold]

        self.X_test           = self.X_folds_validate[i_fold]
        self.y_test           = self.y_folds_validate[i_fold]
        self.test_weights     = self.w_folds_validate[i_fold]

    def compare_rocs(self, roc_file, hp_string):
        hp_roc = roc_file.readlines()
        avg_val_auc = np.average(self.validation_rocs)
        print 'avg. validation roc is: {}'.format(avg_val_auc)
        if len(hp_roc)==0: 
            roc_file.write('{};{:.4f}'.format(hp_string, avg_val_auc))
        elif float(hp_roc[-1].split(';')[-1]) < avg_val_auc:
            roc_file.write('\n')
            roc_file.write('{};{:.4f}'.format(hp_string, avg_val_auc))

    def compute_roc(self):
        '''
        Compute the area under the associated ROC curve, with mc weights
        '''
        self.y_pred_train = self.clf.predict_proba(self.X_train)[:,1:]
        print 'Area under ROC curve for train set is: {:.4f}'.format(roc_auc_score(self.y_train, self.y_pred_train, sample_weight=self.train_weights))

        self.y_pred_test = self.clf.predict_proba(self.X_test)[:,1:]
        print 'Area under ROC curve for test set is: {:.4f}'.format(roc_auc_score(self.y_test, self.y_pred_test, sample_weight=self.test_weights))
        return roc_auc_score(self.y_test, self.y_pred_test, sample_weight=self.test_weights)

    def plot_roc(self, out_tag):
        ''' 
        Method to plot the roc curve, using method from Plotter() class
        '''
        roc_fig = self.plotter.plot_roc(self.y_train, self.y_pred_train, self.train_weights, 
                                   self.y_test, self.y_pred_test, self.test_weights)

        Utils.check_dir('{}/plotting/plots/{}'.format(os.getcwd(), out_tag))
        roc_fig.savefig('{0}/plotting/plots/{1}/{1}_ROC_curve.pdf'.format(os.getcwd(),out_tag))
        print('saving: {0}/plotting/plots/{1}/{1}_ROC_curve.pdf'.format(os.getcwd(),out_tag))
        plt.close()

    def plot_output_score(self, out_tag, ratio_plot=False, norm_to_data=False):
        ''' 
        Method to plot the roc curve and compute the integral of the roc as a performance metric
        '''
        output_score_fig = self.plotter.plot_output_score(self.y_test, self.y_pred_test, self.test_weights, 
                                                          self.proc_arr_test, self.clf.predict_proba(self.X_data_test.values)[:,1:],
                                                          ratio_plot=ratio_plot, norm_to_data=norm_to_data)

        Utils.check_dir('{}/plotting/plots/{}'.format(os.getcwd(),out_tag))
        output_score_fig.savefig('{0}/plotting/plots/{1}/{1}_output_score.pdf'.format(os.getcwd(), out_tag))
        print('saving: {0}/plotting/plots/{1}/{1}_output_score.pdf'.format(os.getcwd(), out_tag))
        plt.close()


class LSTM_DNN(object):
    '''
    Train a DNN that uses LSTM and fully connected layers

    '''

    def __init__(self, data_obj, low_level_vars, high_level_vars, train_frac, eq_weights=True, batch_boost=False):
        '''
        :batch_boost: option to increase batch size based on ROC improvement. Needed for HP opt.
        '''
        self.data_obj            = data_obj
        self.low_level_vars      = low_level_vars
        self.low_level_vars_flat = [var for sublist in low_level_vars for var in sublist]
        self.high_level_vars     = high_level_vars
        self.train_frac          = train_frac
        self.batch_boost         = batch_boost #needed for HP opt
        self.eq_train            = eq_weights
        self.max_epochs          = 100

        self.X_tot               = None
        self.y_tot               = None

        self.X_train_low_level   = None
        self.X_train_high_level  = None
        self.y_train             = None
        self.train_weights       = None
        self.train_eqw           = None
        self.proc_arr_train      = None
        self.y_pred_train        = None

        self.X_test_low_level    = None
        self.X_test_high_level   = None
        self.y_test              = None
        self.test_weights        = None
        self.proc_arr_test       = None
        self.y_pred_test         = None

        self.X_train_low_level   = None
        self.X_valid_low_level   = None
        self.y_valid             = None
        self.valid_weights       = None

        self.X_data_train_low_level  = None
        self.X_data_train_high_level = None

        self.X_data_test_low_level   = None
        self.X_data_test_high_level  = None
        

        #baseline set:  shows little overtraining and good performance
        #self.set_model(n_lstm_layers=1, n_lstm_nodes=150, n_dense_1=2, n_nodes_dense_1=300, 
        #               n_dense_2=3, n_nodes_dense_2=200, dropout_rate=0.2,
        #               learning_rate=0.00001, batch_norm=True, batch_momentum=0.99)

        self.set_model(n_lstm_layers=1, n_lstm_nodes=150, n_dense_1=2, n_nodes_dense_1=300, 
                       n_dense_2=3, n_nodes_dense_2=200, dropout_rate=0.2,
                       learning_rate=0.001, batch_norm=True, batch_momentum=0.99)

        self.hp_grid_rnge           = {'n_lstm_layers': [1,2,3], 'n_lstm_nodes':[100,150,200], 
                                       'n_dense_1':[1,2,3], 'n_nodes_dense_1':[100,200,300],
                                       'n_dense_2':[1,2,3,4], 'n_nodes_dense_2':[100,200,300], 
                                       'dropout_rate':[0.1,0.2,0.3]
                                      }
        #self.hp_grid_rnge           = {'n_lstm_layers': [1], 'n_lstm_nodes':[150,200], 
        #                               'n_dense_1':[2], 'n_nodes_dense_1':[100],
        #                               'n_dense_2':[2], 'n_nodes_dense_2':[100], 
        #                               'dropout_rate':[0.3]
        #                              }

        #assign plotter attribute before data_obj is deleted for mem
        self.plotter = Plotter(data_obj, self.low_level_vars_flat+self.high_level_vars)
        del data_obj

    def var_transform(self, do_data=False):
        '''
        Takes pandas dataframe of X features. Apply natural log to GeV variables, and change empty variable default values
        '''
        if 'subsubleadJetPt' in (self.low_level_vars_flat+self.high_level_vars):
            self.data_obj.mc_df_sig['subsubleadJetPt'] = self.data_obj.mc_df_sig['subsubleadJetPt'].replace(-9999., 1) #zero after logging
            self.data_obj.mc_df_bkg['subsubleadJetPt'] = self.data_obj.mc_df_bkg['subsubleadJetPt'].replace(-9999., 1) #zero after logging
            if do_data: self.data_obj.data_df['subsubleadJetPt'] = self.data_obj.data_df['subsubleadJetPt'].replace(-9999., 1) #zero after logging

        #df['subsubleadJetEta'] = df['subsubleadJetEta'].replace(-9999., -10) #angles can't be zero because its still meaningfull. ?
        #df['subsubleadJetPhi'] = df['subsubleadJetPhi'].replace(-9999., -10)
        #df['subsubleadJetQGL'] = df['subsubleadJetQGL'].replace(-9999., -10) 

        for var in gev_vars:
            if var in (self.low_level_vars_flat+self.high_level_vars):
                self.data_obj.mc_df_sig[var] = np.log(self.data_obj.mc_df_sig[var].values)
                self.data_obj.mc_df_bkg[var] = np.log(self.data_obj.mc_df_bkg[var].values)
                if do_data: self.data_obj.data_df[var]   = np.log(self.data_obj.data_df[var].values)

    def create_X_y(self, mass_res_reweight=True):
        '''
        Create X and y matrices for training and testing. Apply Z-scaling, and save scaler that is for on train data, for use later

        :do_data: option to also create train and test matrices for data. Used only for plotting, even if running cat opt!
        :mass_res_reweight: re-weight signal events by 1/sigma(m_ee), in training only
        '''
        
        if self.eq_train:
            b_to_s_ratio = np.sum(self.data_obj.mc_df_bkg['weight'].values)/np.sum(self.data_obj.mc_df_sig['weight'].values)
            self.data_obj.mc_df_sig['eq_weight'] = self.data_obj.mc_df_sig['weight'] * b_to_s_ratio 
            self.data_obj.mc_df_bkg['eq_weight'] = self.data_obj.mc_df_bkg['weight'] 
        self.data_obj.mc_df_sig.reset_index(drop=True, inplace=True)
        self.data_obj.mc_df_bkg.reset_index(drop=True, inplace=True)
        X_tot = pd.concat([self.data_obj.mc_df_sig, self.data_obj.mc_df_bkg], ignore_index=True)

        #add y_target label (1 for signal, 0 for background). Keep separate from X-train until after Z-scaling
        y_sig = np.ones(self.data_obj.mc_df_sig.shape[0])
        y_bkg = np.zeros(self.data_obj.mc_df_bkg.shape[0])
        y_tot = np.concatenate((y_sig,y_bkg))
        

        return X_tot, y_tot

    def split_X_y(self, X_tot, y_tot, do_data=False):
        if not self.eq_train:
            self.all_vars_X_train, self.all_vars_X_test, self.train_weights, self.test_weights, self.y_train, self.y_test, self.proc_arr_train, self.proc_arr_test =  train_test_split(X_tot[self.low_level_vars_flat+self.high_level_vars], 
                                                                                                                                                           X_tot['weight'], 
                                                                                                                                                           y_tot,
                                                                                                                                                           X_tot['proc'],
                                                                                                                                                           train_size=self.train_frac, test_size=1-self.train_frac, shuffle=True, random_state=1357
                                                                                                                                                          )
        else:
            self.all_vars_X_train, self.all_vars_X_test, self.train_weights, self.test_weights, self.train_eqw, self.test_eqw, self.y_train, self.y_test, self.proc_arr_train, self.proc_arr_test = train_test_split(X_tot[self.low_level_vars_flat+self.high_level_vars], 
                                                                                                                                                                                        X_tot['weight'],
                                                                                                                                                                                        X_tot['eq_weight'], 
                                                                                                                                                                                        y_tot, 
                                                                                                                                                                                        X_tot['proc'],
                                                                                                                                                                                        train_size=self.train_frac, test_size=1-self.train_frac, shuffle=True, random_state=1357
                                                                                                                                                                                        )
            self.train_weights_eq = self.train_eqw.values


        if do_data: #for plotting purposes
            self.all_X_data_train, self.all_X_data_test  = train_test_split(self.data_obj.data_df[self.low_level_vars_flat+self.high_level_vars],
                                                                  train_size=self.train_frac, 
                                                                  test_size=1-self.train_frac, shuffle=True, random_state=1357)

    def get_X_scaler(self, X_train, out_tag='lstm_scaler'):
        '''
        derive transform on X features to give to zero mean and unit std. Derive on train set. Save for use later
        '''

        X_scaler = StandardScaler()
        X_scaler.fit(X_train.values)
        self.X_scaler = X_scaler
        print('saving X scaler: models/{}_X_scaler.pkl'.format(out_tag))
        dump(X_scaler, open('models/{}_X_scaler.pkl'.format(out_tag),'wb'))

    def load_X_scaler(self, out_tag='lstm_scaler'): 
        '''
        load X feature scaler, where the transform has been derived from training sample
        '''

        self.X_scaler = load(open('models/{}_X_scaler.pkl'.format(out_tag),'rb'))
    
    def X_scale_train_test(self, do_data=False):
        '''
        scale train and test X matrices to give zero mean and unit std. Annoying conversions between numpy <-> pandas but necessary for keeping feature names
        '''

        X_scaled_all_vars_train     = self.X_scaler.transform(self.all_vars_X_train) #returns np array so need to re-cast into pandas to get colums/variables
        X_scaled_all_vars_train     = pd.DataFrame(X_scaled_all_vars_train, columns=self.low_level_vars_flat+self.high_level_vars)
        self.X_train_low_level      = X_scaled_all_vars_train[self.low_level_vars_flat].values #will get changed to 2D arrays later
        self.X_train_high_level     = X_scaled_all_vars_train[self.high_level_vars].values

        X_scaled_all_vars_test      = self.X_scaler.transform(self.all_vars_X_test) #important to use scaler tuned on X train
        X_scaled_all_vars_test      = pd.DataFrame(X_scaled_all_vars_test, columns=self.low_level_vars_flat+self.high_level_vars)
        self.X_test_low_level       = X_scaled_all_vars_test[self.low_level_vars_flat].values #will get changed to 2D arrays later
        self.X_test_high_level      = X_scaled_all_vars_test[self.high_level_vars].values

        if do_data: #for plotting purposes
            X_scaled_data_all_vars_train      = self.X_scaler.transform(self.all_X_data_train)
            X_scaled_data_all_vars_train      = pd.DataFrame(X_scaled_data_all_vars_train, columns=self.low_level_vars_flat+self.high_level_vars)
            self.X_data_train_high_level      = X_scaled_data_all_vars_train[self.high_level_vars].values 
            self.X_data_train_low_level       = X_scaled_data_all_vars_train[self.low_level_vars_flat].values

            X_scaled_data_all_vars_test       = self.X_scaler.transform(self.all_X_data_test)
            X_scaled_data_all_vars_test       = pd.DataFrame(X_scaled_data_all_vars_test, columns=self.low_level_vars_flat+self.high_level_vars)
            self.X_data_test_high_level       = X_scaled_data_all_vars_test[self.high_level_vars].values
            self.X_data_test_low_level        = X_scaled_data_all_vars_test[self.low_level_vars_flat].values
       
    def set_low_level_2D_test_train(self, do_data=False, ignore_train=False):
        '''
        Ignore train means do not join 2D train objects. useful if we want to keep low level as a 1D array
        when splitting train into train+validate. We may do 2D transform on output 1D train and valid sets
        '''
        if not ignore_train: self.X_train_low_level = self.join_objects(self.X_train_low_level)
        self.X_test_low_level   = self.join_objects(self.X_test_low_level)
        if do_data:
            self.X_data_train_low_level  = self.join_objects(self.X_data_train_low_level)
            self.X_data_test_low_level   = self.join_objects(self.X_data_test_low_level)


    def join_objects(self, X_low_level):
        '''
        Function take take all low level objects for each event, and transform into a matrix:
           [ [jet1-pt, jet1-eta, ...,
              jet2-pt, jet2-eta, ...,
              jet3-pt, jet3-eta, ... ]_evt1 ,

             [jet1-pt, jet1-eta, ...,
              jet2-pt, jet2-eta, ...,
              jet3-pt, jet3-eta, ...]_evt2 ,

             ...
           ]
        
        Note that the order of the low level inputs is important, and should be jet objects in descending pT
        '''

        print 'Creating 2D object vars...'
        l_to_convert = []
        for index, row in pd.DataFrame(X_low_level, columns=self.low_level_vars_flat).iterrows(): #very slow
            l_event = []
            for i_object_list in self.low_level_vars:
                l_object = []
                for i_var in i_object_list:
                    l_object.append(row[i_var])
                l_event.append(l_object)
            l_to_convert.append(l_event)
        print 'Finished creating train object vars'
        return np.array(l_to_convert, np.float32)

        
    def set_model(self, n_lstm_layers=3, n_lstm_nodes=150, n_dense_1=1, n_nodes_dense_1=300, n_dense_2=4, n_nodes_dense_2=200, dropout_rate=0.1, learning_rate=0.001, batch_norm=True, batch_momentum=0.99):

        input_objects = keras.layers.Input(shape=(len(self.low_level_vars), len(self.low_level_vars[0])), name='input_objects') 
        input_global  = keras.layers.Input(shape=(len(self.high_level_vars),), name='input_global')
        lstm = input_objects
        for i_layer in range(n_lstm_layers):
            lstm = keras.layers.LSTM(n_lstm_nodes, activation='tanh', return_sequences=(i_layer!=(n_lstm_layers-1)), name='lstm_{}'.format(i_layer))(lstm)

        #inputs to dense layers are output of lstm and global-event variables. Also batch norm the FC layers
        dense = keras.layers.concatenate([input_global, lstm])
        for i in range(n_dense_1):
            dense = keras.layers.Dense(n_nodes_dense_1, activation='relu', kernel_initializer='lecun_uniform', name = 'dense1_%d' % i)(dense)
            if batch_norm:
                dense = keras.layers.BatchNormalization(name = 'dense_batch_norm1_%d' % i)(dense)
        dense = keras.layers.Dropout(rate = dropout_rate, name = 'dense_dropout1_%d' % i)(dense)

        for i in range(n_dense_2):
            dense = keras.layers.Dense(n_nodes_dense_2, activation='relu', kernel_initializer='lecun_uniform', name = 'dense2_%d' % i)(dense)
            #add droput and norm if not on last layer
            if batch_norm and i < (n_dense_2 - 1):
                dense = keras.layers.BatchNormalization(name = 'dense_batch_norm2_%d' % i)(dense) 
            if i < (n_dense_2 - 1):
                dense = keras.layers.Dropout(rate = dropout_rate, name = 'dense_dropout2_%d' % i)(dense)

        output = keras.layers.Dense(1, activation = 'sigmoid', name = 'output')(dense)
        optimiser = keras.optimizers.Nadam(lr = learning_rate)
        #optimiser = keras.optimizers.Adam(lr = learning_rate)

        model = keras.models.Model(inputs = [input_global, input_objects], outputs = [output])
        model.compile(optimizer = optimiser, loss = 'binary_crossentropy')
        self.model = model

    def train_w_batch_boost(self, k_folds=3, out_tag='my_lstm', save=True, auc_threshold=0.01, max_bad_epochs=5):
        '''
        Increase the batch size during training, if the improvement in (1-AUC) is above some threshold.
        Terminate the training if no improvement is seen after max batch size update
        '''

        self.create_train_valid_set()

        #paramaters that control batch size
        best_auc           = 0.5
        current_batch_size = 1024
        max_batch_size     = 50000

        #keep track of epochs for plotting loss vs epoch, and for gettint best model
        epoch_counter      = 0 
        best_epoch         = 1 

        keep_training = True

        while keep_training:
            epoch_counter += 1
            print('beginning training iteration for epoch {}'.format(epoch_counter))
            self.train_network(epochs=1, batch_size=current_batch_size, out_tag=out_tag)

            self.save_model(epoch_counter, out_tag)
            val_roc = self.compute_roc(batch_size=current_batch_size, valid_set=True)  #FIXME: what is the best BS here? final BS from batch boost... initial BS? current BS??

            #get average of validation rocs and clear list entries 
            improvement  = ((1-best_auc) - (1-val_roc)) / (1-best_auc)

            #FIXME: if the validation roc does not improve after n bad "epochs", then update the batch size accordingly. Rest bad epochs to zero each time the batch size increases, if it does

            #do checks to see if batch size needs to change etc
            if improvement > auc_threshold:
                print('Improvement in (1-AUC) of {:.4f} percent. Keeping batch size at {}'.format(improvement*100, current_batch_size))
                best_auc = val_roc
                best_epoch = epoch_counter
            elif current_batch_size*4 < max_batch_size:
                print('Improvement in (1-AUC) of only {:.4f} percent. Increasing batch size to {}'.format(improvement*100, current_batch_size*4))
                current_batch_size *= 4
                if val_roc > best_auc: 
                    best_auc = val_roc
                    best_epoch = epoch_counter
            elif current_batch_size < max_batch_size: 
                print('Improvement in (1-AUC) of only {:.4f} percent. Increasing to max batch size of {}'.format(improvement*100, max_batch_size))
                current_batch_size = max_batch_size
                if val_roc > best_auc: 
                    best_auc = val_roc
                    best_epoch = epoch_counter
            elif improvement > 0:
                print('Improvement in (1-AUC) of only {:.4f} percent. Cannot increase batch further'.format(improvement*100))
                best_auc = val_roc
                best_epoch = epoch_counter
            else: 
                print('AUC did not improve and batch size cannot be increased further. Stopping training...')
                keep_training = False

            if epoch_counter > self.max_epochs:
                print('At the maximum number of training epochs ({}). Stopping training...'.format(self.max_epochs))
                keep_training = False
                best_epoch = self.max_epochs
            
        print 'best epoch was: {}'.format(best_epoch)
        print 'best validation auc was: {}'.format(best_auc)
        self.val_roc = best_auc
      

        #delete all models that aren't from the best training. Re-load best model for predicting on test set 
        for epoch in range(1,epoch_counter+1):
            if epoch is not best_epoch:
                os.system('rm {}/models/{}_model_epoch_{}.hdf5'.format(os.getcwd(), out_tag, epoch))
                os.system('rm {}/models/{}_model_architecture_epoch_{}.json'.format(os.getcwd(), out_tag, epoch))
        os.system('mv {0}/models/{1}_model_epoch_{2}.hdf5 {0}/models/{1}_model.hdf5'.format(os.getcwd(), out_tag, best_epoch))
        os.system('mv {0}/models/{1}_model_architecture_epoch_{2}.json {0}/models/{1}_model_architecture.json'.format(os.getcwd(), out_tag, best_epoch))

        #reset model state and load in best weights
        with open('{}/models/{}_model_architecture.json'.format(os.getcwd(), out_tag), 'r') as model_json:
            best_model_architecture = model_json.read()
        self.model = keras.models.model_from_json(best_model_architecture)
        self.model.load_weights('{}/models/{}_model.hdf5'.format(os.getcwd(), out_tag))

        if not save:
            os.system('rm {}/models/{}_model_architecture.json'.format(os.getcwd(), out_tag))
            os.system('rm {}/models/{}_model.hdf5'.format(os.getcwd(), out_tag))
        
    def train_network(self, batch_size, epochs, out_tag='my_lstm'):
        if self.eq_train: self.model.fit([self.X_train_high_level, self.X_train_low_level], self.y_train, epochs=epochs, batch_size=batch_size, sample_weight=self.train_weights_eq)       
        else: self.model.fit([self.X_train_high_level, self.X_train_low_level], self.y_train, epochs=epochs, batch_size=batch_size, sample_weight=self.train_weights)       
    
    def save_model(self, epoch, out_tag):
        Utils.check_dir('./models/')
        self.model.save_weights('{}/models/{}_model_epoch_{}.hdf5'.format(os.getcwd(), out_tag, epoch))
        with open("{}/models/{}_model_architecture_epoch_{}.json".format(os.getcwd(), out_tag, epoch), "w") as f_out:
            f_out.write(self.model.to_json())

    def set_hyper_parameters(self, hp_string):
        hp_dict = {}
        for params in hp_string.split(','):
            hp_name = params.split(':')[0]
            hp_value =params.split(':')[1]
            try: hp_value = int(hp_value)
            except ValueError: hp_value = float(hp_value)
            hp_dict[hp_name] = hp_value
            self.set_model(**hp_dict)

    def create_train_valid_set(self):
        '''
        Partition the X and y training matrix into a train + validation set (i.e. X_train -> X_train + X_validate, and same for y and w)
        Note that validation weights should always be the nominal MC weights
        This also means turning ordinary arrays into 2D arrays, which we should be careful to keep as 1D arrays earlier

        '''

        if not self.eq_train:
            X_train_high_level, X_valid_high_level, X_train_low_level, X_valid_low_level, train_w, valid_w, y_train, y_valid  = train_test_split(self.X_train_high_level, self.X_train_low_level, self.train_weights, self.y_train,
                                                                                                                                                 train_size=0.7, test_size=0.3
                                                                                                                                                 )
        else:
            X_train_high_level, X_valid_high_level, X_train_low_level, X_valid_low_level, train_w, valid_w, w_train_eq, w_valid_eq, y_train, y_valid  = train_test_split(self.X_train_high_level, self.X_train_low_level,
                                                                                                                                                                         self.train_weights, self.train_weights_eq, self.y_train,
                                                                                                                                                                         train_size=0.7, test_size=0.3
                                                                                                                                                                        )
            self.train_weights_eq = w_train_eq

        #FIXME: need to re-equalise weights in each folds as sumW_sig != sumW_bkg anymroe!
        self.train_weights = train_w
        self.valid_weights = valid_w #validation weights should never be equalised weights!

        self.X_train_high_level = X_train_high_level
        self.X_train_low_level  = self.join_objects(X_train_low_level)

        self.X_valid_high_level = X_valid_high_level
        self.X_valid_low_level  = self.join_objects(X_valid_low_level)

        self.y_train            = y_train
        self.y_valid            = y_valid


    def compute_roc(self, batch_size=64, valid_set=False):
        '''
        Compute the area under the associated ROC curve, with usual mc weights

        Return the score on the test set (validation set if performing any model selection)
        '''

        self.y_pred_train = self.model.predict([self.X_train_high_level, self.X_train_low_level], batch_size=batch_size).flatten()
        roc_train = roc_auc_score(self.y_train, self.y_pred_train, sample_weight=self.train_weights)
        print 'ROC train score: {}'.format(roc_train)

        if valid_set:
            self.y_pred_valid = self.model.predict([self.X_valid_high_level, self.X_valid_low_level], batch_size=batch_size).flatten()
            roc_test  = roc_auc_score(self.y_valid, self.y_pred_valid, sample_weight=self.valid_weights)
            print 'ROC valid score: {}'.format(roc_test)
        else:
            self.y_pred_test = self.model.predict([self.X_test_high_level, self.X_test_low_level], batch_size=batch_size).flatten()
            roc_test  = roc_auc_score(self.y_test, self.y_pred_test, sample_weight=self.test_weights)
            print 'ROC test score: {}'.format(roc_test)

        return roc_test

    def compare_rocs(self, roc_file, hp_string):
        hp_roc = roc_file.readlines()
        val_auc = self.val_roc
        print 'validation roc is: {}'.format(val_auc)
        if len(hp_roc)==0: 
            roc_file.write('{};{:.4f}'.format(hp_string, val_auc))
        elif float(hp_roc[-1].split(';')[-1]) < val_auc:
            roc_file.write('\n')
            roc_file.write('{};{:.4f}'.format(hp_string, val_auc))

    def batch_gs_cv(self):
        '''
        Submit a sets of hyperparameters permutations (based on attribute hp_grid_rnge) to the IC batch.
        Take care to separate training weights, which may be modified w.r.t nominal weights, 
        and the weights used when evaluating on the validation set which should be the nominal weights
        '''
        #get all possible HP sets from permutations of the above dict
        hp_perms = self.get_hp_perms()
        #submit job to the batch for the given HP range:
        for hp_string in hp_perms:
            Utils.sub_lstm_hp_script(self.eq_train, self.batch_boost, hp_string)

    def get_hp_perms(self):
        from itertools import product
        ''''
        returns list of all possible hyper parameter combinations in format 'hp1:val1,hp2:val2, ...'
        '''
        hp_perms  = [perm for perm in apply(product, self.hp_grid_rnge.values())]
        final_hps = []
        counter   = 0
        for hp_perm in hp_perms:
            l_entry = ''
            for hp_name, hp_value in zip(self.hp_grid_rnge.keys(), hp_perm):
                l_entry+='{}:{},'.format(hp_name,hp_value)
                counter+=1
                if (counter % len(self.hp_grid_rnge.keys())) == 0: final_hps.append(l_entry[:-1])
        return final_hps

    def plot_roc(self,out_tag):
        ''' 
        Method to plot the roc curve, using method from Plotter() class
        '''
        roc_fig = self.plotter.plot_roc(self.y_train, self.y_pred_train, self.train_weights, 
                                        self.y_test, self.y_pred_test, self.test_weights
                                       )

        Utils.check_dir('{}/plotting/plots/{}'.format(os.getcwd(), out_tag))
        roc_fig.savefig('{0}/plotting/plots/{1}/{1}_ROC_curve.pdf'.format(os.getcwd(),out_tag))
        print('saving: {0}/plotting/plots/{1}/{1}_ROC_curve.pdf'.format(os.getcwd(),out_tag))
        plt.close()

    def plot_output_score(self, out_tag, batch_size=64, ratio_plot=False, norm_to_data=False):
        ''' 
        Method to plot the roc curve and compute the integral of the roc as a performance metric
        '''
        output_score_fig = self.plotter.plot_output_score(self.y_test, self.y_pred_test, self.test_weights, self.proc_arr_test,
                                                          self.model.predict([self.X_data_test_high_level, self.X_data_test_low_level], batch_size=batch_size).flatten(),
                                                          MVA='DNN', ratio_plot=ratio_plot, norm_to_data=norm_to_data)

        Utils.check_dir('{}/plotting/plots/{}'.format(os.getcwd(),out_tag))
        output_score_fig.savefig('{0}/plotting/plots/{1}/{1}_output_score.pdf'.format(os.getcwd(), out_tag))
        print('saving: {0}/plotting/plots/{1}/{1}_output_score.pdf'.format(os.getcwd(), out_tag))
        plt.close()

class Plotter(object):
    '''
    Class to plot input variables and output scores
    '''
    def __init__(self, data_obj, input_vars, sig_col='red', normalise=False, log=False, norm_to_data=False): 
        self.sig_df       = data_obj.mc_df_sig
        self.bkg_df       = data_obj.mc_df_bkg
        self.data_df      = data_obj.data_df
        del data_obj

        self.sig_labels   = np.unique(self.sig_df['proc'].values).tolist()
        self.bkg_labels   = np.unique(self.bkg_df['proc'].values).tolist()

        self.sig_colour   = sig_col
        self.bkg_colours  = ['#91bfdb', '#ffffbf', '#fc8d59']
        self.normalise    = normalise

        self.sig_scaler   = 5*10**7
        self.log_axis     = log

        #get xrange from yaml config
        with open('plotting/var_to_xrange.yaml', 'r') as plot_config_file:
            plot_config        = yaml.load(plot_config_file)
            self.var_to_xrange = plot_config['var_to_xrange']
        missing_vars = [x for x in input_vars if x not in self.var_to_xrange.keys()]
        if len(missing_vars)!=0: raise IOError('Missing variables: {}'.format(missing_vars))

    @classmethod 
    def num_to_str(self, num):
        ''' 
        Convert basic number into scientific form e.g. 1000 -> 10^{3}.
        Not considering decimal inputs for now. Also ignores first unit.
        '''
        str_rep = str(num) 
        if str_rep[0] == 0: return num 
        exponent = len(str_rep)-1
        return r'$\times 10^{%s}$'%(exponent)

    def plot_input(self, var, n_bins, out_label, ratio_plot=False, norm_to_data=False):
        if ratio_plot: 
            plt.rcParams.update({'figure.figsize':(6,5.8)})
            fig, axes = plt.subplots(nrows=2, ncols=1, dpi=200, sharex=True,
                                     gridspec_kw ={'height_ratios':[3,0.8], 'hspace':0.08})   
            ratio = axes[1]
            axes = axes[0]
        else:
            fig  = plt.figure(1)
            axes = fig.gca()

        bkg_stack      = []
        bkg_w_stack    = []
        bkg_proc_stack = []
        
        var_sig     = self.sig_df[var].values
        sig_weights = self.sig_df['weight'].values
        for bkg in self.bkg_labels:
            var_bkg     = self.bkg_df[self.bkg_df.proc==bkg][var].values
            bkg_weights = self.bkg_df[self.bkg_df.proc==bkg]['weight'].values
            bkg_stack.append(var_bkg)
            bkg_w_stack.append(bkg_weights)
            bkg_proc_stack.append(bkg)

        if self.normalise:
            sig_weights /= np.sum(sig_weights)
            bkg_weights /= np.sum(bkg_weights) #FIXME: set this up for multiple bkgs

        bins = np.linspace(self.var_to_xrange[var][0], self.var_to_xrange[var][1], n_bins)

        #add sig mc
        axes.hist(var_sig, bins=bins, label=self.sig_labels[0]+r' ($\mathrm{H}\rightarrow\mathrm{ee}$) '+self.num_to_str(self.sig_scaler), weights=sig_weights*(self.sig_scaler), histtype='step', color=self.sig_colour, zorder=10)

        #data
        data_binned, bin_edges = np.histogram(self.data_df[var].values, bins=bins)
        bin_centres = (bin_edges[:-1] + bin_edges[1:])/2
        x_err    = (bin_edges[-1] - bin_edges[-2])/2
        data_down, data_up = self.poisson_interval(data_binned, data_binned)
        axes.errorbar( bin_centres, data_binned, yerr=[data_binned-data_down, data_up-data_binned], label='Data', fmt='o', ms=4, color='black', capsize=0, zorder=1)

        #add stacked bkg
        if norm_to_data: 
            rew_stack = []
            k_factor = np.sum(self.data_df['weight'].values)/np.sum(self.bkg_df['weight'].values)
            for w_arr in bkg_w_stack:
                rew_stack.append(w_arr*k_factor)
            axes.hist(bkg_stack, bins=bins, label=bkg_proc_stack, weights=rew_stack, histtype='stepfilled', color=self.bkg_colours[0:len(bkg_proc_stack)], log=self.log_axis, stacked=True, zorder=0)
            bkg_stack_summed, _ = np.histogram(np.concatenate(bkg_stack), bins=bins, weights=np.concatenate(rew_stack))
        else: 
            axes.hist(bkg_stack, bins=bins, label=bkg_proc_stack, weights=bkg_w_stack, histtype='stepfilled', color=self.bkg_colours[0:len(bkg_proc_stack)], log=self.log_axis, stacked=True, zorder=0)
            bkg_stack_summed, _ = np.histogram(np.concatenate(bkg_stack), bins=bins, weights=np.concatenate(bkg_w_stack))

        if self.normalise: axes.set_ylabel('Arbitrary Units', ha='right', y=1, size=13)
        else: axes.set_ylabel('Events', ha='right', y=1, size=13)

        current_bottom, current_top = axes.get_ylim()
        axes.set_ylim(bottom=10, top=current_top*1.4)
        axes.legend(bbox_to_anchor=(0.97,0.97), ncol=2)
        self.plot_cms_labels(axes)
           
        var_name_safe = var.replace('_',' ')
        if ratio_plot:
            ratio.errorbar(bin_centres, (data_binned/bkg_stack_summed), fmt='o', ms=4, color='black', capsize=0)
            ratio.set_xlabel('{}'.format(var_name_safe), ha='right', x=1, size=13)
            ratio.set_ylim(0, 2)
            ratio.grid(True, linestyle='dotted')
        else: axes.set_xlabel('{}'.format(var_name_safe), ha='right', x=1, size=13)
       
        Utils.check_dir('{}/plotting/plots/{}'.format(os.getcwd(), out_label))
        fig.savefig('{0}/plotting/plots/{1}/{1}_{2}.pdf'.format(os.getcwd(), out_label, var))
        plt.close()

    @classmethod 
    def plot_cms_labels(self, axes, label='Work in progress', energy='(13 TeV)'):
        axes.text(0, 1.01, r'\textbf{CMS} %s'%label, ha='left', va='bottom', transform=axes.transAxes, size=14)
        axes.text(1, 1.01, r'{}'.format(energy), ha='right', va='bottom', transform=axes.transAxes, size=14)

    def plot_roc(self, y_train, y_pred_train, train_weights, y_test, y_pred_test, test_weights):
        bkg_eff_train, sig_eff_train, _ = roc_curve(y_train, y_pred_train, sample_weight=train_weights)
        bkg_eff_test, sig_eff_test, _ = roc_curve(y_test, y_pred_test, sample_weight=test_weights)

        fig = plt.figure(1)
        axes = fig.gca()
        axes.plot(bkg_eff_train, sig_eff_train, color='red', label='Train')
        axes.plot(bkg_eff_test, sig_eff_test, color='blue', label='Test')
        axes.set_xlabel('Background efficiency', ha='right', x=1, size=13)
        axes.set_xlim((0,1))
        axes.set_ylabel('Signal efficiency', ha='right', y=1, size=13)
        axes.set_ylim((0,1))
        axes.legend(bbox_to_anchor=(0.97,0.97))
        self.plot_cms_labels(axes)
        axes.grid(True, 'major', linestyle='solid', color='grey', alpha=0.5)
        self.fig = fig
        return fig

    def plot_output_score(self, y_test, y_pred_test, test_weights, proc_arr_test, data_pred_test, MVA='BDT', ratio_plot=False, norm_to_data=False):
        if ratio_plot: 
            plt.rcParams.update({'figure.figsize':(6,5.8)})
            fig, axes = plt.subplots(nrows=2, ncols=1, dpi=200, sharex=True,
                                     gridspec_kw ={'height_ratios':[3,0.8], 'hspace':0.08})   
            ratio = axes[1]
            axes = axes[0]
        else:
            fig  = plt.figure(1)
            axes = fig.gca()

        bins = np.linspace(0,1,41)

        bkg_stack      = []
        bkg_w_stack    = []
        bkg_proc_stack = []

        sig_scores = y_pred_test.ravel()  * (y_test==1)
        sig_w_true = test_weights.ravel() * (y_test==1)

        bkg_scores = y_pred_test.ravel()  * (y_test==0)
        bkg_w_true = test_weights.ravel() * (y_test==0)

        if self.normalise:
            sig_w_true /= np.sum(sig_w_true)
            bkg_w_true /= np.sum(bkg_w_true)

        for bkg in self.bkg_labels:
            bkg_score     = bkg_scores * (proc_arr_test==bkg)
            bkg_weights   = bkg_w_true * (proc_arr_test==bkg)
            bkg_stack.append(bkg_score)
            bkg_w_stack.append(bkg_weights)
            bkg_proc_stack.append(bkg)

        #sig
        axes.hist(sig_scores, bins=bins, label=self.sig_labels[0]+r' ($\mathrm{H}\rightarrow\mathrm{ee}$) '+self.num_to_str(self.sig_scaler), weights=sig_w_true*(self.sig_scaler), histtype='step', color=self.sig_colour)

        #data - need to take test frac of data
        data_binned, bin_edges = np.histogram(data_pred_test, bins=bins)
        bin_centres = (bin_edges[:-1] + bin_edges[1:])/2
        x_err    = (bin_edges[-1] - bin_edges[-2])/2
        data_down, data_up = self.poisson_interval(data_binned, data_binned)
        axes.errorbar( bin_centres, data_binned, yerr=[data_binned-data_down, data_up-data_binned], label='Data', fmt='o', ms=4, color='black', capsize=0, zorder=1)

        if norm_to_data: 
            rew_stack = []
            k_factor = np.sum(np.ones_like(data_pred_test))/np.sum(bkg_w_true)
            for w_arr in bkg_w_stack:
                rew_stack.append(w_arr*k_factor)
            axes.hist(bkg_stack, bins=bins, label=bkg_proc_stack, weights=rew_stack, histtype='stepfilled', color=self.bkg_colours[0:len(bkg_proc_stack)], log=self.log_axis, stacked=True, zorder=0)
            bkg_stack_summed, _ = np.histogram(np.concatenate(bkg_stack), bins=bins, weights=np.concatenate(rew_stack))
        else: 
            axes.hist(bkg_stack, bins=bins, label=bkg_proc_stack, weights=bkg_w_stack, histtype='stepfilled', color=self.bkg_colours[0:len(bkg_proc_stack)], log=self.log_axis, stacked=True, zorder=0)
            bkg_stack_summed, _ = np.histogram(np.concatenate(bkg_stack), bins=bins, weights=np.concatenate(bkg_w_stack))
        axes.legend(bbox_to_anchor=(0.98,0.98), ncol=2)

        current_bottom, current_top = axes.get_ylim()
        axes.set_ylim(bottom=0, top=current_top*1.3)
        if self.normalise: axes.set_ylabel('Arbitrary Units', ha='right', y=1, size=13)
        else: axes.set_ylabel('Events', ha='right', y=1, size=13)

        if ratio_plot:
            ratio.errorbar(bin_centres, (data_binned/bkg_stack_summed), fmt='o', ms=4, color='black', capsize=0)
            ratio.set_xlabel('{} Score'.format(MVA), ha='right', x=1, size=13)
            ratio.set_ylim(0, 2)
            ratio.grid(True, linestyle='dotted')
        else: axes.set_xlabel('{} Score'.format(MVA), ha='right', x=1, size=13)
        self.plot_cms_labels(axes)

        #ggH
        #axes.axvline(0.751, ymax=0.75, color='black', linestyle='--')
        #axes.axvline(0.554, ymax=0.75, color='black', linestyle='--')
        #axes.axvline(0.331, ymax=0.75, color='black', linestyle='--')
        #axes.axvspan(0, 0.331, ymax=0.75, color='grey', alpha=0.7)
        #VBF
        #axes.axvline(0.884, ymax=0.75, color='black', linestyle='--')
        #axes.axvline(0.612, ymax=0.75, color='black', linestyle='--')
        #axes.axvspan(0, 0.612, ymax=0.75, color='grey', alpha=0.6)
        return fig


    @classmethod
    def cats_vs_ams(self, cats, AMS, out_tag):
        fig  = plt.figure(1)
        axes = fig.gca()
        axes.plot(cats,AMS, 'ro')
        axes.set_xlim((0, cats[-1]+1))
        axes.set_xlabel('$N_{\mathrm{cat}}$', ha='right', x=1, size=13)
        axes.set_ylabel('Combined AMS', ha='right', y=1, size=13)
        Plotter.plot_cms_labels(axes)
        fig.savefig('{}/categoryOpt/nCats_vs_AMS_{}.pdf'.format(os.getcwd(), out_tag))

    def poisson_interval(self, x, variance, level=0.68):                                                                      
        neff = x**2/variance
        scale = x/neff
     
        # CMS statcomm recommendation
        l = scipy.stats.gamma.interval(
            level, neff, scale=scale,
        )[0]
        u = scipy.stats.gamma.interval(
            level, neff+1, scale=scale
        )[1]
     
        # protect against no effecitve entries
        l[neff==0] = 0.
     
        # protect against no variance
        l[variance==0.] = 0.
        u[variance==0.] = np.inf
        return l, u


class Utils(object):
    def __init__(self): pass

    @classmethod 
    def check_dir(self, file_dir):
        '''
        Check directory exists; if not make it.
        '''
        if not path.isdir(file_dir):
            print 'making directory: {}'.format(file_dir)
            system('mkdir -p %s' %file_dir)

    @classmethod 
    def sub_hp_script(self, eq_weights, hp_string, k_folds, job_dir='{}/submissions/bdt_hp_opts_jobs'.format(os.getcwd())):
        '''
        Submits train_bdt.py with option -H hp_string -k, to IC batch
        When run this way, a BDT gets trained with HPs = hp_string, and cross validated on k_folds 
        '''

        file_safe_string = hp_string
        for p in [':',',','.']:
            file_safe_string = file_safe_string.replace(p,'_')

        system('mkdir -p {}'.format(job_dir))
        sub_file_name = '{}/sub_bdt_hp_{}.sh'.format(job_dir,file_safe_string)
        #FIXME: add config name as a function argument to make it general
        sub_command   = "python train_bdt.py -c bdt_config.yaml -H {} -k {}".format(hp_string, k_folds)
        if eq_weights: sub_command += ' -w'
        with open('{}/submissions/sub_bdt_opt_template.sh'.format(os.getcwd())) as f_template:
            with open(sub_file_name,'w') as f_sub:
                for line in f_template.readlines():
                    if '!CWD!' in line: line = line.replace('!CWD!', os.getcwd())
                    if '!CMD!' in line: line = line.replace('!CMD!', '"{}"'.format(sub_command))
                    f_sub.write(line)
        system( 'qsub -o {} -e {} -q hep.q -l h_rt=1:00:00 -l h_vmem=4G {}'.format(sub_file_name.replace('.sh','.out'), sub_file_name.replace('.sh','.err'), sub_file_name ) )

    @classmethod 
    def sub_lstm_hp_script(self, eq_weights, batch_boost, hp_string, job_dir='{}/submissions/lstm_hp_opts_jobs'.format(os.getcwd())):
        '''
        Submits train_bdt.py with option -H hp_string -k, to IC batch
        When run this way, a LSTM gets trained with HPs = hp_string
        '''

        file_safe_string = hp_string
        for p in [':',',','.']:
            file_safe_string = file_safe_string.replace(p,'_')

        system('mkdir -p {}'.format(job_dir))
        sub_file_name = '{}/sub_lstm_hp_{}.sh'.format(job_dir,file_safe_string)
        #FIXME: add config name as a function argument to make it general. Do not need file paths here as copt everything into one dir
        sub_command   = "python train_lstm.py -c lstm_config.yaml -H {}".format(hp_string)
        if eq_weights: sub_command += ' -w'
        if batch_boost: sub_command += ' -B'
        with open('{}/submissions/sub_hp_opt_template.sh'.format(os.getcwd())) as f_template:
            with open(sub_file_name,'w') as f_sub:
                for line in f_template.readlines():
                    if '!CWD!' in line: line = line.replace('!CWD!', os.getcwd())
                    if '!CMD!' in line: line = line.replace('!CMD!', '"{}"'.format(sub_command))
                    f_sub.write(line)
        system( 'qsub -o {} -e {} -q hep.q -l h_rt=12:00:00 -l h_vmem=12G {}'.format(sub_file_name.replace('.sh','.out'), sub_file_name.replace('.sh','.err'), sub_file_name ) )

