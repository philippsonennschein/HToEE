#data handling imports
import uproot as upr
import numpy as np
import pandas as pd
import os
from ROOT import TLorentzVector as lv
from numpy import pi
from os import path, system
from variables import nominal_vars, gen_vars

#BDT imports
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import pickle
import warnings

#NN imports. Will eventually migrate NN to separate file
from keras.models import Sequential
from keras.initializers import RandomNormal
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import Nadam
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from keras.utils import np_utils

#plotting imports
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
            #FIXME: temp fix until ptOm in samples
            df = df_tree.pandas.df(data_vars.remove('genWeight')).query('dielectronMass>110 and dielectronMass<150')
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
        #df['year'] = year

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


class BDTHelpers(object):

    def __init__(self, data_obj, train_vars, train_frac, eq_weights=False):
        #if using multiple years, should be concatted by now and in ROOTHelpers data_object argument
      
        #attributes for the dataset formation
        mc_df_sig = data_obj.mc_df_sig
        self.sig_procs  = np.unique(mc_df_sig['proc']).tolist()
        mc_df_bkg = data_obj.mc_df_bkg
        self.bkg_procs  = np.unique(mc_df_bkg['proc']).tolist()
        df_data = data_obj.data_df

        self.train_frac = train_frac

        #add y_target label (1 for signal, 0 for background)
        mc_df_sig['y'] = np.ones(mc_df_sig.shape[0]).tolist()
        mc_df_bkg['y'] = np.zeros(mc_df_bkg.shape[0]).tolist()

        if eq_weights: 
            b_to_s_ratio = np.sum(mc_df_bkg['weight'].values)/np.sum(mc_df_sig['weight'].values)
            mc_df_sig['eq_weight']  = mc_df_sig['weight'] * b_to_s_ratio
            mc_df_bkg['eq_weight'] = mc_df_bkg['weight']
            self.eq_train = True
        else: self.eq_train = False

        Z_tot = pd.concat([mc_df_sig, mc_df_bkg], ignore_index=True)

        if not eq_weights:
            X_train, X_test, train_w, test_w, y_train, y_test, proc_arr_train, proc_arr_test = train_test_split(Z_tot[train_vars], 
                                                                                               Z_tot['weight'],
                                                                                               Z_tot['y'], Z_tot['proc'],
                                                                                               train_size=train_frac, 
                                                                                               test_size=1-train_frac,
                                                                                               shuffle=True, random_state=1357
                                                                                               )
        else:
            X_train, X_test, train_w, test_w, train_eqw, test_eqw, y_train, y_test, proc_arr_train, proc_arr_test = train_test_split(Z_tot[train_vars], Z_tot['weight'], 
                                                                                                                    Z_tot['eq_weight'], Z_tot['y'], Z_tot['proc'],
                                                                                                                    train_size=train_frac, 
                                                                                                                    test_size=1-train_frac,
                                                                                                                    shuffle=True, 
                                                                                                                    random_state=1357
                                                                                                                    )
            self.train_weights_eq = train_eqw.values
            #NB: will never test/evaluate with equalised weights. This is explicitly why we set another train weight attribute, 
            #    because for overtraining we need to evaluate on the train set (and hence need nominal MC train weights)
      
        self.train_vars       = train_vars
        self.X_train          = X_train.values
        self.y_train          = y_train.values
        self.train_weights    = train_w.values
        self.y_pred_train     = None
        self.proc_arr_train   = proc_arr_train

        self.X_test           = X_test.values
        self.y_test           = y_test.values
        self.test_weights     = test_w.values
        self.y_pred_test      = None
        self.proc_arr_test    = proc_arr_test

        #get data test set for plotting bkg/data agreement
        self.X_data_train, self.X_data_test = train_test_split(df_data[train_vars], train_size=train_frac, test_size=1-train_frac, shuffle=True, random_state=1357)


        self.clf              = xgb.XGBClassifier(objective='binary:logistic', n_estimators=100, 
                                                  eta=0.05, maxDepth=4, min_child_weight=0.01, 
                                                  subsample=0.6, colsample_bytree=0.6, gamma=1)

        #attributes for the hp optmisation and cross validation
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

        self.plotter          = Plotter(data_obj, train_vars)
        del data_obj
        

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

    def plot_output_score(self, out_tag):
        ''' 
        Method to plot the roc curve and compute the integral of the roc as a performance metric
        '''
        output_score_fig = self.plotter.plot_output_score(self.y_test, self.y_pred_test, self.test_weights, self.proc_arr_test, self.clf.predict_proba(self.X_data_test.values)[:,1:])

        Utils.check_dir('{}/plotting/plots/{}'.format(os.getcwd(),out_tag))
        output_score_fig.savefig('{0}/plotting/plots/{1}/{1}_output_score.pdf'.format(os.getcwd(), out_tag))
        print('saving: {0}/plotting/plots/{1}/{1}_output_score.pdf'.format(os.getcwd(), out_tag))
        plt.close()

class LSTM_DNN(object):
    '''
    Train a DNN that uses LSTM and fully connected layers
    '''

    def __init__(self, data_obj, low_level_vars, high_level_vars, train_frac, eq_weights=False):
        #attributes for the dataset formation
        mc_df_sig = data_obj.mc_df_sig
        mc_df_bkg = data_obj.mc_df_bkg

        self.low_vars    = low_level_vars
        self.high_vars   = high_level_vars

        #add y_target label (1 for signal, 0 for background)
        mc_df_sig['y'] = np.ones(mc_df_sig.shape[0]).tolist()
        mc_df_bkg['y'] = np.zeros(mc_df_bkg.shape[0]).tolist()

        if eq_weights: 
            b_to_s_ratio = np.sum(mc_df_bkg['weight'].values)/np.sum(mc_df_sig['weight'].values)
            mc_df_sig['eq_weight'] = mc_df_sig['weight'] * b_to_s_ratio 
            mc_df_bkg['eq_weight'] = mc_df_bkg['weight'] 
            self.eq_train = True
        else: self.eq_train = False

        Z_tot = pd.concat([mc_df_sig, mc_df_bkg], ignore_index=True)

        if not eq_weights:
            low_vars_train, low_vars_test, train_w, test_w, y_train, y_test, = train_test_split(Z_tot[low_level_vars], Z_tot['weight'], Z_tot['y'], train_size=train_frac, test_size=1-train_frac, shuffle=True, random_state=1357)
            high_vars_train, high_vars_test, train_w, test_w, y_train, y_test, = train_test_split(Z_tot[high_level_vars], Z_tot['weight'], Z_tot['y'], train_size=train_frac, test_size=1-train_frac, shuffle=True, random_state=1357)
        else:
            low_vars_train, low_vars_test, train_eqw, test_eqw, y_train, y_test, = train_test_split(Z_tot[low_level_vars], Z_tot['weight'], Z_tot['eq_weight'], Z_tot['y'], train_size=train_frac, test_size=1-train_frac, shuffle=True, random_state=1357)
            high_vars_train, high_vars_test, train_eqw, test_eqw, y_train, y_test, = train_test_split(Z_tot[high_level_vars], Z_tot['weight'], Z_tot['eq_weight'], Z_tot['y'], train_size=train_frac, test_size=1-train_frac, shuffle=True, random_state=1357)
            self.train_weights_eq = train_eqw.values


        #develop scaler to standardise input features (and re-use to scale test data)
        #print ('Array contains NaN: ', np.isnan(X_train).any())
        #X_scaler = StandardScaler()
        #X_scaler.fit(X_train.values)
        #self.X_train          = X_scaler.transform(X_train)
        #self.y_train          = np_utils.to_categorical(y_train.values, num_classes=2)
        #self.train_weights    = train_w.values #needed for calc of train ROC even if training wth eq weights
        #self.y_pred_train     = None

        #self.X_test           = X_scaler.transform(X_test)
        #self.y_test           = np_utils.to_categorical(y_test.values, num_classes=2)
        #self.test_weights     = test_w.values
        #self.y_pred_test      = None
    

class Plotter(object):
    '''
    Class to plot input variables and output scores
    '''
    #def __init__(self, data_obj, input_vars, sig_col='firebrick',  bkg_col='violet',  normalise=True): 
    def __init__(self, data_obj, input_vars, sig_col='red', normalise=False, log=False, reweight=True): 
        self.sig_df     = data_obj.mc_df_sig
        self.bkg_df     = data_obj.mc_df_bkg
        self.data_df    = data_obj.data_df
        del data_obj

        self.sig_labels = np.unique(self.sig_df['proc'].values).tolist()
        self.bkg_labels = np.unique(self.bkg_df['proc'].values).tolist()

        self.sig_colour = sig_col
        self.bkg_colours= ['#91bfdb', '#ffffbf', '#fc8d59']
        self.normalise  = normalise

        self.input_vars = input_vars
        self.sig_scaler = 5*10**8
        self.log_axis   = log
        self.reweight   = reweight

    @classmethod 
    def num_to_str(self, num):
        ''' 
        Convert basic number into scientific form e.g. 1000 -> 1 x 10^{3}.
        Not considering decimal inputs for now
        '''
        str_rep = str(num) 
        if str_rep[0] == 0: return num 
        exponent = len(str_rep)-1
        return r'$\times 10^{%s}$'%(exponent)

    def plot_input(self, var, n_bins, out_label):
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

        #plot with np first to get consistent ranges and modify last bin to avoid empty space at end of x-axis
        binned_data, bin_edges = np.histogram(var_sig, n_bins, weights=sig_weights)
        bkw_index=0
        sumw_all_bins=0
        for ibin_sum in reversed(binned_data):
            sumw_all_bins+=ibin_sum
            if sumw_all_bins < 0.02*np.sum(binned_data): bkw_index+=1
            else: break
        if bkw_index!=0: bin_edges = bin_edges[:-bkw_index]     
        bins = np.linspace(bin_edges[0], bin_edges[-1], n_bins)

        #add sig mc
        axes.hist(var_sig, bins=bins, label=self.sig_labels[0]+r' ($\mathrm{H}\rightarrow\mathrm{ee}$) '+self.num_to_str(self.sig_scaler), weights=sig_weights*(self.sig_scaler), histtype='step', color=self.sig_colour, zorder=10)

        #data
        data_binned, bin_edges = np.histogram(self.data_df[var].values, bins=bins)
        bin_centres = (bin_edges[:-1] + bin_edges[1:])/2
        x_err    = (bin_edges[-1] - bin_edges[-2])/2
        data_down, data_up = self.poisson_interval(data_binned, data_binned)
        axes.errorbar( bin_centres, data_binned, yerr=[data_binned-data_down, data_up-data_binned], label='Data', fmt='o', ms=4, color='black', capsize=0, zorder=1)

        #add stacked bkg
        if self.reweight: 
            rew_stack = []
            k_factor = np.sum(self.data_df['weight'].values)/np.sum(self.bkg_df['weight'].values)
            for w_arr in bkg_w_stack:
                rew_stack.append(w_arr*k_factor)
            axes.hist(bkg_stack, bins=bins, label=bkg_proc_stack, weights=rew_stack, histtype='stepfilled', color=self.bkg_colours[0:len(bkg_proc_stack)], log=self.log_axis, stacked=True, zorder=0)
        else: axes.hist(bkg_stack, bins=bins, label=bkg_proc_stack, weights=bkg_w_stack, histtype='stepfilled', color=self.bkg_colours[0:len(bkg_proc_stack)], log=self.log_axis, stacked=True, zorder=0)


        var_name_safe = var.replace('_',' ')
        axes.set_xlabel('{}'.format(var_name_safe), ha='right', x=1, size=13)
 
        if self.normalise: axes.set_ylabel('Arbitrary Units', ha='right', y=1, size=13)
        else: axes.set_ylabel('Events', ha='right', y=1, size=13)

        current_bottom, current_top = axes.get_ylim()
        axes.set_ylim(bottom=10, top=current_top*1.2)
        axes.legend(bbox_to_anchor=(0.97,0.97))
        self.plot_cms_labels(axes)
       
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

    def plot_output_score(self, y_test, y_pred_test, test_weights, proc_arr_test, data_pred_test):
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

        #axes.hist(bkg_scores, bins=bins, label='Background', weights=bkg_w_true, histtype='step', color=self.bkg_colours)
        if self.reweight: 
            rew_stack = []
            k_factor = np.sum(np.ones_like(data_pred_test))/np.sum(bkg_w_true)
            for w_arr in bkg_w_stack:
                rew_stack.append(w_arr*k_factor)
            axes.hist(bkg_stack, bins=bins, label=bkg_proc_stack, weights=rew_stack, histtype='stepfilled', color=self.bkg_colours[0:len(bkg_proc_stack)], log=self.log_axis, stacked=True, zorder=0)
        else: axes.hist(bkg_stack, bins=bins, label=bkg_proc_stack, weights=bkg_w_stack, histtype='stepfilled', color=self.bkg_colours[0:len(bkg_proc_stack)], log=self.log_axis, stacked=True, zorder=0)
        axes.legend(bbox_to_anchor=(0.98,0.98), ncol=2)

        current_bottom, current_top = axes.get_ylim()
        axes.set_ylim(bottom=0, top=current_top*1.5)
        if self.normalise: axes.set_ylabel('Arbitrary Units', ha='right', y=1, size=13)
        else: axes.set_ylabel('Events', ha='right', y=1, size=13)
        axes.set_xlabel('BDT Score', ha='right', x=1, size=13)
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
        with open('{}/submissions/sub_bdt_hp_template.sh'.format(os.getcwd())) as f_template:
            with open(sub_file_name,'w') as f_sub:
                for line in f_template.readlines():
                    if '!CWD!' in line: line = line.replace('!CWD!', os.getcwd())
                    if '!CMD!' in line: line = line.replace('!CMD!', '"{}"'.format(sub_command))
                    f_sub.write(line)
        system( 'qsub -o {} -e {} -q hep.q -l h_rt=1:00:00 -l h_vmem=4G {}'.format(sub_file_name.replace('.sh','.out'), sub_file_name.replace('.sh','.err'), sub_file_name ) )

