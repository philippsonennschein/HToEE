import uproot as upr
import numpy as np
import pandas as pd
import xgboost as xgb
import os
from os import path, system
from variables import nominal_vars, gen_vars
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score, roc_curve

import warnings
import matplotlib.pyplot as plt
try:
     plt.style.use("cms10_6")
except IOError:
     warnings.warn('Could not import user defined matplot style file. Using default style settings...')
plt.rcParams.update({'legend.fontsize':10})

class ROOTHelpers(object):
    '''
    Class containing methods to train various ML algorithms to discriminate the
    VBF H -> ee signal from the standard DY background.

    :years: the years for which data and simulation will be read in for
    :mc_dir: directory where root files for simulation are held. Files for all years should be in this directory
    :data_dir: directory where root files for data are held. Files for all years should be in this directory
    :mc_trees_sig_str: name of the trees in simulated signal. Should be identical across years
    :mc_trees_bkg_str: name of the trees in simulated background. Should be identical across years
    :data_trees_str: name of the trees in data. Should be identical across years

    :mc_sig_year_fnames: list of tuples for mc sig with format [ (yearX : sampleX), (yearY : sampleY), ... ]
    '''
  
    def __init__(self, mc_dir, mc_trees_sig_str, mc_trees_bkg_str, mc_fnames, data_dir, data_trees_str, data_fnames, train_vars, vars_to_add, presel_str):

        self.mc_dir             = mc_dir #FIXME: remove '\' using if_ends_with()
        #self.mc_df_dir      = mc_dir+'/DataFrames'
        self.mc_trees_sig       = mc_trees_sig_str
        self.mc_sig_year_fnames = mc_fnames['sig'].items()
        self.mc_trees_bkg       = mc_trees_bkg_str
        self.mc_bkg_year_fnames = mc_fnames['bkg'].items()
        self.data_dir           = data_dir
        #self.data_df_dir        = data_dir+'/DataFrames'
        self.data_trees         = data_trees_str
        self.data_year_fnames   = data_fnames.items()

        self.mc_df_sig          = []
        self.mc_df_bkg          = []
        self.data_df            = []

        assert set([year[0] for year in self.mc_sig_year_fnames]) == set([year[0] for year in self.mc_bkg_year_fnames]) == set([year[0] for year in self.data_year_fnames]), 'Inconsistency in sample years!'
        self.lumi_map           = {'2016':35.9, '2017':41.5, '2018':59.7}

        assert all(x in (nominal_vars+list(vars_to_add.keys())) for x in train_vars), 'All training variables were not in nominal variables!'

        self.nominal_vars       = nominal_vars
        self.train_vars         = train_vars
        self.vars_to_add        = vars_to_add
        self.cut_string         = presel_str


    def load_mc(self, year, file_name, bkg=False, reload_data=False):
        '''
        Try to load mc dataframe. If it doesn't exist, read in the root file.
        This should be used once per year, if reading in multiple years.
        '''
        try: 
            if reload_data: raise IOError
            elif not bkg: self.mc_df_sig.append( self.load_df(self.mc_dir+'DataFrames/', 'sig', year) )
            else: self.mc_df_bkg.append( self.load_df(self.mc_dir+'DataFrames/', 'bkg', year) )
        except IOError: 
            if not bkg: self.mc_df_sig.append( self.root_to_df(self.mc_dir, file_name, self.mc_trees_sig, 'sig', year) )
            else: self.mc_df_bkg.append( self.root_to_df(self.mc_dir, file_name, self.mc_trees_bkg, 'bkg', year) )

    def load_data(self, file_name, year):
        '''
        Try to load Data dataframe. If it doesn't exist, read in the root file.
        This should be used once per year, if reading in multiple years.
        '''
        try: 
            self.data_df.append( self.load_df(self.data_df_dir, 'data', year) )
        except IOError: 
            self.data_df.append( self.root_to_df(self.data_dir, self.df_trees, 'data', year) )

    def load_df(self, df_dir, flag, year):
        df = pd.read_hdf('{}/{}_df_{}.h5'.format(df_dir, flag, year))
        print('Sucessfully loaded DataFrame: {}/{}_df_{}.h5'.format(df_dir, flag, year))
        return df    

    def root_to_df(self, file_dir, file_name, tree_name, flag, year):
        '''
        Load a single .root dataset for simulation. Apply any preselection and lumi scaling
        If reading in simulated samples, apply lumi scaling and read in gen-level variables too
        '''

        print('Reading {} file: {}, for year: {}'.format(flag, file_dir+file_name, year))
        df_file = upr.open(file_dir+file_name)
        df_tree = df_file[tree_name]
        del df_file
        if (flag=='sig') or (flag=='bkg'): df = df_tree.pandas.df(self.nominal_vars + gen_vars).query(self.cut_string)
        df = df_tree.pandas.df(self.nominal_vars).query(self.cut_string)

        print('Reshuffling events')
        df = df.sample(frac=1).reset_index(drop=True)

        if (flag=='sig') or(flag=='bkg'): df = self.scale_by_lumi(file_name, df, year)
        print('Number of events in final dataframe: {}'.format(np.sum(df['weight'].values)))

        Utils.check_dir(file_dir+'DataFrames/') 
        df.to_hdf('{}/{}_df_{}.h5'.format(file_dir+'DataFrames', flag, year), 'df',mode='w',format='t')
        print('Saved dataframe: {}/{}_df_{}.h5'.format(file_dir+'DataFrames', flag, year))

        return df

    def scale_by_lumi(self, file_name, df, year):
        '''
        Scale simulation by the lumi of the year
        '''
        print('scaling weights for file: {} from year: {}, by {}'.format(file_name, year, self.lumi_map[year]))
        df['weight']*=self.lumi_map[year]
        return df


    def concat_years(self):
        #FIXME: add functionality concat list with more than one entry
        if len(self.mc_df_sig) == 1: self.mc_df_sig = self.mc_df_sig[0]
        else: pass
        if len(self.mc_df_bkg) ==1:  self.mc_df_bkg = self.mc_df_bkg[0] 
        else: pass
        if len(self.data_df) ==1:  self.data_df = self.data_df[0] 
        else: pass 

class BDTHelpers(object):

    def __init__(self, data_obj, train_vars, train_frac, eq_weights=False):
        #if using multiple years, should be concatted by now and in ROOTHelpers data_object argument
      
        #attributes for the dataset formation
        mc_df_sig = data_obj.mc_df_sig
        mc_df_bkg = data_obj.mc_df_bkg

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

        #FIXME:
        # feature egineering here, by calling a method from above class
        #using literal_eval or sth
        Z_tot['dijet_centrality'] = np.exp(-4.*((Z_tot['dijet_Zep']/Z_tot['dijet_abs_dEta'])**2))

        if not eq_weights:
            X_train, X_test, train_w, test_w, y_train, y_test, = train_test_split(Z_tot[train_vars], Z_tot['weight'], Z_tot['y'], train_size=train_frac, test_size=1-train_frac, shuffle=True)
        else:
            X_train, X_test, train_w, test_w, train_eqw, test_eqw, y_train, y_test, = train_test_split(Z_tot[train_vars], Z_tot['weight'], Z_tot['eq_weight'], Z_tot['y'], train_size=train_frac, test_size=1-train_frac, shuffle=True)
            self.train_weights_eq = train_eqw.values
            #NB: will never test/evaluate with equalised weights. This is explicitly why we set another train weight attribute, because for overtraining we need to evaluate on the train set (and hence need nominal MC train weights)
      
        self.train_vars       = train_vars
        self.X_train          = X_train.values
        self.y_train          = y_train.values
        self.train_weights    = train_w.values
        self.y_pred_train     = None

        self.X_test           = X_test.values
        self.y_test           = y_test.values
        self.test_weights     = test_w.values
        self.y_pred_test      = None

        self.clf              = xgb.XGBClassifier(objective='binary:logistic', n_estimators=300, 
                                                  eta=0.1, maxDepth=6, min_child_weight=0.5, 
                                                  subsample=0.6, colsample_bytree=0.6, gamma=1)

        #attributes for the hp optmisation and cross validation
        self.hp_grid_rnge     = {'learning_rate': [0.01, 0.05, 0.1, 0.3],
                                 'max_depth':[x for x in range(3,10)],
                                 'min_child_weight':[x for x in range(0,3)],
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

        #FIXME: better to formally inherit from plotter?
        self.plotter          = Plotter(data_obj, train_vars)
        del data_obj
        

    def train_classifier(self, file_path, save=False):
        if self.eq_train: train_weights = self.train_weights_eq
        else: train_weights = self.train_weights

        print 'Training classifier... '
        clf = self.clf.fit(self.X_train, self.y_train, sample_weight=train_weights)
        print 'Finished Training classifier!'
        self.clf = clf

        #ROOTHelpers.check_dir(file_path + 'models')
        #if save: clf.save_model(file_path + 'models')

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
                hp_dict[params[0]] = params[1]
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


    def compute_roc(self):
        '''
        Compute the area under the associated ROC curve, with mc weights
        '''
        self.y_pred_train = self.clf.predict_proba(self.X_train)[:,1:]
        print 'Area under ROC curve for train set is: {:.4f}'.format(roc_auc_score(self.y_train, self.y_pred_train, sample_weight=self.train_weights*1000))

        self.y_pred_test = self.clf.predict_proba(self.X_test)[:,1:]
        print 'Area under ROC curve for test set is: {:.4f}'.format(roc_auc_score(self.y_test, self.y_pred_test, sample_weight=self.test_weights*1000))
        return roc_auc_score(self.y_test, self.y_pred_test, sample_weight=self.test_weights*1000)

    def plot_roc(self):
        ''' 
        Method to plot the roc curve, using method from Plotter() class
        '''
        roc_fig = self.plotter.plot_roc(self.y_train, self.y_pred_train, self.train_weights, 
                                   self.y_test, self.y_pred_test, self.test_weights)

        Utils.check_dir('{}/plots'.format(os.getcwd()))
        roc_fig.savefig('plots/ROC_curve_best_model.pdf')
        plt.close()

    def plot_output_score(self):
        ''' 
        Method to plot the roc curve and compute the integral of the roc as a 
        performance metric
        '''
        output_score_fig = self.plotter.plot_output_score(self.y_test, self.y_pred_test, self.test_weights)

        Utils.check_dir('{}/plots'.format(os.getcwd()))
        output_score_fig.savefig('plots/output_score_best_model.pdf')
        plt.close()
     
    def train_old_classifier(self, file_path, save=True):
        '''
        cross check the performance using DMatrix and xgb.train wrapper (mainly for sanity)
        '''
        training_matrix  = xgb.DMatrix(self.X_train, label=self.y_train, weight=self.train_weights, feature_names=self.train_vars)
        if self.eq_train: 
            alt_train_matrix = xgb.DMatrix(self.X_train, label=self.y_train, weight=self.train_weights_eq, feature_names=self.train_vars)
        testing_matrix   = xgb.DMatrix(self.X_test,  label=self.y_test,  weight=self.test_weights,  feature_names=self.train_vars)

        print 'Training classifier... '
        if self.eq_train:
            clf = xgb.train({'objective':'binary:logistic'}, alt_train_matrix)
        else:
            clf = xgb.train({'objective':'binary:logistic'}, training_matrix)
        print 'Finished Training classifier!'
        
        y_pred_train = clf.predict(training_matrix)
	print 'Area under ROC curve for train set is: {:.4f}'.format(roc_auc_score(self.y_train, y_pred_train, sample_weight=self.train_weights*1000))
        y_pred_test = clf.predict(testing_matrix)
        print 'Area under ROC curve for test set is: {:.4f}'.format(roc_auc_score(self.y_test, y_pred_test, sample_weight=self.test_weights*1000))


class Plotter(object):
    '''
    Class to plot input variables and output scores
    '''
    def __init__(self, data_obj, input_vars, sig_col='firebrick', sig_label='VBF', bkg_col='violet', bkg_label='DYMC',  normalise=True): 
        self.sig_df     = data_obj.mc_df_sig
        self.bkg_df     = data_obj.mc_df_bkg

        #FIXME:
        # feature egineering here, by calling a method from above class (should be done above really)
        self.sig_df['dijet_centrality'] = np.exp(-4.*((self.sig_df['dijet_Zep']/self.sig_df['dijet_abs_dEta'])**2))
        self.bkg_df['dijet_centrality'] = np.exp(-4.*((self.bkg_df['dijet_Zep']/self.bkg_df['dijet_abs_dEta'])**2))

        self.sig_colour = sig_col
        self.sig_label  = sig_label
        self.bkg_colour = bkg_col
        self.bkg_label  = bkg_label
        self.normalise  = normalise

        self.input_vars = input_vars
        del data_obj

    def plot_input(self, var):
        fig  = plt.figure(1)
        axes = fig.gca()
        
        var_sig     = self.sig_df[var].values
        sig_weights = self.sig_df['weight'].values
        var_bkg     = self.bkg_df[var].values
        bkg_weights = self.bkg_df['weight'].values

        if self.normalise:
            sig_weights /= np.sum(sig_weights)
            bkg_weights /= np.sum(bkg_weights)

        #plot with np first to get consistent ranges and  modify last bin to avoid empty space 
        binned_data, bin_edges = np.histogram(var_sig, weights=sig_weights)
        bkw_index=0
        for ibin_sum in reversed(binned_data):
            if ibin_sum < 0.05*np.sum(binned_data): bkw_index+=1
            else: break
        if bkw_index!=0: bin_edges = bin_edges[:-bkw_index]     
        bins = np.linspace(bin_edges[0], bin_edges[-1], 21)

        axes.hist(var_sig, bins=bins, label=self.sig_label, weights=sig_weights, histtype='step', color=self.sig_colour)
        axes.hist(var_bkg, bins=bins, label=self.bkg_label, weights=bkg_weights, histtype='step', color=self.bkg_colour)

        var_name_safe = var.replace('_',' ')
        axes.set_xlabel('{}'.format(var_name_safe), ha='right', x=1)
        axes.set_ylabel('Arbitrary Units', ha='right', y=1)
        current_bottom, current_top = axes.get_ylim()
        axes.set_ylim(bottom=0, top=current_top*1.2)
        axes.legend(bbox_to_anchor=(0.97,0.97))
        self.plot_cms_labels(axes)
       
        Utils.check_dir('{}/plots'.format(os.getcwd()))
        fig.savefig('plots/{}.pdf'.format(var))
        plt.close()

    def plot_cms_labels(self, axes, label='Work in progress', energy='(13 TeV)'):
        axes.text(0, 1.01, r'\textbf{CMS} %s'%label, ha='left', va='bottom', transform=axes.transAxes)
        axes.text(1, 1.01, r'{}'.format(energy), ha='right', va='bottom', transform=axes.transAxes)

    def plot_roc(self, y_train, y_pred_train, train_weights, y_test, y_pred_test, test_weights):
        print 'plotting ROC'
        bkg_eff_train, sig_eff_train, _ = roc_curve(y_train, y_pred_train, sample_weight=train_weights)
        bkg_eff_test, sig_eff_test, _ = roc_curve(y_test, y_pred_test, sample_weight=test_weights)

        fig = plt.figure(1)
        axes = fig.gca()
        axes.plot(bkg_eff_train, sig_eff_train, color='red', label='Train')
        axes.plot(bkg_eff_test, sig_eff_test, color='blue', label='Test')                                            
        axes.set_xlabel('Background efficiency', ha='right', x=1)
        axes.set_xlim((0,1))
        axes.set_ylabel('Signal efficiency', ha='right', y=1)
        axes.set_ylim((0,1))
        axes.legend(bbox_to_anchor=(0.97,0.97))
        self.plot_cms_labels(axes)
        return fig

    def plot_output_score(self, y_test, y_pred_test, test_weights):
        fig  = plt.figure(1)
        axes = fig.gca()
        bins = np.linspace(0,1,31)

        sig_scores = y_pred_test.ravel()  * (y_test==1)
        sig_w_true = test_weights.ravel() * (y_test==1)

        bkg_scores = y_pred_test.ravel()  * (y_test==0)
        bkg_w_true = test_weights.ravel() * (y_test==0)


        if self.normalise:
            sig_w_true /= np.sum(sig_w_true)
            bkg_w_true /= np.sum(bkg_w_true)

        axes.hist(sig_scores, bins=bins, label=self.sig_label, weights=sig_w_true, histtype='step', color=self.sig_colour)
        axes.hist(bkg_scores, bins=bins, label=self.bkg_label, weights=bkg_w_true, histtype='step', color=self.bkg_colour)
        axes.legend(bbox_to_anchor=(0.97,0.97))

        current_bottom, current_top = axes.get_ylim()
        axes.set_ylim(bottom=0, top=current_top*1.2)
        axes.set_ylabel('Arbitrary Units', ha='right', y=1)
        axes.set_xlabel('BDT Score', ha='right', x=1)
        self.plot_cms_labels(axes)
        return fig


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
    def sub_hp_script(self, eq_weights, hp_string, k_folds, job_dir='{}/bdt_hp_opts_jobs'.format(os.getcwd())):
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
        with open('sub_bdt_hp_template.sh') as f_template:
            with open(sub_file_name,'w') as f_sub:
                for line in f_template.readlines():
                    if '!CWD!' in line: line = line.replace('!CWD!', os.getcwd())
                    if '!CMD!' in line: line = line.replace('!CMD!', '"{}"'.format(sub_command))
                    f_sub.write(line)

        system( 'qsub -o {} -e {} -q hep.q -l h_rt=1:00:00 -l h_vmem=4G {}'.format(sub_file_name.replace('.sh','.out'), sub_file_name.replace('.sh','.err'), sub_file_name ) )


class Logger(object):
    '''
    Class to log info for training and to print options/training configs
    '''
    def __init__(self): pass
