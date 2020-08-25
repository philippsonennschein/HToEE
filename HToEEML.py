import uproot as upr
import numpy as np
import pandas as pd
import xgboost as xgb
from os import path, system
from variables import nominal_vars, gen_vars
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

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


    def load_mc(self, year, file_name, bkg=False):
        '''
        Try to load mc dataframe. If it doesn't exist, read in the root file.
        This should be used once per year, if reading in multiple years.
        '''
        try: 
            if not bkg: self.mc_df_sig.append( self.load_df(self.mc_dir+'DataFrames/', 'sig', year) )
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
        if (flag=='sig') or(flag=='bkg'): df = df_tree.pandas.df(self.nominal_vars + gen_vars).query(self.cut_string)
        df = df_tree.pandas.df(self.nominal_vars).query(self.cut_string)

        print('Reshuffling events')
        df = df.sample(frac=1).reset_index(drop=True)

        if (flag=='sig') or(flag=='bkg'): df = self.scale_by_lumi(file_name, df, year)
        print('Number of events in final dataframe: {}'.format(np.sum(df['weight'].values)))

        self.check_dir(file_dir+'DataFrames/') 
        df.to_hdf('{}/{}_df_{}.h5'.format(file_dir+'DataFrames', flag, year), 'df',mode='w',format='t')
        print('Saved dataframe: {}/{}_df_{}.h5'.format(file_dir+'DataFrames', flag, year))

    def scale_by_lumi(self, file_name, df, year):
        '''
        Scale simulation by the lumi of the year
        '''
        print('scaling weights for file: {} from year: {}, by {}'.format(file_name, year, self.lumi_map[year]))
        df['weight']*=self.lumi_map[year]
        return df

    @classmethod #allows method to be called without "self" argument
    def check_dir(self, file_dir):
        '''
        Check directory exists; if not make it.
        '''
        if not path.isdir(file_dir):
            print 'making directory: {}'.format(file_dir)
            system('mkdir -p %s' %file_dir)

    def concat_years(self):
        #FIXME: add functionality concat list with more than one entry
        if len(self.mc_df_sig) == 1: self.mc_df_sig = self.mc_df_sig[0]
        else: pass
        if len(self.mc_df_bkg) ==1:  self.mc_df_bkg = self.mc_df_bkg[0] 
        else: pass
        if len(self.data_df) ==1:  self.data_df = self.data_df[0] 
        else: pass 

class DNN(object):
    '''
    Class to train DNN
    '''
    def __init__(self): pass


class BDT(object):
    '''
    Class to train BDT
    '''
    def __init__(self, sig_df, bkg_df, train_vars, train_frac):
        #if using multiple years, should be concatted by now and in ROOTHelpers attribute 

        #concat X DF frame for sig and df
        X_tot = pd.concat([sig_df, bkg_df], axis=0, ignore_index=True)

        #FIXME:
        #could do feature egineering here, but call a method from above class
        X_tot['dijet_centrality'] = np.exp(-4.*((X_tot['dijet_Zep']/X_tot['dijet_abs_dEta'])**2))

        # make target column of 1 (VBF) and 0 (bkg) then join sig and bkg together into one df
        y_tot = np.hstack( (np.ones(sig_df.shape[0]), np.zeros(bkg_df.shape[0])) ).ravel()

        X_train, X_test, y_train, y_test = train_test_split(X_tot, y_tot, train_size=train_frac, test_size=1-train_frac, shuffle=True)

        self.X_train          = X_train[train_vars].values
        self.y_train          = y_train
        self.train_weights    = X_train['weight'].values

        self.X_test           = X_test[train_vars].values
        self.y_test           = y_test
        self.test_weights     = X_test['weight'].values
        #FIXME: add equalised weights options
        #self.eq_weights       = X_test['weight'].values

        self.y_pred           = None
        self.clf              = None
        self.hp_dict          = {}

        self.logger           = Logger()

    def train_classifier(self, file_path, save=True, eq_weights=False):
        #FIXME: set EQ weights as training option

        print 'Training classifier... '
        clf = xgb.XGBClassifier(n_estimators=300, lr=0.05, maxDepth=10, gamma=0, subsample=0.5, verbosity=2)
        clf.fit(self.X_train, self.y_train)
        print 'Finished Training classifier!'
        self.clf = clf

        ROOTHelpers.check_dir(file_path + 'models')
        #if save: clf.save_model(file_path + 'models')
        
        
    def compute_roc(self):

        y_pred_train = self.clf.predict(self.X_train)
        print self.train_weights
        print 'Area under ROC curve for test set is: {:.4f}'.format(roc_auc_score(self.y_train, y_pred_train, sample_weight=self.train_weights))

        print self.test_weights
        y_pred_test = self.clf.predict(self.X_test)
        print 'Area under ROC curve for test set is: {:.4f}'.format(roc_auc_score(self.y_test, y_pred_test, sample_weight=self.test_weights))
     

class Logger(object):
    '''
    Class to log info for training and to print options/training configs
    '''
    def __init__(self): pass


class Utils(object):
    def __init__(self): pass

    def plot_roc(self, y_pred, y_true, weight):
        ''' 
        Method to plot the roc curve and compute the integral of the roc as a 
        performance metric
        '''

        pass
