import os
from os import path, system

import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score, roc_curve
import pickle

from DataHandling import ROOTHelpers
from PlottingUtils import Plotter
from Utils import Utils

class BDTHelpers(object):
    """
    Functions to train a binary class BDT for signal/background separation

    :param data_obj: instance of ROOTHelpers class. containing Dataframes for simulated signal, simulated background, and possibly data
    :type data_obj: ROOTHelpers
    :param train_vars: list of variables to train the BDT with
    :type train_vars: list
    :param train_frac: fraction of events to train the network. Test on 1-train_frac
    :type train_frac: float
    :param eq_train: whether to train with the sum of signal weights scaled equal to the sum of background weights
    :type eq_train: bool
    """

    def __init__(self, data_obj, train_vars, train_frac, eq_train=True):

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
        """
        Create X train/test and y train/test

        Arguments
        ---------
        mass_res_reweight: bool
           re-weight signal events by 1/sigma(m_ee), in training only.
        """
        
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
                                                                                                                                         #test_size=1-self.train_frac,
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
                                                                                                                                        #test_size=1-self.train_frac,
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
                                                                                                               #test_size=1-self.train_frac,
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
        """
        Train the BDT and save the resulting classifier

        Arguments
        ---------
        file_path: string
            base file path used when saving model
        save: bool
            option to save the classifier
        model_name: string
            name of the model to be saved
        """

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

    def batch_gs_cv(self, k_folds=3, pt_rew=False):
        """
        Submit a sets of hyperparameters permutations (based on attribute hp_grid_rnge) to the IC batch.
        Perform k-fold cross validation; take care to separate training weights, which
        may be modified w.r.t nominal weights, and the weights used when evaluating on the
        validation set which should be the nominal weights

        Arguments
        ---------
        k_folds: int
            number of folds that the training+validation set are partitioned into
        """

        #get all possible HP sets from permutations of the above dict
        hp_perms = self.get_hp_perms()

        #submit job to the batch for the given HP range:
        for hp_string in hp_perms:
            Utils.sub_hp_script(self.eq_train, hp_string, k_folds, pt_rew)
            
    def get_hp_perms(self):
        """
        Return a list of all possible hyper parameter combinations (permutation template set in constructor) in format 'hp1:val1,hp2:val2, ...'

        Returns
        -------
        final_hps: all possible combinaions of hyper parameters given in self.hp_grid_rnge
        """

        from itertools import product

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
        """
        Set a given set hyper-parameters for the classifier. Append the resulting classifier as a class attribute

        Arguments
        --------
        hp_string: string
            string of hyper-parameter values, with format 'hp1:val1,hp2:val2, ...'
        """

        hp_dict = {}
        for params in hp_string.split(','):
            hp_name = params.split(':')[0]
            hp_value =params.split(':')[1]
            try: hp_value = int(hp_value)
            except ValueError: hp_value = float(hp_value)
            hp_dict[hp_name] = hp_value
        self.clf = xgb.XGBClassifier(**hp_dict)
 
    def set_k_folds(self, k_folds):
        """
        Partition the X and Y matrix into folds = k_folds, and append to list (X and y separate) attribute for the class, from the training samples (i.e. X_train -> X_train + X_validate, and same for y and w)
        Used in conjunction with the get_i_fold function to pull one fold out for training+validating
        Note that validation weights should always be the nominal MC weights

        Arguments
        --------
        k_folds: int
            number of folds that the training+validation set are partitioned into
        """

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
        """
        Gets the training and validation fold for a given CV iteration from class attribute,
        and overwrites the self.X_train, self.y_train and self.X_train, self.y_train respectively, and the weights, to train
        Note that for these purposes, our "test" sets are really the "validation" sets

        Arguments
        --------
        i_folds: int
            the index describing the train+validate datasets
        """

        self.X_train          = self.X_folds_train[i_fold]
        self.train_weights    = self.w_folds_train[i_fold] #nominal MC weights needed for computing roc on train set (overtraining test)
        if self.eq_train:
            self.train_weights_eq = self.w_folds_train_eq[i_fold] 
        self.y_train          = self.y_folds_train[i_fold]

        self.X_test           = self.X_folds_validate[i_fold]
        self.y_test           = self.y_folds_validate[i_fold]
        self.test_weights     = self.w_folds_validate[i_fold]

    def compare_rocs(self, roc_file, hp_string):
        """
        Compare the AUC for the current model, to the current best AUC saved in a .txt file 
        Arguments
        ---------
        roc_file: string
            path for the file holding the current best AUC (as the final line)
        hp_string: string
            string contraining each hyper_paramter for the network, with the following syntax: 'hp_1_name:hp_1_value, hp_2_name:hp_2_value, ...'
        """

        hp_roc = roc_file.readlines()
        avg_val_auc = np.average(self.validation_rocs)
        print 'avg. validation roc is: {}'.format(avg_val_auc)
        if len(hp_roc)==0: 
            roc_file.write('{};{:.4f}'.format(hp_string, avg_val_auc))
        elif float(hp_roc[-1].split(';')[-1]) < avg_val_auc:
            roc_file.write('\n')
            roc_file.write('{};{:.4f}'.format(hp_string, avg_val_auc))

    def compute_roc(self):
        """
        Compute the area under the associated ROC curve, with mc weights
        """

        self.y_pred_train = self.clf.predict_proba(self.X_train)[:,1:]
        print 'Area under ROC curve for train set is: {:.4f}'.format(roc_auc_score(self.y_train, self.y_pred_train, sample_weight=self.train_weights))

        self.y_pred_test = self.clf.predict_proba(self.X_test)[:,1:]
        print 'Area under ROC curve for test set is: {:.4f}'.format(roc_auc_score(self.y_test, self.y_pred_test, sample_weight=self.test_weights))
        return roc_auc_score(self.y_test, self.y_pred_test, sample_weight=self.test_weights)

    def plot_roc(self, out_tag):
        """
        Method to plot the roc curve, using method from Plotter() class
        """

        roc_fig = self.plotter.plot_roc(self.y_train, self.y_pred_train, self.train_weights, 
                                   self.y_test, self.y_pred_test, self.test_weights, out_tag=out_tag)

        Utils.check_dir('{}/plotting/plots/{}'.format(os.getcwd(), out_tag))
        roc_fig.savefig('{0}/plotting/plots/{1}/{1}_ROC_curve.pdf'.format(os.getcwd(),out_tag))
        print('saving: {0}/plotting/plots/{1}/{1}_ROC_curve.pdf'.format(os.getcwd(),out_tag))
        plt.close()

    def plot_output_score(self, out_tag, ratio_plot=False, norm_to_data=False):
        """
        Plot the output score for the classifier, for signal, background, and data

        Arguments
        ---------
        out_tag: string
            output tag used as part of the image name, when saving
        ratio_plot: bool
            whether to plot the ratio between simulated background and data
        norm_to_data: bool
            whether to normalise the integral of the simulated background, to the integral in data
        """

        output_score_fig = self.plotter.plot_output_score(self.y_test, self.y_pred_test, self.test_weights, 
                                                          self.proc_arr_test, self.clf.predict_proba(self.X_data_test.values)[:,1:],
                                                          ratio_plot=ratio_plot, norm_to_data=norm_to_data)

        Utils.check_dir('{}/plotting/plots/{}'.format(os.getcwd(),out_tag))
        output_score_fig.savefig('{0}/plotting/plots/{1}/{1}_output_score.pdf'.format(os.getcwd(), out_tag))
        print('saving: {0}/plotting/plots/{1}/{1}_output_score.pdf'.format(os.getcwd(), out_tag))
        plt.close()

