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

    def __init__(self, data_obj, train_vars, train_frac, eq_train=False,hp_n_estimators=100,hp_learning_rate=0.05,hp_max_depth=4,hp_min_child_weight=0.01,hp_subsample=0.6,hp_colsample_bytree=0.6,hp_gamma=1):

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

        self.hp_n_estimators = hp_n_estimators
        self.hp_learning_rate = hp_learning_rate #Eta!
        self.hp_max_depth = hp_max_depth
        self.hp_min_child_weight = hp_min_child_weight
        self.hp_subsample = hp_min_child_weight
        self.hp_colsample_bytree = hp_colsample_bytree
        self.hp_gamma = hp_gamma     

        #attributes for the hp optmisation and cross validation
        self.clf              = xgb.XGBClassifier(objective='binary:logistic', n_estimators=self.hp_n_estimators, 
                                                  learning_rate=self.hp_learning_rate, max_depth=self.hp_max_depth, min_child_weight=self.hp_min_child_weight, 
                                                  subsample=self.hp_subsample, colsample_bytree=self.hp_colsample_bytree, gamma=self.hp_gamma)

        self.hp_grid_rnge     = {'learning_rate': [0.01, 0.05, 0.1, 0.3],
                                 'max_depth':[x for x in range(6,7)],
                                 'min_child_weight':[x for x in range(0,3)], #FIXME: probs not appropriate range for a small sumw!
                                 #'gamma': np.linspace(2,4,3).tolist(),
                                 #'subsample': [0.5, 0.8, 1.0],
                                 'n_estimators':[300,400]
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
        
    def create_X_and_y(self, mass_res_reweight=False):
        """
        Create X train/test and y train/test

        Arguments
        ---------
        mass_res_reweight: bool
           re-weight signal events by 1/sigma(m_ee), in training only.
        """
        
        mc_df_sig = self.data_obj.mc_df_sig
        mc_df_bkg = self.data_obj.mc_df_bkg

        #add y_target label
        mc_df_sig['y'] = np.ones(self.data_obj.mc_df_sig.shape[0]).tolist()
        mc_df_bkg['y'] = np.zeros(self.data_obj.mc_df_bkg.shape[0]).tolist()



        if self.eq_train: 
            if mass_res_reweight:
                #be careful not to change the real weight variable, or test scores will be invalid
                mc_df_sig['MoM_weight'] = (mc_df_sig['weight']) * (1./mc_df_sig['diphotonSigmaMoM'])
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

            #Second instance is just to generate X_test2 that contains 100% of the events so that we can then run predict_proba on the full dataset
            X_train2, X_test2, train_w2, test_w2, y_train2, y_test2, proc_arr_train2, proc_arr_test2 = train_test_split(Z_tot[self.train_vars], 
                                                                                                               Z_tot['weight'],
                                                                                                               Z_tot['y'], Z_tot['proc'],
                                                                                                               train_size=0, 
                                                                                                               shuffle=False
                                                                                                               )                                                                                                                             
            #NB: will never test/evaluate with equalised weights. This is explicitly why we set another train weight attribute, 
            #    because for overtraining we need to evaluate on the train set (and hence need nominal MC train weights)
            self.train_weights_eq = train_w_eqw.values
        elif mass_res_reweight:
           mc_df_sig['MoM_weight'] = (mc_df_sig['weight']) * (1./mc_df_sig['diphotonSigmaMoM'])

           Z_tot = pd.concat([mc_df_sig, mc_df_bkg], ignore_index=True)
           X_train, X_test, train_w, test_w, train_w_eqw, test_w_eqw, y_train, y_test, proc_arr_train, proc_arr_test = train_test_split(Z_tot[self.train_vars], Z_tot['weight'], 
                                                                                                                                        Z_tot['MoM_weight'], Z_tot['y'], Z_tot['proc'],
                                                                                                                                        train_size=self.train_frac, 
                                                                                                                                        #test_size=1-self.train_frac,
                                                                                                                                        shuffle=True, 
                                                                                                                                        random_state=1357
                                                                                                                                        )
           X_train2, X_test2, train_w2, test_w2, y_train2, y_test2, proc_arr_train2, proc_arr_test2 = train_test_split(Z_tot[self.train_vars], 
                                                                                                               Z_tot['weight'],
                                                                                                               Z_tot['y'], Z_tot['proc'],
                                                                                                               train_size=0, 
                                                                                                               shuffle=False
                                                                                                               )
           
           self.train_weights_eq = train_w_eqw.values
           self.eq_train = True #use alternate weight in training. could probs rename this to something better
        else:
           print ('not applying any reweighting...')
           Z_tot = pd.concat([mc_df_sig, mc_df_bkg], ignore_index=True)
           X_train, X_test, train_w, test_w, y_train, y_test, proc_arr_train, proc_arr_test = train_test_split(Z_tot[self.train_vars], 
                                                                                                               Z_tot['weight'],
                                                                                                               Z_tot['y'], Z_tot['proc'],
                                                                                                               train_size=self.train_frac, 
                                                                                                               #test_size=1-self.train_frac,
                                                                                                               shuffle=True, random_state=1357
                                                                                                               )

           X_train2, X_test2, train_w2, test_w2, y_train2, y_test2, proc_arr_train2, proc_arr_test2 = train_test_split(Z_tot[self.train_vars], 
                                                                                                               Z_tot['weight'],
                                                                                                               Z_tot['y'], Z_tot['proc'],
                                                                                                               train_size=0, 
                                                                                                               shuffle=False
                                                                                                               )

#Change
        self.X_test2          = X_test2

        self.X_train          = X_train #.values
        self.y_train          = y_train #.values
        self.train_weights    = train_w #.values
        self.proc_arr_train   = proc_arr_train

        self.X_test           = X_test  #.values
        self.y_test           = y_test  #.values
        self.test_weights     = test_w  #.values
        self.proc_arr_test    = proc_arr_test

        self.X_data_train, self.X_data_test = train_test_split(self.data_obj.data_df[self.train_vars], train_size=self.train_frac, test_size=1-self.train_frac, shuffle=True, random_state=1357)

    def create_X_and_y_three_class(self, third_class, mass_res_reweight=True):
        """
        Create X train/test and y train/test for three class BDR

        Arguments
        ---------
        mass_res_reweight: bool
           re-weight signal events by 1/sigma(m_ee), in training only.
        third_class: str
           name third bkg class for the classifier. Remaining classes: 1) all other bkgs, 2) signal
        """

        mc_df_sig = self.data_obj.mc_df_sig
        mc_df_bkg = self.data_obj.mc_df_bkg

        #add y_target label

        bkg_procs_key = [0,1]
        bkg_procs_mask = []
        bkg_procs_mask.append( mc_df_bkg['proc'].ne(third_class) )
        bkg_procs_mask.append( mc_df_bkg['proc'].eq(third_class) )
        mc_df_bkg['y'] = np.select(bkg_procs_mask, bkg_procs_key)
        mc_df_sig['y'] = np.full(self.data_obj.mc_df_sig.shape[0], 2).tolist()

        if self.eq_train: 
            if mass_res_reweight:
                #be careful not to change the real weight variable, or test scores will be invalid
                mc_df_sig['MoM_weight'] = (mc_df_sig['weight']) * (1./mc_df_sig['diphotonSigmaMoM'])                
                bkg_sumw                = np.sum(mc_df_bkg[mc_df_bkg.y==0]['weight'].values)
                third_class_sumw        = np.sum(mc_df_bkg[mc_df_bkg.y==1]['weight'].values)
                sig_sumw                = np.sum(mc_df_sig['MoM_weight'].values)
                mc_df_sig['eq_weight']  = (mc_df_sig['MoM_weight']) * (bkg_sumw/sig_sumw)
                mc_df_bkg.loc[mc_df_bkg.y==1,'weight'] = mc_df_bkg.loc[mc_df_bkg.y==1,'weight'] * (bkg_sumw/third_class_sumw)
            else: 
                #b_to_s_ratio = np.sum(mc_df_bkg['weight'].values)/np.sum(mc_df_sig['weight'].values)
                #mc_df_sig['eq_weight'] = (mc_df_sig['weight']) * (b_to_s_ratio) 
                bkg_sumw                = np.sum(mc_df_bkg[mc_df_bkg.y==0]['weight'].values)
                third_class_sumw        = np.sum(mc_df_bkg[mc_df_bkg.y==1]['weight'].values)
                sig_sumw                = np.sum(mc_df_sig['weight'].values)
                mc_df_sig['eq_weight']  = (mc_df_sig['weight']) * (bkg_sumw/sig_sumw)
                mc_df_bkg.loc[mc_df_bkg.y==1,'weight'] = mc_df_bkg.loc[mc_df_bkg.y==1,'weight'] * (bkg_sumw/third_class_sumw)
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
           mc_df_sig['MoM_weight'] = (mc_df_sig['weight']) * (1./mc_df_sig['diphotonSigmaMoM'])

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
           print ('not applying any reweighting...')
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

        print ('Training classifier... ')
        clf = self.clf.fit(self.X_train, self.y_train) #, sample_weight=train_weights)
        print ('Finished Training classifier!')
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
        print ('avg. validation roc is: {}'.format(avg_val_auc))
        if len(hp_roc)==0: 
            roc_file.write('{};{:.4f}'.format(hp_string, avg_val_auc))
        elif float(hp_roc[-1].split(';')[-1]) < avg_val_auc:
            roc_file.write('\n')
            roc_file.write('{};{:.4f}'.format(hp_string, avg_val_auc))

    def compute_roc(self):
        """
        Compute the area under the associated ROC curve, with mc weights. Also compute with blinded data as bkg

        Returns
        -------
        roc_auc_score: float
            area under the roc curve evluated on test set
        """

        self.y_pred_train = self.clf.predict_proba(self.X_train)[:,1:]
        print ('Area under ROC curve for train set is: {:.4f}'.format(roc_auc_score(self.y_train, self.y_pred_train))) #, sample_weight=self.train_weights
        self.y_pred_test = self.clf.predict_proba(self.X_test)[:,1:]
        print ('Area under ROC curve for test set is: {:.4f}'.format(roc_auc_score(self.y_test, self.y_pred_test))) #, sample_weight=self.test_weights

        self.y_pred = self.clf.predict_proba(self.X_test2)[:,1:]
        np.savetxt('models/Output_BDT.csv', self.y_pred, delimiter=',')

        #get auc for bkg->data
        sig_y_pred_test  = self.y_pred_test[self.y_test==1]
        sig_weights_test = self.test_weights[self.y_test==1]
        sig_y_true_test  = self.y_test[self.y_test==1]
        data_weights_test = np.ones(self.X_data_test.values.shape[0])
        data_y_true_test  = np.zeros(self.X_data_test.values.shape[0])
        #Removed .values for feature importance plots. .values converts dfs to np.arrays
        #data_y_pred_test  = self.clf.predict_proba(self.X_data_test.values)[:,1:]
        data_y_pred_test  = self.clf.predict_proba(self.X_data_test)[:,1:]
        
        print ('Area under ROC curve with data as bkg is: {:.4f}'.format(roc_auc_score( np.concatenate((sig_y_true_test, data_y_true_test), axis=None),
                                                                                       np.concatenate((sig_y_pred_test, data_y_pred_test), axis=None),
                                                                                       sample_weight=np.concatenate((sig_weights_test, data_weights_test), axis=None) 
                                                                                     )
                                                                        ))

        return roc_auc_score(self.y_test, self.y_pred_test) #, sample_weight=self.test_weights

    def compute_roc_three_class(self, third_class):
        """
        Compute the area under the associated ROC curves for three class problem, with mc weights. Also compute with blinded data as bkg

        """

        self.y_pred_train = self.clf.predict_proba(self.X_train)
        self.y_pred_test  = self.clf.predict_proba(self.X_test)

        sig_y_train = np.where(self.y_train==2, 1, 0)
        sig_y_test  = np.where(self.y_test==2, 1, 0)
                  
        bkg_y_train = np.where(self.y_train>0, 0, 1)
        bkg_y_test  = np.where(self.y_test>0, 0, 1)
                  
        third_class_y_train = np.where(self.y_train==1, 1, 0)
        third_class_y_test  = np.where(self.y_test==1, 1, 0)

        print ('area under roc curve for training set (sig vs rest) = %1.3f'%( roc_auc_score(sig_y_train, self.y_pred_train[:,2], sample_weight=self.train_weights) ))
        print ('area under roc curve for test set = %1.3f \n'%( roc_auc_score(sig_y_test, self.y_pred_test[:,2], sample_weight=self.test_weights) ))
        print ('area under roc curve for training set (bkg vs rest) = %1.3f'%( roc_auc_score(bkg_y_train, self.y_pred_train[:,0], sample_weight=self.train_weights) ))
        print ('area under roc curve for test set = %1.3f \n'%( roc_auc_score(bkg_y_test, self.y_pred_test[:,0], sample_weight=self.test_weights) ))
        print ('area under roc curve for training set (third class vs rest) = %1.3f'%( roc_auc_score(third_class_y_train, self.y_pred_train[:,1], sample_weight=self.train_weights) ))
        print ('area under roc curve for test set = %1.3f'%( roc_auc_score(third_class_y_test, self.y_pred_test[:,1], sample_weight=self.test_weights) ))

        #get auc for bkg->data
        #sig_y_pred_test  = self.y_pred_test[self.y_test==1]
        #sig_weights_test = self.test_weights[self.y_test==1]
        #sig_y_true_test  = self.y_test[self.y_test==1]
        #data_weights_test = np.ones(self.X_data_test.values.shape[0])
        #data_y_true_test  = np.zeros(self.X_data_test.values.shape[0])
        #data_y_pred_test  = self.clf.predict_proba(self.X_data_test.values)[:,1:]
        #print 'Area under ROC curve with data as bkg is: {:.4f}'.format(roc_auc_score( np.concatenate((sig_y_true_test, data_y_true_test), axis=None),
        #                                                                               np.concatenate((sig_y_pred_test, data_y_pred_test), axis=None),
        #                                                                               sample_weight=np.concatenate((sig_weights_test, data_weights_test), axis=None) 
        #                                                                             )
        #                                                               )
        

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

    def plot_output_score(self, out_tag, ratio_plot=False, norm_to_data=False, log=False):
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

        #Removed .value for feature importance plot
        output_score_fig = self.plotter.plot_output_score(self.y_test, self.y_pred_test, self.test_weights, 
                                                          self.proc_arr_test, self.clf.predict_proba(self.X_data_test)[:,1:],
                                                          ratio_plot=ratio_plot, norm_to_data=norm_to_data, log=log)        

        Utils.check_dir('{}/plotting/plots/{}'.format(os.getcwd(),out_tag))
        output_score_fig.savefig('{0}/plotting/plots/{1}/{1}_output_score.pdf'.format(os.getcwd(), out_tag))
        print('saving: {0}/plotting/plots/{1}/{1}_output_score.pdf'.format(os.getcwd(), out_tag))
        plt.close()


    def plot_feature_importance(self,num_plots='single',num_feature=20,imp_type='weight',values = False):
        
        if num_plots=='single':
            xgb.plot_importance(self.clf, max_num_features=num_feature, grid = False, height = 0.4, importance_type = imp_type, title = 'Feature importance ({})'.format(imp_type), show_values = values, color ='blue')
            plt.savefig('{0}/plotting/plots/{1}/{1}_feature_importance_{2}.pdf'.format(os.getcwd(), out_tag, imp_type))
            print('saving: {0}/plotting/plots/{1}/{1}_feature_importance_{2}.pdf'.format(os.getcwd(), out_tag, imp_type))
        
        else:
            imp_types = ['weight','gain','cover']
            for i in imp_types:
                xgb.plot_importance(self.clf, max_num_features=num_feature, grid = False, height = 0.4, importance_type = i, title = 'Feature importance ({})'.format(i), show_values = values, color ='blue')
                plt.savefig('{0}/plotting/plots/{1}/{1}_feature_importance_{2}.pdf'.format(os.getcwd(), out_tag, i))
                print('saving: {0}/plotting/plots/{1}/{1}_feature_importance_{2}.pdf'.format(os.getcwd(), out_tag, i))
        '''
        if importance_type=='weight':
            xgb.plot_importance(self.clf, max_num_features=20, grid = False, height = 0.4, importance_type = "weight", title = "Feature importance (weight)", show_values = False, color ='blue')
            plt.savefig('{0}/plotting/plots/{1}/{1}_feature_importance_weight.pdf'.format(os.getcwd(), out_tag))
            print('saving: {0}/plotting/plots/{1}/{1}_feature_importance_weight.pdf'.format(os.getcwd(), out_tag))

        if importance_type=='gain':
            xgb.plot_importance(self.clf, max_num_features=20, grid = False, height = 0.4, importance_type = "gain", title = "Feature importance (gain)", show_values = False, color ='blue')
            plt.savefig('{0}/plotting/plots/{1}/{1}_feature_importance_gain.pdf'.format(os.getcwd(), out_tag))
            print('saving: {0}/plotting/plots/{1}/{1}_feature_importance_gain.pdf'.format(os.getcwd(), out_tag))

        if importance_type=='cover':
            xgb.plot_importance(self.clf, max_num_features=20, grid = False, height = 0.4, importance_type = "cover", title = "Feature importance (cover)", show_values = False, color ='blue')
            plt.savefig('{0}/plotting/plots/{1}/{1}_feature_importance_cover.pdf'.format(os.getcwd(), out_tag))
            print('saving: {0}/plotting/plots/{1}/{1}_feature_importance_cover.pdf'.format(os.getcwd(), out_tag))
        '''


    def plot_output_score_three_class(self, out_tag, ratio_plot=False, norm_to_data=False, log=False, third_class=''):
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

        #class_id = {'Background':0, 'Third_Class':1 ,'Signal':2}
        class_id = {'Other_backgrounds':0, 'VBF_Z':1 ,'VBF_Signal':2}
        for clf_class, _id in class_id.iteritems():
            #plot all processes for each predicted class
            y_pred_test  = self.y_pred_test[:,_id]
            output_score_fig = self.plotter.plot_output_score_three_class(self.y_test, y_pred_test, self.test_weights, 
                                                                          norm_to_data=norm_to_data, log=log, clf_class=clf_class)

            Utils.check_dir('{}/plotting/plots/{}'.format(os.getcwd(),out_tag))
            output_score_fig.savefig('{0}/plotting/plots/{1}/{1}_output_score_clf_class_{2}.pdf'.format(os.getcwd(), out_tag, clf_class))
            print('saving: {0}/plotting/plots/{1}/{1}_output_score_clf_class_{2}.pdf'.format(os.getcwd(), out_tag, clf_class))
            plt.close()

