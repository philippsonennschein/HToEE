import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml
import sys
from os import path,system

from DataHandling import ROOTHelpers
from BDTs_HP_opt import BDTHelpers

def main(options):

    #take options from the yaml config
    with open(options.config, 'r') as config_file:
        config            = yaml.load(config_file)
        output_tag        = config['output_tag']

        mc_dir            = config['mc_file_dir']
        mc_fnames         = config['mc_file_names']
  
        #data not needed yet, but stil specify in the config for compatibility with constructor
        data_dir          = config['data_file_dir']
        data_fnames       = config['data_file_names']

        proc_to_tree_name = config['proc_to_tree_name']

        train_vars        = config['train_vars']
        vars_to_add       = config['vars_to_add']
        presel            = config['preselection']

                                           #Data handling stuff#
 
        #load the mc dataframe for all years
        if options.pt_reweight: 
            cr_selection = config['reweight_cr']
            output_tag += '_pt_reweighted'
            root_obj = ROOTHelpers(output_tag, mc_dir, mc_fnames, data_dir, data_fnames, proc_to_tree_name, train_vars, vars_to_add, cr_selection)
        else: root_obj = ROOTHelpers(output_tag, mc_dir, mc_fnames, data_dir, data_fnames, proc_to_tree_name, train_vars, vars_to_add, presel)

        for sig_obj in root_obj.sig_objects:
            root_obj.load_mc(sig_obj, reload_samples=options.reload_samples)
        for bkg_obj in root_obj.bkg_objects:
            root_obj.load_mc(bkg_obj, bkg=True, reload_samples=options.reload_samples)
        for data_obj in root_obj.data_objects:
            root_obj.load_data(data_obj, reload_samples=options.reload_samples)
        root_obj.concat() 



        #reweight samples in bins of pT (and maybe Njets), for each year separely. Note targetted selection
        # is applied here and all df's are resaved for smaller mem
        if options.pt_reweight and options.reload_samples:  #FIXME what about reading files in first time, wanting to pT rew, but not including options.reload samples? It wont reweight and save the reweighted df's
            root_obj.apply_pt_rew('DYMC', presel)
            #root_obj.pt_njet_reweight('DYMC', year, presel)

        
                                                #BDT stuff#

        #set up X, w and y, train-test 

        #Hyperparameters:
        #n_estimators=100
        #learning_rate=0.05
        #max_depth=4 
        #min_child_weight=0.01
        subsample=0.6
        colsample_bytree=1.0
        gamma=3

        n_estimators=[400,500]
        max_depth=[6,7]
        learning_rate = [0.05,0.07]
        min_child_weight=[0.1,1.0]
        #gamma = [0.0,1.0,2.0,3.0,4.0,5.0]
        #subsample = [0.5,0.6,0.7,0.8,0.9,1.0]
        #colsample_bytree=[0.2,0.4,0.6,0.8,1.0]

        #gamma_roc_scores = []
        #n_estimators_roc_scores = []
        #max_depth_roc_scores = []
        #subsample_roc_scores = []
        #learning_rate_roc_scores = []
        #min_child_weight_roc_scores = []
        #colsample_bytree_roc_scores = []

        param_comb = []
        roc_values = []

        for n_estimators_value in n_estimators:
            for max_depth_value in max_depth:
                for learning_rate_value in learning_rate:
                    for min_child_weight_value in min_child_weight:
                        
                        bdt_hee = BDTHelpers(root_obj, train_vars, options.train_frac, eq_train=options.eq_train,hp_n_estimators=n_estimators_value,hp_learning_rate=learning_rate_value,hp_max_depth=max_depth_value,hp_min_child_weight=min_child_weight_value,hp_subsample=subsample,hp_colsample_bytree=colsample_bytree,hp_gamma=gamma)
                        bdt_hee.create_X_and_y(mass_res_reweight=True)

                        bdt_hee.train_classifier(root_obj.mc_dir, save=True, model_name=output_tag+'_clf')
                        bdt_hee.compute_roc()
                        param_comb.append([n_estimators_value,max_depth_value,learning_rate_value,min_child_weight_value])
                        roc_values.append(bdt_hee.compute_roc())
                        print('Parameter combination',n_estimators_value,max_depth_value,learning_rate_value,min_child_weight_value)
                        print('ROC Score:',bdt_hee.compute_roc())
        
        max_value = np.max(roc_values)
        index_max_value = roc_values.index(max_value)
        best_param_comb = param_comb[index_max_value]
        print('Best parameter combination:',best_param_comb)
        print('ROC Score:',max_value)

'''
        fig, ax = plt.subplots(1)
        plt.scatter(colsample_bytree,colsample_bytree_roc_scores,label='Colsample by Tree')
        name = 'plotting/plots/HP_Optimization_colsamplebytree'
        plt.title('Hp Optimization Colsample By Tree')
        plt.legend()
        plt.xlabel('Colsample by Tree')
        plt.ylabel('ROC Values')
        fig.savefig(name)
'''

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    required_args = parser.add_argument_group('Required Arguments')
    required_args.add_argument('-c','--config', action='store', required=True)
    opt_args = parser.add_argument_group('Optional Arguements')
    opt_args.add_argument('-r','--reload_samples', action='store_true', default=False)
    opt_args.add_argument('-w','--eq_train', action='store_true', default=False)
    opt_args.add_argument('-o','--opt_hps', action='store_true', default=False)
    opt_args.add_argument('-H','--hp_perm', action='store', default=None)
    opt_args.add_argument('-k','--k_folds', action='store', default=3, type=int)
    opt_args.add_argument('-b','--train_best', action='store_true', default=False)
    opt_args.add_argument('-t','--train_frac', action='store', default=0.7, type=float)
    opt_args.add_argument('-P','--pt_reweight', action='store_true',default=False)
    options=parser.parse_args()
    main(options)
