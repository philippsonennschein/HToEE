import argparse
import numpy as np
import yaml
import sys
from os import path,system

from DataHandling import ROOTHelpers
from BDTs import BDTHelpers

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
        bdt_hee = BDTHelpers(root_obj, train_vars, options.train_frac, eq_train=options.eq_train)
        bdt_hee.create_X_and_y(mass_res_reweight=True)
        #df_Z_tot, df_y_test, df_y_train = bdt_hee.create_X_and_y(mass_res_reweight=True)

        #submit the HP search if option true
        if options.hp_perm is not None:
            if options.opt_hps and options.train_best:
                raise Exception('Cannot optimise HPs and train best model. Run optimal training after hyper paramter optimisation')
            elif options.opt_hps and options.hp_perm:
                raise Exception('opt_hps option submits scripts with the hp_perm option; Cannot submit a script with both!')
            else: 
                print( 'About to train + validate on dataset with {} fold splitting'.format(options.k_folds))
                bdt_hee.set_hyper_parameters(options.hp_perm)
                bdt_hee.set_k_folds(options.k_folds)
                for i_fold in range(options.k_folds):
                    bdt_hee.set_i_fold(i_fold)
                    bdt_hee.train_classifier(root_obj.mc_dir, save=False)
                    bdt_hee.validation_rocs.append(bdt_hee.compute_roc())
                with open('{}/bdt_hp_opt_{}.txt'.format(mc_dir, output_tag),'a+') as val_roc_file:
                    bdt_hee.compare_rocs(val_roc_file, options.hp_perm)
                    val_roc_file.close()
           
        elif options.opt_hps:
            #FIXME: add warning that many jobs are about to be submiited
            if options.k_folds<2: raise ValueError('K-folds option must be at least 2')
            if path.isfile('{}/bdt_hp_opt_{}.txt'.format(mc_dir, output_tag)): 
                system('rm {}/bdt_hp_opt_{}.txt'.format(mc_dir, output_tag))
                print ('deleting: {}/bdt_hp_opt_{}.txt'.format(mc_dir, output_tag))
            bdt_hee.batch_gs_cv(k_folds=options.k_folds, pt_rew=options.pt_reweight)

        elif options.train_best:
            output_tag+='_best'
            with open('{}/bdt_hp_opt_{}.txt'.format(mc_dir, output_tag),'r') as val_roc_file:
                hp_roc = val_roc_file.readlines()
                best_params = hp_roc[-1].split(';')[0]
                print ('Best classifier params are: {}'.format(best_params))
                bdt_hee.set_hyper_parameters(best_params)
                bdt_hee.train_classifier(root_obj.mc_dir, save=True, model_name=output_tag)
                bdt_hee.compute_roc()
                bdt_hee.plot_roc(output_tag)
                bdt_hee.plot_output_score(output_tag, ratio_plot=True, norm_to_data=(not options.pt_reweight))

        #else just train BDT with default HPs
        else:
            bdt_hee.train_classifier(root_obj.mc_dir, save=True, model_name=output_tag+'_clf')
            bdt_hee.compute_roc()
            bdt_hee.plot_roc(output_tag)
            bdt_hee.plot_output_score(output_tag, ratio_plot=True, norm_to_data=(not options.pt_reweight), log=False)
            #bdt_hee.plot_feature_importance(num_plots='single',num_feature=20,imp_type='gain',values = False)
            #bdt_hee.plot_feature_importance(num_plots='all',num_feature=20,values = False)

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
