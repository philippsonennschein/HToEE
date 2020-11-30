import argparse
import numpy as np
import yaml
import sys
from HToEEML import ROOTHelpers, BDTHelpers
from os import path,system

def main(options):

    #take options from the yaml config
    with open(options.config, 'r') as config_file:
        config        = yaml.load(config_file)
        output_tag    = config['signal_process']

        mc_dir        = config['mc_file_dir']
        mc_fnames     = config['mc_file_names']
  
        #data not needed yet, but stil specify in the config for compatibility with constructor
        data_dir      = config['data_file_dir']
        data_fnames   = config['data_file_names']

        proc_to_tree_name = config['proc_to_tree_name']

        train_vars   = config['train_vars']
        vars_to_add  = config['vars_to_add']
        presel       = config['preselection']

                                           #Data handling stuff#
 
        #load the mc dataframe for all years
        root_obj = ROOTHelpers(output_tag, mc_dir, mc_fnames, data_dir, data_fnames, proc_to_tree_name, train_vars, vars_to_add, presel)

        for sig_obj in root_obj.sig_objects:
            root_obj.load_mc(sig_obj, reload_samples=options.reload_samples)
        for bkg_obj in root_obj.bkg_objects:
            root_obj.load_mc(bkg_obj, bkg=True, reload_samples=options.reload_samples)
        root_obj.concat() #automatically conacat years if more than one detected
  

                                                #BDT stuff#

        #set up X, w and y, train-test 
        bdt_hee = BDTHelpers(root_obj, train_vars, options.train_frac, options.eq_weights)

        #submit the HP search if option true
        if options.hp_perm is not None:
            if options.opt_hps and options.train_best:
                raise Exception('Cannot optimise HPs and train best model. Run optimal training after hyper paramter optimisation')
            elif options.opt_hps and options.hp_perm:
                raise Exception('opt_hps option submits scripts with the hp_perm option; Cannot submit a script with both!')
            else: 
                print 'About to train + validate on dataset with {} fold splitting'.format(options.k_folds)
                bdt_hee.set_hyper_parameters(options.hp_perm)
                bdt_hee.set_k_folds(options.k_folds)
                for i_fold in range(options.k_folds):
                    bdt_hee.set_i_fold(i_fold)
                    bdt_hee.train_classifier(root_obj.mc_dir, save=False)
                    bdt_hee.validation_rocs.append(bdt_hee.compute_roc())
                with open('{}/bdt_hp_opt.txt'.format(mc_dir),'a+') as val_roc_file:
                    bdt_hee.compare_rocs(val_roc_file, options.hp_perm)
                    val_roc_file.close()
           
        elif options.opt_hps:
            #FIXME: add warning that many jobs are about to be submiited
            if options.k_folds<2: raise ValueError('K-folds option must be at least 2')
            if path.isfile('{}/bdt_hp_opt.txt'.format(mc_dir)): 
                system('rm {}/bdt_hp_opt.txt'.format(mc_dir))
                print ('deleting: {}/bdt_hp_opt.txt'.format(mc_dir))
            bdt_hee.batch_gs_cv(k_folds=3)

        elif options.train_best:
            output_tag+='_best'
            with open('{}/bdt_hp_opt.txt'.format(mc_dir),'r') as val_roc_file:
                hp_roc = val_roc_file.readlines()
                best_params = hp_roc[-1].split(';')[0]
                print 'Best classifier params are: {}'.format(best_params)
                bdt_hee.set_hyper_parameters(best_params)
                bdt_hee.train_classifier(root_obj.mc_dir, save=True, model_name=output_tag)
                bdt_hee.compute_roc()
                bdt_hee.plot_roc(output_tag)
                bdt_hee.plot_output_score(output_tag)

        #else just train BDT with default HPs
        else:
            bdt_hee.train_classifier(root_obj.mc_dir, save=True, model_name=output_tag+'_clf')
            bdt_hee.compute_roc()
            bdt_hee.plot_roc(output_tag)
            bdt_hee.plot_output_score(output_tag)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    required_args = parser.add_argument_group('Required Arguments')
    required_args.add_argument('-c','--config', action='store', required=True)
    opt_args = parser.add_argument_group('Optional Arguements')
    opt_args.add_argument('-r','--reload_samples', action='store_true', default=False)
    opt_args.add_argument('-w','--eq_weights', action='store_true', default=False)
    opt_args.add_argument('-o','--opt_hps', action='store_true', default=False)
    opt_args.add_argument('-H','--hp_perm', action='store', default=None)
    opt_args.add_argument('-k','--k_folds', action='store', default=3, type=int)
    opt_args.add_argument('-b','--train_best', action='store_true', default=False)
    opt_args.add_argument('-t','--train_frac', action='store', default=0.7, type=float)
    options=parser.parse_args()
    main(options)
