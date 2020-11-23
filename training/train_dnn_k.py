import argparse
import numpy as np
import yaml
from HToEEML import ROOTHelpers, DNN_keras
from os import path,system

def main(options):

    #take options from the yaml config
    with open(options.config, 'r') as config_file:
        config        = yaml.load(config_file)
        mc_dir        = config['mc_file_dir']
        mc_tree_sig   = config['mc_tree_name_sig']
        mc_tree_bkg   = config['mc_tree_name_bkg']
        mc_fnames     = config['mc_file_names']
  
        #data not needed yet, but still specify in the config for compatibility with constructor
        data_dir      = config['data_file_dir']
        data_fnames   = config['data_file_names']
        data_tree     = config['data_tree_name']

        train_vars   = config['train_vars']
        vars_to_add  = config['vars_to_add']
        presel       = config['preselection']

                                           #Data handling stuff#
 
        #load the mc dataframe for all years
        data_obj = ROOTHelpers(mc_dir, mc_tree_sig, mc_tree_bkg, mc_fnames, data_dir, data_tree, data_fnames, train_vars, vars_to_add, presel)

        for year, file_name in data_obj.mc_sig_year_fnames:
            data_obj.load_mc(year, file_name, reload_data=options.reload_data)
        for year, file_name in data_obj.mc_bkg_year_fnames:
            data_obj.load_mc(year, file_name, bkg=True, reload_data=options.reload_data)
        data_obj.concat_years()

                                                #BDT stuff#
        #set up X, w and y, train-test 
        bdt_hee = DNN_keras(data_obj, train_vars, options.train_frac, options.eq_weights)
        bdt_hee.set_model_params(hidden_n=500, num_layers=5, dropout=0.4)
        bdt_hee.fit(batch_size=64, epochs=50)#, validate=False) #not validating as sample is too small!
        bdt_hee.get_predictions()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    required_args = parser.add_argument_group('Required Arguments')
    required_args.add_argument('-c','--config', action='store', required=True)
    opt_args = parser.add_argument_group('Optional Arguements')
    opt_args.add_argument('-r','--reload_data', action='store_true', default=False)
    opt_args.add_argument('-w','--eq_weights', action='store_true', default=False)
    opt_args.add_argument('-o','--opt_hps', action='store_true', default=False)
    opt_args.add_argument('-H','--hp_perm', action='store', default=None)
    opt_args.add_argument('-k','--k_folds', action='store', default=3, type=int)
    opt_args.add_argument('-b','--train_best', action='store_true', default=False)
    opt_args.add_argument('-t','--train_frac', action='store', default=0.7, type=float)
    options=parser.parse_args()
    main(options)
