import argparse
import numpy as np
import yaml
from HToEEML import ROOTHelpers, LSTM_DNN
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

        object_vars   = config['object_vars']
        event_vars    = config['event_vars']
        vars_to_add   = config['vars_to_add']
        presel        = config['preselection']

                                           #Data handling stuff#
 
        #load the mc dataframe for all years
        data_obj = ROOTHelpers(mc_dir, mc_tree_sig, mc_tree_bkg, mc_fnames, data_dir, data_tree, data_fnames, object_vars+event_vars, vars_to_add, presel)

        for year, file_name in data_obj.mc_sig_year_fnames:
            data_obj.load_mc(year, file_name, reload_data=options.reload_data)
        for year, file_name in data_obj.mc_bkg_year_fnames:
            data_obj.load_mc(year, file_name, bkg=True, reload_data=options.reload_data)
        data_obj.concat_years()

                                                #LSTM stuff#
        lstm = LSTM_DNN(data_obj, object_vars, event_vars, options.train_frac, options.eq_weights)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    required_args = parser.add_argument_group('Required Arguments')
    required_args.add_argument('-c','--config', help='configuration file (yaml) for training', action='store', required=True)
    opt_args = parser.add_argument_group('Optional Arguements')
    opt_args.add_argument('-r','--reload_data', help='re-load the .root files and convert into pandas DataFrames', action='store_true', default=False)
    opt_args.add_argument('-w','--eq_weights', help='equalise the sum weights between signala nd background classes', action='store_true', default=False)
    opt_args.add_argument('-t','--train_frac', help='fraction of events used for training. 1-test_frac used for testing', action='store', default=0.7, type=float)
    opt_args.add_argument('--no_lstm', help='dont use object-level features (LSTM layers)', action='store_true', default=False)
    opt_args.add_argument('--no_global', help='dont use event-level features', action='store_true', default=False)

    options=parser.parse_args()
    main(options)
