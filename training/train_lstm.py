import argparse
import numpy as np
import yaml
from HToEEML import ROOTHelpers, LSTM_DNN
from os import path,system

def main(options):

    #take options from the yaml config
    with open(options.config, 'r') as config_file:
        config        = yaml.load(config_file)
        output_tag        = config['output_tag']
 
        mc_dir            = config['mc_file_dir']
        mc_fnames         = config['mc_file_names']
  
        #data not needed yet, but stil specify in the config for compatibility with constructor
        data_dir          = config['data_file_dir']
        data_fnames       = config['data_file_names']
 
        proc_to_tree_name = config['proc_to_tree_name']

        object_vars   = config['object_vars']
        flat_obj_vars = [var for i_object in object_vars for var in i_object]
        event_vars    = config['event_vars']
        vars_to_add   = config['vars_to_add']
        presel        = config['preselection']

                                           #Data handling stuff#
 
        #load the mc dataframe for all years
        root_obj = ROOTHelpers(output_tag, mc_dir, mc_fnames, data_dir, data_fnames, proc_to_tree_name, flat_obj_vars+event_vars, vars_to_add, presel)

        for sig_obj in root_obj.sig_objects:
            root_obj.load_mc(sig_obj, reload_samples=options.reload_samples)
        for bkg_obj in root_obj.bkg_objects:
            root_obj.load_mc(bkg_obj, bkg=True, reload_samples=options.reload_samples)
        root_obj.concat()

                                                #LSTM stuff#
        lstm = LSTM_DNN(root_obj, object_vars, event_vars, options.train_frac, options.eq_weights)
        lstm.set_model()
        lstm.join_objects()
        lstm.train_network()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    required_args = parser.add_argument_group('Required Arguments')
    required_args.add_argument('-c','--config', help='configuration file (yaml) for training', action='store', required=True)
    opt_args = parser.add_argument_group('Optional Arguements')
    opt_args.add_argument('-r','--reload_samples', help='re-load the .root files and convert into pandas DataFrames', action='store_true', default=False)
    opt_args.add_argument('-w','--eq_weights', help='equalise the sum weights between signala nd background classes', action='store_true', default=False)
    opt_args.add_argument('-t','--train_frac', help='fraction of events used for training. 1-test_frac used for testing', action='store', default=0.7, type=float)
    opt_args.add_argument('--no_lstm', help='dont use object-level features (LSTM layers)', action='store_true', default=False)
    opt_args.add_argument('--no_global', help='dont use event-level features', action='store_true', default=False)

    options=parser.parse_args()
    main(options)
