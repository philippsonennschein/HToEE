import argparse
import numpy as np
import yaml
import pickle
from HToEEML import ROOTHelpers, BDTHelpers

def main(options):
    
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
        if options.data_as_bkg:
            for data_obj in root_obj.data_objects:
                root_obj.load_data(data_obj, reload_samples=options.reload_samples)
        else:
            for bkg_obj in root_obj.bkg_objects:
                root_obj.load_mc(bkg_obj, bkg=True, reload_samples=options.reload_samples)
        root_obj.concat()

                                           #Tag sequence stuff#
        
       

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    required_args = parser.add_argument_group('Required Arguments')
    required_args.add_argument('-c','--config', action='store', required=True)
    opt_args = parser.add_argument_group('Optional Arguements')
    opt_args.add_argument('-r','--reload_samples', action='store_true', default=False)
    opt_args.add_argument('-d','--data_as_bkg', action='store_true', default=False)
    options=parser.parse_args()
main(options)
