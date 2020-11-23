import argparse
import numpy as np
import yaml
from HToEEML import ROOTHelpers, Plotter

def main(options):

    #take options from the yaml config
    with open(options.config, 'r') as config_file:
        config        = yaml.load(config_file)
        mc_dir        = config['mc_file_dir']
        mc_tree_sig   = config['mc_tree_name_sig']
        mc_tree_bkg   = config['mc_tree_name_bkg']
        mc_fnames     = config['mc_file_names']
  
        #NOTE:
        #data not needed yet, could use this for validation later. Hence no loading right now 
        #stil specify in the config as format in general
        data_dir      = config['data_file_dir']
        data_fnames   = config['data_file_names']
        data_tree     = config['data_tree_name']

        train_vars   = config['train_vars']
        vars_to_add  = config['vars_to_add']
        presel       = config['preselection']

        sig_colour   = 'firebrick'
        sig_label    = 'VBF'
        bkg_colour   = 'violet'
        bkg_label    = 'DYMC'
 
                                           #Data handling stuff#
 
        #load the mc dataframe for all years
        data_obj = ROOTHelpers(mc_dir, mc_tree_sig, mc_tree_bkg, mc_fnames, data_dir, data_tree, data_fnames, train_vars, vars_to_add, presel)

        for year, file_name in data_obj.mc_sig_year_fnames:
            data_obj.load_mc(str(year), file_name, reload_data=options.reload_data)
        for year, file_name in data_obj.mc_bkg_year_fnames:
            data_obj.load_mc(str(year), file_name, bkg=True, reload_data=options.reload_data)
        #for year, file_name in data_obj.data_year_fnames:
        #    HToEEBDT.load_data(year, file_name, bkg=True)
        data_obj.concat_years()

                                                #BDT stuff#

        #set up X, w and y, train-test 
        plotter = Plotter(data_obj, train_vars, sig_colour, sig_label, bkg_colour, bkg_label)
        for var in train_vars:
            plotter.plot_input(var)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    required_args = parser.add_argument_group('Required Arguments')
    required_args.add_argument('-c','--config', action='store', required=True)
    opt_args = parser.add_argument_group('Optional Arguements')
    opt_args.add_argument('-r','--reload_data', action='store_true', default=False)
    options=parser.parse_args()
    main(options)
