import argparse
import numpy as np
import yaml
from HToEEML import ROOTHelpers, Plotter

def main(options):

    #take options from the yaml config
    with open(options.config, 'r') as config_file:
        config            = yaml.load(config_file)
        output_tag        = config['output_tag']

        mc_dir            = config['mc_file_dir']
        mc_fnames         = config['mc_file_names']
  
        #data not needed yet, could use this for validation later. keep for compat with class
        data_dir          = config['data_file_dir']
        data_fnames       = config['data_file_names']

        train_vars        = config['train_vars']
        vars_to_add       = config['vars_to_add']
        presel            = config['preselection']

        proc_to_tree_name = config['proc_to_tree_name']

        #sig_colour        = 'forestgreen'
        sig_colour        = 'red'
        bkg_colour        = 'violet'
 
                                           #Data handling stuff#

        #load the mc dataframe for all years
        root_obj = ROOTHelpers(output_tag, mc_dir, mc_fnames, data_dir, data_fnames, proc_to_tree_name, train_vars, vars_to_add, presel)

        for sig_obj in root_obj.sig_objects:
            root_obj.load_mc(sig_obj, reload_samples=options.reload_samples)
        for bkg_obj in root_obj.bkg_objects:
            root_obj.load_mc(bkg_obj, bkg=True, reload_samples=options.reload_samples)
        for data_obj in root_obj.data_objects:
            root_obj.load_data(data_obj, reload_samples=options.reload_samples)
        root_obj.concat()

                                            #Plotter stuff#
 
        #set up X, w and y, train-test 
        plotter = Plotter(root_obj, train_vars, sig_col=sig_colour)
        for var in train_vars:
            plotter.plot_input(var, options.n_bins, output_tag)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    required_args = parser.add_argument_group('Required Arguments')
    required_args.add_argument('-c','--config', action='store', required=True)
    opt_args = parser.add_argument_group('Optional Arguements')
    opt_args.add_argument('-r','--reload_samples', action='store_true', default=False)
    opt_args.add_argument('-b','--n_bins',  default=26, type=int)
    options=parser.parse_args()
    main(options)
