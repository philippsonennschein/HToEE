import argparse
import yaml
from HToEEML import ROOTHelpers, BDT

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

        #ROC without centrality was: 82.23%
        #ROC with centrality is: 83.03%
        train_vars   = config['train_vars']
        vars_to_add  = config['vars_to_add']
        presel       = config['preselection']


        #put this into a function later
        
        #load the mc dataframe for all years
        data_obj = ROOTHelpers(mc_dir, mc_tree_sig, mc_tree_bkg, mc_fnames, data_dir, data_tree, data_fnames, train_vars, vars_to_add, presel)

        for year, file_name in data_obj.mc_sig_year_fnames:
            data_obj.load_mc(str(year), file_name)
        for year, file_name in data_obj.mc_bkg_year_fnames:
            data_obj.load_mc(str(year), file_name, bkg=True)
        #for year, file_name in data_obj.data_year_fnames:
        #    HToEEBDT.load_data(year, file_name, bkg=True)
        data_obj.concat_years()


        #train bdt, optimise HPS and save best model
        bdt = BDT(data_obj.mc_df_sig, data_obj.mc_df_bkg, train_vars, train_frac=0.7)
        bdt.train_classifier(data_obj.mc_dir, save=True)
        bdt.compute_roc()

        #evaluate model using roc


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    required_args = parser.add_argument_group('Required Arguements')
    required_args.add_argument('-c','--config', action='store', required=True)
    required_args.add_argument('-y','--concat_years', action='store', required=True)
    options=parser.parse_args()
    main(options)
