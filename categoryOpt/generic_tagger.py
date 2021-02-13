import argparse
import numpy as np
import yaml
import pandas as pd
from os import path, system

from DataHandling import ROOTHelpers
from tag_seq_base import taggerBase

def main(options):

    
    with open(options.config, 'r') as config_file:
        config             = yaml.load(config_file)
        output_tag         = config['output_tag']

        mc_dir             = config['mc_file_dir']
        mc_fnames          = config['mc_file_names']
  
        #data not needed yet, but stil specify in the config for compatibility with constructor
        data_dir           = config['data_file_dir']
        data_fnames        = config['data_file_names']

        proc_to_tree_name  = config['proc_to_tree_name']

        proc_to_train_vars = config['train_vars']
        all_train_vars     = [item for sublist in proc_to_train_vars.values() for item in sublist]
        vars_to_add        = config['vars_to_add']

        if options.syst_name is not None: 
            syst = options.syst_name
            read_syst=True
        else: read_syst=False

                                           #Data handling stuff#
        #apply loosest selection (ggh) first, else memory requirements are ridiculous. Fine to do this since all cuts all looser than VBF (not removing events with higher priority)
        #also note we norm the MC before applying this cut. In data we apply it when reading in.
        loosest_selection = 'dielectronMass > 110 and dielectronMass < 150 and leadElectronPtOvM > 0.333 and subleadElectronPtOvM > 0.25' 
 
        #load the mc dataframe for all years. Do not apply any specific preselection to sim samples
        root_obj = ROOTHelpers(output_tag, mc_dir, mc_fnames, data_dir, data_fnames, proc_to_tree_name, all_train_vars, vars_to_add, loosest_selection, read_systs=read_syst) 
        root_obj.no_lumi_scale()
        for sig_obj in root_obj.sig_objects:
            root_obj.load_mc(sig_obj, reload_samples=options.reload_samples)
        if not read_syst:
            if options.data_as_bkg:
                for data_obj in root_obj.data_objects:
                    root_obj.load_data(data_obj, reload_samples=options.reload_samples)
            else:
                for bkg_obj in root_obj.bkg_objects:
                    root_obj.load_mc(bkg_obj, bkg=True, reload_samples=options.reload_samples)
        root_obj.concat()

    if read_syst: combined_df = root_obj.mc_df_sig
    elif options.data_as_bkg: combined_df = pd.concat([root_obj.mc_df_sig, root_obj.data_df])
    else: combined_df = pd.concat([root_obj.mc_df_sig, root_obj.mc_df_bkg])

    del root_obj

                                       #Tag sequence stuff#
    #specify sequence of tags and preselection targetting each

    tag_sequence      = ['VBF','ggH']
    true_procs        = ['VBF','ggH']
    if not read_syst: true_procs.append('Data') 

    tag_preselection  = {'VBF': [combined_df['dielectronMass'].gt(110) & 
                                 combined_df['dielectronMass'].lt(150) &
                                 combined_df['leadElectronPtOvM'].gt(0.333) &
                                 combined_df['subleadElectronPtOvM'].gt(0.25) &
                                 combined_df['dijetMass'].gt(350) &
                                 combined_df['leadJetPt'].gt(40) &
                                 combined_df['subleadJetPt'].gt(30)
                                ],
                         'ggH': [combined_df['dielectronMass'].gt(110) & 
                                 combined_df['dielectronMass'].lt(150) &
                                 combined_df['leadElectronPtOvM'].gt(0.333) &
                                 combined_df['subleadElectronPtOvM'].gt(0.25)
                                ]       
                        }

    #create tag object 
    tag_obj = taggerBase(tag_sequence, true_procs, combined_df, syst_name=options.syst_name)
    if read_syst: tag_obj.relabel_syst_vars()

    #get number models and tag boundaries from config
    with open(options.mva_config, 'r') as mva_config_file:
        config            = yaml.load(mva_config_file)
        proc_to_model     = config['models']
        tag_boundaries    = config['boundaries']

        #evaluate MVA scores used in categorisation
        for proc, model in proc_to_model.iteritems():
            if 'BDT' in model: tag_obj.eval_bdt(proc, model, proc_to_train_vars[proc])
            elif 'NN' in model: tag_obj.eval_dnn(proc, model, proc_to_train_vars[proc])
            else: raise IOError('Did not get a classifier with BDT or DNN in model name!')

    #set up tag boundaries for each process being targeted
    tag_obj.decide_tag(tag_preselection, tag_boundaries)
    tag_obj.decide_priority()
    branch_names = tag_obj.get_tree_names(tag_boundaries)
    tag_obj.set_tree_names(tag_boundaries)
    tag_obj.fill_trees(branch_names)
    if not read_syst: tag_obj.plot_matrix(branch_names, output_tag) 
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    required_args = parser.add_argument_group('Required Arguments')
    required_args.add_argument('-c','--config', action='store', required=True)
    required_args.add_argument('-M','--mva_config', action='store', required=True)
    opt_args = parser.add_argument_group('Optional Arguements')
    required_args.add_argument('-S','--syst_name', action='store', default=None)
    opt_args.add_argument('-r','--reload_samples', help='re-load the .root files and convert into pandas DataFrames', action='store_true', default=False)
    opt_args.add_argument('-d','--data_as_bkg', action='store_true', default=False)
    options=parser.parse_args()
    main(options)
