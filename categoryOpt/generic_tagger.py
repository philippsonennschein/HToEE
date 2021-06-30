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

        #check if dnn (lstm) variables need to be read in 
        proc_to_train_vars = config['train_vars']
        all_train_vars     = []
        for proc, varrs in proc_to_train_vars.iteritems():
            if isinstance(varrs, dict):
                object_vars     = proc_to_train_vars[proc]['object_vars']
                flat_obj_vars   = [var for i_object in object_vars for var in i_object]
                event_vars      = proc_to_train_vars[proc]['event_vars']
                all_train_vars += (flat_obj_vars + event_vars)
            else: 
                all_train_vars += varrs

        vars_to_add        = config['vars_to_add']

        if options.syst_name is not None: 
            syst = options.syst_name
            read_syst=True
        else: read_syst=False

        if read_syst and options.dump_weight_systs: raise IOError('Cannot dump weight variations and tree systematics at the same time. Please run separately for each.')
        if options.data_only and (read_syst or options.dump_weight_systs): raise IOError('Cannot read Data and apply sysetmatic shifts')

                                           #Data handling stuff#
        #apply loosest selection (ggh) first, else memory requirements are ridiculous. Fine to do this since all cuts all looser than VBF (not removing events with higher priority)
        #also note we norm the MC before applying this cut. In data we apply it when reading in.
        #loosest_selection = 'dielectronMass > 110 and dielectronMass < 150 and leadElectronPtOvM > 0.333 and subleadElectronPtOvM > 0.25' cant do this since these vars change with systematics!
        loosest_selection = 'dielectronMass > 100' 
 
        #load the mc dataframe for all years. Do not apply any specific preselection to sim samples
        root_obj = ROOTHelpers(output_tag, mc_dir, mc_fnames, data_dir, data_fnames, proc_to_tree_name, all_train_vars, vars_to_add, loosest_selection, read_systs=(read_syst or options.dump_weight_systs)) 
        root_obj.no_lumi_scale()
        for sig_obj in root_obj.sig_objects:
            root_obj.load_mc(sig_obj, reload_samples=options.reload_samples)
        #if not read_syst:
        if not options.data_as_bkg:
            for bkg_obj in root_obj.bkg_objects:
                root_obj.load_mc(bkg_obj, bkg=True, reload_samples=options.reload_samples)
        else:
            for data_obj in root_obj.data_objects:
                root_obj.load_data(data_obj, reload_samples=options.reload_samples)
            #overwrite background attribute, for compat with DNN class
            root_obj.mc_df_bkg = root_obj.data_df
        root_obj.concat()

    #get year of samples for roob obj and check we didn't accidentally read in more than 1 year
    if len(root_obj.years)!=1: raise IOError('Reading in more than one year at a time! Tagging should be split by year')
    else: year = list(root_obj.years)[0]

    #if read_syst: combined_df = root_obj.mc_df_sig doesnt work with DNN set up since need bkg class in _init_ 
    #else: combined_df = pd.concat([root_obj.mc_df_sig, root_obj.mc_df_bkg])
    combined_df = pd.concat([root_obj.mc_df_sig, root_obj.mc_df_bkg])


                                       #Tag sequence stuff#
    #specify sequence of tags and preselection targetting each

    tag_sequence      = ['VBF','ggH']
    true_procs        = ['VBF','ggH']
    if (not read_syst) and (not options.dump_weight_systs) : true_procs.append('Data') 
    if options.data_only: true_procs = ['Data']

    #create tag object 
    tag_obj = taggerBase(tag_sequence, true_procs, combined_df, syst_name=options.syst_name)
    if read_syst: tag_obj.relabel_syst_vars() #not run if reading weight systematics


    #get number models and tag boundaries from config
    with open(options.mva_config, 'r') as mva_config_file:
        config            = yaml.load(mva_config_file)
        proc_to_model     = config['models']
        tag_boundaries    = config['boundaries']

        #evaluate MVA scores used in categorisation
        for proc, model in proc_to_model.iteritems():
            #for BDT - proc:[var list]. For DNN - proc:{var_type1:[var_list_type1], var_type2: [...], ...}
            if isinstance(model,dict):
                object_vars     = proc_to_train_vars[proc]['object_vars']
                flat_obj_vars   = [var for i_object in object_vars for var in i_object]
                event_vars      = proc_to_train_vars[proc]['event_vars']

                dnn_loaded = tag_obj.load_dnn(proc, model)
                train_tag = model['architecture'].split('_model')[0]
                tag_obj.eval_lstm(dnn_loaded, train_tag, root_obj, proc, object_vars, flat_obj_vars, event_vars)

            elif isinstance(model,str): tag_obj.eval_bdt(proc, model, proc_to_train_vars[proc])
            else: raise IOError('Did not get a classifier models in correct format in config')

    del root_obj

    #need to do this after eval MVAs, since LSTM class used in eval_lstm needs some Data in df for constructor
    if (read_syst or options.dump_weight_systs): 
        tag_obj.combined_df = tag_obj.combined_df[tag_obj.combined_df.proc!='Data'].copy() #avoid copy warnings later
    tag_preselection = tag_obj.get_tag_preselection()

    #set up tag boundaries for each process being targeted
    tag_obj.decide_tag(tag_preselection, tag_boundaries)
    tag_obj.decide_priority()
    branch_names = tag_obj.get_tree_names(tag_boundaries, year)
    tag_obj.set_tree_names(tag_boundaries,options.dump_weight_systs,year)
    tag_obj.fill_trees(branch_names, year, print_yields=not read_syst)
    if not read_syst: 
        pass #tag_obj.plot_matrix(branch_names, output_tag)  #struct error?
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    required_args = parser.add_argument_group('Required Arguments')
    required_args.add_argument('-c','--config', action='store', required=True)
    required_args.add_argument('-M','--mva_config', action='store', required=True)
    opt_args = parser.add_argument_group('Optional Arguements')
    required_args.add_argument('-S','--syst_name', action='store', default=None)
    opt_args.add_argument('-r','--reload_samples', help='re-load the .root files and convert into pandas DataFrames', action='store_true', default=False)
    opt_args.add_argument('-d','--data_as_bkg', action='store_true', default=False)
    opt_args.add_argument('-D','--data_only', action='store_true', default=False)
    opt_args.add_argument('-W','--dump_weight_systs', help='Dump all weight variations in the nominal output trees e.g. effect of pre-firing', action='store_true', default=False)
    options=parser.parse_args()
    main(options)
