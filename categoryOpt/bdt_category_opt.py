import argparse
import numpy as np
import yaml
import pickle
from HToEEML import ROOTHelpers, BDTHelpers
from catOptim import CatOptim

def main(options):

    #take options from the yaml config
    with open(options.config, 'r') as config_file:
        config        = yaml.load(config_file)
        mc_dir        = config['mc_file_dir']
        mc_tree_sig   = config['mc_tree_name_sig']
        mc_tree_bkg   = config['mc_tree_name_bkg']
        mc_fnames     = config['mc_file_names']
        proc_tag      = config['signal_process']
  
        data_dir      = config['data_file_dir']
        data_fnames   = config['data_file_names']
        data_tree     = config['data_tree_name']

        train_vars   = config['train_vars']
        vars_to_add  = config['vars_to_add']
        presel       = config['preselection']

        #load the mc dataframe for all years
        data_obj = ROOTHelpers(proc_tag, mc_dir, mc_tree_sig, mc_tree_bkg, mc_fnames, data_dir, data_tree, data_fnames, train_vars, vars_to_add, presel)

        for year, file_name in data_obj.mc_sig_year_fnames:
            data_obj.load_mc(year, file_name, reload_samples=options.reload_samples)
        for year, file_name in data_obj.mc_bkg_year_fnames:
            data_obj.load_mc(year, file_name, bkg=True, reload_samples=options.reload_samples)
        for year, file_name in data_obj.data_year_fnames:
            data_obj.load_data(year, file_name, reload_samples=options.reload_samples)
        data_obj.concat_years()

        print 'loading classifier: {}'.format(options.model)
        clf = pickle.load(open("{}".format(options.model), "rb"))

        #apply cut-based selection if not optimising BDT score (pred probs still evaluated for compatability w exisiting constructor). 
        if len(options.cut_based_str)>0:
            data_obj.apply_more_cuts(options.cut_based_str)

        sig_weights   = data_obj.mc_df_sig['weight'].values
        sig_m_ee      = data_obj.mc_df_sig['dipho_mass'].values
        pred_prob_sig = clf.predict_proba(data_obj.mc_df_sig[train_vars].values)[:,1:].ravel()

        if options.data_as_bkg: 
            bkg_weights   = data_obj.data_df['weight'].values
            bkg_m_ee      = data_obj.data_df['dipho_mass'].values
            pred_prob_bkg = clf.predict_proba(data_obj.data_df[train_vars].values)[:,1:].ravel()

        else: 
            bkg_weights   = data_obj.mc_df_bkg['weight'].values
            bkg_m_ee      = data_obj.mc_df_bkg['dipho_mass'].values
            pred_prob_bkg = clf.predict_proba(data_obj.mc_df_bkg[train_vars].values)[:,1:].ravel()

        #set up optimiser ranges and no. categories to test if non-cut based
        ranges    = [ [0.3,1.] ]
        names     = ['{} BDT score'.format(proc_tag)] #arbitrary
        print_str = ''
        cats = [1,2,3,4]

        #just to use class methods here
        if len(options.cut_based_str)>0:
            optimiser = CatOptim(sig_weights, sig_m_ee, [pred_prob_sig], bkg_weights, bkg_m_ee, [pred_prob_bkg], 0, ranges, names)
            AMS = optimiser.cutBasedAMS()
            print 'String for cut based optimimastion: {}'.format(options.cut_based_str)
            print 'Cut-based optimimsation gives AMS = {:1.8f}'.format(AMS)

        else:
            for n_cats in cats:
                optimiser = CatOptim(sig_weights, sig_m_ee, [pred_prob_sig], bkg_weights, bkg_m_ee, [pred_prob_bkg], n_cats, ranges, names)
                optimiser.optimise(1, options.n_iters) #set lumi to 1 as already scaled when loading in
                print_str += 'Results for {} categories : \n'.format(n_cats)
                print_str += optimiser.getPrintableResult()
            print '\n {}'.format(print_str)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    required_args = parser.add_argument_group('Required Arguments')
    required_args.add_argument('-c','--config', action='store', required=True)
    required_args.add_argument('-m','--model', action='store', required=True)
    opt_args = parser.add_argument_group('Optional Arguements')
    opt_args.add_argument('-r','--reload_samples', action='store_true', default=False)
    opt_args.add_argument('-i','--n_iters', action='store', default=3000, type=int)
    opt_args.add_argument('-d','--data_as_bkg', action='store_true', default=False)
    opt_args.add_argument('-k','--cut_based_str', action='store',type=str, default='')
    options=parser.parse_args()
    main(options)
