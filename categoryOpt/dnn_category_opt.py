import argparse
import numpy as np
import pandas as pd
import yaml
#import pickle
import keras

from catOptim import CatOptim
from DataHandling import ROOTHelpers
from PlottingUtils import Plotter
from NeuralNets import LSTM_DNN

def main(options):

    #take options from the yaml config
    with open(options.config, 'r') as config_file:
        config            = yaml.load(config_file)
        output_tag        = config['output_tag']

        mc_dir            = config['mc_file_dir']
        mc_fnames         = config['mc_file_names']
  
        data_dir          = config['data_file_dir']
        data_fnames       = config['data_file_names']

        proc_to_tree_name = config['proc_to_tree_name']       

        object_vars       = config['object_vars']
        flat_obj_vars     = [var for i_object in object_vars for var in i_object]
        event_vars        = config['event_vars']
        vars_to_add       = config['vars_to_add']
        presel            = config['preselection']
                 
                                           #Data handling stuff#
                 
        #load the mc dataframe for all years
        root_obj = ROOTHelpers(output_tag, mc_dir, mc_fnames, data_dir, data_fnames, proc_to_tree_name, flat_obj_vars+event_vars, vars_to_add, presel) 

        for sig_obj in root_obj.sig_objects:
            root_obj.load_mc(sig_obj, reload_samples=options.reload_samples)
        if not options.data_as_bkg:
            for bkg_obj in root_obj.bkg_objects:
                root_obj.load_mc(bkg_obj, bkg=True, reload_samples=options.reload_samples)
        else:
            for data_obj in root_obj.data_objects:
                root_obj.load_data(data_obj, reload_samples=options.reload_samples)
            #overwrite background attribute, for compat with DNN class
            root_obj.mc_df_bkg = root_obj.data_df
        root_obj.concat()

        #apply cut-based selection if not optimising BDT score (pred probs still evaluated for compatability w exisiting catOpt constructor). 
        if len(options.cut_based_str)>0:
            root_obj.apply_more_cuts(options.cut_based_str)

                                           # DNN evaluation stuff #

        #load architecture and model weights
        print 'loading DNN: {}'.format(options.model_architecture)
        with open('{}'.format(options.model_architecture), 'r') as model_json:
            model_architecture = model_json.read()
        model = keras.models.model_from_json(model_architecture)
        model.load_weights('{}'.format(options.model))

        LSTM = LSTM_DNN(root_obj, object_vars, event_vars, 1.0, False, True)

        # set up X and y Matrices. Log variables that have GeV units
        LSTM.var_transform(do_data=False) #bkg=data here. This option is for plotting purposes
        X_tot, y_tot     = LSTM.create_X_y()
        X_tot            = X_tot[flat_obj_vars+event_vars] #filter unused vars

        #scale X_vars to mean=0 and std=1. Use scaler fit during previous dnn training
        LSTM.load_X_scaler(out_tag=output_tag)
        X_tot            = LSTM.X_scaler.transform(X_tot)

        #make 2D vars for LSTM layers
        X_tot            = pd.DataFrame(X_tot, columns=flat_obj_vars+event_vars)
        X_tot_high_level = X_tot[event_vars].values
        X_tot_low_level  = LSTM.join_objects(X_tot[flat_obj_vars])

        #predict probs
        pred_prob_tot    = model.predict([X_tot_high_level, X_tot_low_level], batch_size=1024).flatten()

        sig_weights   = root_obj.mc_df_sig['weight'].values
        sig_m_ee      = root_obj.mc_df_sig['dielectronMass'].values
        pred_prob_sig = pred_prob_tot[y_tot==1] 

        bkg_weights   = root_obj.data_df['weight'].values
        bkg_m_ee      = root_obj.data_df['dielectronMass'].values
        pred_prob_bkg = pred_prob_tot[y_tot==0]

                                             #category optimisation stuff#

        #set up optimiser ranges and no. categories to test if non-cut based
        ranges    = [ [0.3,1.] ]
        names     = ['{} score'.format(output_tag)] #arbitrary
        print_str = ''
        cats = [1,2,3,4]
        AMS  = []

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
                AMS.append(optimiser.bests.totSignif)
            print '\n {}'.format(print_str)

        #make nCat vs AMS plots
        Plotter.cats_vs_ams(cats, AMS, output_tag)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    required_args = parser.add_argument_group('Required Arguments')
    required_args.add_argument('-c','--config', action='store', required=True)
    required_args.add_argument('-a','--model_architecture', action='store', required=True)
    required_args.add_argument('-m','--model', action='store', required=True)
    opt_args = parser.add_argument_group('Optional Arguements')
    opt_args.add_argument('-r','--reload_samples', action='store_true', default=False)
    opt_args.add_argument('-i','--n_iters', action='store', default=3000, type=int)
    opt_args.add_argument('-d','--data_as_bkg', action='store_true', default=False)
    opt_args.add_argument('-k','--cut_based_str', action='store',type=str, default='')
    options=parser.parse_args()
    main(options)
