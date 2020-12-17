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
        for data_obj in root_obj.data_objects: # for plotting
            root_obj.load_data(data_obj, reload_samples=options.reload_samples)
        root_obj.concat()

                                                #LSTM stuff#

        LSTM = LSTM_DNN(root_obj, object_vars, event_vars, options.train_frac, options.eq_weights)

        #functions called in subbed job, if options.opt_hps was true
        if options.hp_perm is not None:
            if options.opt_hps and options.train_best:
                 raise Exception('Cannot optimise HPs and train best model. Run optimal training after hyper paramter optimisation')
            elif options.opt_hps and options.hp_perm:
                raise Exception('opt_hps option submits scripts with the hp_perm option; Cannot submit a script with both!')
            else: 
                print 'About to train + validate on dataset with {} fold splitting'.format(options.k_folds)
                LSTM.set_hyper_parameters(options.hp_perm)
                LSTM.model.summary()
                LSTM.set_k_folds(options.k_folds) #handles train/valid/test data manip into list attributes
                for i_fold in range(options.k_folds):
                    LSTM.set_i_fold(i_fold)
                    LSTM.train_network(root_obj.mc_dir, save=False)
                    LSTM.validation_rocs.append(LSTM.compute_roc(batch_size=64))
                with open('{}/lstm_hp_opt_{}.txt'.format(mc_dir, output_tag),'a+') as val_roc_file:
                    LSTM.compare_rocs(val_roc_file, options.hp_perm)
                    val_roc_file.close()

        elif options.opt_hps:
            #FIXME: add warning that many jobs are about to be submiited
            if options.k_folds<2: raise ValueError('K-folds option must be at least 2')
            if path.isfile('{}/lstm_hp_opt_{}.txt'.format(mc_dir, output_tag)): 
                system('rm {}/lstm_hp_opt_{}.txt'.format(mc_dir, output_tag))
                print ('deleting: {}/lstm_hp_opt_{}.txt'.format(mc_dir, output_tag))
            LSTM.batch_gs_cv(k_folds=options.k_folds)

        elif options.train_best:
            output_tag+='_best'
            with open('{}/lstm_hp_opt_{}.txt'.format(mc_dir, output_tag),'r') as val_roc_file:
                hp_roc = val_roc_file.readlines()
                best_params = hp_roc[-1].split(';')[0]
                print 'Best classifier params are: {}'.format(best_params)
                LSTM.set_hyper_parameters(best_params)
                LSTM.model.summary()
                LSTM.X_train_low_level = LSTM.join_objects(LSTM.X_train_low_level)
                LSTM.X_test_low_level  = LSTM.join_objects(LSTM.X_test_low_level)
                LSTM.train_network(root_obj.mc_dir, save=True)
                LSTM.compute_roc(batch_size=64)
                LSTM.plot_roc(output_tag)
                LSTM.plot_output_score(output_tag, batch_size=14)

        #else train with basic parameters/architecture
        else: 
           LSTM.model.summary()
           LSTM.X_train_low_level = LSTM.join_objects(LSTM.X_train_low_level)
           LSTM.X_test_low_level  = LSTM.join_objects(LSTM.X_test_low_level)
           LSTM.train_network(root_obj.mc_dir, epochs=5, save=True)
           LSTM.compute_roc(batch_size=64)
           LSTM.plot_roc(output_tag)
           LSTM.plot_output_score(output_tag, batch_size=64)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    required_args = parser.add_argument_group('Required Arguments')
    required_args.add_argument('-c','--config', help='configuration file (yaml) for training', action='store', required=True)
    opt_args = parser.add_argument_group('Optional Arguements')
    opt_args.add_argument('-r','--reload_samples', help='re-load the .root files and convert into pandas DataFrames', action='store_true', default=False)
    opt_args.add_argument('-w','--eq_weights', help='equalise the sum weights between signala nd background classes', action='store_true', default=False)
    opt_args.add_argument('-o','--opt_hps', action='store_true', default=False)
    opt_args.add_argument('-H','--hp_perm', action='store', default=None)
    opt_args.add_argument('-k','--k_folds', action='store', default=3, type=int)
    opt_args.add_argument('-b','--train_best', action='store_true', default=False)
    opt_args.add_argument('-t','--train_frac', help='fraction of events used for training. 1-test_frac used for testing', action='store', default=0.7, type=float)
    #opt_args.add_argument('-L', '--no_lstm', help='dont use object-level features (LSTM layers)', action='store_true', default=False)
    #opt_args.add_argument('-G', '--no_global', help='dont use event-level features', action='store_true', default=False)

    options=parser.parse_args()
    main(options)
