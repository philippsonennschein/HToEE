import argparse
import numpy as np
import yaml
from os import path, system

from DataHandling import ROOTHelpers
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
 
        if options.pt_reweight: 
            cr_selection = config['reweight_cr']
            output_tag += '_pt_reweighted'
            root_obj = ROOTHelpers(output_tag, mc_dir, mc_fnames, data_dir, data_fnames, proc_to_tree_name, flat_obj_vars+event_vars, vars_to_add, cr_selection)
        else: root_obj = ROOTHelpers(output_tag, mc_dir, mc_fnames, data_dir, data_fnames, proc_to_tree_name, flat_obj_vars+event_vars, vars_to_add, presel)

        #load the dataframes for all years
        for sig_obj in root_obj.sig_objects:
            root_obj.load_mc(sig_obj, reload_samples=options.reload_samples)
        for bkg_obj in root_obj.bkg_objects:
            root_obj.load_mc(bkg_obj, bkg=True, reload_samples=options.reload_samples)
        for data_obj in root_obj.data_objects: # for plotting
            root_obj.load_data(data_obj, reload_samples=options.reload_samples)
        root_obj.concat()

        #reweight samples in bins of pT (and maybe Njets), for each year separely. 
        if options.pt_reweight and options.reload_samples: 
            root_obj.apply_pt_rew('DYMC', presel)

                                                #LSTM stuff#

        LSTM = LSTM_DNN(root_obj, object_vars, event_vars, options.train_frac, options.eq_weights, options.batch_boost)
        if not options.opt_hps:
            LSTM.var_transform(do_data=True)
            X_tot, y_tot = LSTM.create_X_y(mass_res_reweight=True)
            LSTM.split_X_y(X_tot, y_tot, do_data=True)
            if options.hp_perm is not None: LSTM.get_X_scaler(LSTM.all_vars_X_train, out_tag=output_tag, save=False)
            else: LSTM.get_X_scaler(LSTM.all_vars_X_train, out_tag=output_tag)
            LSTM.X_scale_train_test(do_data=True)
            LSTM.set_low_level_2D_test_train(do_data=True, ignore_train=options.batch_boost) 

        #functions called in subbed job, if options.opt_hps was true
        if options.hp_perm is not None:
            if options.opt_hps and options.train_best:
                 raise Exception('Cannot optimise HPs and train best model. Run optimal training after hyper paramter optimisation')
            elif options.opt_hps and options.hp_perm:
                raise Exception('opt_hps option submits scripts with the hp_perm option; Cannot submit a script with both!')
            else: 
                LSTM.set_hyper_parameters(options.hp_perm)
                LSTM.model.summary()
                LSTM.train_w_batch_boost(out_tag=output_tag, save=False)
                with open('{}/lstm_hp_opt_{}.txt'.format(mc_dir, output_tag),'a+') as val_roc_file:
                    LSTM.compare_rocs(val_roc_file, options.hp_perm)
                    val_roc_file.close()

        elif options.opt_hps:
            #FIXME: add warning that many jobs are about to be submiited
            if path.isfile('{}/lstm_hp_opt_{}.txt'.format(mc_dir, output_tag)): 
                system('rm {}/lstm_hp_opt_{}.txt'.format(mc_dir, output_tag))
                print ('deleting: {}/lstm_hp_opt_{}.txt'.format(mc_dir, output_tag))
            LSTM.batch_gs_cv(pt_rew=options.pt_reweight)

        elif options.train_best:
            output_tag+='_best'
            with open('{}/lstm_hp_opt_{}.txt'.format(mc_dir, output_tag),'r') as val_roc_file:
                hp_roc = val_roc_file.readlines()
                best_params = hp_roc[-1].split(';')[0]
                print 'Best classifier params are: {}'.format(best_params)
                LSTM.set_hyper_parameters(best_params)
                LSTM.model.summary()
                LSTM.train_w_batch_boost(out_tag=output_tag)
                #compute final roc on test set
                LSTM.compute_roc(batch_size=1024) 
                LSTM.plot_roc(output_tag)
                LSTM.plot_output_score(output_tag, batch_size=1024, ratio_plot=True, norm_to_data=(not options.pt_reweight))

        #else train with basic parameters/architecture
        else: 
           LSTM.model.summary()
           if options.batch_boost: #type of model selection so need validation set
               LSTM.train_w_batch_boost(out_tag=output_tag) #handles creating validation set and 2D vars and sequential saving
           else: 
               LSTM.train_network(epochs=5, batch_size=1024)
               #LSTM.train_network(epochs=7, batch_size=32)
               LSTM.save_model(out_tag=output_tag)
           LSTM.compute_roc(batch_size=1024)
           #compute final roc on test set
           LSTM.plot_roc(output_tag)
           LSTM.plot_output_score(output_tag, batch_size=1024, ratio_plot=True, norm_to_data=(not options.pt_reweight))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    required_args = parser.add_argument_group('Required Arguments')
    required_args.add_argument('-c','--config', help='configuration file (yaml) for training', action='store', required=True)
    opt_args = parser.add_argument_group('Optional Arguements')
    opt_args.add_argument('-r','--reload_samples', help='re-load the .root files and convert into pandas DataFrames', action='store_true', default=False)
    opt_args.add_argument('-w','--eq_weights', help='equalise the sum weights between signala nd background classes', action='store_true', default=False)
    opt_args.add_argument('-o','--opt_hps', action='store_true', default=False)
    opt_args.add_argument('-H','--hp_perm', action='store', default=None)
    opt_args.add_argument('-b','--train_best', action='store_true', default=False)
    opt_args.add_argument('-t','--train_frac', help='fraction of events used for training. 1-test_frac used for testing', action='store', default=0.7, type=float)
    opt_args.add_argument('-P','--pt_reweight', action='store_true',default=False)
    opt_args.add_argument('-B','--batch_boost', action='store_true',default=False)
    #opt_args.add_argument('-L', '--no_lstm', help='dont use object-level features (LSTM layers)', action='store_true', default=False)
    #opt_args.add_argument('-G', '--no_global', help='dont use event-level features', action='store_true', default=False)

    options=parser.parse_args()
    main(options)
