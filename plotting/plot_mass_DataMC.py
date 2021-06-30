import argparse
import numpy as np
import yaml
import pickle

from DataHandling import ROOTHelpers
from PlottingUtils import Plotter

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
 
                                           #Data handling stuff#

        #load the mc dataframe for all years
        if options.pt_reweight: 
            cr_selection = config['reweight_cr']
            output_tag += '_pt_reweighted'
            root_obj = ROOTHelpers(output_tag, mc_dir, mc_fnames, data_dir, data_fnames, proc_to_tree_name, train_vars, vars_to_add, cr_selection)
        else: root_obj = ROOTHelpers(output_tag, mc_dir, mc_fnames, data_dir, data_fnames, proc_to_tree_name, train_vars, vars_to_add, presel)

        for sig_obj in root_obj.sig_objects:
            root_obj.load_mc(sig_obj, reload_samples=options.reload_samples)
        for bkg_obj in root_obj.bkg_objects:
            root_obj.load_mc(bkg_obj, bkg=True, reload_samples=options.reload_samples)
        for data_obj in root_obj.data_objects:
            root_obj.load_data(data_obj, reload_samples=options.reload_samples)
        root_obj.concat()

        if options.pt_reweight and options.reload_samples: 
            root_obj.apply_pt_rew('DYMC', presel)

    #load MVA
        with open(options.mva_config, 'r') as mva_config_file:
            config            = yaml.load(mva_config_file)
            model             = config['models'][options.mva_proc]
            boundaries        = config['boundaries'][options.mva_proc]

            #add DNN later
            if isinstance(model,str): 
                print 'evaluating BDT: {}'.format(model)
                clf = pickle.load(open('models/{}'.format(model), "rb"))
                root_obj.mc_df_sig[options.mva_proc+'_mva'] = clf.predict_proba(root_obj.mc_df_sig[train_vars].values)[:,1:].ravel()
                root_obj.mc_df_bkg[options.mva_proc+'_mva'] = clf.predict_proba(root_obj.mc_df_bkg[train_vars].values)[:,1:].ravel()
                root_obj.data_df[options.mva_proc+'_mva']   = clf.predict_proba(root_obj.data_df[train_vars].values)[:,1:].ravel()

            else: raise IOError('Did not get a classifier models in correct format in config')

                                            #Plotter stuff#

        plotter = Plotter(root_obj, train_vars, sig_col=sig_colour, norm_to_data=True)
        cat_counter = 0
        for b in boundaries:
            if cat_counter==0: extra_cuts = options.mva_proc+'_mva >' + str(boundaries['tag_0'])
            else: extra_cuts = (options.mva_proc+'_mva <' + str(boundaries['tag_'+str(cat_counter-1)])) + ' and ' + (options.mva_proc+'_mva >' + str(boundaries['tag_'+str(cat_counter)]))
            plotter.plot_input(options.mass_var_name, options.n_bins, output_tag, options.ratio_plot, norm_to_data=True, extra_cuts=extra_cuts, extra_tag=cat_counter, blind=True)
            cat_counter += 1


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    required_args = parser.add_argument_group('Required Arguments')
    required_args.add_argument('-c','--config', action='store', required=True)
    required_args.add_argument('-m','--mass_var_name',  action='store', type=str, required=True)
    required_args.add_argument('-M','--mva_config', action='store', required=True)
    opt_args = parser.add_argument_group('Optional Arguements')
    required_args.add_argument('-p','--mva_proc', action='store', default='VBF')
    opt_args.add_argument('-r','--reload_samples', action='store_true', default=False)
    opt_args.add_argument('-b','--n_bins',  default=26, type=int)
    opt_args.add_argument('-P','--pt_reweight',  action='store_true', default=False)
    opt_args.add_argument('-R','--ratio_plot',  action='store_true', default=False)
    options=parser.parse_args()
    main(options)
