import argparse
import numpy as np
import yaml
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
try:
     plt.style.use("cms10_6_HP")
except IOError:
     warnings.warn('Could not import user defined matplot style file. Using default style settings...')

from DataHandling import ROOTHelpers
from DYPlotter import DYPlotter
from Utils import Utils


def main(options):

    
    with open(options.config, 'r') as config_file:
        config             = yaml.load(config_file)
        output_tag         = config['output_tag']

        mc_dir             = config['mc_file_dir']
        mc_fnames          = config['mc_file_names']
  
        data_dir           = config['data_file_dir']
        data_fnames        = config['data_file_names']

        proc_to_tree_name  = config['proc_to_tree_name']

        #check if dnn (lstm) variables need to be read in 
        varrs              = config['train_vars']
        all_train_vars     = []
        if isinstance(varrs, dict):
            object_vars     = varrs['object_vars']
            flat_obj_vars   = [var for i_object in object_vars for var in i_object]
            event_vars      = varrs['event_vars']
            all_train_vars += (flat_obj_vars + event_vars)
        else: 
            all_train_vars  = varrs

        vars_to_add         = config['vars_to_add']
        presel              = config['preselection']
        cut_map             = config['cut_map']

                                           #Data handling stuff#
 
        #get the dataframe for all years. Do not apply any specific preselection to sim samples
        root_obj = ROOTHelpers(output_tag, mc_dir, mc_fnames, data_dir, data_fnames, proc_to_tree_name, all_train_vars, vars_to_add, presel, read_systs=True)

        #for sig_obj in root_obj.sig_objects:
        #    root_obj.load_mc(sig_obj, reload_samples=options.reload_samples)
        for bkg_obj in root_obj.bkg_objects:
            root_obj.load_mc(bkg_obj, bkg=True, reload_samples=options.reload_samples)
        for data_obj in root_obj.data_objects:
            root_obj.load_data(data_obj, reload_samples=options.reload_samples)
        root_obj.concat()


        #--------------------------------------------------------------------------------------------------


        dy_plotter = DYPlotter(root_obj,cut_map)
        if options.reload_samples: #FIXME: reading in for the first time  wont re-weight sample!
            dy_plotter.pt_reweight()

        dy_plotter.manage_memory(options.systematics)
        if (options.var_name is None) or ('mva' in options.var_name.lower()): dy_plotter.eval_mva(options.mva_config, output_tag) #little bit hard coded - be careful if 'mva' not in MVA ouput name. Below line is safer but longer.
        #dy_plotter.eval_mva(options.mva_config, output_tag)
        #--------------------------------------------------------------------------------------------------

        with open('plotting/var_to_xrange.yaml', 'r') as plot_config_file:
            plot_config        = yaml.load(plot_config_file)
            var_to_xrange      = plot_config['var_to_xrange'] 

        if options.var_name is not None: vars_to_plot = [options.var_name]
        else: var_to_plot = all_train_vars+[dy_plotter.proc+'_mva']

        for var in vars_to_plot:

            if 'mva' in var: var_bins = np.linspace(0, 1, options.n_bins)
            else: var_bins = np.linspace(var_to_xrange[var][0], var_to_xrange[var][1], options.n_bins)
            print 'plotting var: {}'.format(var)
            fig, axes = plt.subplots(nrows=2, ncols=1, dpi=200, sharex=True,
                                     gridspec_kw ={'height_ratios':[3,0.8], 'hspace':0.08})    

            cut_str = dy_plotter.get_cut_string(var)

            #data stuff
            data_binned, bin_centres, data_stat_down_up = dy_plotter.plot_data(cut_str, axes, var, var_bins)
            dy_plotter.plot_bkgs(cut_str, axes, var, var_bins, data_binned, bin_centres, data_stat_down_up)

            #syst stuff
            dy_plotter.plot_systematics(cut_str, axes, var, var_bins, options.systematics, do_mva=(options.var_name is None))

            axes = dy_plotter.set_canv_style(axes, var, var_bins)
            axes[0].legend(bbox_to_anchor=(0.97,0.97), ncol=1)
            Utils.check_dir('{}/plotting/plots/{}'.format(os.getcwd(), output_tag))
            #fig.savefig('{0}/plotting/plots/{1}/{1}_{2}.pdf'.format(os.getcwd(), output_tag, var)) 
            fig.savefig('/vols/cms/jwd18/Hee/MLCategorisation/CMSSW_10_2_0/src/HToEE/plotting/plots/{0}/{0}_{1}.pdf'.format(output_tag, var))  #temp hardcode


        
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    required_args = parser.add_argument_group('Required Arguments')
    required_args.add_argument('-c','--config', action='store', required=True)
    required_args.add_argument('-M','--mva_config', action='store', required=True)
    required_args.add_argument('-s','--systematics', nargs='+', required=True, default=['jec'], type=str)
    opt_args = parser.add_argument_group('Optional Arguements')
    opt_args.add_argument('-r','--reload_samples', help='re-load the .root files and convert into pandas DataFrames', action='store_true', default=False)
    opt_args.add_argument('-n','--n_bins', help='number of bins for plotting each variables', action='store', default=41, type=int)
    opt_args.add_argument('-v','--var_name', help='Name of single variable if wanting to plot just one', action='store', default=None, type=str)
    options=parser.parse_args()
    main(options)
