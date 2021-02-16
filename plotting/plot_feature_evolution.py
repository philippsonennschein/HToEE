import argparse
import numpy as np
import yaml
import pickle
import os

#plotting imports
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
try:
     plt.style.use("cms10_6_HP")
except IOError:
     warnings.warn('Could not import user defined matplot style file. Using default style settings...')
plt.rcParams.update({'legend.fontsize':10}) 

from DataHandling import ROOTHelpers
from PlottingUtils import Plotter
from Utils import Utils

def annotate_and_save(axes, plotter, var):
    axes.set_ylabel('Arbitrary Units', ha='right', y=1, size=13)
    current_bottom, current_top = axes.get_ylim()
    axes.set_ylim(bottom=0, top=current_top*1.3)
    #axes.legend(bbox_to_anchor=(0.97,0.97), ncol=2)
    axes.legend(loc='upper center', bbox_to_anchor=(0.5,0.97), ncol=2)
    plotter.plot_cms_labels(axes)

    var_name_safe = var.replace('_',' ')
    #axes.set_xlim(left=plotter.var_to_xrange[var][0], right=plotter.var_to_xrange[var][1])
    axes.set_xlabel('{}'.format(var_name_safe), ha='right', x=1, size=13)

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
        colours           = ['#d7191c', '#fdae61', '#f2f229', '#abdda4', '#2b83ba']

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
        #for data_obj in root_obj.data_objects:
        #    root_obj.load_data(data_obj, reload_samples=options.reload_samples)
        root_obj.concat()

        if options.pt_reweight and options.reload_samples: 
            for year in root_obj.years:
                root_obj.pt_reweight('DYMC', year, presel)

                                            #Plotter stuff#

        #add model predictions to sig df
        print 'loading classifier: {}'.format(options.model)
        clf = pickle.load(open("{}".format(options.model), "rb"))
        sig_df = root_obj.mc_df_sig
        sig_df['bdt_score'] = clf.predict_proba(sig_df[train_vars].values)[:,1:].ravel()
        bkg_df = root_obj.mc_df_bkg
        bkg_df['bdt_score'] = clf.predict_proba(bkg_df[train_vars].values)[:,1:].ravel()
 
        plotter  = Plotter(root_obj, train_vars)
        #for VBF, good set is: [0.30 0.50 0.70 0.80 0.90 1.0]
        #for ggH, good set is: [0.10 0.30 0.45 0.53 0.60 0.8]
        bdt_bins = np.array(options.boundaries)
        Utils.check_dir('{}/plotting/plots/{}_sig_bkg_evo'.format(os.getcwd(), output_tag))
        i_hist = 0

        for var in train_vars+['dielectronMass']:
            fig  = plt.figure(1)
            axes = fig.gca()
            var_bins = np.linspace(plotter.var_to_xrange[var][0], plotter.var_to_xrange[var][1], options.n_bins)
            for ibin in range(len(bdt_bins)-1):
                sig_cut = sig_df[np.logical_and( sig_df['bdt_score'] > bdt_bins[ibin], sig_df['bdt_score'] < bdt_bins[ibin+1])][var]
                weights_cut = sig_df[np.logical_and( sig_df['bdt_score'] > bdt_bins[ibin], sig_df['bdt_score'] < bdt_bins[ibin+1])]['weight']
                axes.hist(sig_cut, bins=var_bins, label='{:.2f} $<$ MVA $<$ {:.2f}'.format(bdt_bins[ibin], bdt_bins[ibin+1]), weights=weights_cut, histtype='step', color=colours[i_hist], normed=True)
                i_hist += 1
            i_hist=0
            annotate_and_save(axes, plotter, var)
            fig.savefig('{0}/plotting/plots/{1}_sig_bkg_evo/{1}_{2}.pdf'.format(os.getcwd(), output_tag, var))
            print('saving: {0}/plotting/plots/{1}_sig_bkg_evo/{1}_{2}.pdf'.format(os.getcwd(), output_tag, var))
            plt.close()

        #plot background (check mass is not being sculpted)
        for var in ['dielectronMass']:
            fig  = plt.figure(1)
            axes = fig.gca()
            var_bins = np.linspace(plotter.var_to_xrange[var][0], plotter.var_to_xrange[var][1], options.n_bins)
            for ibin in range(len(bdt_bins)-1):
                bkg_cut = bkg_df[np.logical_and( bkg_df['bdt_score'] > bdt_bins[ibin], bkg_df['bdt_score'] < bdt_bins[ibin+1])][var]
                bkg_weights_cut = bkg_df[np.logical_and( bkg_df['bdt_score'] > bdt_bins[ibin], bkg_df['bdt_score'] < bdt_bins[ibin+1])]['weight']
                axes.hist(bkg_cut, bins=var_bins, label='{:.2f} $<$ MVA $<$ {:.2f}'.format(bdt_bins[ibin], bdt_bins[ibin+1]), weights=bkg_weights_cut, histtype='step', color=colours[i_hist], normed=True)
                i_hist+=1
            i_hist=0

            annotate_and_save(axes, plotter, var)
            fig.savefig('{0}/plotting/plots/{1}_sig_bkg_evo/{1}_{2}_bkg.pdf'.format(os.getcwd(), output_tag, var))
            plt.close()




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    required_args = parser.add_argument_group('Required Arguments')
    required_args.add_argument('-c','--config', action='store', required=True)
    required_args.add_argument('-m','--model', action='store', required=True)
    required_args.add_argument('-B','--boundaries', nargs='+', required=True, default=[0.3,0.5,0.7,1.0], type=float)
    opt_args = parser.add_argument_group('Optional Arguements')
    opt_args.add_argument('-r','--reload_samples', action='store_true', default=False)
    opt_args.add_argument('-b','--n_bins',  default=26, type=int)
    opt_args.add_argument('-P','--pt_reweight',  action='store_true', default=False)
    options=parser.parse_args()
    main(options)
