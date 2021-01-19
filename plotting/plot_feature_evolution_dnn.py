import argparse
import numpy as np
import pandas as pd
import yaml
import keras
import os
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
from NeuralNets import LSTM_DNN
from Utils import Utils


def annotate_and_save(axes, plotter, var):
    axes.set_ylabel('Arbitrary Units', ha='right', y=1, size=13)
    current_bottom, current_top = axes.get_ylim()
    axes.set_ylim(bottom=0, top=current_top*1.3)
    #axes.legend(bbox_to_anchor=(0.97,0.97), ncol=2)
    axes.legend(loc='upper center', bbox_to_anchor=(0.5,0.97), ncol=2)
    plotter.plot_cms_labels(axes)

    var_name_safe = var.replace('_',' ')
    axes.set_xlim(left=plotter.var_to_xrange[var][0], right=plotter.var_to_xrange[var][1])
    axes.set_xlabel('{}'.format(var_name_safe), ha='right', x=1, size=13)

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
        colours           = ['#d7191c', '#fdae61', '#f2f229', '#abdda4', '#2b83ba']
                 
                                           #Data handling stuff#
                 
        #load the mc dataframe for all years
        root_obj = ROOTHelpers(output_tag, mc_dir, mc_fnames, data_dir, data_fnames, proc_to_tree_name, flat_obj_vars+event_vars, vars_to_add, presel) 
 
        for sig_obj in root_obj.sig_objects:
            root_obj.load_mc(sig_obj, reload_samples=options.reload_samples)
        for bkg_obj in root_obj.bkg_objects:
            root_obj.load_mc(bkg_obj, bkg=True, reload_samples=options.reload_samples)

        root_obj.concat()

                                            #Plotter stuff#

        #add model predictions to sig df
        print 'loading DNN: {}'.format(options.model_architecture)
        with open('{}'.format(options.model_architecture), 'r') as model_json:
            model_architecture = model_json.read()
        model = keras.models.model_from_json(model_architecture)
        model.load_weights('{}'.format(options.model))

        LSTM = LSTM_DNN(root_obj, object_vars, event_vars, 1.0, False, True)
        unscaled_sig_df = root_obj.mc_df_sig.copy()
        unscaled_bkg_df = root_obj.mc_df_bkg.copy()

        # set up X and y Matrices 
        LSTM.var_transform(do_data=False)  
        X_tot, y_tot     = LSTM.create_X_y()

        X_tot            = X_tot[flat_obj_vars+event_vars] #filter unused vars
        LSTM.load_X_scaler(out_tag=output_tag)
        X_tot            = LSTM.X_scaler.transform(X_tot)

        X_tot            = pd.DataFrame(X_tot, columns=flat_obj_vars+event_vars)
        X_tot_high_level = X_tot[event_vars].values
        X_tot_low_level  = LSTM.join_objects(X_tot[flat_obj_vars])
        pred_prob_tot    = model.predict([X_tot_high_level, X_tot_low_level], batch_size=1024).flatten()

        unscaled_sig_df['bdt_score'] = pred_prob_tot[y_tot==1]
        unscaled_bkg_df['bdt_score'] = pred_prob_tot[y_tot==0]
 
        train_vars = flat_obj_vars+event_vars
        plotter  = Plotter(root_obj, train_vars, norm_to_data=True)
        #for VBF, good set is: [0.30 0.50 0.70 0.80 0.90 1.0]
        bdt_bins = np.array(options.boundaries)
        Utils.check_dir('{}/plotting/plots/{}_sig_bkg_evo'.format(os.getcwd(), output_tag))
        i_hist = 0

        for var in train_vars+['dielectronMass']:
            fig  = plt.figure(1)
            axes = fig.gca()
            var_bins = np.linspace(plotter.var_to_xrange[var][0], plotter.var_to_xrange[var][1], options.n_bins)
            for ibin in range(len(bdt_bins)-1):
                sig_cut = unscaled_sig_df[np.logical_and( unscaled_sig_df['bdt_score'] > bdt_bins[ibin], unscaled_sig_df['bdt_score'] < bdt_bins[ibin+1])][var]
                weights_cut = unscaled_sig_df[np.logical_and( unscaled_sig_df['bdt_score'] > bdt_bins[ibin], unscaled_sig_df['bdt_score'] < bdt_bins[ibin+1])]['weight']
                weights_cut /= np.sum(weights_cut)
                axes.hist(sig_cut, bins=var_bins, label='{:.2f} $<$ MVA $<$ {:.2f}'.format(bdt_bins[ibin], bdt_bins[ibin+1]), weights=weights_cut, histtype='step', color=colours[i_hist])
                i_hist += 1
            i_hist=0
            annotate_and_save(axes, plotter, var)
            axes.text(0.95, 0.6, 'Simulated VBF signal', ha='right', va='bottom', transform=axes.transAxes, size=14)
            fig.savefig('{0}/plotting/plots/{1}_sig_bkg_evo/{1}_{2}.pdf'.format(os.getcwd(), output_tag, var))
            plt.close()

        #plot background (check mass is not being sculpted)
        for var in ['dielectronMass']:
            fig  = plt.figure(1)
            axes = fig.gca()
            var_bins = np.linspace(plotter.var_to_xrange[var][0], plotter.var_to_xrange[var][1], options.n_bins)
            for ibin in range(len(bdt_bins)-1):
                bkg_cut = unscaled_bkg_df[np.logical_and( unscaled_bkg_df['bdt_score'] > bdt_bins[ibin], unscaled_bkg_df['bdt_score'] < bdt_bins[ibin+1])][var]
                bkg_weights_cut = unscaled_bkg_df[np.logical_and( unscaled_bkg_df['bdt_score'] > bdt_bins[ibin], unscaled_bkg_df['bdt_score'] < bdt_bins[ibin+1])]['weight']
                bkg_weights_cut /= np.sum(bkg_weights_cut)
                axes.hist(bkg_cut, bins=var_bins, label='{:.2f} $<$ MVA $<$ {:.2f}'.format(bdt_bins[ibin], bdt_bins[ibin+1]), weights=bkg_weights_cut, histtype='step', color=colours[i_hist])
                i_hist+=1
            i_hist=0

            annotate_and_save(axes, plotter, var)
            axes.text(0.95, 0.6, 'Simulated background', ha='right', va='bottom', transform=axes.transAxes, size=14)
            fig.savefig('{0}/plotting/plots/{1}_sig_bkg_evo/{1}_{2}_bkg.pdf'.format(os.getcwd(), output_tag, var))
            plt.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    required_args = parser.add_argument_group('Required Arguments')
    required_args.add_argument('-c','--config', action='store', required=True)
    required_args.add_argument('-a','--model_architecture', action='store', required=True)
    required_args.add_argument('-m','--model', action='store', required=True)
    required_args.add_argument('-B','--boundaries', nargs='+', required=True, default=[0.3,0.5,0.7,1.0], type=float)
    opt_args = parser.add_argument_group('Optional Arguements')
    opt_args.add_argument('-r','--reload_samples', action='store_true', default=False)
    opt_args.add_argument('-b','--n_bins',  default=26, type=int)
    options=parser.parse_args()
    main(options)
