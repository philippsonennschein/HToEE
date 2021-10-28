import numpy as np
import yaml
import os
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
try:
     plt.style.use("cms10_6_HP")
except IOError:
     warnings.warn('Could not import user defined matplot style file. Using default style settings...')
import scipy.stats

import sys

from Utils import Utils

class Plotter(object):
    '''
    Class to plot input variables and output scores
    '''
    #def __init__(self, data_obj, input_vars, sig_col='forestgreen', normalise=False, log=False, norm_to_data=False): 
    def __init__(self, data_obj, input_vars, sig_col='red', normalise=False, norm_to_data=False): 
        self.sig_df       = data_obj.mc_df_sig
        self.bkg_df       = data_obj.mc_df_bkg
        self.data_df      = data_obj.data_df
        del data_obj

        self.sig_labels   = np.unique(self.sig_df['proc'].values).tolist()
        self.bkg_labels   = np.unique(self.bkg_df['proc'].values).tolist()

        #self.bkg_labels   = ['EWKZ', 'DYMC', 'TT2L2Nu', 'TTSemiL'] #FIXME: temp re-ordering of procs for VBF
        #self.bkg_colours  = ['firebrick','mediumblue', 'lime','olive','red','blue','green','orange','yellow','violet','white'] #better for ggH
        self.bkg_colours  = ['silver','indianred','yellowgreen','lightgreen','green','mediumturquoise','darkslategrey','skyblue','steelblue','lightsteelblue','mediumslateblue']

        self.sig_colour   = sig_col
        self.normalise    = normalise
        #This can be change:
        self.sig_scaler   = 10**2

        #get xrange from yaml config
        with open('plotting/var_to_xrange.yaml', 'r') as plot_config_file:
            plot_config        = yaml.load(plot_config_file)
            self.var_to_xrange = plot_config['var_to_xrange']
            
        missing_vars = [x for x in input_vars if x not in self.var_to_xrange.keys()]
        if len(missing_vars)!=0: raise IOError('Missing variables in var_to_xrange.py: {}'.format(missing_vars))

    @classmethod 
    def num_to_str(self, num):
        ''' 
        Convert basic number into scientific form e.g. 1000 -> 10^{3}.
        Not considering decimal inputs (see decimal_to_str). Also ignores first unit.
        '''
        str_rep = str(num) 
        if str_rep[0] == 0: return num 
        exponent = len(str_rep)-1
        return r'$\times 10^{%s}$'%(exponent)

    @classmethod 
    def decimal_to_str(self, num):
        ''' 
        Convert basic number into scientific form e.g. 0.001 -> 10^{-3}.
        Not considering non-decimal inputs for now. Also ignores first unit.
        '''
        str_rep = str(num) 
        if str_rep[0] is not '0': raise IOError('Need a decimal input for string formatting, rather than:'.format(num))
        exponent = len(str_rep)-2 #minus one for the decimal point counted in string length
        return r'$\times 10^{-%s}$'%(exponent)

#Katharina thinks that this is the key to combine the background

    def plot_input(self, var, n_bins, out_label, ratio_plot=False, norm_to_data=False, extra_cuts=None, extra_tag=None, blind=False):
        if blind and ('diphotonMass' not in var): raise IOError('blinding only configured for plotting diphoton Mass!')
        if ratio_plot: 
            plt.rcParams.update({'figure.figsize':(6,5.8)})
            fig, axes = plt.subplots(nrows=2, ncols=1, dpi=200, sharex=True,
                                     gridspec_kw ={'height_ratios':[3,0.8], 'hspace':0.08})   
            ratio = axes[1]
            axes = axes[0]
        else:
            fig  = plt.figure(1)
            axes = fig.gca()

        bkg_stack      = []
        bkg_w_stack    = []
        bkg_proc_stack = []
        
        if extra_cuts is not None:
            var_sig     = self.sig_df.query(extra_cuts)[var].values
            sig_weights = self.sig_df.query(extra_cuts)['weight'].values
        else:
            var_sig     = self.sig_df[var].values
            sig_weights = self.sig_df['weight'].values

        bkg_stack_qcd = []
        bkg_w_stack_qcd = []
        bkg_proc_stack_qcd = []
        bkg_stack_diphoton = []
        bkg_w_stack_diphoton = []
        bkg_proc_stack_diphoton = []
        bkg_stack_gjet = []
        bkg_w_stack_gjet = []
        bkg_proc_stack_gjet = []

        for bkg in self.bkg_labels:
            if extra_cuts is not None:
                tmp_bkg     = self.bkg_df[self.bkg_df.proc==bkg]
                var_bkg     = tmp_bkg.query(extra_cuts)[var].values
                bkg_weights = tmp_bkg.query(extra_cuts)['weight'].values
            else: 
                var_bkg     = self.bkg_df[self.bkg_df.proc==bkg][var].values
                bkg_weights = self.bkg_df[self.bkg_df.proc==bkg]['weight'].values

            if 'QCD' in bkg:
                bkg_stack_qcd.append(var_bkg)
                bkg_w_stack_qcd.append(bkg_weights)
                bkg_proc_stack_qcd.append('QCD')
            elif 'Diphoton' in bkg:
                bkg_stack_diphoton.append(var_bkg)
                bkg_w_stack_diphoton.append(bkg_weights)
                bkg_proc_stack_diphoton.append('Diphoton')
            else:
                bkg_stack_gjet.append(var_bkg)
                bkg_w_stack_gjet.append(bkg_weights)
                bkg_proc_stack_gjet.append('GJet')

        bkg_stack.extend(bkg_stack_qcd)
        bkg_stack.extend(bkg_stack_diphoton)
        bkg_stack.extend(bkg_stack_gjet)

        bkg_w_stack.extend(bkg_w_stack_qcd)
        bkg_w_stack.extend(bkg_w_stack_diphoton)
        bkg_w_stack.extend(bkg_w_stack_gjet)

        bkg_proc_stack.extend(bkg_proc_stack_qcd)
        bkg_proc_stack.extend(bkg_proc_stack_diphoton)
        bkg_proc_stack.extend(bkg_proc_stack_gjet)

        #for bkg in self.bkg_labels:
        #    if extra_cuts is not None:
        #        tmp_bkg     = self.bkg_df[self.bkg_df.proc==bkg]
        #        var_bkg     = tmp_bkg.query(extra_cuts)[var].values
        #        bkg_weights = tmp_bkg.query(extra_cuts)['weight'].values
        #    else: 
        #        var_bkg     = self.bkg_df[self.bkg_df.proc==bkg][var].values
        #        bkg_weights = self.bkg_df[self.bkg_df.proc==bkg]['weight'].values

        #    bkg_stack.append(var_bkg)
        #    bkg_w_stack.append(bkg_weights)
        #    bkg_proc_stack.append(bkg)

        if self.normalise:
            sig_weights /= np.sum(sig_weights)
            bkg_weights /= np.sum(bkg_weights) #FIXME: set this up for multiple bkgs

        bins = np.linspace(self.var_to_xrange[var][0], self.var_to_xrange[var][1], n_bins)

        #add sig mc
        #Changed
        axes.hist(var_sig, bins=bins, label=self.sig_labels[0]+r' ($\mathrm{H}\rightarrow\mathrm{Gamma Gamma}$) '+self.num_to_str(self.sig_scaler), weights=sig_weights*(self.sig_scaler), histtype='step', color=self.sig_colour, zorder=10)
        #Before:
        #axes.hist(var_sig, bins=bins, label=self.sig_labels[0]+r' ($\mathrm{H}\rightarrow\mathrm{ee}$) '+self.num_to_str(self.sig_scaler), weights=sig_weights*(self.sig_scaler), histtype='step', color=self.sig_colour, zorder=10)
        
        #data
        if extra_cuts is not None:  data_binned, bin_edges = np.histogram(self.data_df.query(extra_cuts)[var].values, bins=bins)
        else: data_binned, bin_edges = np.histogram(self.data_df[var].values, bins=bins)
        bin_centres = (bin_edges[:-1] + bin_edges[1:])/2
        x_err    = (bin_edges[-1] - bin_edges[-2])/2
        data_down, data_up = self.poisson_interval(data_binned, data_binned)

        if blind: 
            blinded_mask = []
            for i_bin, bin_val in enumerate(bin_centres): #use list comprehension
                if bin_val > 115 and bin_val < 135: blinded_mask.append(np.NaN)
                else: blinded_mask.append(1)
                  
            blinded_data_binned = np.asarray(data_binned) * (blinded_mask)
            blinded_data_down   = np.asarray(data_down) * (blinded_mask)
            blinded_data_up     = np.asarray(data_up) * (blinded_mask)
            axes.errorbar( bin_centres, blinded_data_binned, yerr=[blinded_data_binned-blinded_data_down, blinded_data_up-blinded_data_binned], label='Data', fmt='o', ms=3, color='black', capsize=0, zorder=1)
        else: axes.errorbar( bin_centres, data_binned, yerr=[data_binned-data_down, data_up-data_binned], label='Data', fmt='o', ms=3, color='black', capsize=0, zorder=1)

        #add stacked bkg
        if norm_to_data: 
            rew_stack = []
            bkg_stack_summed, _ = np.histogram(np.concatenate(bkg_stack), bins=bins, weights=np.concatenate(bkg_w_stack))
            k_factor = np.sum(data_binned) / np.sum(bkg_stack_summed) #important to do this after binning, since norm may be different than before (if var has -999's)
            for w_arr in bkg_w_stack:
                rew_stack.append(w_arr*k_factor)
            axes.hist(bkg_stack, bins=bins, label=bkg_proc_stack, weights=rew_stack, histtype='stepfilled', color=self.bkg_colours[0:len(bkg_proc_stack)], stacked=True, zorder=0)

            
            bkg_stack_summed *= k_factor
            sumw2_bkg, _  = np.histogram(np.concatenate(bkg_stack), bins=bins, weights=np.concatenate(rew_stack)**2)
        else: 
            axes.hist(bkg_stack, bins=bins, label=bkg_proc_stack, weights=bkg_w_stack, histtype='stepfilled', color=self.bkg_colours[0:len(bkg_proc_stack)], stacked=True, zorder=0)
            bkg_stack_summed, _ = np.histogram(np.concatenate(bkg_stack), bins=bins, weights=np.concatenate(bkg_w_stack))
            sumw2_bkg, _  = np.histogram(np.concatenate(bkg_stack), bins=bins, weights=np.concatenate(bkg_w_stack)**2)

        if self.normalise: axes.set_ylabel('Arbitrary Units', ha='right', y=1, size=13)
        else: axes.set_ylabel('Events', ha='right', y=1, size=13)

        #plot mc error 
        bkg_std_down, bkg_std_up  = self.poisson_interval(bkg_stack_summed, sumw2_bkg)                                                   
        axes.fill_between(bins, list(bkg_std_down)+[bkg_std_down[-1]], list(bkg_std_up)+[bkg_std_up[-1]], alpha=0.3, step="post", color="grey", lw=1, zorder=4, label='Simulation stat. unc.')

        #change axes limits
        current_bottom, current_top = axes.get_ylim()
        if self.var_to_xrange[var][2]:
            axes.set_yscale('log', nonposy='clip')
            axes.set_ylim(bottom=100, top=current_top*20)
        else: axes.set_ylim(bottom=0, top=current_top*1.35)

        #Changed!
        #axes.legend(bbox_to_anchor=(0.9,0.97), ncol=2, prop={'size':10})
        axes.legend(loc='best', ncol=2, prop={'size':10})
        self.plot_cms_labels(axes)
           
        var_name_safe = var.replace('_',' ')
        if ratio_plot:
            if blind: 
                ratio.errorbar(bin_centres, (blinded_data_binned/bkg_stack_summed), yerr=[ (blinded_data_binned-blinded_data_down)/bkg_stack_summed, (blinded_data_up-blinded_data_binned)/bkg_stack_summed], fmt='o', ms=3, color='black', capsize=0)
                bkg_std_down_ratio = (np.ones_like(bkg_std_down) - ((bkg_stack_summed - bkg_std_down)/bkg_stack_summed)) * blinded_mask
                bkg_std_up_ratio   = (np.ones_like(bkg_std_up)   + ((bkg_std_up - bkg_stack_summed)/bkg_stack_summed)) * blinded_mask
            else:
                ratio.errorbar(bin_centres, (data_binned/bkg_stack_summed), yerr=[ (data_binned-data_down)/bkg_stack_summed, (data_up-data_binned)/bkg_stack_summed], fmt='o', ms=3, color='black', capsize=0)
                bkg_std_down_ratio = np.ones_like(bkg_std_down) - ((bkg_stack_summed - bkg_std_down)/bkg_stack_summed)
                bkg_std_up_ratio   = np.ones_like(bkg_std_up)   + ((bkg_std_up - bkg_stack_summed)/bkg_stack_summed)
            ratio.fill_between(bins, list(bkg_std_down_ratio)+[bkg_std_down_ratio[-1]], list(bkg_std_up_ratio)+[bkg_std_up_ratio[-1]], alpha=0.3, step="post", color="grey", lw=1, zorder=4)

            ratio.set_xlabel('{}'.format(var_name_safe), ha='right', x=1, size=13)
            ratio.set_ylabel('Data/MC', size=13)
            #ratio.set_ylim(0, 2)
            ratio.set_ylim(0.8, 1.2)
            ratio.grid(True, linestyle='dotted')
        else: axes.set_xlabel('{}'.format(var_name_safe), ha='right', x=1, size=13)
       
        Utils.check_dir('{}/plotting/plots/{}'.format(os.getcwd(), out_label))
        if extra_tag is not None: 
            fig.savefig('{0}/plotting/plots/{1}/{1}_{2}_cat{3}.pdf'.format(os.getcwd(), out_label, var, extra_tag))
        else:
            fig.savefig('{0}/plotting/plots/{1}/{1}_{2}.pdf'.format(os.getcwd(), out_label, var))
        plt.close()

    @classmethod 
    def plot_cms_labels(self, axes, label='Work in progress', lumi='41.5',energy='(13 TeV)'):
        axes.text(0, 1.01, r'\textbf{CMS} %s'%label, ha='left', va='bottom', transform=axes.transAxes, size=14)
        axes.text(1, 1.01, r'%s fb\textsuperscript{-1} %s'%(lumi,energy), ha='right', va='bottom', transform=axes.transAxes, size=14)

    def plot_roc(self, y_train, y_pred_train, train_weights, y_test, y_pred_test, test_weights, out_tag='MVA'):
        bkg_eff_train, sig_eff_train, _ = roc_curve(y_train, y_pred_train, sample_weight=train_weights)
        bkg_eff_test, sig_eff_test, _ = roc_curve(y_test, y_pred_test, sample_weight=test_weights)

        fig = plt.figure(1)
        axes = fig.gca()
        axes.plot(bkg_eff_train, sig_eff_train, color='red', label='Train')
        axes.plot(bkg_eff_test, sig_eff_test, color='blue', label='Test')
        axes.set_xlabel('Background efficiency', ha='right', x=1, size=13)
        axes.set_xlim((0,1))
        axes.set_ylabel('Signal efficiency', ha='right', y=1, size=13)
        axes.set_ylim((0,1))
        axes.legend(bbox_to_anchor=(0.97,0.97))
        self.plot_cms_labels(axes)
        axes.grid(True, 'major', linestyle='solid', color='grey', alpha=0.5)
        self.fig = fig

        np.savez('{}/models/{}_ROC_sig_bkg_arrays.npz'.format(os.getcwd(), out_tag), sig_eff_test=sig_eff_test, bkg_eff_test=bkg_eff_test)
        #np.savez('{}/models/{}_ROC_sig_bkg_arrays_NOJETVARS.npz'.format(os.getcwd(), out_tag), sig_eff_test=sig_eff_test, bkg_eff_test=bkg_eff_test)
        return fig

    def plot_output_score(self, y_test, y_pred_test, test_weights, proc_arr_test, data_pred_test, MVA='BDT', ratio_plot=False, norm_to_data=False, log=False):
        if ratio_plot: 
            plt.rcParams.update({'figure.figsize':(6,5.8)})
            fig, axes = plt.subplots(nrows=2, ncols=1, dpi=200, sharex=True,
                                     gridspec_kw ={'height_ratios':[3,0.8], 'hspace':0.08})   
            ratio = axes[1]
            axes = axes[0]
        else:
            fig  = plt.figure(1)
            axes = fig.gca()

        bins = np.linspace(0,1,41)

        bkg_stack      = []
        bkg_w_stack    = []
        bkg_proc_stack = []

        sig_scores = y_pred_test.ravel()  * (y_test==1)
        sig_w_true = test_weights.ravel() * (y_test==1)

        bkg_scores = y_pred_test.ravel()  * (y_test==0)
        bkg_w_true = test_weights.ravel() * (y_test==0)

        if self.normalise:
            sig_w_true /= np.sum(sig_w_true)
            bkg_w_true /= np.sum(bkg_w_true)

        for bkg in self.bkg_labels:
            bkg_score     = bkg_scores * (proc_arr_test==bkg)
            bkg_weights   = bkg_w_true * (proc_arr_test==bkg)
            bkg_stack.append(bkg_score)
            bkg_w_stack.append(bkg_weights)
            bkg_proc_stack.append(bkg)

        #sig
        axes.hist(sig_scores, bins=bins, label=self.sig_labels[0]+r' ($\mathrm{H}\rightarrow\mathrm{ee}$) '+self.num_to_str(self.sig_scaler), weights=sig_w_true*(self.sig_scaler), histtype='step', color=self.sig_colour)

        #data - need to take test frac of data
        data_binned, bin_edges = np.histogram(data_pred_test, bins=bins)
        bin_centres = (bin_edges[:-1] + bin_edges[1:])/2
        x_err    = (bin_edges[-1] - bin_edges[-2])/2
        data_down, data_up = self.poisson_interval(data_binned, data_binned)
        axes.errorbar( bin_centres, data_binned, yerr=[data_binned-data_down, data_up-data_binned], label='Data', fmt='o', ms=4, color='black', capsize=0, zorder=1)

        if norm_to_data: 
            rew_stack = []
            k_factor = np.sum(np.ones_like(data_pred_test))/np.sum(bkg_w_true)
            for w_arr in bkg_w_stack:
                rew_stack.append(w_arr*k_factor)
            bkg_mc_binned, _, _ = axes.hist(bkg_stack, bins=bins, label=bkg_proc_stack, weights=rew_stack, histtype='stepfilled', color=self.bkg_colours[0:len(bkg_proc_stack)], stacked=True, zorder=0)
            bkg_stack_summed, _ = np.histogram(np.concatenate(bkg_stack), bins=bins, weights=np.concatenate(rew_stack))
            sumw2_bkg, _  = np.histogram(np.concatenate(bkg_stack), bins=bins, weights=np.concatenate(rew_stack)**2)
        else: 
            axes.hist(bkg_stack, bins=bins, label=bkg_proc_stack, weights=bkg_w_stack, histtype='stepfilled', color=self.bkg_colours[0:len(bkg_proc_stack)], stacked=True, zorder=0)
            bkg_stack_summed, _ = np.histogram(np.concatenate(bkg_stack), bins=bins, weights=np.concatenate(bkg_w_stack))
            sumw2_bkg, _  = np.histogram(np.concatenate(bkg_stack), bins=bins, weights=np.concatenate(bkg_w_stack)**2)
        #plot mc error 
        bkg_std_down, bkg_std_up  = self.poisson_interval(bkg_stack_summed, sumw2_bkg)                                                   
        axes.fill_between(bins, list(bkg_std_down)+[bkg_std_down[-1]], list(bkg_std_up)+[bkg_std_up[-1]], alpha=0.3, step="post", color="grey", lw=1, zorder=4, label='Simulation stat. unc.')

        #axes.legend(bbox_to_anchor=(0.97,0.97), ncol=2)
        axes.legend(bbox_to_anchor=(0.9,0.97), ncol=2, prop={'size':10})
        if self.normalise: axes.set_ylabel('Arbitrary Units', ha='right', y=1, size=13)
        else: axes.set_ylabel('Events', ha='right', y=1, size=13)

        if ratio_plot:
            ratio.errorbar(bin_centres, (data_binned/bkg_stack_summed), yerr=[ (data_binned-data_down)/bkg_stack_summed, (data_up-data_binned)/bkg_stack_summed], fmt='o', ms=4, color='black', capsize=0)
            bkg_std_down_ratio = np.ones_like(bkg_std_down) - ((bkg_stack_summed - bkg_std_down)/bkg_stack_summed)
            bkg_std_up_ratio   = np.ones_like(bkg_std_up)   + ((bkg_std_up - bkg_stack_summed)/bkg_stack_summed)
            ratio.fill_between(bins, list(bkg_std_down_ratio)+[bkg_std_down_ratio[-1]], list(bkg_std_up_ratio)+[bkg_std_up_ratio[-1]], alpha=0.3, step="post", color="grey", lw=1, zorder=4)

            ratio.set_xlabel('{} Score'.format(MVA), ha='right', x=1, size=13)
            ratio.set_ylim(0, 2)
            ratio.grid(True, linestyle='dotted')
        else: axes.set_xlabel('{} Score'.format(MVA), ha='right', x=1, size=13)
        self.plot_cms_labels(axes, lumi='137')

        current_bottom, current_top = axes.get_ylim()
        if log: 
            axes.set_yscale('log', nonposy='clip')
            axes.set_ylim(bottom=1, top=current_top*100)
        else: 
            axes.set_ylim(bottom=0, top=current_top*1.45)

        #ggH
        #axes.axvline(0.890, ymax=0.7, color='black', linestyle='--')
        #axes.axvline(0.741, ymax=0.7, color='black', linestyle='--')
        #axes.axvline(0.577, ymax=0.7, color='black', linestyle='--')
        #axes.axvline(0.213, ymax=0.7, color='black', linestyle='--')
        #axes.axvspan(0, 0.213, ymax=0.7, color='grey', alpha=0.35)

        #VBF BDT
        axes.axvline(0.890, ymax=0.7, color='black', linestyle='--')
        axes.axvline(0.741, ymax=0.7, color='black', linestyle='--')
	axes.axvspan(0, 0.741, ymax=0.7, color='grey', alpha=0.35)

	#VBF DNN
	#axes.axvline(0.907, ymax=0.71, color='black', linestyle='--')
	#axes.axvline(0.750, ymax=0.71, color='black', linestyle='--')
        #axes.axvspan(0, 0.750, ymax=0.71, color='grey', lw=0, alpha=0.5)

        return fig

    def plot_output_score_three_class(self, y_test, y_pred_test, test_weights, MVA='BDT', norm_to_data=False, log=False, clf_class=''):
        fig  = plt.figure(1)
        axes = fig.gca()

        bins = np.linspace(0,1,41)

        #bkg_stack      = []
        #bkg_w_stack    = []
        #bkg_proc_stack = []


        bkg_scores = y_pred_test.ravel()  * (y_test==0)
        bkg_w_true = test_weights.ravel() * (y_test==0)

        third_class_scores = y_pred_test.ravel()  * (y_test==1)
        third_class_w_true   = test_weights.ravel() * (y_test==1)

        sig_scores = y_pred_test.ravel()  * (y_test==2)
        sig_w_true = test_weights.ravel() * (y_test==2)

        #normalise snice only care about shape for now
        sig_w_true         /= np.sum(sig_w_true)
        bkg_w_true         /= np.sum(bkg_w_true)
        third_class_w_true /= np.sum(third_class_w_true)

        #for bkg in self.bkg_labels:
        #    bkg_score     = bkg_scores * (proc_arr_test==bkg)
        #    bkg_weights   = bkg_w_true * (proc_arr_test==bkg)
        #    bkg_stack.append(bkg_score)
        #    bkg_w_stack.append(bkg_weights)
        #    bkg_proc_stack.append(bkg)

        #sig
        axes.hist(sig_scores, bins=bins, label=self.sig_labels[0]+r' ($\mathrm{H}\rightarrow\mathrm{ee}$) ', weights=sig_w_true, histtype='step', color=self.sig_colour)

        #data - need to take test frac of data
        #data_binned, bin_edges = np.histogram(data_pred_test, bins=bins)
        #bin_centres = (bin_edges[:-1] + bin_edges[1:])/2
        #x_err    = (bin_edges[-1] - bin_edges[-2])/2
        #data_down, data_up = self.poisson_interval(data_binned, data_binned)
        #axes.errorbar( bin_centres, data_binned, yerr=[data_binned-data_down, data_up-data_binned], label='Data', fmt='o', ms=4, color='black', capsize=0, zorder=1)

        #if norm_to_data: 
        #    rew_stack = []
        #    k_factor = np.sum(np.ones_like(data_pred_test))/np.sum(bkg_w_true)
        #    for w_arr in bkg_w_stack:
        #        rew_stack.append(w_arr*k_factor)
        #    bkg_mc_binned, _, _ = axes.hist(bkg_stack, bins=bins, label=bkg_proc_stack, weights=rew_stack, histtype='stepfilled', color=self.bkg_colours[0:len(bkg_proc_stack)], stacked=True, zorder=0)
        #    bkg_stack_summed, _ = np.histogram(np.concatenate(bkg_stack), bins=bins, weights=np.concatenate(rew_stack))
        #    sumw2_bkg, _  = np.histogram(np.concatenate(bkg_stack), bins=bins, weights=np.concatenate(rew_stack)**2)
        #else: 
        #    axes.hist(bkg_stack, bins=bins, label=bkg_proc_stack, weights=bkg_w_stack, histtype='stepfilled', color=self.bkg_colours[0:len(bkg_proc_stack)], stacked=True, zorder=0)
        #    bkg_stack_summed, _ = np.histogram(np.concatenate(bkg_stack), bins=bins, weights=np.concatenate(bkg_w_stack))
        #    sumw2_bkg, _  = np.histogram(np.concatenate(bkg_stack), bins=bins, weights=np.concatenate(bkg_w_stack)**2)

        axes.hist(bkg_scores, bins=bins, label='Other backgrounds', weights=bkg_w_true, histtype='step', color=self.bkg_colours[0], zorder=0)
        axes.hist(third_class_scores, bins=bins, label='VBF Z', weights=third_class_w_true, histtype='step', color=self.bkg_colours[1], zorder=0)

        #plot mc error 
        #bkg_std_down, bkg_std_up  = self.poisson_interval(bkg_stack_summed, sumw2_bkg)                                                   
        #axes.fill_between(bins, list(bkg_std_down)+[bkg_std_down[-1]], list(bkg_std_up)+[bkg_std_up[-1]], alpha=0.3, step="post", color="grey", lw=1, zorder=4, label='Simulation stat. unc.')

        #axes.legend(bbox_to_anchor=(0.97,0.97), ncol=2)
        axes.legend(bbox_to_anchor=(0.9,0.97), ncol=2, prop={'size':10})
        axes.set_ylabel('Arbitrary Units', ha='right', y=1, size=13)
        axes.set_xlabel('{} Score'.format(clf_class.replace('_',' ')), ha='right', x=1, size=13)

        #if ratio_plot:
        #    ratio.errorbar(bin_centres, (data_binned/bkg_stack_summed), yerr=[ (data_binned-data_down)/bkg_stack_summed, (data_up-data_binned)/bkg_stack_summed], fmt='o', ms=4, color='black', capsize=0)
        #    bkg_std_down_ratio = np.ones_like(bkg_std_down) - ((bkg_stack_summed - bkg_std_down)/bkg_stack_summed)
        #    bkg_std_up_ratio   = np.ones_like(bkg_std_up)   + ((bkg_std_up - bkg_stack_summed)/bkg_stack_summed)
        #    ratio.fill_between(bins, list(bkg_std_down_ratio)+[bkg_std_down_ratio[-1]], list(bkg_std_up_ratio)+[bkg_std_up_ratio[-1]], alpha=0.3, step="post", color="grey", lw=1, zorder=4)

        #    ratio.set_xlabel('{} Score'.format(MVA), ha='right', x=1, size=13)
        #    ratio.set_ylim(0, 2)
        #    ratio.grid(True, linestyle='dotted')
        #else: axes.set_xlabel('{} Score'.format(MVA), ha='right', x=1, size=13)
        #self.plot_cms_labels(axes)

        #current_bottom, current_top = axes.get_ylim()
        #if log: 
        #    axes.set_yscale('log', nonposy='clip')
        #    axes.set_ylim(bottom=1, top=current_top*20)
        #else: 
        #    axes.set_ylim(bottom=0, top=current_top*1.45)

        return fig


    @classmethod
    def cats_vs_ams(self, cats, AMS, out_tag, scaler=0.0001):
        fig  = plt.figure(1)
        axes = fig.gca()

        exponent_label = self.decimal_to_str(scaler)
        scaled_AMS = [point / scaler for point in AMS]

        axes.plot(cats,scaled_AMS, '-ro')
        axes.set_xlim((0, cats[-1]+1))
        #current_bottom, current_top = axes.get_ylim()
        #axes.set_ylim(bottom=0, top=current_top*1.5)
        axes.set_xlabel('$N_{\mathrm{cat}}$', ha='right', x=1, size=13)
        axes.set_ylabel('Combined AMS', ha='right', y=1, size=13)
        axes.text(-0.15, 1.01, '{}'.format(exponent_label), ha='left', va='bottom', transform=axes.transAxes, size=11)
        Plotter.plot_cms_labels(axes)
        fig.savefig('{}/categoryOpt/nCats_vs_AMS_{}.pdf'.format(os.getcwd(), out_tag))

    def poisson_interval(self, x, variance, level=0.68): 
        #print('variance',variance)
        neff = x**2/variance
        #print('neff=,',neff)
        scale = x/neff
     
        # CMS statcomm recommendation
        l = scipy.stats.gamma.interval(
            level, neff, scale=scale,
        )[0]
        u = scipy.stats.gamma.interval(
            level, neff+1, scale=scale
        )[1]
     
        # protect against no effecitve entries
        l[neff==0] = 0.
     
        # protect against no variance
        l[variance==0.] = 0.
        u[variance==0.] = np.inf
        return l, u
