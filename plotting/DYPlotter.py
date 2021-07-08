import pandas as pd
import numpy as np
import scipy.stats
import yaml
import pickle



class DYPlotter(object):
    """
    Class with useful functions for making DYData/DYMC validation plots

    :param root_obj: object containing dfs needed for plotting, with no preselection applied
    :type : ROOTHelpers
    :param cut_map: dictionary of form {variable_name:cut_applied}
    :type : dict
    :param mc_total: binned mc for the variable being plot
    :type : numpy 1D array
    :param mc_total: binned mc for the variable being plot for a given bkg process 
    :type : numpy 1D array
    :param mc_stat_uncs: statistical uncertainties for each bin. First entry is array of lower bounds,
     second is array of upper bounds
    :type : list
    :param k_factor: normalisation between data and MC for given variable
    :type : float
    :param clf: classifier being evaluated
    :type : XGBClassifier() (or similar)
    :param proc: name of the process that the classifier targets e.g. VBF (BDT/NN)
    :type: str
    """

    def __init__(self, root_obj, cut_map): 
        self.root_obj       = root_obj
        self.cut_map        = cut_map

        self.mc_total       = None
        self.mc_totals      = {}
	self.mc_stat_uncs   = None
	self.k_factor       = None
        self.clf            = None
        self.proc           = None
        #self.colours        = ['lightgrey','#CBCBE5'] #VBF
        self.colours        = ['#CBCBE5'] #ggH

    def pt_reweight(self):
        """
        Derive pt reweighting factors for the full applied preselection. Apply this to dfs 
        """

        print 'reweighting MC to Data in pT(Z) bins. Inclusive in year atm...'
        print 'DEBUG: cut map looks like: {}'.format(self.cut_map)
        #selection_str = [var_name+cut for var_name,cut in self.cut_map.iteritems() if cut != '']
        selection_str = []
        for var_name, cuts in self.cut_map.iteritems():
            if len(cuts)>1: selection_str += [var_name+cut for cut in cuts]
            else: selection_str.append(var_name+cuts[0])
        separator = ' and '
        all_selection = separator.join(selection_str)
        print 'DEBUG: final selection looks like: {}'.format(self.cut_map.iteritems())

        presel_mc   = self.root_obj.mc_df_bkg.query(all_selection)
        presel_data = self.root_obj.data_df.query(all_selection)
        dy_mc_pt    = presel_mc['dielectronPt'].values
        dy_w        = presel_mc['weight'].values
        dy_data_pt  = presel_data['dielectronPt'].values

        del presel_mc
        del presel_data
            
        pt_bins = np.linspace(0,180,101)
        mc_pt_summed, _ = np.histogram(dy_mc_pt, bins=pt_bins, weights=dy_w)
        data_pt_summed, bin_edges = np.histogram(dy_data_pt, bins=pt_bins)
        bin_edges = (bin_edges[:-1] + bin_edges[1:])/2
            
        scale_factors = data_pt_summed/mc_pt_summed
            
        scaled_dfs = []
        for i_bin in range(len(scale_factors)-1):
            temp_df = self.root_obj.mc_df_bkg[self.root_obj.mc_df_bkg.dielectronPt > bin_edges[i_bin]] 
            temp_df = temp_df[temp_df.dielectronPt < bin_edges[i_bin+1]] 
            temp_df['weight'] *= scale_factors[i_bin]
            scaled_dfs.append(temp_df)  
 
        the_rest = self.root_obj.mc_df_bkg[self.root_obj.mc_df_bkg.dielectronPt > bin_edges[-1]]
        scaled_dfs.append(the_rest)
              
        self.root_obj.mc_df_bkg = pd.concat(scaled_dfs, ignore_index=True)
        del scaled_dfs

 
        for year in self.root_obj.years:
            self.root_obj.save_modified_dfs(year, ignore_sig=True)

    def manage_memory(self):
        """
        Delete dataframe columns that are not needed for data and mc. 

        Could also delete systs that are not used (since atm only removing nominal vars but still reading nonimal+syst in because of config format)
        """

        used_variables = self.root_obj.train_vars+self.cut_map.keys()+['weight']+['proc']
        #used_variables_w_systs = self.root_obj.train_vars+self.cut_map.keys()+['weight']
        #FIXME: add syst variables to this too ^
        for col in self.root_obj.data_df.columns: #looping through data columns wont so we dont delete systs
            if col not in used_variables: 
                del self.root_obj.mc_df_bkg[col]
                del self.root_obj.data_df[col]

    def plot_data(self, cut_string, axes, variable, bins):
        """ self explanatory """

        cut_df             = self.root_obj.data_df.query(cut_string)
        var_to_plot        = cut_df[variable].values
        var_weights        = cut_df['weight'].values
        del cut_df

        data_binned, bin_edges = np.histogram(var_to_plot, bins=bins, weights=var_weights)
        print '--> Integral of hist: {}, for data is: {}'.format(variable,np.sum(data_binned))
        bin_centres = (bin_edges[:-1] + bin_edges[1:])/2
        data_stat_down, data_stat_up = self.poisson_interval(data_binned, data_binned)

        #FIXME: sort this niche issue out
        #dataUp[dataUp==np.inf] == 0

        data_no_zeros = data_binned.copy() 
        data_no_zeros[data_no_zeros==0] = np.nan #removes markers at zero i.e. no entries

        axes[0].errorbar(bin_centres, data_no_zeros, 
                         yerr=[data_binned-data_stat_down, data_stat_up-data_binned],
                         label='Data', fmt='o', ms=4, color='black', capsize=0)

        return data_binned, bin_centres, (data_stat_down, data_stat_up)

    def plot_bkgs(self, cut_string, axes, variable, bins, data_binned, bin_centres, data_stat_down_up):
        """ self explanatory """

       
        bkg_frame = self.root_obj.mc_df_bkg.query(cut_string)
        print 'DEBUG: sumW nominal after nominal presel: {}'.format(np.sum(bkg_frame['weight']))

        #get norm factor
        var_to_plot_all_bkgs  = bkg_frame[variable].values
        var_weights_all_bkgs  = bkg_frame['weight'].values 
        sumw_all_bkgs, _      = np.histogram(var_to_plot_all_bkgs, bins=bins, weights=var_weights_all_bkgs)
        self.k_factor         = np.sum(data_binned)/np.sum(sumw_all_bkgs)
        sumw_all_bkgs        *= self.k_factor

        #Set up stat unc arrays to add to
        stat_down_all_bkgs = np.zeros(len(bins)-1)
        stat_up_all_bkgs   = np.zeros(len(bins)-1)

        #plot each proc and add up stat uncertainties
        for counter, bkg_proc in enumerate(self.root_obj.bkg_procs):
            proc_frame        = bkg_frame[bkg_frame.proc==bkg_proc]
            var_to_plot       = proc_frame[variable].values
            var_weights       = proc_frame['weight'].values 
            del proc_frame

            var_weights              *= self.k_factor
            sumw, _                   = np.histogram(var_to_plot, bins=bins, weights=var_weights)
            self.mc_totals[bkg_proc]  = sumw
            sumw2, _                  = np.histogram(var_to_plot, bins=bins, weights=var_weights**2)
            stat_down, stat_up        = self.poisson_interval(sumw, sumw2)
            stat_down_all_bkgs       += stat_down
            stat_up_all_bkgs         += stat_up

            print '--> Integral of hist: {}, for background proc {} is: {}'.format(variable, bkg_proc, np.sum(sumw))
            axes[0].hist(var_to_plot, bins=bins, label=bkg_proc, weights=var_weights, color=self.colours[counter], histtype='stepfilled')


        self.mc_stat_unc          = [sumw_all_bkgs-stat_down_all_bkgs, stat_up_all_bkgs-sumw_all_bkgs]
        self.mc_total             = sumw_all_bkgs

        data_bkg_ratio   = data_binned/sumw_all_bkgs
        axes[1].errorbar( bin_centres, data_bkg_ratio, yerr=[(data_binned-data_stat_down_up[0])/sumw_all_bkgs,(data_stat_down_up[1] - data_binned)/sumw_all_bkgs], fmt='o', ms=4, color='black', capsize=0, zorder=3)


    def plot_systematics(self, cut_string, axes, variable, bins, systematics, do_mva=True):
        """ self explanatory """
        
        #create and fill one Systematic object info for each syst FIXME FIXME (for each sample?)
        syst_objects = {}
        for syst_name in systematics:
            syst_dfs = self.relabel_syst_vars(syst_name, cut_string, plot_var=variable)
            print 'DEBUG: nominal vars '
            print self.root_obj.mc_df_bkg[['leadJetEn', 'dijetMass', 'leadJetEta']]
            print 'DEBUG: syst up vars '
            print syst_dfs['Up'][['leadJetEn', 'dijetMass', 'leadJetEta']]
            print 'DEBUG: syst down vars '
            print syst_dfs['Down'][['leadJetEn', 'dijetMass', 'leadJetEta']]
            for syst_type in syst_dfs.keys():
                syst_dfs[syst_type]['weight'] = syst_dfs[syst_type]['weight'].copy() * self.k_factor
                if do_mva: syst_dfs[syst_type][self.proc+'_mva'] = self.eval_bdt(self.clf, syst_dfs[syst_type], self.root_obj.train_vars)
            syst_objects[syst_name] = Systematic(syst_name, down_frame=syst_dfs['Down'], up_frame=syst_dfs['Up'])
            #print 'DEBUG: for syst: {}, MVA up/down diff are equal: {} !!'.format(syst_name, np.array_equal(syst_dfs['Up'][self.proc+'_mva'],syst_dfs['Down'][self.proc+'_mva']))
            #print 'DEBUG: for syst: {}, leadJetEn up/down diff are equal: {} !!'.format(syst_name, np.array_equal(syst_dfs['Up']['leadJetEn'],syst_dfs['Down']['leadJetEn']))
            del syst_dfs
            
        for syst_name, syst_obj in syst_objects.iteritems():
            print 'DEBUG: sys name: {}'.format(syst_name)
            for syst_type, i_frame in syst_obj.up_down_frames.iteritems():
                for bkg_proc in self.root_obj.bkg_procs:
                    proc_frame       = i_frame[i_frame.proc==bkg_proc]
                    var_to_plot      = proc_frame[variable].values
                    weight           = proc_frame['weight'].values
                    i_syst_binned, _ = np.histogram(var_to_plot, bins=bins, weights=weight)
 
                    #compare variation to the nominal for given sample and fill bin list
                    true_down_variations  = []
                    true_up_variations    = []
 
                    print 'bkg proc: {}'.format(bkg_proc)
                    print 'syst type: {}'.format(syst_type)
                    print 'i_syst binned:'
                    print i_syst_binned
                    print 'mc total for proc'
                    print  self.mc_totals[bkg_proc]
                    #compare the systematic change to the !nominal! bin entries for that proc.
                    #for ybin_syst, ybin_nominal in zip(i_syst_binned, self.mc_total):
                    for ybin_syst, ybin_nominal in zip(i_syst_binned, self.mc_totals[bkg_proc]):
                      if ybin_syst > ybin_nominal: 
                        true_up_variations.append(ybin_syst - ybin_nominal)
                        true_down_variations.append(0)
                      elif ybin_syst < ybin_nominal:
                        true_down_variations.append(ybin_nominal - ybin_syst)
                        true_up_variations.append(0)
                      else: #sometimes in low stat cases we get no change either way wrt nominal
                        true_up_variations.append(0)
                        true_down_variations.append(0)

                    print 'true down variations'
                    print true_down_variations
                    print 'true up variations'
                    print true_up_variations
 
                    if syst_type=='Down':
                        syst_obj.down_syst_binned[bkg_proc] = [np.asarray(true_down_variations), 
                                                               np.asarray(true_up_variations)]
                    else:
                        syst_obj.up_syst_binned[bkg_proc]   = [np.asarray(true_down_variations), 
                                                               np.asarray(true_up_variations)]

        #add all the up/down variations (separately) for each systematic in quadrature for each bin, 
        #for each proc 

        down_squares = [] 
        up_squares   = [] 

        for syst_name, syst_obj in syst_objects.iteritems():
            for bkg_proc in self.root_obj.bkg_procs:
                down_squares.append( syst_obj.down_syst_binned[bkg_proc][0]**2 )
                down_squares.append( syst_obj.up_syst_binned[bkg_proc][0]**2 )

                up_squares.append( syst_obj.down_syst_binned[bkg_proc][1]**2 )
                up_squares.append( syst_obj.up_syst_binned[bkg_proc][1]**2 )

        #print 'down squares'
        #print down_squares
        #print 'up squares'
        #print up_squares

        #now add up each bin that has been squared (will add element wise since np array)
        syst_merged_downs = np.zeros(len(bins)-1)
        syst_merged_ups   = np.zeros(len(bins)-1)

        for down_array in down_squares:
            syst_merged_downs += down_array
        for up_array in up_squares:
            syst_merged_ups   += up_array


        #combined with correpsonding stat error. note that if we are considering a sample set, the name and set attributes are identical now
        #NOTE: syst have already been squared above in prep for this step!

        #syst_merged_downs = np.sqrt( syst_merged_downs + self.mc_stat_uncs[sample_obj.name][0]**2) 
        #syst_merged_ups   = np.sqrt( syst_merged_ups   + self.mc_stat_uncs[sample_obj.name][1]**2) 

        #merged_syst_obj.merged_syst_stat_down  = syst_merged_downs
        #merged_syst_obj.merged_syst_stat_up    = syst_merged_ups

        #merged_downs = sample_obj.systematics['merged_systs'].merged_syst_stat_down
        #merged_ups   = sample_obj.systematics['merged_systs'].merged_syst_stat_up

        #FIXME add back in!
        merged_downs = np.sqrt( syst_merged_downs + self.mc_stat_unc[0]**2) 
        merged_ups   = np.sqrt( syst_merged_ups   + self.mc_stat_unc[1]**2) 

        #FIXME switch below and above fixme's if wanting stat only
        #merged_downs = np.sqrt( self.mc_stat_unc[0]**2) 
        #merged_ups   = np.sqrt( self.mc_stat_unc[1]**2) 

        print 'mc total'
        print self.mc_total
        print 'syst downs'
        print np.sqrt(syst_merged_downs)
        print 'syst ups'
        print np.sqrt(syst_merged_ups)
        print 'mc stat down: {}'.format(self.mc_stat_unc[0])
        print 'mc stat up: {}'.format(self.mc_stat_unc[1])

        up_yield   = self.mc_total + merged_ups
        #FIXME: fix this niche issue below with poiss err function
        up_yield[up_yield==np.inf] = 0
        down_yield = self.mc_total - merged_downs

        axes[0].fill_between(bins, list(down_yield)+[down_yield[-1]], list(up_yield)+[up_yield[-1]], alpha=0.3, step="post", color="lightcoral", lw=1, edgecolor='red', zorder=4, label='Simulation stat. $\oplus$ syst. unc.')

        #total_mc             = self.mc_total
        sigma_tot_ratio_down = merged_downs/self.mc_total
        sigma_tot_ratio_up   = merged_ups/self.mc_total
                
        ratio_down_excess    = np.ones(len(self.mc_total)) - sigma_tot_ratio_down
        ratio_up_excess      = np.ones(len(self.mc_total)) + sigma_tot_ratio_up
                
        #1. if we have no entries, the upper limit is inf and lower is nan
        #2. hence we set both to nan, so they aren't plot in the ratio plot
        #3  BUT if we have [nan, nan, 1 ,2 ,,, ] and/or [1, 2, ... nan, nan] 
        #   i.e. multiple Nan's at each end, then we have to set to Nan closest
        #   to the filled numbers to 1, such that error on the closest filled value
        #   doesn't mysteriously disappear
        #EDIT: gave up and did this dumb fix:

        #print ratio_up_excess
        #print ratio_down_excess
        #ratio_up_excess[ratio_up_excess==np.inf] = 1 
        #ratio_down_excess = np.nan_to_num(ratio_down_excess)
        #ratio_down_excess[ratio_down_excess==0] =1
        
        axes[1].fill_between(bins, list(ratio_down_excess)+[ratio_down_excess[-1]], list(ratio_up_excess)+[ratio_up_excess[-1]] , alpha=0.3, step="post", color="lightcoral", lw=1 , zorder=2)


    def relabel_syst_vars(self, syst_name, cut_string, plot_var, syst_types=['Up','Down']):
        """  
        Overwrite the nominal branches, with the analagous branch but with a systematic variation.
        For example if syst = jec, we may overwrite "leadJetPt" with "leadJetPt_JecUp/Down"
        Arguments
        ---------
        """  
             
        #import variables that may change with each systematic
        from syst_maps import syst_map 
        syst_dfs = {}
        print '\n\n'
        print 'DEBUG: reading systematic: {}'.format(syst_name)
        for ext in syst_types:
            print 'DEBUG: reading ext: {}'.format(ext)
            nominal_vars = syst_map[syst_name+ext]
            replacement_vars = [var_name+'_'+syst_name+ext for var_name in syst_map[syst_name+ext]] 

            #need to remove events asap else memory kills jobs. Hence apply preselection to syst vars before doing renaming stuff
            syst_cut_map = self.cut_map.copy()
            counter = 0
            print 'DEBUG: nominal cut map: {}'.format(self.cut_map)
	    print 'DEBUG: nominal vars: {}'.format(nominal_vars)
            print 'DEBUG: replacement vars: {}'.format(replacement_vars)

            #delete plot var from cut map (dont want to cut on variable we are plotting)
            if plot_var in syst_cut_map.keys(): del syst_cut_map[plot_var] 

            #replace nominal vars in the cut map with syst vars
            for var in nominal_vars:
                if var in syst_cut_map.keys():
                    print 'DEBUG: changing cut_var from {} to {}'.format(var, replacement_vars[counter])
                    del syst_cut_map[var]
                    syst_cut_map[replacement_vars[counter]] = self.cut_map[var] #if cut has syst variation for syst being considered. Format is syst_varies_name : cut (same as nominal)
                counter+=1
            print 'DEBUG: syst varied ({}) cut map: {}'.format(ext,syst_cut_map)

            #syst_cut_list = [var_name+cut for var_name,cut in syst_cut_map.iteritems()]
            syst_cut_list = []
            for var_name, cuts in syst_cut_map.iteritems():
                if len(cuts)>1: syst_cut_list += [var_name+cut for cut in cuts]
                else: syst_cut_list.append(var_name+cuts[0])
            syst_cut_string = ' and '.join(syst_cut_list)
            print 'DEBUG syst {} df sumW before cuts: {}'.format(ext, np.sum(self.root_obj.mc_df_bkg['weight']))
            df_copy = self.root_obj.mc_df_bkg.query(syst_cut_string)
             
            #relabel. Delete nominal column frst else pandas throws an exception. Then rename syst col name -> nominal col name
            for n_var, replacement_var in zip(nominal_vars,replacement_vars):
                #print 'replacing syst title: {} with nominal title: {}'.format(replacement_var, n_var)
                #FIXME: (CHECKME) check this for certain: e.g. lead jet mass has syst variation but is removed in earlier step since its not used anywhere in the analysis
                #FIXME  in general: if the variable isnt used in the classifier, we dont need to read in the syst varied versions of it (removed in manage_memory)
                if n_var in df_copy.columns:
                    del df_copy[n_var]
                    #df_copy.drop(labels=n_var, inplace=True)
                    df_copy.rename(columns={replacement_var : n_var}, inplace=True) #wont always be in col since removed unused vars!  
            syst_dfs[ext] = df_copy
            print 'DEBUG syst {} df sumW after cuts: {}'.format(ext, np.sum(df_copy['weight']))
            #print 'DEBUG: for syst: {}, after cuts, nominal and {} diff are equal: {} !!'.format(syst_name, ext, np.array_equal(df_copy['leadJetEn'],self.root_obj.mc_df_bkg['leadJetEn']))
        return syst_dfs


    def eval_mva(self, mva_config, output_tag):
        """ 
        evaluate score on whatever mva is passed, on whatever df is passed. Not used at the moment.
        """

        #FIXME fix DNN eval

        with open(mva_config, 'r') as mva_config_file:
            config            = yaml.load(mva_config_file)
            proc_to_model     = config['models']
            for proc, model in proc_to_model.iteritems():
                if proc not in output_tag: continue

                #for BDT - proc:[var list]. For DNN - proc:{var_type1:[var_list_type1], var_type2: [...], ...}
                if isinstance(model,dict):
                    object_vars     = proc_to_train_vars[proc]['object_vars']
                    flat_obj_vars   = [var for i_object in object_vars for var in i_object]
                    event_vars      = proc_to_train_vars[proc]['event_vars']
 
                    dnn_loaded = tag_obj.load_dnn(proc, model)
                    train_tag = model['architecture'].split('_model')[0]
                    tag_obj.eval_lstm(dnn_loaded, train_tag, root_obj, proc, object_vars, flat_obj_vars, event_vars)
 
                elif isinstance(model,str): 
                    clf = pickle.load(open('models/{}'.format(model), "rb"))
                    self.root_obj.mc_df_bkg[proc+'_mva'] = self.eval_bdt(clf, self.root_obj.mc_df_bkg, self.root_obj.train_vars)
                    self.root_obj.data_df[proc+'_mva'] =  self.eval_bdt(clf, self.root_obj.data_df, self.root_obj.train_vars)
                    self.clf = clf
                    self.proc = proc
                else: raise IOError('Did not get a classifier models in correct format in config')


    def eval_bdt(self, clf, df, train_vars):
        """ 
        evaluate score for BDT, on whatever df is passed
        """
        return clf.predict_proba(df[train_vars].values)[:,1:].ravel()   

    @classmethod
    def set_canv_style(self, axes, variable, bins, label='Work in progress', energy='13 TeV'):
        x_err = abs(bins[-1] - bins[-2])
        axes[0].set_ylabel('Events / {0:.2g}'.format(2*x_err) , size=14, ha='right', y=1)
        #if variable.norm: axes[0].set_ylabel('1/N dN/d(%s) /%.2f' % (variable.xlabel,x_err, ha='right', y=1)
        axes[0].text(0, 1.01, r'\textbf{CMS} %s'%label, ha='left', va='bottom', transform=axes[0].transAxes, size=14)
        #axes[0].text(1, 1.01, r'137 fb\textsuperscript{-1} (%s)'%(energy), ha='right', va='bottom', transform=axes[0].transAxes, size=14)
        axes[0].text(1, 1.01, r'41.5 fb\textsuperscript{-1} (%s)'%(energy), ha='right', va='bottom', transform=axes[0].transAxes, size=14)
       
        x_label = variable.replace("_", " ")
        axes[1].set_xlabel(x_label, size=14, ha='right', x=1)
        axes[1].set_ylim(0.52,1.48)
        axes[1].grid(True, linestyle='dotted')

        current_bottom, current_top = axes[0].get_ylim()
        #axes[0].set_ylim(top=current_top*1.45)
        axes[0].set_ylim(top=current_top*1.4)

        return axes

    def get_cut_string(self, var_to_plot):
        """
           Form a string of cuts to query samples. Take care to remove
           the cuts on the variable being plot
        """

        cut_dict = self.cut_map.copy()
        print 'cut dict is {}:'.format(cut_dict)
        if var_to_plot in cut_dict.keys(): del cut_dict[var_to_plot]
        #cut_list_non_null = [var_name+cut for var_name,cut in cut_dict.iteritems() if cut != '']
        cut_list_non_null = []
        for var_name, cuts in cut_dict.iteritems():
            if len(cuts)>1: cut_list_non_null += [var_name+cut for cut in cuts]
            else: cut_list_non_null.append(var_name+cuts[0])
        separator = ' and '
        cut_string = separator.join(cut_list_non_null)
        print 'cut string is {}:'.format(cut_string)
        return cut_string

    def poisson_interval(self, x, variance, level=0.68): #FIXME: dont copy, take this from PlottingUtils class
        neff = x**2/variance
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

class Systematic(object):
    """ 
       Object containing attributes related to systematic variations.
       One object is created per systematic uncertainty, for all bkg processes inclusively.
    """ 
        
    def __init__(self, name, up_frame=None, down_frame=None):
            self.name                  = name
            self.up_down_frames        = {'Up':up_frame, 'Down':down_frame}
            self.up_syst_binned        = {} #{proc1: [true_downs, true_ups], proc2: ...}
            self.down_syst_binned      = {} #{proc1: [true_downs, true_ups], proc2: ...}
            self.true_up               = {}
            self.true_down             = {}
