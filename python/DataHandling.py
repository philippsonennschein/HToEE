#data handling imports
import uproot as upr
import numpy as np
import pandas as pd
import os
from numpy import pi
from os import path, system
import yaml

import pickle
from pickle import load, dump

from variables import nominal_vars, gen_vars, gev_vars
from syst_maps import syst_map, weight_systs
from Utils import Utils


class SampleObject(object):
    '''
    Book-keeping class to store attributes of each sample. One object to be used per year, per sample -
    practically this means one per ROOT file

    :param proc_tag: physics process name for the sample being read in
    :type proc_tag: string
    :param year: year for the sample being read in 
    :type year: int
    :param file_name: name of the ROOT file being read in
    :type file_name: string
    :param tree_path: name of the TTree for the sample, contained in the ROOT TDirectory
    :type tree_path: string
    ''' 

    def __init__(self, proc_tag, year, file_name, tree_path):
        self.proc_tag  = proc_tag
        self.year      = year
        self.file_name = file_name
        self.tree_name = tree_path

class ROOTHelpers(object):
    """
    Class produce dataframes from any number of signal, background, or data processes 
    for multiple years of data taking

    :param out_tag: output string to be added to saved objectsm e.g. plots, dataframes, models, etc.
    :type out_tag: string
    :param mc_dir: directory where root files for simulation are held. Files for all years should be in this directory
    :type mc_dir: string
    :param mc_fnames: file names for simulated signal and background samples. Each has its own process key. 
                      Each process key has its own year keys. See any example training config for more detail.
    :type mc_fnames: dict
    :param data_dir: directory where root files for data are held. Files for all years should be in this directory
    :type: data_dir: string
    :param data_fnames: file names for Data samples. The key for all samples should be 'Data'.
                        This key has its own year keys. See any example training config for more detail.
    :type data_fnames: dict
    :param proc_to_tree_name: tree name split by physics process. Useful if trees have a process dependent name.
                              Should have one string per physics process name (per key)
    :type proc_to_tree_name: dict
    :param train_vars: variables to be used when training a classifier.
    :type train_vars: list
    :param vars_to_add: variables that are not in the input ROOT files, but will be added during sample processing.
                        Should become redundant when all variables are eventually in input files.
    :type vars_to_add: list
    :param presel_str: selection to be applied to all samples.
    :type presel_str: string
    :param read_systs: option to read in variables resulting from systematic variations e.g. JEC, JER, ...
    :type read_systs: bool
    """
  
    def __init__(self, out_tag, mc_dir, mc_fnames, data_dir, data_fnames, proc_to_tree_name, train_vars, vars_to_add, presel_str='', read_systs=False):
        self.years              = set()
        self.lumi_map           = {'2016':35.9, '2017':41.5, '2018':59.7}
        self.lumi_scale         = True
        self.XS_map             = {'ggH':48.58*5E-9, 'VBF':3.782*5E-9, 'DYMC': 6225.4, 'TT2L2Nu':86.61, 'TTSemiL':358.57} #all in pb. also have BR for signals
        self.eff_acc            = {'2016':{'ggH':0.3736318, 'VBF':0.3766126, 'DYMC':0.0599756, 'TT2L2Nu':0.0160820, 'TTSemiL':0.0000938}, #Pass8 from dumper, year dependent. update if selection changes
                                   '2017':{'ggH':0.3736318, 'VBF':0.3766126, 'DYMC':0.0628066, 'TT2L2Nu':0.0165499, 'TTSemiL':0.0000897},
                                   '2018':{'ggH':0.3736318, 'VBF':0.3766126, 'DYMC':0.0633516, 'TT2L2Nu':0.0167196, 'TTSemiL':0.0000873}
                                  }     
        #self.eff_acc            = {'ggH':0.3736318, 'VBF':0.3766126, 'DYMC':0.0628066, 'TT2L2Nu':0.0165516, 'TTSemiL':0.0000897} #Pass5 from dumper. update if selection changes
        #self.eff_acc            = {'ggH':0.4515728, 'VBF':0.4670169, 'DYMC':0.0748512, 'TT2L2Nu':0.0405483, 'TTSemiL':0.0003810} #from dumper. Pass2
        #self.eff_acc            = {'ggH':0.4014615, 'VBF':0.4044795, 'DYMC':0.0660393, 'TT2L2Nu':0.0176973, 'TTSemiL':0.0001307} #from dumper. Pass3

        self.out_tag            = out_tag
        self.mc_dir             = mc_dir #FIXME: remove '\' using if_ends_with()
        self.data_dir           = data_dir
       
        self.sig_procs          = []
        self.sig_objects        = []
        for proc, year_to_file in mc_fnames['sig'].items():
            if proc not in self.sig_procs: self.sig_procs.append(proc) 
            else: raise IOError('Multiple versions of same signal proc trying to be read')
            for year, file_name in year_to_file.iteritems():
                self.years.add(year)
                self.sig_objects.append( SampleObject(proc, year, file_name, proc_to_tree_name[proc]) )
 
        self.bkg_procs          = []
        self.bkg_objects        = []
        for proc, year_to_file in mc_fnames['bkg'].items():
            if proc not in self.bkg_procs: self.bkg_procs.append(proc) 
            else: raise IOError('Multiple versions of same background proc trying to be read')
            for year, file_name in year_to_file.iteritems():
                if year not in self.years:  raise IOError('Incompatible sample years')
                self.bkg_objects.append( SampleObject(proc, year, file_name, proc_to_tree_name[proc]) )

        self.data_objects       = []
        for proc, year_to_file in data_fnames.items():
            for year, file_name in year_to_file.iteritems():
                if year not in self.years:  raise IOError('Incompatible sample years')
                self.data_objects.append( SampleObject(proc, year, file_name, proc_to_tree_name[proc]) )

        self.mc_df_sig          = []
        self.mc_df_bkg          = []
        self.data_df            = []

        if vars_to_add is None: vars_to_add = {}
        self.vars_to_add        = vars_to_add

        self.data_vars = nominal_vars
        self.nominal_vars = nominal_vars[:] #return copy so self.data_vars isn't modified

        if read_systs: 
            for syst_type in syst_map.keys():
                self.nominal_vars += [var_name+'_'+syst_type for var_name in syst_map[syst_type]]
                #self.nominal_vars += [var_name+syst_type for var_name in syst_map[syst_type]]

        #if read_weight_systs:
            for weight_syst in weight_systs.keys():
                self.nominal_vars += [weight_syst+'_'+ext for ext in weight_systs[weight_syst]]

        missing_vars = [x for x in train_vars if x not in (nominal_vars+list(vars_to_add.keys()))]
        if len(missing_vars)!=0: raise IOError('Missing variables: {}'.format(missing_vars))
        self.train_vars         = train_vars
        self.cut_string         = presel_str

    def no_lumi_scale(self):
        """ 
        Toggle lumi scaling on/off. Useful when producing ROOT files for worksapces/fits.
        """

        self.lumi_scale=False

    def load_mc(self, sample_obj, bkg=False, reload_samples=False):
        """
        Try to load mc dataframe. If it doesn't exist, read in the root file.
        This should be used once per year, if reading in multiple years.

        Arguments
        ---------
        sample_obj: SampleObject
            an instance of the SampleObject class. Used to unpack attributes of the sample.
        bkg: bool
            indicates if the simulated sample being processed is background.
        reload_samples: bool
            force all samples to be read in from the input ROOT files, even if the dataframes already exist.
            Useful if changes to input files have been made.
        """

        try: 
            if reload_samples: raise IOError
            elif not bkg: self.mc_df_sig.append( self.load_df(self.mc_dir+'DataFrames/', sample_obj.proc_tag, sample_obj.year) )
            else: self.mc_df_bkg.append( self.load_df(self.mc_dir+'DataFrames/', sample_obj.proc_tag, sample_obj.year) )
        except IOError: 
            if not bkg: self.mc_df_sig.append( self.root_to_df(self.mc_dir, 
                                                               sample_obj.proc_tag,
                                                               sample_obj.file_name,
                                                               sample_obj.tree_name,
                                                               'sig', sample_obj.year
                                                              )
                                             )
            else: self.mc_df_bkg.append( self.root_to_df(self.mc_dir,
                                                         sample_obj.proc_tag,
                                                         sample_obj.file_name, 
                                                         sample_obj.tree_name,
                                                         'bkg', sample_obj.year
                                                        )
                                       )

    def load_data(self, sample_obj, reload_samples=False):
        """
        Try to load Data dataframe. If it doesn't exist, read in the root file.
        This should be used once per year, if reading in multiple years.

        Arguments
        ---------
        sample_obj: SampleObject
            an instance of the SampleObject class. Used to unpack attributes of the sample.
        reload_samples: bool
            force all samples to be read in from the input ROOT files, even if the dataframes alreadt exist.
            Useful if changes to input files have been made.
        """

        try: 
            if reload_samples: raise IOError
            else: self.data_df.append( self.load_df(self.data_dir+'DataFrames/', 'Data', sample_obj.year) )
        except IOError: 
            self.data_df.append( self.root_to_df(self.data_dir, sample_obj.proc_tag, sample_obj.file_name, sample_obj.tree_name, 'Data', sample_obj.year) )

    def load_df(self, df_dir, proc, year):
        """
        Load pandas dataframe, for a given process and year. Check all variables need for training are in columns.

        Arguments
        ---------
        df_dir: string
            directory where pandas dataframes for each process x year are kept. 
        proc: string
            physics process name for dataframe being read in
        year: int
            year for dataframe being read in
            
        Returns
        -------
        df: pandas dataframe that was read in.
        """

        #print 'loading {}{}_{}_df_{}.pkl'.format(df_dir, proc, self.out_tag, year)
        #df = pd.read_hdf('{}{}_{}_df_{}.h5'.format(df_dir, proc, self.out_tag, year))
        #df = pd.read_pickle('{}{}_{}_df_{}.pkl'.format(df_dir, proc, self.out_tag, year))
        df = pd.read_csv('{}{}_{}_df_{}.csv'.format(df_dir, proc, self.out_tag, year))
        missing_vars = [x for x in self.train_vars if x not in df.columns]
        if len(missing_vars)!=0: raise IOError('Missing variables in dataframe: {}. Reload with option -r and try again'.format(missing_vars))
        else: print('Sucessfully loaded DataFrame: {}{}_{}_df_{}.csv'.format(df_dir, proc, self.out_tag, year))

        return df    

    def root_to_df(self, file_dir, proc_tag, file_name, tree_name, flag, year):
        """
        Load a single root file for signal, background or data, for a given year. Apply any preselection.
        If reading in simulated samples, apply lumi scaling and read in gen-level variables too

        Arguments
        ---------
        file_dir: string
            directory that the ROOT file being read in is contained.
        proc_tag: string
            name of the physics process for the sample.
        file_name: string
            name of ROOT file being read in.
        tree_name: string
            name of TTree contrained within the ROOT file being read in.
        flag: string
            flag to indicate signal ('sig'), background ('bkg'), or Data ('Data')

        Returns
        -------
        df: pandas dataframe created from the input ROOT file
        """

        print('Reading {} file: {}, for year: {}'.format(proc_tag, file_dir+file_name, year))
        df_file = upr.open(file_dir+file_name)
        df_tree = df_file[tree_name]
        del df_file

        if flag == 'Data':
            #can cut on data here as dont need to run MC_norm
            df = df_tree.pandas.df(self.data_vars).query(self.cut_string)
        else:
            #cannot cut on sim now as need to run MC_norm and need sumGenW before selection!
            df = df_tree.pandas.df(self.nominal_vars+gen_vars)
            #NOTE: dont apply cuts yet as need to do MC norm!


        if len(self.cut_string)>0:
            if flag != 'Data':
                df = self.MC_norm(df, proc_tag, year)
                df = df.query(self.cut_string)
        else:
            if flag != 'Data':
                df = self.MC_norm(df, proc_tag, year)

        df = df.sample(frac=1).reset_index(drop=True)
        df = df.dropna()
        df['proc'] = proc_tag
        df['year'] = year

        print('Number of events in final dataframe: {}'.format(np.sum(df['weight'].values)))
        #save everything
        Utils.check_dir(file_dir+'DataFrames/') 
        df.to_csv('{}/{}_{}_df_{}.csv'.format(file_dir+'DataFrames', proc_tag, self.out_tag, year))
        print('Saved dataframe: {}/{}_{}_df_{}.csv'.format(file_dir+'DataFrames', proc_tag, self.out_tag, year))
        #df.to_pickle('{}/{}_{}_df_{}.pkl'.format(file_dir+'DataFrames', proc_tag, self.out_tag, year))
        #df.to_hdf('{}/{}_{}_df_{}.h5'.format(file_dir+'DataFrames', proc_tag, self.out_tag, year), 'df', mode='w', format='t')

        return df

    def MC_norm(self, df, proc_tag, year):
        """
        Apply normalisation to get expected number of events (perform before prelection)
        Note that "weight" column already has exp SFs applied i.e. weight = genWeight * centralObjWeight

        End up with a normalised weight: weight = (event_weight * xs * eff * acc)/sum(genWeights)

        Arguments 
        ---------
        :param df: pandas Dataframe
            dataframe for simulated signal or background with weights to be normalised
        :param proc_tag: string
            name of the physics process for the dataframe
        :param year: int
            year corresponding to the dataframe being read in

        Returns
        -------
        df: normalised dataframes
        """

        #Do scaling that used to happen in flashgg: XS * BR(for sig only) eff * acc
        sum_w_gen = np.sum(df['genWeight'].values)
        print 'scaling by {} by XS: {}'.format(proc_tag, self.XS_map[proc_tag])
        df['weight'] *= (self.XS_map[proc_tag]) 

        if self.lumi_scale: #should not be doing this in the final Tag producer
            print 'scaling by {} by Lumi: {} * 1000 /pb'.format(proc_tag, self.lumi_map[year])
            df['weight'] *= self.lumi_map[year]*1000 #lumi is added earlier but XS is in pb, so need * 1000

        print 'scaling by {} by eff*acc: {}'.format(proc_tag, self.eff_acc[year][proc_tag])
        df['weight'] *= (self.eff_acc[year][proc_tag])
        df['weight'] /= sum_w_gen

        print 'sumW for proc {}: {}'.format(proc_tag, np.sum(df['weight'].values))

        return df

    def apply_more_cuts(self, cut_string):
        """
        Apply some additional cuts, after nominal preselection (which was applied when file was read in)

        Arguments
        ---------
        cut_string: string
            set of cuts to be applied to all variables
        """

        self.mc_df_sig          = self.mc_df_sig.query(cut_string)
        self.mc_df_bkg          = self.mc_df_bkg.query(cut_string)
        self.data_df            = self.data_df.query(cut_string)

    def concat(self):
        """
        Concat sample types (sig, bkg, data) together, if more than one df in the associated sample type list.
        Years will also be automatically concatennated over. Could split this up into another function if desired
        but year info is only needed for lumi scaling.

        If the list is empty (not reading anything), leave it empty
        """

        if len(self.mc_df_sig) == 1: self.mc_df_sig = self.mc_df_sig[0]
        elif len(self.mc_df_sig) == 0: pass
        else: self.mc_df_sig = pd.concat(self.mc_df_sig)

        if len(self.mc_df_bkg) == 1: self.mc_df_bkg = self.mc_df_bkg[0] 
        elif len(self.mc_df_bkg) == 0: pass
        else: self.mc_df_bkg = pd.concat(self.mc_df_bkg)

        if len(self.data_df) == 1: self.data_df = self.data_df[0] 
        elif len(self.data_df) == 0 : pass
        else: self.data_df = pd.concat(self.data_df)
   
    def apply_pt_rew(self, bkg_proc, presel, norm=True):
        """
        Derive a reweighting for a single bkg process in a m(ee) control region around the Z-peak, in bins on pT(ee),
        to map bkg process to Data. SFs for each year are derived separately. Then apply to SR.

        Note that if norming, we must apply the flat k-factor in the SR after pt-reweighting, since the latter
        doesn't preserve the normalisation

        Arguments
        ---------
        bkg_proc: string
            name of the physics process we want to re-weight. Nominally this is for Drell-Yan.
        year: float
            year to be re-weighted (perform this separately for each year)
        presel: string
            preselection to apply to go from the CR -> SR
        norm: bool
            normalise the simulated background to data. Results in a shape-only correction
        """

        #derive SFs
        year_to_scale_factors = {}
        for year in self.years:

            pt_bins = np.linspace(0,180,101)
            bkg_df = self.mc_df_bkg.query('proc=="{}" and year=="{}" and dielectronMass>80 and dielectronMass<100'.format(bkg_proc,year))
            data_df = self.data_df.query('year=="{}" and dielectronMass>80 and dielectronMass<100'.format(year))       

            #FIXME: here only norming DY events to data which is mainly DY...
            if norm: bkg_df['weight'] *= (np.sum(data_df['weight'])/np.sum(bkg_df['weight']))

            bkg_pt_binned, _ = np.histogram(bkg_df['dielectronPt'], bins=pt_bins, weights=bkg_df['weight'])
            data_pt_binned, bin_edges = np.histogram(data_df['dielectronPt'], bins=pt_bins)
            year_to_scale_factors[year] = data_pt_binned/bkg_pt_binned

        print 'DEBUG: scale factors for year are', year_to_scale_factors

        #put samples into SR phase space. 
        self.apply_more_cuts(presel)

        #apply pt-reweighting inSR
        #self.mc_df_bkg['weight'] = self.mc_df_bkg.apply(self.pt_reweight_helper, axis=1, args=[bkg_proc, bin_edges, year_to_scale_factors])

        #do not get an inedxing error when we try do index = length + 1, since len SFs = len bin_edges -1
        scaled_bkg_dfs = []
        for year in self.years:
            df_year_i = self.mc_df_bkg.query('year=="{}"'.format(year))
            scale_factors = year_to_scale_factors[year]
            for i_bin in range(len(scale_factors)):
                temp_df = df_year_i[df_year_i.dielectronPt > bin_edges[i_bin]] 
                temp_df = temp_df[temp_df.dielectronPt < bin_edges[i_bin+1]] 
                temp_df['weight'] *= scale_factors[i_bin]
                scaled_bkg_dfs.append(temp_df)  
            outside_rng_df = df_year_i[df_year_i.dielectronPt > bin_edges[-1]]
            scaled_bkg_dfs.append(outside_rng_df)
        
        self.mc_df_bkg = pd.concat(scaled_bkg_dfs)
        del scaled_bkg_dfs

        #use numpy select to apply year dependent k-factor to events. Much quicker than using pandas apply! can probs do it the same way as above for clarity but whatev
        if norm:
            normed_year_dfs_bkg = []
            for year in self.years:
                    df_bkg_year_i = self.mc_df_bkg.query('year=="{}"'.format(year))
                    df_data_year_i = self.data_df.query('year=="{}"'.format(year))
                    norm_factor = np.sum(df_data_year_i['weight']) / np.sum(df_bkg_year_i['weight'])
                    df_bkg_year_i['weight'] *= norm_factor
                    normed_year_dfs_bkg.append(df_bkg_year_i)

            self.mc_df_bkg = pd.concat( normed_year_dfs_bkg )

        for year in self.years:
            self.save_modified_dfs(year)


    def pt_reweight_helper(self, row, bkg_proc, bin_edges, scale_factors):
        """
        Function called in pandas apply() function, looping over rows and testing conditions. Can be called for
        single or double differential re-weighting.

        Tests which pT a bkg proc is, and if it is the proc to reweight, before
        applying a pT dependent scale factor to apply (derived from CR)
        
        If dielectron pT is above the max pT bin, just return the nominal weight (very small num of events)

        Arguments
        ---------
        row: pandas Series
            a single row of the dataframe being looped through. Automatically generated as first argument
            when using pandas apply()
        bkg_proc: string
            name of the physics process we want to re-weight. Nominally this is for Drell-Yan.
        bin_edges: numpy array
            edges of each pT bin, in which the re-weighting is applied
        scale_factors: numpy array
            scale factors to be applied in each pT bin

        Returns
        -------
        row['weight'] * scale-factor : float of the (modified) MC weight for a single event/dataframe row
        """

        if row['proc']==bkg_proc and row['dielectronPt']<bin_edges[-1]:
            rew_factors = scale_factors[row['year']]
            for i_bin in range(len(bin_edges)):
                if (row['dielectronPt'] > bin_edges[i_bin]) and (row['dielectronPt'] < bin_edges[i_bin+1]):
                    return row['weight'] * rew_factors[i_bin] 
        else:
            return row['weight'] 


    def pt_njet_reweight(self, bkg_proc, year, presel, norm_first=True):
        """
        Derive a reweighting for a single bkg process in a m(ee) control region around the Z-peak, double differentially 
        in bins on pT(ee) and nJets, to map bkg process to Data. Then apply this in the signal region.

        Arguments
        ---------
        bkg_proc: string
            name of the physics process we want to re-weight. Nominally this is for Drell-Yan.
        year: float
            year to be re-weighted (perform this separately for each year)
        presel: string
            preselection to apply to go from the CR -> SR
        norm_first: bool
            normalise the simulated background to data. Results in a shape-only correction
        """
        #FIXME: function is out of date with rest of code

        #can remove this once nJets is put in ntuples from dumper
        outcomes_mc_bkg = [ self.mc_df_bkg['leadJetPt'].lt(0),
                            self.mc_df_bkg['leadJetPt'].gt(0) & self.mc_df_bkg['subleadJetPt'].lt(0), 
                            self.mc_df_bkg['leadJetPt'].gt(0) & self.mc_df_bkg['subleadJetPt'].gt(0)
                          ]

        outcomes_data   = [ self.data_df['leadJetPt'].lt(0),
                            self.data_df['leadJetPt'].gt(0) & self.data_df['subleadJetPt'].lt(0), 
                            self.data_df['leadJetPt'].gt(0) & self.data_df['subleadJetPt'].gt(0)
                          ]
        jets    = [0, 1, 2] # 2 really means nJet >= 2

        self.mc_df_bkg['nJets'] = np.select(outcomes_mc_bkg, jets) 
        self.data_df['nJets'] = np.select(outcomes_data, jets) 

        #apply re-weighting
        pt_bins = np.linspace(0,200,101)
        jet_bins = [0,1,2]
        n_jets_to_sfs_map = {}


        #derive pt and njet based SFs
        for n_jets in jet_bins:
            if not n_jets==jet_bins[-1]: 
                bkg_df = self.mc_df_bkg.query('proc=="{}" and year=="{}" and dielectronMass>80 and dielectronMass<100 and nJets=={}'.format(bkg_proc,year, n_jets))
                data_df = self.data_df.query('year=="{}" and dielectronMass>80 and dielectronMass<100 and nJets=={}'.format(year,n_jets))       
            else: 
                bkg_df = self.mc_df_bkg.query('proc=="{}" and year=="{}" and dielectronMass>80 and dielectronMass<100 and nJets>={}'.format(bkg_proc,year, n_jets))
                data_df = self.data_df.query('year=="{}" and dielectronMass>80 and dielectronMass<100 and nJets>={}'.format(year,n_jets))       

            if norm_first:
                CR_norm_i_jet_bin = (np.sum(data_df['weight'])/np.sum(bkg_df['weight']))
                bkg_df['weight'] *= CR_norm_i_jet_bin

            bkg_pt_binned, _ = np.histogram(bkg_df['dielectronPt'], bins=pt_bins, weights=bkg_df['weight'])
            data_pt_binned, bin_edges = np.histogram(data_df['dielectronPt'], bins=pt_bins)
            n_jets_to_sfs_map[n_jets] = data_pt_binned/bkg_pt_binned

        #now apply the proc targeting selection on all dfs, and re-save. Then apply derived SFs
        self.apply_more_cuts(presel)
        if norm_first:
            SR_i_jet_to_norm = {}
            for n_jets in jet_bins:
                SR_i_jet_to_norm[n_jets] = np.sum(self.data_df['weight']) / np.sum(self.mc_df_bkg['weight'])
            self.mc_df_bkg['weight'] = self.mc_df_bkg.apply(self.pt_njet_reweight_helper, axis=1, args=[bkg_proc, year, bin_edges, n_jets_to_sfs_map, True, SR_i_jet_to_norm])

        else: self.mc_df_bkg['weight'] = self.mc_df_bkg.apply(self.pt_njet_reweight_helper, axis=1, args=[bkg_proc, year, bin_edges, n_jets_to_sfs_map, True, None])
        self.save_modified_dfs(year)
         


    def save_modified_dfs(self,year):
        """
        Save dataframes again. Useful if modifications were made since reading in and saving e.g. pT reweighting or applying more selection
        (or both).

        Arguments
        ---------
        year: int
            year for which all samples being saved correspond to
        """

        print 'saving modified dataframes...'
        for sig_proc in self.sig_procs:
            sig_df = self.mc_df_sig[np.logical_and(self.mc_df_sig.proc==sig_proc, self.mc_df_sig.year==year)]
            #sig_df.to_hdf('{}/{}_{}_df_{}.h5'.format(self.mc_dir+'DataFrames', sig_proc, self.out_tag, year), 'df', mode='w', format='t')
            #sig_df.to_pickle('{}/{}_{}_df_{}.pkl'.format(self.mc_dir+'DataFrames', sig_proc, self.out_tag, year))
            sig_df.to_csv('{}/{}_{}_df_{}.csv'.format(self.mc_dir+'DataFrames', sig_proc, self.out_tag, year))
            print('saved dataframe: {}/{}_{}_df_{}.csv'.format(self.mc_dir+'DataFrames', sig_proc, self.out_tag, year))

        for bkg_proc in self.bkg_procs:
            bkg_df = self.mc_df_bkg[np.logical_and(self.mc_df_bkg.proc==bkg_proc,self.mc_df_bkg.year==year)]
            #bkg_df.to_hdf('{}/{}_{}_df_{}.h5'.format(self.mc_dir+'DataFrames', bkg_proc, self.out_tag, year), 'df', mode='w', format='t')
            #bkg_df.to_pickle('{}/{}_{}_df_{}.pkl'.format(self.mc_dir+'DataFrames', bkg_proc, self.out_tag, year))
            bkg_df.to_csv('{}/{}_{}_df_{}.csv'.format(self.mc_dir+'DataFrames', bkg_proc, self.out_tag, year))
            print('saved dataframe: {}/{}_{}_df_{}.csv'.format(self.mc_dir+'DataFrames', bkg_proc, self.out_tag, year))

        data_df = self.data_df[self.data_df.year==year]
        #data_df.to_hdf('{}/{}_{}_df_{}.h5'.format(self.data_dir+'DataFrames', 'Data', self.out_tag, year), 'df', mode='w', format='t')
        #data_df.to_pickle('{}/{}_{}_df_{}.pkl'.format(self.data_dir+'DataFrames', 'Data', self.out_tag, year))
        data_df.to_csv('{}/{}_{}_df_{}.csv'.format(self.data_dir+'DataFrames', 'Data', self.out_tag, year))
        print('saved dataframe: {}/{}_{}_df_{}.csv'.format(self.data_dir+'DataFrames', 'Data', self.out_tag, year))

