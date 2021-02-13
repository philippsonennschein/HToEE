import numpy as np
import yaml
import pandas as pd
import pickle
from os import path, system
import root_pandas
import matplotlib.pyplot as plt

from variables import syst_vars

class taggerBase(object):
    """
    Class to handle the production of analysis categories based off MVA scores, and some tag sequence (priority)
    Also handles systematic variations.

    :param tag_seq: sequence of analysis categories targetting each process. Has format [highest_priority, ..., lowest_priority]
    :type tag_teq: list
    :param true_procs: list of the process being considered. May include Data, if not handling systematics
    :type true_procs: list
    :param combined_df: DataFrame containing al information needed for tagging, including output scores of any MVAs
    :type combined_df: pandas DataFrame
    :param syst_name: name of the systematic being read in. Must also be an "Up" and "Down" type variation
    :type syst_name: str
    :param tree_vars: list of variables to be stored in all output trees
    :type tree_vars: list
    """

    def __init__(self, tag_seq, true_procs, combined_df, syst_name=None):
        self.syst_name   = syst_name
        self.tag_seq     = tag_seq 
        self.true_procs  = true_procs 
        self.combined_df = combined_df
        self.syst_name   = syst_name

        self.tree_vars   = ['dZ', 'CMS_hgg_mass', 'weight']
        self.default_tag = -999


    def relabel_syst_vars(self, del_other_systs=False):
        """
        Overwrite the nominal branches, with the analagous branch but with a systematic variation.
        For example if syst = JecUp, we may overwrite "leadJetPt" with "leadJetPtJecUp"

        Expect this function to evolve, since hope to have systematics map to variables that change with them e.g. {Jec: [leadPt, leadPhi..., ]}

        Arguments
        ---------
        del_other_systs: bool
            delete other columns of systematics we read in, but will not use. Useful for memory management.
        """

        #import variables that may change with each systematic
        from variables import syst_vars

        #relabel. Delete nominal column frst else pandas throws an exception
        for syst_var in syst_vars:
            del self.combined_df[syst_var]
            self.combined_df.rename(columns={syst_var+self.syst_name : syst_var}, inplace=True)

        if del_other_systs: pass

    def eval_bdt(self, proc, model, train_vars):
        """
        Evaluate a given BDT corresponding to categorisation targeting a given process.
        Add the resulting scores to self.combined_df

        Arguments
        ---------
        model: str
            file path to BDT model
        proc: str
            process name that the BDT develops categories to target
        train_vars: list
            list of variables that were used in the training of the BDT, in the same order!
        """

        print 'evaluating BDT: {}'.format(model)
        clf = pickle.load(open('models/{}'.format(model), "rb"))
        self.combined_df[proc+'_mva'] = clf.predict_proba(self.combined_df[train_vars].values)[:,1:].ravel()
   

    def eval_dnn(self): pass
        #be careful to normalise vars etc. Will need to import stuff from DNN class and use it here (as we did it the previous tagger script)
        #print 'evaluating NN: {}'.format(model)

    def decide_tag(self, tag_preselection, tag_boundaries):
        """
        Consider an event for each tag, for all process being targeted.

        Fill each DataFrame column with tag info e.g. 0, 1, ..., or -999, for both VBF and ggH
        """

        #set up tag boundaries for each process being targeted i.e. create mask
        for tag in self.tag_seq:
            presel     = tag_preselection[tag]
            tag_bounds = tag_boundaries[tag].values()
            tag_masks = []
            for i_bound in range(len(tag_bounds)): #get indexes
                if i_bound==0: #first/tightest bound, tag 0
                    tag_masks.append( presel[0] & self.combined_df['{}_mva'.format(tag)].gt(tag_bounds[i_bound]) )
                else: #intermed bounds
                    tag_masks.append( presel[0] & self.combined_df['{}_mva'.format(tag)].lt(tag_bounds[i_bound-1]) & 
                                      self.combined_df['{}_mva'.format(tag)].gt(tag_bounds[i_bound])
                                    )

            mask_key = [icat for icat in range(len(tag_bounds))]

            #test event for categories targeting given proc
            self.combined_df['{}_analysis_tag'.format(tag)] = np.select(tag_masks, mask_key, default=self.default_tag)


    def decide_priority(self):
        """
        Decide which tag an event should preferentially fall in, based on the tag priority
        """

        self.combined_df['priority_tag'] = self.default_tag
        for tag_code, tag in enumerate(self.tag_seq):
            #if an event has tag satisifed, and hasn't already been assigned a tag, set tag
            mask = [ self.combined_df['{}_analysis_tag'.format(tag)].ne(self.default_tag) & self.combined_df['priority_tag'].eq(self.default_tag) ]
            key  = [ tag_code ]
            current_priority = self.combined_df['priority_tag']
            self.combined_df['priority_tag'] = np.select(mask, key, default=current_priority)

        #NOTE: np select cant handle string and float comparisons. Therefore need to encode proc to number (above) and decode again later
        for tag_code, tag_name in enumerate(self.tag_seq):
            self.combined_df['priority_tag'].replace(tag_code, tag_name, inplace=True)
        self.combined_df['priority_tag'].replace(self.default_tag, 'NOTAG', inplace=True)
         
        #debug_cols = ['dielectronMass', 'leadElectronPtOvM', 'subleadElectronPtOvM', 'dijetMass', 'leadJetPt', 'subleadJetPt', 'ggH_mva', 'VBF_mva', 'VBF_analysis_tag', 'ggH_analysis_tag', 'priority_tag']
        #print self.combined_df[debug_cols]

        #other option is to use pandas apply with (priv) function, but its v slow on large dfs
        #def tag_priority_helper(row):
        #    """
        #    go through sequence in order of priority and return highest priority tag
        #    """
        #    
        #    priority_tag = 'NOTAG'
        #    for tag in self.tag_seq:
        #        if row[tag+'_analysis_tag'] != self.default_tag : priority_tag = tag
        #    return priority_tag
        #self.combined_df['priority_tag'] = self.combined_df.apply(tag_priority_tree, axis=1)

    def get_tree_names(self, tag_boundaries):
        """
        Automatically get branch names, for each true proc.
        """

        #FIXME: account for data in systs!
        #set tree names for output files with format: <true_proc>_<misc_info>_<tag>_cat<tag_num>
        branch_names = {}
        for true_proc in self.true_procs: 
            branch_names[true_proc] = []
            for target_proc in self.tag_seq:  #for all events with proc = true_proc, which tag do they fall into?
                for i_tag in range(len(tag_boundaries[target_proc].values())):#for each tag corresponding to the category we target, which events go in which tag
                     if self.syst_name is not None: branch_names[true_proc].append('{}_125_13TeV_{}cat{}_{}01sigma'.format(true_proc.lower(), target_proc.lower(), i_tag, self.syst_name))
                     else:      
                         if true_proc is not 'Data': branch_names[true_proc].append('{}_125_13TeV_{}cat{}'.format(true_proc.lower(), target_proc.lower(), i_tag ))
                         else: branch_names[true_proc].append('{}_13TeV_{}cat{}'.format(true_proc, target_proc.lower(), i_tag ))

        if not path.isdir('output_trees/'):
            print 'making directory: {}'.format('output_trees/')
            system('mkdir -p %s' %'output_trees/')

        return branch_names

    def set_tree_names(self, tag_boundaries):
        """
        Use the tag priority and tag number (for each process) to decide on a final tag
        """

        mask = []
        keys = []
        for proc in self.true_procs:

            if proc is not 'Data':
                proc_label = proc.lower()#data needs captial D in tree name
                label = '125_13TeV'
            else:
                proc_label = proc 
                label = '13TeV'

            for tag in self.tag_seq:
                tag_bounds = tag_boundaries[tag].values()
                for i_tag in range(len(tag_bounds)): #get indexes
                    mask.append(self.combined_df['proc'].eq(proc) & self.combined_df['priority_tag'].eq(tag) & self.combined_df['{}_analysis_tag'.format(tag)].eq(i_tag))
                    if self.syst_name is not None: keys.append('{}_{}_{}cat{}_{}01sigma'.format(proc_label,label,tag.lower(),i_tag,self.syst_name))
                    else: keys.append('{}_{}_{}cat{}'.format(proc_label,label,tag.lower(),i_tag))

        self.combined_df['tree_name'] = np.select(mask, keys, default='NOTAG')
        self.combined_df['dZ'] = float(0.)
        self.combined_df['CMS_hgg_mass'] = self.combined_df['dielectronMass'] 


        debug_cols = ['dielectronMass', 'leadElectronPtOvM', 'subleadElectronPtOvM', 'dijetMass', 'leadJetPt', 'subleadJetPt', 'ggH_mva', 'VBF_mva', 'VBF_analysis_tag', 'ggH_analysis_tag', 'priority_tag', 'proc', 'tree_name']
        print self.combined_df[debug_cols]


    def fill_trees(self, branch_names):
        #have to save individual trees as root files (fn=bn), then hadd over single proc on the command line, to get one proc file with all tag trees
        debug_cols = ['dielectronMass', 'leadElectronPtOvM', 'subleadElectronPtOvM', 'dijetMass', 'leadJetPt', 'subleadJetPt', 'ggH_mva', 'VBF_mva', 'VBF_analysis_tag', 'ggH_analysis_tag', 'priority_tag', 'proc', 'tree_name']

        for proc in self.true_procs:
            selected_df = self.combined_df[self.combined_df.proc==proc]
            for bn in branch_names[proc]:
                print bn
                branch_selected_df = selected_df[selected_df.tree_name==bn]
                print branch_selected_df[debug_cols].head(10)
                root_pandas.to_root(branch_selected_df[self.tree_vars], 'output_trees/{}.root'.format(bn), key=bn)
                print
    
    def plot_matrix(self, branch_names, output_tag):
        """
        create categorisation matrix of (rows=reco cats, cols=procs), using branch name of each event
        """
    
        def plot_helper(matrix, norm, branch_names, output_tag):
            """
            draw categorisation matrix; make it look nice
            """

            plt.set_cmap('Blues')
            fig = plt.figure()
            axes = fig.gca()
            mat = axes.matshow(matrix)
        
            for i in range(0,matrix.shape[0]):
                for j in range(0,matrix.shape[1]):
                    c = matrix[i,j]
                    if np.abs(c)>=1:
                        axes.text(j,i,'{:.0f}'.format(c),fontdict={'size': 8},va='center',ha='center')
        
            axes.set_xticks(np.arange(2))
            axes.set_xticklabels(self.true_procs,rotation='vertical')
            axes.set_yticks(np.arange(len(branch_names['VBF'])))
            categories = [cat.split('_')[-1] for cat in branch_names['VBF']]
            axes.set_yticklabels(categories)
            #axes.xaxis.tick_top()
            cbar = fig.colorbar(mat)
            cbar.set_label(r'Category Composition (%)')
            file_dir = './plotting/plots/{}'.format(output_tag)
            if not path.isdir(file_dir):
                print 'making directory: {}'.format(file_dir)
                system('mkdir -p %s' %file_dir)
            fig.savefig('{}/{}_purity_matrix_{}_norm.pdf'.format(file_dir, output_tag, norm), bbox_inches='tight')
            plt.close()

        confusion_matrix = []
        for x_proc in self.tag_seq:
            proc_selected_df = self.combined_df[self.combined_df.proc==x_proc]
            matrix_col = []
            for reco_branch in branch_names[x_proc]:
                proc_reco_selected_sumw = np.sum(proc_selected_df[proc_selected_df.tree_name==reco_branch]['weight'])
                matrix_col.append(proc_reco_selected_sumw)
            confusion_matrix.append(np.array(matrix_col))
        confusion_matrix = np.stack(confusion_matrix)
        confusion_matrix = np.transpose(confusion_matrix)
        matrix_col_norm  = 100 * confusion_matrix / (confusion_matrix.sum(axis=0).reshape(1,-1)) #broadcast
        matrix_row_norm  = 100 * confusion_matrix / (confusion_matrix.sum(axis=1).reshape(-1,1)) #broadcast
                 
        plot_helper(matrix_col_norm, 'col', branch_names, output_tag)
        plot_helper(matrix_row_norm, 'row', branch_names, output_tag)


