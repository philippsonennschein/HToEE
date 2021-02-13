import argparse
import numpy as np
import yaml
import pickle
import pandas as pd
import root_pandas
from os import path, system
import matplotlib.pyplot as plt

from DataHandling import ROOTHelpers

def main(options):
    
    with open(options.config, 'r') as config_file:
        config             = yaml.load(config_file)
        output_tag         = config['output_tag']

        mc_dir             = config['mc_file_dir']
        mc_fnames          = config['mc_file_names']
  
        #data not needed yet, but stil specify in the config for compatibility with constructor
        data_dir           = config['data_file_dir']
        data_fnames        = config['data_file_names']

        proc_to_tree_name  = config['proc_to_tree_name']

        proc_to_train_vars = config['train_vars']
        all_train_vars     = [item for sublist in proc_to_train_vars.values() for item in sublist]

        vars_to_add        = config['vars_to_add']


                                           #Data handling stuff#
        #apply loosest selection (ggh) first, else memory requirements are ridiculous. Fine to do this since all cuts all looser than VBF (not removing events with higher priority)
        loosest_selection = 'dielectronMass > 110 and dielectronMass < 150'
 
        #load the mc dataframe for all years. Do not apply any specific preselection
        root_obj = ROOTHelpers(output_tag, mc_dir, mc_fnames, data_dir, data_fnames, proc_to_tree_name, all_train_vars, vars_to_add, loosest_selection) 
        root_obj.no_lumi_scale()
        for sig_obj in root_obj.sig_objects:
            root_obj.load_mc(sig_obj, reload_samples=options.reload_samples)
        if options.data_as_bkg:
            for data_obj in root_obj.data_objects:
                root_obj.load_data(data_obj, reload_samples=options.reload_samples)
        else:
            for bkg_obj in root_obj.bkg_objects:
                root_obj.load_mc(bkg_obj, bkg=True, reload_samples=options.reload_samples)
        root_obj.concat()


                                       #Tag sequence stuff#
    if options.data_as_bkg: combined_df = pd.concat([root_obj.mc_df_sig, root_obj.data_df])
    else: combined_df = pd.concat([root_obj.mc_df_sig, root_obj.mc_df_bkg])
    del root_obj

    #decide sequence of tags and specify preselection for use with numpy.select:
    tag_sequence      = ['VBF','ggH']
    true_procs        = ['VBF','ggH','Data']
    tag_preselection  = {'VBF': [combined_df['dielectronMass'].gt(110) & 
                                     combined_df['dielectronMass'].lt(150) &
                                     combined_df['leadElectronPtOvM'].gt(0.333) &
                                     combined_df['subleadElectronPtOvM'].gt(0.25) &
                                     combined_df['dijetMass'].gt(350) &
                                     combined_df['leadJetPt'].gt(40) &
                                     combined_df['subleadJetPt'].gt(30)
                                    ],
                            'ggH':  [combined_df['dielectronMass'].gt(110) & 
                                     combined_df['dielectronMass'].lt(150) &
                                     combined_df['leadElectronPtOvM'].gt(0.333) &
                                     combined_df['subleadElectronPtOvM'].gt(0.25)
                                    ]       
                            }

    #get number models and tag boundaries from config
    with open(options.bdt_config, 'r') as bdt_config_file:
        config            = yaml.load(bdt_config_file)
        proc_to_model     = config['models']
        tag_boundaries    = config['boundaries']

        #evaluate MVA scores used in categorisation
        for proc, model in proc_to_model.iteritems():
            print 'evaluating classifier: {}'.format(model)
            clf = pickle.load(open('models/{}'.format(model), "rb"))
            train_vars = proc_to_train_vars[proc]
            combined_df[proc+'_bdt'] = clf.predict_proba(combined_df[train_vars].values)[:,1:].ravel()
       
        #set up tag boundaries for each process being targeted
        for tag in tag_sequence:
            presel     = tag_preselection[tag]
            tag_bounds = tag_boundaries[tag].values()
            tag_masks = []
            for i_bound in range(len(tag_bounds)): #get indexes
                if i_bound==0: #first bound, tag 0
                    tag_masks.append( presel[0] & combined_df['{}_bdt'.format(tag)].gt(tag_bounds[i_bound]) )
                else: #intermed bounds
                    tag_masks.append( presel[0] & combined_df['{}_bdt'.format(tag)].lt(tag_bounds[i_bound-1]) & 
                                      combined_df['{}_bdt'.format(tag)].gt(tag_bounds[i_bound])
                                    )
            mask_key     = [icat for icat in range(len(tag_bounds))]
            #fill column with tag info e.g. 0, 1, ..., or -999
            combined_df['{}_analysis_tag'.format(tag)] = np.select(tag_masks, mask_key, default=-999)


        # deduce tag priority: if two or more tags satisfied then set final tag to highest priority tag. make this non hardcoded i.e. compare proc in position 1 to all lower prioty positions. then compare proc in pos 2 ...
        tag_priority_filter = [ combined_df['VBF_analysis_tag'].ne(-999) & combined_df['ggH_analysis_tag'].ne(-999), # 1) if both filled...
                                combined_df['VBF_analysis_tag'].ne(-999) & combined_df['ggH_analysis_tag'].eq(-999), # 2) if VBF filled and ggH not, take VBF
                                combined_df['VBF_analysis_tag'].eq(-999) & combined_df['ggH_analysis_tag'].ne(-999), # 3) if ggH filled and VBF not, take ggH
                              ]

        tag_priority_key    = [ 'VBF', #1) take VBF
                                'VBF', #2) take VBF
                                'ggH', #3) take ggH
                              ]
        combined_df['priority_tag'] = np.select(tag_priority_filter, tag_priority_key, default='NOTAG') # 4) keep -999 i.e. NOTAG

        #use the (proc + tag number) and tag priority to assign the correct tree name to each event
        combined_df['tree_name'] = combined_df.apply(assign_tree, axis=1, args=[true_procs])

        debug_vars = ['proc', 'VBF_analysis_tag', 'ggH_analysis_tag', 'priority_tag']
        print combined_df[debug_vars+['tree_name']] #checking logic is correct

        # FIXME: use numpy select instead of pandas apply. Make it less hardcoded too!
        #for true_proc in tag_sequence+['Data']: 
        #    #isolate true proc
        #    true_proc_df = combined_df[combined_df.proc==true_proc.lower()]
        #    #how much true proc landed in each of our analysis cats?
        #    for target_proc in tag_sequence:  #for all events that got the proc tag, which tag did they fall into?
        #        true_proc_target_proc_df = true_proc_df[true_proc_df.priority_tag==target_proc]
        #        for i_tag in range(len(proc_to_tags[target_proc].values())):#for each tag corresponding to the category we target, which events go in which tag
        #             true_procs_target_proc_tag_i  = true_proc_target_proc_df[true_proc_target_proc_df['{}_analysis_tag'.format(target_proc)].eq(i_tag)]
        #                 
        #             branch_name = '{}_125_13TeV_{}cat{} :'.format(true_proc.lower(), target_proc.lower(), i_tag )
        #             print true_procs_target_proc_tag_i[['dipho_mass', 'dipho_leadIDMVA', 'dipho_subleadIDMVA', 'dipho_lead_ptoM', 'dipho_sublead_ptoM', 'dijet_Mjj', 'dijet_LeadJPt', 'dijet_SubJPt', 'ggH_bdt', 'VBF_bdt', 'VBF_analysis_tag', 'ggH_analysis_tag', 'priority_tag']].head(10)
        #             print branch_name
       
        #some debug checks:
        #debug_procs = ['dipho_mass', 'dipho_leadIDMVA', 'dipho_subleadIDMVA', 'dipho_lead_ptoM', 'dipho_sublead_ptoM', 'dijet_Mjj', 'dijet_LeadJPt', 'dijet_SubJPt', 'ggH_bdt', 'VBF_bdt', 'VBF_analysis_tag', 'ggH_analysis_tag', 'priority_tag']
        #print combined_df[debug_procs]
        #print combined_df[combined_df.VBF_analysis_tag>-1][debug_procs]
        #print combined_df[combined_df.ggH_analysis_tag>-1][debug_procs]

        #fill trees based on both on tree_name variable set above. Dump vars needed in fits
        tree_vars = ['dZ', 'CMS_hgg_mass', 'weight']
        combined_df['dZ'] = float(0.)
        combined_df['CMS_hgg_mass'] = combined_df['dielectronMass'] 

        #set tree names for output files with format: <true_proc>_<misc_info>_<tag>_cat<tag_num>
        branch_names = {}
        for true_proc in true_procs: 
            branch_names[true_proc] = []
            for target_proc in tag_sequence:  #for all events with proc = true_proc, which tag do they fall into?
                for i_tag in range(len(tag_boundaries[target_proc].values())):#for each tag corresponding to the category we target, which events go in which tag
                     if true_proc is not 'Data': branch_names[true_proc].append('{}_125_13TeV_{}cat{}'.format(true_proc.lower(), target_proc.lower(), i_tag ))
                     else: branch_names[true_proc].append('{}_13TeV_{}cat{}'.format(true_proc, target_proc.lower(), i_tag ))

        if not path.isdir('output_trees/'):
            print 'making directory: {}'.format('output_trees/')
            system('mkdir -p %s' %'output_trees/')

        #have to save individual trees as root files (fn=bn), then hadd over single proc on the command line, to get one proc file with all tag trees
        print branch_names
        for proc in true_procs:
            selected_df = combined_df[combined_df.proc==proc]
            for bn in branch_names[proc]:
                print bn
                branch_selected_df = selected_df[selected_df.tree_name==bn]
                print branch_selected_df[debug_vars+['tree_name']].head(10)
                root_pandas.to_root(branch_selected_df[tree_vars], 'output_trees/{}.root'.format(bn), key=bn)
                print

        #create 5 x 2 purity matrix of (reco , proc)
        confusion_matrix = []
        for true_proc in ['VBF', 'ggH']:
            proc_selected_df = combined_df[combined_df.proc==true_proc]
            matrix_col = []
            for reco_branch in branch_names[true_proc]:
                proc_reco_selected_sumw = np.sum(proc_selected_df[proc_selected_df.tree_name==reco_branch]['weight'])
                matrix_col.append(proc_reco_selected_sumw)
            confusion_matrix.append(np.array(matrix_col))
        confusion_matrix = np.stack(confusion_matrix)
        confusion_matrix = np.transpose(confusion_matrix)
        matrix_col_norm  = 100 * confusion_matrix / (confusion_matrix.sum(axis=0).reshape(1,-1)) #broadcast
        matrix_row_norm  = 100 * confusion_matrix / (confusion_matrix.sum(axis=1).reshape(-1,1)) #broadcast


        plot_matrix(matrix_col_norm, 'col', branch_names, true_procs, output_tag)
        plot_matrix(matrix_row_norm, 'row', branch_names, true_procs, output_tag)

def assign_tree(row, true_procs):
    label = '13TeV'
    for proc in true_procs:
        if row['proc'] == proc: 
            if proc is not 'Data':
                proc = proc.lower()#data needs captial D in tree name
                label = '125_13TeV'
            #for all true events with true_proc = proc, which went in what analysis category
            if row['priority_tag'] == 'VBF':
                if row['VBF_analysis_tag']==0 : return '{}_{}_vbfcat0'.format(proc,label)
                elif row['VBF_analysis_tag']==1 : return '{}_{}_vbfcat1'.format(proc,label)
            elif row['priority_tag'] == 'ggH':
                if row['ggH_analysis_tag']==0 : return '{}_{}_gghcat0'.format(proc,label)
                elif row['ggH_analysis_tag']==1 : return '{}_{}_gghcat1'.format(proc,label)
                elif row['ggH_analysis_tag']==2 : return '{}_{}_gghcat2'.format(proc,label)
            elif row['priority_tag'] == 'NOTAG': return 'NOTAG'
            else: raise KeyError('Did not have one of the correct tags')


def plot_matrix(matrix, norm, branch_names, true_procs, output_tag):
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
    axes.set_xticklabels(true_procs,rotation='vertical')
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
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    required_args = parser.add_argument_group('Required Arguments')
    required_args.add_argument('-c','--config', action='store', required=True)
    required_args.add_argument('-M','--bdt_config', action='store', required=True)
    opt_args = parser.add_argument_group('Optional Arguements')
    opt_args.add_argument('-r','--reload_samples', help='re-load the .root files and convert into pandas DataFrames', action='store_true', default=False)
    opt_args.add_argument('-d','--data_as_bkg', action='store_true', default=False)
    options=parser.parse_args()
    main(options)
