import argparse
import numpy as np
import yaml
import pickle
from HToEEML import ROOTHelpers 
import pandas as pd
import root_pandas


def main(options):
    
    with open(options.config, 'r') as config_file:
        config             = yaml.load(config_file)
        output_tag         = config['signal_process']

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
        loosest_selection = 'dipho_mass > 110 and dipho_mass < 150 and dipho_leadIDMVA > -0.9 and dipho_subleadIDMVA > -0.9 and dipho_lead_ptoM > 0.333 and dipho_sublead_ptoM > 0.25'
 
        #load the mc dataframe for all years. Do not apply any specific preselection
        root_obj = ROOTHelpers(output_tag, mc_dir, mc_fnames, data_dir, data_fnames, proc_to_tree_name, all_train_vars, vars_to_add, loosest_selection) 

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
    tag_sequence          = ['VBF','ggH']
    proc_to_preselection  = {'VBF': [combined_df['dipho_mass'].gt(110) & 
                                     combined_df['dipho_mass'].lt(150) &
                                     combined_df['dipho_leadIDMVA'].gt(-0.2) &
                                     combined_df['dipho_subleadIDMVA'].gt(-0.2) &
                                     combined_df['dipho_lead_ptoM'].gt(0.333) &
                                     combined_df['dipho_sublead_ptoM'].gt(0.25) &
                                     combined_df['dijet_Mjj'].gt(350) &
                                     combined_df['dijet_LeadJPt'].gt(40) &
                                     combined_df['dijet_SubJPt'].gt(30)
                                    ],
                            'ggH':  [combined_df['dipho_mass'].gt(110) & 
                                     combined_df['dipho_mass'].lt(150) &
                                     combined_df['dipho_leadIDMVA'].gt(-0.9) &
                                     combined_df['dipho_subleadIDMVA'].gt(-0.9) &
                                     combined_df['dipho_lead_ptoM'].gt(0.333) &
                                     combined_df['dipho_sublead_ptoM'].gt(0.25)
                                    ]       
                            }


    with open(options.bdt_config, 'r') as bdt_config_file:
        config            = yaml.load(bdt_config_file)
        proc_to_model     = config['models']
        proc_to_tags      = config['boundaries']

        #evaluate MVA scores used in categorisation
        for proc, model in proc_to_model.iteritems():
            print 'evaluating classifier: {}'.format(model)
            clf = pickle.load(open('models/{}'.format(model), "rb"))
            train_vars = proc_to_train_vars[proc]
            combined_df[proc+'_bdt'] = clf.predict_proba(combined_df[train_vars].values)[:,1:].ravel()
       
        # TAG NUMBER #

        #decide on tag
        for proc in tag_sequence:
            presel     = proc_to_preselection[proc]
            tag_bounds = proc_to_tags[proc].values()
            tag_masks = []
            for i_bound in range(len(tag_bounds)): #c++ type looping for index reasons
                if i_bound==0: #first bound, tag 0
                    tag_masks.append( presel[0] & combined_df['{}_bdt'.format(proc)].gt(tag_bounds[i_bound]) )
                else: #intermed bound
                    tag_masks.append( presel[0] & combined_df['{}_bdt'.format(proc)].lt(tag_bounds[i_bound-1]) & 
                                      combined_df['{}_bdt'.format(proc)].gt(tag_bounds[i_bound])
                                    )

            mask_key     = [icat for icat in range(len(tag_bounds))]

            combined_df['{}_analysis_tag'.format(proc)] = np.select(tag_masks, mask_key, default=-999)


        # PROC PRIORITY #

        # deduce tag priority: if two or more tags satisfied then set final tag to highest priority tag. make this non hardcoded i.e. compare proc in position 1 to all lower prioty positions. then compare proc in pos 2 ...
        tag_priority_filter = [ combined_df['VBF_analysis_tag'].ne(-999) & combined_df['ggH_analysis_tag'].ne(-999), # 1) if both filled...
                                combined_df['VBF_analysis_tag'].ne(-999) & combined_df['ggH_analysis_tag'].eq(-999), # 2) if VBF filled and ggH not, take VBF
                                combined_df['VBF_analysis_tag'].eq(-999) & combined_df['ggH_analysis_tag'].ne(-999), # 3) if ggH filled and VBF not, take ggH
                              ]

        tag_priority_key    = [ 'VBF', #1) take VBF
                                'VBF', #2) take VBF
                                'ggH', #3) take ggH
                              ]
        combined_df['priority_tag'.format(proc)] = np.select(tag_priority_filter, tag_priority_key, default='NOTAG') # else keep -999 i.e. NOTAG
       
        #some debug checks:
        #print combined_df[['dipho_mass', 'dipho_leadIDMVA', 'dipho_subleadIDMVA', 'dipho_lead_ptoM', 'dipho_sublead_ptoM', 'dijet_Mjj', 'dijet_LeadJPt', 'dijet_SubJPt', 'ggH_bdt', 'VBF_bdt', 'VBF_analysis_tag', 'ggH_analysis_tag', 'priority_tag']]
        #print combined_df[combined_df.VBF_analysis_tag>-1][['dipho_mass', 'dipho_leadIDMVA', 'dipho_subleadIDMVA', 'dipho_lead_ptoM', 'dipho_sublead_ptoM', 'dijet_Mjj', 'dijet_LeadJPt', 'dijet_SubJPt', 'ggH_bdt', 'VBF_bdt', 'VBF_analysis_tag', 'ggH_analysis_tag', 'priority_tag']]
        #print combined_df[combined_df.ggH_analysis_tag>-1][['dipho_mass', 'dipho_leadIDMVA', 'dipho_subleadIDMVA', 'dipho_lead_ptoM', 'dipho_sublead_ptoM', 'dijet_Mjj', 'dijet_LeadJPt', 'dijet_SubJPt', 'ggH_bdt', 'VBF_bdt', 'VBF_analysis_tag', 'ggH_analysis_tag', 'priority_tag']]

        # FILL TREES BASED ON BOTH OF ABOVE  
        tree_vars = ['dZ', 'CMS_hgg_mass', 'weight']
        combined_df['dZ'] = 0 
        combined_df['CMS_hgg_mass'] = combined_df['dipho_mass'] 

        # FIXME: dont loop through events eventually but for now I cba to use numpy to vectorise it again
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
         
        #get tree names
        branch_names = {}
        #print 'DEBUG: {}'.format(np.unique(combined_df['proc']))
        for true_proc in tag_sequence+['Data']: 
            branch_names[true_proc] = []
            for target_proc in tag_sequence:  #for all events that got the proc tag, which tag did they fall into?
                for i_tag in range(len(proc_to_tags[target_proc].values())):#for each tag corresponding to the category we target, which events go in which tag
                     if true_proc is not 'Data': branch_names[true_proc].append('{}_125_13TeV_{}cat{}'.format(true_proc.lower(), target_proc.lower(), i_tag ))
                     else: branch_names[true_proc].append('{}_125_13TeV_{}cat{}'.format(true_proc, target_proc.lower(), i_tag ))

        #debug_procs = ['dipho_mass', 'dipho_leadIDMVA', 'dipho_subleadIDMVA', 'dipho_lead_ptoM', 'dipho_sublead_ptoM', 'dijet_Mjj', 'dijet_LeadJPt', 'dijet_SubJPt', 'ggH_bdt', 'VBF_bdt', 'VBF_analysis_tag', 'ggH_analysis_tag', 'priority_tag']
        debug_vars = ['proc', 'VBF_analysis_tag', 'ggH_analysis_tag', 'priority_tag']
        combined_df['tree_name'] = combined_df.apply(assign_tree, axis=1)
        print combined_df[debug_vars+['tree_name']]

        #have to save individual trees then hadd procs together on the command line.
        for proc in tag_sequence+['Data']:
            selected_df = combined_df[combined_df.proc==proc]
            for bn in branch_names[proc]:
                print bn
                branch_selected_df = selected_df[selected_df.tree_name==bn]
                print branch_selected_df[debug_vars+['tree_name']].head(20)
                root_pandas.to_root(branch_selected_df[tree_vars], '{}.root'.format(bn), key=bn)
                print

        #if worst comes to worst then just iter over df rows and fill a root tree the old school way

def assign_tree(row):
    if row['proc'] == 'VBF': 
        #for all true vbf processes, which went in what analysis category
        if row['priority_tag'] == 'VBF':
            if row['VBF_analysis_tag']==0 : return 'vbf_125_13TeV_vbfcat0'
            elif row['VBF_analysis_tag']==1 : return 'vbf_125_13TeV_vbfcat1'
        elif row['priority_tag'] == 'ggH':
            if row['ggH_analysis_tag']==0 : return 'vbf_125_13TeV_gghcat0'
            elif row['ggH_analysis_tag']==1 : return 'vbf_125_13TeV_gghcat1'
            elif row['ggH_analysis_tag']==2 : return 'vbf_125_13TeV_gghcat2'
        elif row['priority_tag'] == 'NOTAG': return 'NOTAG'
        else: raise KeyError('Did not have one of the correct tags')

    elif row['proc'] == 'ggH':
        if row['priority_tag'] == 'VBF':
            if row['VBF_analysis_tag']==0 : return 'ggh_125_13TeV_vbfcat0'
            elif row['VBF_analysis_tag']==1 : return 'ggh_125_13TeV_vbfcat1'
        elif row['priority_tag'] == 'ggH':
            if row['ggH_analysis_tag']==0 : return 'ggh_125_13TeV_gghcat0'
            elif row['ggH_analysis_tag']==1 : return 'ggh_125_13TeV_gghcat1'
            elif row['ggH_analysis_tag']==2 : return 'ggh_125_13TeV_gghcat2'
        elif row['priority_tag'] == 'NOTAG': return 'NOTAG'
        else: raise KeyError('Did not have one of the correct tags')

        #for all true ggh processes, which went in what analysis category
    elif row['proc'] == 'Data':
        if row['priority_tag'] == 'VBF':
            if row['VBF_analysis_tag']==0 : return 'Data_125_13TeV_vbfcat0'
            elif row['VBF_analysis_tag']==1 : return 'Data_125_13TeV_vbfcat1'
        elif row['priority_tag'] == 'ggH':
            if row['ggH_analysis_tag']==0 : return 'Data_125_13TeV_gghcat0'
            elif row['ggH_analysis_tag']==1 : return 'Data_125_13TeV_gghcat1'
            elif row['ggH_analysis_tag']==2 : return 'Data_125_13TeV_gghcat2'
        elif row['priority_tag'] == 'NOTAG': return 'NOTAG'
        else: raise KeyError('Did not have one of the correct tags')
        #for all true data processes, which went in what analysis category

    else: raise KeyError('Did not have one of the correct prosc')
            


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    required_args = parser.add_argument_group('Required Arguments')
    required_args.add_argument('-c','--config', action='store', required=True)
    required_args.add_argument('-B','--bdt_config', action='store', required=True)
    opt_args = parser.add_argument_group('Optional Arguements')
    opt_args.add_argument('-r','--reload_samples', action='store_true', default=False)
    opt_args.add_argument('-d','--data_as_bkg', action='store_true', default=False)
    options=parser.parse_args()
    main(options)
