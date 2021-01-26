import argparse
import numpy as np
import yaml
import pickle
import pandas as pd
import root_pandas
from os import path, system
import keras

from DataHandling import ROOTHelpers
from NeuralNets import LSTM_DNN

def main(options):
    
    with open(options.config, 'r') as config_file:
        config             = yaml.load(config_file)
        output_tag         = config['output_tag']

        mc_dir             = config['mc_file_dir']
        mc_fnames          = config['mc_file_names']
  
        data_dir           = config['data_file_dir']
        data_fnames        = config['data_file_names']

        proc_to_tree_name  = config['proc_to_tree_name']

        proc_to_train_vars = config['train_vars']
        object_vars        = proc_to_train_vars['VBF']['object_vars']
        flat_obj_vars      = [var for i_object in object_vars for var in i_object]
        event_vars         = proc_to_train_vars['VBF']['event_vars']

        #used to check all vars we need for categorisation are in our dfs
        all_train_vars     = proc_to_train_vars['ggH'] + flat_obj_vars + event_vars

        vars_to_add        = config['vars_to_add']


                                           #Data handling stuff#
        #apply loosest selection (ggh) first, else memory requirements are ridiculous. Fine to do this since all cuts all looser than VBF (not removing events with higher priority)
        loosest_selection = 'dielectronMass > 110 and dielectronMass < 150'
 
        #load the mc dataframe for all years. Do not apply any specific preselection
        root_obj = ROOTHelpers(output_tag, mc_dir, mc_fnames, data_dir, data_fnames, proc_to_tree_name, all_train_vars, vars_to_add, loosest_selection) 
        root_obj.no_lumi_scale()
        for sig_obj in root_obj.sig_objects:
            root_obj.load_mc(sig_obj, reload_samples=options.reload_samples)
        if not options.data_as_bkg:
            for bkg_obj in root_obj.bkg_objects:
                root_obj.load_mc(bkg_obj, bkg=True, reload_samples=options.reload_samples)
        else:
            for data_obj in root_obj.data_objects:
                root_obj.load_data(data_obj, reload_samples=options.reload_samples)
            #overwrite background attribute, for compat with DNN class
            root_obj.mc_df_bkg = root_obj.data_df
        root_obj.concat()


                                       #Tag sequence stuff#
    #NOTE: these must be concatted in the same way they are concatted in LSTM.create_X_y(), else predicts are misaligned
    combined_df = pd.concat([root_obj.mc_df_sig, root_obj.mc_df_bkg])

    #decide sequence of tags and specify preselection for use with numpy.select:
    tag_sequence          = ['VBF','ggH']
    proc_to_preselection  = {'VBF': [combined_df['dielectronMass'].gt(110) & 
                                     combined_df['dielectronMass'].lt(150) &
                                     combined_df['leadElectronPToM'].gt(0.333) &
                                     combined_df['subleadElectronPToM'].gt(0.25) &
                                     combined_df['dijetMass'].gt(350) &
                                     combined_df['leadJetPt'].gt(40) &
                                     combined_df['subleadJetPt'].gt(30)
                                    ],
                            'ggH':  [combined_df['dielectronMass'].gt(110) & 
                                     combined_df['dielectronMass'].lt(150) &
                                     combined_df['leadElectronPToM'].gt(0.333) &
                                     combined_df['subleadElectronPToM'].gt(0.25)
                                    ]       
                            }


        # GET MVA SCORES #    

    with open(options.mva_config, 'r') as mva_config_file:
        config            = yaml.load(mva_config_file)
        proc_to_model     = config['models']
        proc_to_tags      = config['boundaries']

        #evaluate ggH BDT scores
        print 'evaluating ggH classifier: {}'.format(proc_to_model['ggH'])
        clf = pickle.load(open('models/{}'.format(proc_to_model['ggH']), "rb"))
        train_vars = proc_to_train_vars['ggH']
        combined_df['ggH_mva'] = clf.predict_proba(combined_df[train_vars].values)[:,1:].ravel()

        #Evaluate VBF LSTM
        print 'loading VBF DNN:'
        with open('models/{}'.format(proc_to_model['VBF']['architecture']), 'r') as model_json:
            model_architecture = model_json.read()
        model = keras.models.model_from_json(model_architecture)
        model.load_weights('models/{}'.format(proc_to_model['VBF']['model']))
         
        LSTM = LSTM_DNN(root_obj, object_vars, event_vars, 1.0, False, True)
         
        # set up X and y Matrices. Log variables that have GeV units
        LSTM.var_transform(do_data=False)  
        X_tot, y_tot     = LSTM.create_X_y()
        X_tot            = X_tot[flat_obj_vars+event_vars] #filter unused vars
        print np.isnan(X_tot).any()
         
        #scale X_vars to mean=0 and std=1. Use scaler fit during previous dnn training
        LSTM.load_X_scaler(out_tag='VBF_DNN')
        X_tot            = LSTM.X_scaler.transform(X_tot)
            
        #make 2D vars for LSTM layers
        X_tot            = pd.DataFrame(X_tot, columns=flat_obj_vars+event_vars)
        X_tot_high_level = X_tot[event_vars].values
        X_tot_low_level  = LSTM.join_objects(X_tot[flat_obj_vars])
            
        #predict probs. Corresponds to same events, since dfs are concattened internally in the same 
        combined_df['VBF_mva']    = model.predict([X_tot_high_level, X_tot_low_level], batch_size=1).flatten()

        # TAG NUMBER #

        #decide on tag
        for proc in tag_sequence:
            presel     = proc_to_preselection[proc]
            tag_bounds = proc_to_tags[proc].values()
            tag_masks = []
            for i_bound in range(len(tag_bounds)): #c++ type looping for index reasons
                if i_bound==0: #first bound, tag 0
                    tag_masks.append( presel[0] & combined_df['{}_mva'.format(proc)].gt(tag_bounds[i_bound]) )
                else: #intermed bound
                    tag_masks.append( presel[0] & combined_df['{}_mva'.format(proc)].lt(tag_bounds[i_bound-1]) & 
                                      combined_df['{}_mva'.format(proc)].gt(tag_bounds[i_bound])
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
        combined_df['dZ'] = float(0.)
        combined_df['CMS_hgg_mass'] = combined_df['dielectronMass'] 

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
                     else: branch_names[true_proc].append('{}_13TeV_{}cat{}'.format(true_proc, target_proc.lower(), i_tag ))

        #debug_procs = ['dipho_mass', 'dipho_leadIDMVA', 'dipho_subleadIDMVA', 'dipho_lead_ptoM', 'dipho_sublead_ptoM', 'dijet_Mjj', 'dijet_LeadJPt', 'dijet_SubJPt', 'ggH_bdt', 'VBF_bdt', 'VBF_analysis_tag', 'ggH_analysis_tag', 'priority_tag']
        debug_vars = ['proc', 'VBF_analysis_tag', 'ggH_analysis_tag', 'priority_tag']
        combined_df['tree_name'] = combined_df.apply(assign_tree, axis=1)
        print combined_df[debug_vars+['tree_name']]

        if not path.isdir('output_trees/'):
            print 'making directory: {}'.format('output_trees/')
            system('mkdir -p %s' %'output_trees/')

        #have to save individual trees then hadd procs together on the command line.
        for proc in tag_sequence+['Data']:
            selected_df = combined_df[combined_df.proc==proc]
            for bn in branch_names[proc]:
                print bn
                branch_selected_df = selected_df[selected_df.tree_name==bn]
                print branch_selected_df[debug_vars+['tree_name']].head(20)
                root_pandas.to_root(branch_selected_df[tree_vars], 'output_trees/{}.root'.format(bn), key=bn)
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
            if row['VBF_analysis_tag']==0 : return 'Data_13TeV_vbfcat0'
            elif row['VBF_analysis_tag']==1 : return 'Data_13TeV_vbfcat1'
        elif row['priority_tag'] == 'ggH':
            if row['ggH_analysis_tag']==0 : return 'Data_13TeV_gghcat0'
            elif row['ggH_analysis_tag']==1 : return 'Data_13TeV_gghcat1'
            elif row['ggH_analysis_tag']==2 : return 'Data_13TeV_gghcat2'
        elif row['priority_tag'] == 'NOTAG': return 'NOTAG'
        else: raise KeyError('Did not have one of the correct tags')
        #for all true data processes, which went in what analysis category

    else: raise KeyError('Did not have one of the correct procs')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    required_args = parser.add_argument_group('Required Arguments')
    required_args.add_argument('-c','--config', action='store', required=True)
    required_args.add_argument('-B','--mva_config', action='store', required=True)
    opt_args = parser.add_argument_group('Optional Arguements')
    opt_args.add_argument('-r','--reload_samples', help='re-load the .root files and convert into pandas DataFrames', action='store_true', default=False)
    opt_args.add_argument('-d','--data_as_bkg', action='store_true', default=False)
    options=parser.parse_args()
    main(options)
