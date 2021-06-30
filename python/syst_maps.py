#pass 6 names
syst_map  = {'jerUp'  : 
                      ['leadJetEn', 'leadJetMass', 'leadJetPt',
                       'subleadJetEn', 'subleadJetMass', 'subleadJetPt',
                       'subsubleadJetEn', 'subsubleadJetMass', 'subsubleadJetPt',
                       'dijetPt', 'dijetMass'],
             'jerDown': 
                      ['leadJetEn', 'leadJetMass', 'leadJetPt',
                       'subleadJetEn', 'subleadJetMass', 'subleadJetPt',
                       'subsubleadJetEn', 'subsubleadJetMass', 'subsubleadJetPt',
                       'dijetPt', 'dijetMass'],

             'jesTotalUp': 
                      ['leadJetEn', 'leadJetMass', 'leadJetPt',
                      'subleadJetEn', 'subleadJetMass', 'subleadJetPt',
                      'subsubleadJetEn', 'subsubleadJetMass', 'subsubleadJetPt',
                      'dijetPt', 'dijetMass'],
             'jesTotalDown': 
                      ['leadJetEn', 'leadJetMass', 'leadJetPt',
                       'subleadJetEn', 'subleadJetMass', 'subleadJetPt',
                       'subsubleadJetEn', 'subsubleadJetMass', 'subsubleadJetPt',
                       'dijetPt', 'dijetMass'],

             'ElPtScaleUp': 
                      ['leadElectronEn', 'leadElectronMass', 'leadElectronPt', 
                       #'subleadElectronEn', 'subleadElectronMass', 'subleadElectronPt', #never use these
                       #'subsubleadElectronEn', 'subsubleadElectronMass', 'subsubleadElectronPt', never use these
                       'leadElectronPtOvM', 'subleadElectronPtOvM', 'dielectronMass', 'dielectronPt'
                      ],
             'ElPtScaleDown': 
                      ['leadElectronEn', 'leadElectronMass', 'leadElectronPt', 
                       #'subleadElectronEn','subleadElectronMass', 'subleadElectronPt', #never use these
                       #'subsubleadElectronEn', 'subsubleadElectronMass', 'subsubleadElectronPt', #never use these
                       'leadElectronPtOvM', 'subleadElectronPtOvM', 'dielectronMass', 'dielectronPt'
                      ],
            }

#variables that effect only event weights e.g. pre-firing correction. Fill nominal tree for these
#the nested dict values (list 1) are the down, nominal, and up string exts for the systematic in the key.
#the nested dict values (list 2) are the year this systematic effects
weight_systs = {'L1PreFiringWeight': {'exts':['_Dn', '_Nom', '_Up'], 'years':['2016', '2017']},
                'ElectronIDSF':      {'exts':['Down', '', 'Up'], 'years':['2016', '2017', '2018']},
                'ElectronRecoSF':    {'exts':['Down', '', 'Up'], 'years':['2016', '2017', '2018']},
                'TriggerSF':         {'exts':['Down', '', 'Up'], 'years':['2016', '2017', '2018']}
               }
