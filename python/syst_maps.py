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
                       'dijetPt', 'dijetMass']
            }

#variables that effect only event weights e.g. pre-firing correction. Fill nominal tree for these
weight_systs = {'L1PreFiringWeight': ['Dn', 'Nom', 'Up']
               }
