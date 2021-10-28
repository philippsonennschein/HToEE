#electron vars
nominal_vars = ['diphotonPt', 'diphotonMass', 'dijetMass', 'leadJetPt', 'subleadJetPt', 'weight', 'centralObjectWeight',
                'leadPhotonPtOvM','subleadPhotonPtOvM','leadPhotonEta','subleadPhotonEta','leadPhotonIDMVA','subleadPhotonIDMVA'
                #'leadElectronEta', #'leadElectronEn', 'leadElectronPt', 'leadElectronPhi', 'leadElectronMass',
                #'subleadElectronEta',#'subleadElectronPt', 'subleadElectronEn',  'subleadElectronPhi', 'subleadElectronMass',
                #'dielectronCosPhi','dielectronPt', 'dielectronMass', 'leadJetPt','subleadJetPt',#'leadJetMass', 'subleadJetMass', 
                #'leadJetEn', 'leadJetPt','leadJetEta', 'leadJetPhi','leadJetQGL', #add jet en
                #'subleadJetEn', 'subleadJetPt','subleadJetEta', 'subleadJetPhi', 'subleadJetQGL', #add sublead jet en
                #'subsubleadJetEn','subsubleadJetPt', 'subsubleadJetEta', 'subsubleadJetPhi', 'subsubleadJetQGL', 
                #'dijetAbsDEta', 'dijetMass', 'dijetDieleAbsDEta', 'dijetDieleAbsDPhiTrunc', 
                #'dijetMinDRJetEle', 'dijetCentrality', 'dielectronSigmaMoM', 'dijetDPhi', #'dijetPt',
                #'leadJetDieleDPhi', 'leadJetDieleDEta', 'subleadJetDieleDPhi', 'subleadJetDieleDEta',
                #'leadElectronCharge', 'subleadElectronCharge',
               ]

#for MVA training, hence not including masses
gev_vars     = ['leadJetEn', 'leadJetPt', 'subleadJetEn', 'subleadJetPt', 'subsubleadJetEn', 'subsubleadJetPt', 
                'leadElectronEn', 'leadElectronPt', 'subleadElectronEn', 'subleadElectronPt',
                'leadElectronPToM', 'subleadElectronPToM', 'dijetMass', 'dielectronPt'
               ]

gen_vars     = ['genWeight'] 

