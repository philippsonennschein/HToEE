#electron vars
nominal_vars = ['weight', 'centralObjectWeight', 'leadElectronPtOvM', 'subleadElectronPtOvM', #'leadElectronIDMVA', 'subleadElectronIDMVA',
                'leadElectronEta', #'leadElectronEn', 'leadElectronPt', 'leadElectronPhi', 'leadElectronMass',
                'subleadElectronEta',#'subleadElectronPt', 'subleadElectronEn',  'subleadElectronPhi', 'subleadElectronMass',
                #'subsubleadElectronEta',#'subsubleadElectronPt', 'subsubleadElectronEn', 'subsubleadElectronPhi', 'subsubleadElectronMass',
                'dielectronCosPhi','dielectronPt', 'dielectronMass', 'leadJetPt','subleadJetPt',#'leadJetMass', 'subleadJetMass', 
                'leadJetEn', 'leadJetPt','leadJetEta', 'leadJetPhi','leadJetQGL', #add jet en
                'subleadJetEn', 'subleadJetPt','subleadJetEta', 'subleadJetPhi', 'subleadJetQGL', #add sublead jet en
                'subsubleadJetEn','subsubleadJetPt', 'subsubleadJetEta', 'subsubleadJetPhi', 'subsubleadJetQGL', #'subsubleadJetMass' add subsublead jet en
                'dijetAbsDEta', 'dijetMass', 'dijetDieleAbsDEta', 'dijetDieleAbsDPhiTrunc', 
                'dijetMinDRJetEle', 'dijetCentrality', 'dielectronSigmaMoM', 'dijetDPhi', #'dijetPt',
                'leadJetDieleDPhi', 'leadJetDieleDEta', 'subleadJetDieleDPhi', 'subleadJetDieleDEta',
                'leadElectronCharge', 'subleadElectronCharge',
                #'nSoftJets','metSumET','metPhi','metPt' , 'leadJetBTagScore', 'subleadJetBTagScore', 'subsubleadJetBTagScore',
                'leadJetPUJID','subleadJetPUJID', 'subsubleadJetPUJID'#,'leadJetID','subleadJetID','subsubleadJetID'
               ]

#for MVA training, hence not including masses
gev_vars     = ['leadJetEn', 'leadJetPt', 'subleadJetEn', 'subleadJetPt', 'subsubleadJetEn', 'subsubleadJetPt', 
                'leadElectronEn', 'leadElectronPt', 'subleadElectronEn', 'subleadElectronPt',
                'leadElectronPToM', 'subleadElectronPToM', 'dijetMass', 'dielectronPt'
               ]

gen_vars     = ['genWeight'] 

