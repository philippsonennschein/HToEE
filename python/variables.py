#electron vars
nominal_vars = ['weight', 'centralObjectWeight', 'leadElectronIDMVA', 'subleadElectronIDMVA','leadElectronPtOvM', 'subleadElectronPtOvM',
                'leadElectronPt', 'leadElectronEn', 'leadElectronEta', 'leadElectronPhi', 'leadElectronMass',
                'subleadElectronEta',#'subleadElectronPt', 'subleadElectronEn',  'subleadElectronPhi', 'subleadElectronMass',
                'subsubleadElectronEta',#'subsubleadElectronPt', 'subsubleadElectronEn', 'subsubleadElectronPhi', 'subsubleadElectronMass',
                'dielectronCosPhi','dielectronPt', 'dielectronMass', 'leadJetMass', 'leadJetPt','subleadJetMass', 'subleadJetPt',
                'leadJetEn', 'leadJetPt','leadJetEta', 'leadJetPhi','leadJetQGL', #add jet en
                'subleadJetEn', 'subleadJetPt','subleadJetEta', 'subleadJetPhi', 'subleadJetQGL', #add sublead jet en
                'subsubleadJetEn','subsubleadJetPt','subsubleadJetMass', 'subsubleadJetEta', 'subsubleadJetPhi', 'subsubleadJetQGL', #add subsublead jet en
                'dijetAbsDEta', 'dijetMass', 'dijetDieleAbsDEta', 'dijetDieleAbsDPhiTrunc', # FIXME: dijetAbsDPhiTrunc is actually dijet_dphi. Still need 'dijet_dipho_dphi_trunc' (min dphi I think it is)
                'dijetMinDRJetEle', 'dijetCentrality', 'dielectronSigmaMoM', 'dijetDPhi', 'dijetPt',
                'leadJetDieleDPhi', 'leadJetDieleDEta', 'subleadJetDieleDPhi', 'subleadJetDieleDEta'
               ]

#for MVA training, hence not including masses
gev_vars     = ['leadJetEn', 'leadJetPt', 'subleadJetEn', 'subleadJetPt', 'subsubleadJetEn', 'subsubleadJetPt', 
                'leadElectronEn', 'leadElectronPt', 'subleadElectronEn', 'subleadElectronPt',
                'leadElectronPToM', 'subleadElectronPToM', 'dijetMass', 'dielectronPt'
               ]

gen_vars     = ['genWeight'] 

