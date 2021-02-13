#electron vars
nominal_vars = ['weight', 'leadElectronIDMVA', 'subleadElectronIDMVA','leadElectronPtOvM', 'subleadElectronPtOvM',
                'leadJetEn', 'leadElectronPt', 'leadElectronEta', 'leadElectronPhi',               
                'subleadJetEn', 'subleadElectronPt',  'subleadElectronEta', 'subleadElectronPhi',
                'subsubleadJetEn', 'dielectronCosPhi','dielectronPt', 'dielectronMass', 
                'leadJetPt','leadJetEta', 'leadJetPhi','leadJetQGL', #add jet en
                'subleadJetPt','subleadJetEta', 'subleadJetPhi', 'subleadJetQGL', #add sublead jet en
                'subsubleadJetPt','subsubleadJetEta', 'subsubleadJetPhi', 'subsubleadJetQGL', #add subsublead jet en
                'dijetAbsDEta', 'dijetMass', 'dijetDieleAbsDEta', 'dijetDieleAbsDPhiTrunc', # FIXME: dijetAbsDPhiTrunc is actually dijet_dphi. Still need 'dijet_dipho_dphi_trunc' (min dphi I think it is)
                'dijetMinDRJetEle', 'dijetCentrality', 'dielectronSigmaMoM', 'dijetDPhi'
               ]

#for MVA training, hence not including masses
gev_vars     = ['leadJetEn', 'leadJetPt', 'subleadJetEn', 'subleadJetPt', 'subsubleadJetEn', 'subsubleadJetPt', 
                'leadElectronEn', 'leadElectronPt', 'subleadElectronEn', 'subleadElectronPt',
                'leadElectronPToM', 'subleadElectronPToM', 'dijetMass', 'dielectronPt'
               ]

gen_vars     = ['genWeight'] 

