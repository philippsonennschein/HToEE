#electron vars
nominal_vars = ['genWeight', 'leadElectronIDMVA', 'subleadElectronIDMVA','leadElectronPToM*', 'subleadElectronPToM*',
                'leadElectronPt', 'leadElectronEta', 'leadElectronPhi',
                'subleadElectronPt',  'subleadElectronEta', 'subleadElectronPhi',
                'dielectronCosPhi','dielectronPt', 'dielectronMass', 
                'leadJetPt','leadJetEta', 'leadJetPhi','leadJetQGL', #add jet en
                'subleadJetPt','subleadJetEta', 'subleadJetPhi', 'subleadJetQGL', #add sublead jet en
                'subsubleadJetPt','subsubleadJetEta', 'subsubleadJetPhi', 'subsubleadJetQGL', #add subsublead jet en
                'dijetAbsDEta', 'dijetMass', 'dijetAbsDPhiTrunc', # FIXME: dijetAbsDPhiTrunc is actually dijet_dphi. Still need 'dijet_dipho_dphi_trunc'
                'dijetMinDRJetEle', 'dijetCentrality'
               ]

#for MVA training, hence not including masses
gev_vars =     ['leadJetEn', 'leadJetPt', 'subleadJetEn', 'subleadJetPt', 'subsubleadJetEn', 'subsubleadJetPt', 
                'leadElectronEn', 'leadElectronPt', 'subleadElectronEn', 'subleadElectronPt',
                'leadElectronPToM', 'subleadElectronPToM', 'dijetMass'
               ]


gen_vars     = []  #dont need any gen vars for now



