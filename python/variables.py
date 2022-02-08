#electron vars
nominal_vars = ['diphotonMass','diphotonPt','diphotonEta','diphotonPhi','diphotonCosPhi','diphotonSigmaMoM', 
                'leadPhotonIDMVA','leadPhotonPtOvM','leadPhotonEta','leadPhotonEn','leadPhotonMass','leadPhotonPt','leadPhotonPhi',
                'subleadPhotonIDMVA','subleadPhotonPtOvM','subleadPhotonEta','subleadPhotonEn','subleadPhotonMass','subleadPhotonPt','subleadPhotonPhi',
                #'subsubleadPhotonIDMVA','subsubleadPhotonPtOvM','subsubleadPhotonEta','subsubleadPhotonEn','subsubleadPhotonMass','subsubleadPhotonPt','subsubleadPhotonPhi',
                'dijetMass','dijetPt','dijetEta','dijetPhi','dijetDPhi','dijetAbsDEta','dijetCentrality','dijetMinDRJetPho','dijetDiphoAbsDEta',
                'leadJetPUJID','leadJetPt','leadJetEn','leadJetEta','leadJetPhi','leadJetMass','leadJetBTagScore','leadJetDiphoDEta','leadJetDiphoDPhi',
                'subleadJetPUJID','subleadJetPt','subleadJetEn','subleadJetEta','subleadJetPhi','subleadJetMass','subleadJetBTagScore','subleadJetDiphoDPhi','subleadJetDiphoDEta',
                'subsubleadJetPUJID','subsubleadJetPt','subsubleadJetEn','subsubleadJetEta','subsubleadJetPhi','subsubleadJetMass','subsubleadJetBTagScore',#'subsubleadJetDiphoDPhi','subsubleadJetDiphoDEta',
                'weight', 'centralObjectWeight',
                'nSoftJets','metPt','metPhi','metSumET'

                #'diphotonPt', 'diphotonMass', 'dijetMass', 'leadJetPt', 'subleadJetPt', 'weight', 'centralObjectWeight',
                #'leadPhotonPtOvM','subleadPhotonPtOvM','leadPhotonEta','subleadPhotonEta','leadPhotonIDMVA','subleadPhotonIDMVA'
                #'leadPhotonEn'
                #'leadElectronPt', 'leadElectronPhi', 'leadElectronMass',
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
gev_vars     = ['leadJetEn', 'subleadJetEn', 'subleadJetPt', 'subsubleadJetEn', 'subsubleadJetPt', 
                'leadElectronEn', 'leadElectronPt', 'subleadElectronEn', 'subleadElectronPt',
                'leadElectronPToM', 'subleadElectronPToM', 'dijetMass', 'dielectronPt'
               ]

gen_vars     = ['genWeight','HTXS_stage_0','HTXS_stage1_2_cat_pTjet30GeV'] 

