def addPt(row):
    return row['dipho_mass']*row['dipho_PToM']

def truthDipho(row, sigList): 
    if row['proc'] in sigList: return 1
    else: return 0

def truthDiphoMulticlass(row, sigList): 
    if row['proc'] in sigList: return 0
    elif row['proc'].count('qcd'): return 1
    elif row['proc'].count('gjet'): return 2
    elif row['proc'].count('dipho'): return 3

def truthVhHad(row):
    if row['HTXSstage1p2bin']==204: return 1
    else: return 0
    #elif row['HTXSstage1_1_cat']>107 and row['HTXSstage1_1_cat']<111: return 0
    #else: return -1

def vhHadWeight(row, ratio):
    weight = 1000. * abs(row['weight'])
    if row['proc'].count('qcd'): 
        weight *= 0.04 #downweight bc too few events
    #now account for the resolution
    if row['sigmarv']>0. and row['sigmawv']>0.:
        weight *= ( (row['vtxprob']/row['sigmarv']) + ((1.-row['vtxprob'])/row['sigmawv']) )
    if row['truthVhHad']==1: 
      return ratio * weight
    else: return weight

def truthClass(row): #translate yaceen to mine #NOTE: bins numbered 0->12
   if row['HTXSstage1_1_cat']==100: return -1 #out of acceptance 
   elif row['HTXSstage1_1_cat']==101: return 8 #bsm bin
   elif row['HTXSstage1_1_cat']>=102 and row['HTXSstage1_1_cat']<=109: return int(row['HTXSstage1_1_cat']-102)
   elif row['HTXSstage1_1_cat']==110: return 9 
   elif row['HTXSstage1_1_cat']==111: return 11
   elif row['HTXSstage1_1_cat']==112: return 10

def truthClass1p2(row, mergeBSM=False): #translate yaceen to mine #NOTE: bins numbered 0->15
   if row['HTXSstage1p2bin']==100: return -1 #out of acceptance 
   elif row['HTXSstage1p2bin']>=105 and row['HTXSstage1p2bin']<=112: return int(row['HTXSstage1p2bin']-105)
   # VBF-like ggH
   elif row['HTXSstage1p2bin']==113: return 12
   elif row['HTXSstage1p2bin']==114: return 13
   elif row['HTXSstage1p2bin']==115: return 14
   elif row['HTXSstage1p2bin']==116: return 15
   #BSM bins
   elif row['HTXSstage1p2bin']>100 and row['HTXSstage1p2bin']<105:
       if mergeBSM: return 8
       else:
           if row['HTXSstage1p2bin']==101: return 8 
           elif row['HTXSstage1p2bin']==102: return 9 
           elif row['HTXSstage1p2bin']==103: return 10 
           elif row['HTXSstage1p2bin']==104: return 11 

def truthClass_2016(row):
  if(row['gen_pTH'] < 200):
    if(row['n_gen_jets'] == 0):
      if(row['gen_pTH'] < 10): return 0                                                                         
      else: return 1
    if(row['n_gen_jets'] == 1):
      if(row['gen_pTH'] < 60): return 2
      elif(row['gen_pTH'] < 120): return 3
      elif(row['gen_pTH'] < 200): return 4
    if(row['n_gen_jets'] >= 2):
      if(row['gen_dijet_Mjj'] < 350):
        if(row['gen_pTH'] < 60): return 5
        elif(row['gen_pTH'] < 120): return 6
        elif(row['gen_pTH'] < 200): return 7
      else: #( implicit if Mjj>350)
        if(row['gen_ptHjj'] < 25):
          if(row['gen_dijet_Mjj'] < 700): return 9
          else: return 10
        else:
          if(row['gen_dijet_Mjj'] < 700): return 11
          else: return 12
  elif(row['gen_pTH']>200): return 8
  else: return -1 #everything that doesn't go into a bin

def truthVBF(row):
    if row['HTXSstage1p1bin']>206.5 and row['HTXSstage1p1bin']<210.5: return 2
    elif row['HTXSstage1p1bin']>109.5 and row['HTXSstage1p1bin']<113.5: return 1
    else: return 0
    #elif row['HTXSstage1_1_cat']==0: return 0
    #else: return -1

def truthVBFconstitSamples(row):
    if row['HTXSstage1_1_cat']>206.5 and row['HTXSstage1_1_cat']<210.5: return 2
    elif row['HTXSstage1_1_cat']>109.5 and row['HTXSstage1_1_cat']<113.5: return 1
    else: return 0
    #elif row['HTXSstage1_1_cat']==0: return 0
    #else: return -1

def vbfWeight(row, vbfSumW, gghSumW, bkgSumW):
    weight = abs(row['weight'])
    if row['proc'].count('qcd'): 
        weight *= 0.04 
    if row['sigmarv']>0. and row['sigmawv']>0.:
        weight *= ( (row['vtxprob']/row['sigmarv']) + ((1.-row['vtxprob'])/row['sigmawv']) )
    if row['truthVBF']==2: 
      return (bkgSumW/vbfSumW) * weight
    elif row['truthVBF']==1: 
      return (bkgSumW/gghSumW) * weight
    else: return weight

def applyLumiScale(row, yearToLumi):
    if row['proc'] == 'Data': return 1
    weight = row['weight']
    year = row['year']
    return weight*yearToLumi[year]

def applyLumiScaleForBinaryYears(row, yearToLumi):
    weight = row['weight']
    if row['year_2016']==1:
        return weight*yearToLumi['2016']
    elif row['year_2017']==1:
        return weight*yearToLumi['2017']
    elif row['year_2018']==1:
        return weight*yearToLumi['2018']
    else: raise Exception('Error: did not find a year in the proc column of dataframe')


def applyDataScale(row, yearToDataScale):
    weight = row['weight']
    if '2016' in row['proc']:
        return weight*yearToDataScale['2016']
    elif '2017' in row['proc']:
        return weight*yearToDataScale['2017']
    elif '2018' in row['proc']:
        return weight*yearToDataScale['2018']
    else: raise Exception('Error: did not find a year in the proc column of dataframe')

      
def ptSplits(row):
    if row['diphoR']==8: 
        if (row['diphoP']>200): 
            if (row['diphoP']<300): return 8
            elif (row['diphoP']<450): return 9
            elif (row['diphoP']<650): return 10
            else: return 11
    else: return row['diphoR']
    

def truthJets(row):
    if row['HTXSstage1_1_cat']==3: return 0
    elif row['HTXSstage1_1_cat']>=4 and row['HTXSstage1_1_cat']<=7: return 1
    elif row['HTXSstage1_1_cat']>=8 and row['HTXSstage1_1_cat']<=11: return 2
    else: return -1

def reco(row):
    #Stage1.2 category definitions
  if(row['diphopt'] < 200):
    if(row['gghMVA_n_rec_jets'] == 0):
      if(row['diphopt'] < 10): return 0
      else: return 1
    if(row['gghMVA_n_rec_jets'] == 1): 
      if(row['diphopt'] < 60): return 2
      elif(row['diphopt'] < 120): return 3
      elif(row['diphopt'] < 200): return 4 
    if(row['gghMVA_n_rec_jets'] >= 2): 
      if(row['gghMVA_Mjj'] < 350):
        if(row['diphopt'] < 60): return 5 
        elif(row['diphopt'] < 120): return 6
        elif(row['diphopt'] < 200): return 7
      else: #( implicit if Mjj>350)
        if(row['gghMVA_Mjj'] < 700):
          if(row['dipho_dijet_ptHjj'] < 25): return 12
          else: return 13
        else:
          if(row['dipho_dijet_ptHjj'] < 25): return 14
          else: return 15
  elif(row['diphopt']>200): 
    if (row['diphopt']<300): return 8
    elif (row['diphopt']<450): return 9
    elif (row['diphopt']<650): return 10
    else: return 11

  else: return -1 #everything that doesn't go into a bin

def jetPtToClass(row):
    #Stage1.1 reco/cat definitions
  if(row['dipho_pt'] < 200):
    if(row['n_pred_jets'] == 0):
      if(row['dipho_pt'] < 10): return 0
      else: return 1
    if(row['n_pred_jets'] == 1): 
      if(row['dipho_pt'] < 60): return 2
      elif(row['dipho_pt'] < 120): return 3
      elif(row['dipho_pt'] < 200): return 4 
    if(row['n_pred_jets'] >= 2): 
      if(row['dijet_Mjj'] < 350):
        if(row['dipho_pt'] < 60): return 5 
        elif(row['dipho_pt'] < 120): return 6
        elif(row['dipho_pt'] < 200): return 7
      else: #( implicit if Mjj>350)
        if(row['dipho_dijet_ptHjj'] < 25):
          if(row['dijet_Mjj'] < 700): return 9
          else: return 10
        else: #(implicit if PtHjj > 25)
          if(row['dijet_Mjj'] < 700): return 11
          else: return 12
  elif(row['dipho_pt']>200): return 8
  else: return -1 #everything that doesn't go into a bin

def jetPtToggHClass(row):
    #Stage1.1 reco/cat definitions
  if(row['diphopt'] < 200):
    if(row['n_pred_jets'] == 0): 
      if(row['diphopt'] < 10): return 0
      else: return 1
    if(row['n_pred_jets'] == 1): 
      if(row['diphopt'] < 60): return 2
      elif(row['diphopt'] < 120): return 3
      elif(row['diphopt'] < 200): return 4 
    if(row['n_pred_jets'] >= 2): 
      if(row['diphopt'] < 60): return 5 
          #if(ev.gen_ptHjj > 0 and ev.gen_ptHjj < 25): return 6
          #if(ev.gen_ptHjj > 25): return 7
      elif(row['diphopt'] < 120): return 6
          #if(ev.gen_ptHjj > 0 and ev.gen_ptHjj < 25): return 8
          #if(ev.gen_ptHjj > 25): return 9
      elif(row['diphopt'] < 200): return 7
          #if(ev.gen_ptHjj > 0 and ev.gen_ptHjj < 25): return 10
          #if(ev.gen_ptHjj > 25): return 11
  elif(row['diphopt']>200): return 8
  else: return -1 #everything that doesn't go into a bin 


def downWeightQCD(row, factor=0.04 ):
    weight = row['weight']
    if row['proc'].count('gjet') or row['proc'].count('qcd'): 
      weight *= 0.04 
    return weight

def diphoWeight(row, sigList, sigWeight=1. ):
    weight = row['weight']
    #move this downweighting to its own fn
    #if row['proc'].count('qcd'): 
    #    weight *= 0.04 #downweight bc too few events
    if row['HTXSstage1p2bin'] > 0.01:
        weight *= sigWeight #arbitrary change in signal weight, to be optimised
    #now account for the resolution
    if row['sigmarv']>0. and row['sigmawv']>0.:
        weight *= ( (row['vtxprob']/row['sigmarv']) + ((1.-row['vtxprob'])/row['sigmawv']) )
    weight = abs(weight)
    return weight

def combinedWeight(row):
    weight = row['weight']
    if row['proc'].count('qcd'): 
        weight *= 0.04 #downweight bc too few events
    weight = abs(weight)
    return weight

def normWeight(row, bkgWeight=100., zerojWeight=1.):
    weightFactors = [0.0002994, 0.0000757, 0.0000530, 0.0000099, 0.0000029, 0.0000154, 0.0000235, 0.0000165, 0.0000104] #update these at some point
    weight = row['weight']
    if row['proc'].count('qcd'): 
        weight *= 0.04 / weightFactors[ int(row['truthClass']) ] #reduce because too large by default
    else: 
        weight *= 1. / weightFactors[ int(row['truthClass']) ] #otherwise just reweight by xs
    weight = abs(weight)
    #arbitrary weight changes to be optimised
    if row['proc'] != 'ggh':
        weight *= bkgWeight
    elif row['reco'] == 0: 
        weight *= zerojWeight
    return weight

def jetWeight(row):
    weightFactors = [0.606560, 0.270464, 0.122976]
    weight = row['weight']
    weight *= 1. / weightFactors[ int(row['truthJets']) ] #otherwise just reweight by xs
    weight = abs(weight)
    return weight

def altDiphoWeight(row, weightRatio,sigList):
    weight = row['weight']
    if row['proc'].count('qcd'):
        weight *= 0.04 #downweight bc too few events
    if row['HTXSstage1p2bin'] > 0.01:
        weight *= weightRatio #arbitrary change in signal weight, to be optimised
    #now account for the resolution
    if row['sigmarv']>0. and row['sigmawv']>0.:
        weight *= ( (row['vtxprob']/row['sigmarv']) + ((1.-row['vtxprob'])/row['sigmawv']) )
    weight = abs(weight)
    return weight

def procWeight(row, procFracs): #equalise the weights for the procs i.e. equal sum of weights for the 9 classes
    #weightFactors = [0.146457, 0.517408, 0.151512, 0.081834, 0.014406, 0.020672, 0.036236, 0.017924, 0.013551]
    weight = row['weight']                             
    weight *= 1. / procFracs[ int(row['truthClass']) ] 
    weight = abs(weight)                              
    return weight  

def eightProcWeight(row): #equalise the weights for the procs i.e. equal sum of weights for the 9 classes
    weightFactors = [0.148501, 0.524624, 0.153626, 0.082976, 0.014518, 0.020960, 0.036741, 0.018054]  
    weight = row['weight']                             
    weight *= 1. / weightFactors[ int(row['truthClass']) ] 
    weight = abs(weight)                              
    return weight  
