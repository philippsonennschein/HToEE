#usual imports
import ROOT as r
r.gROOT.SetBatch(True)
import numpy as np
import pandas as pd
import sys
import pickle
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn import cross_validation
from os import path, system
from array import array
from addRowFunctions import addPt, truthDipho, reco, diphoWeight, altDiphoWeight, truthJets, jetWeight, truthClass, jetPtToClass, procWeight
from otherHelpers import prettyHist, getAMS, computeBkg, getRealSigma
from root_numpy import tree2array, fill_hist
import usefulStyle as useSty
from math import pi

from keras.models import Sequential 
from keras.initializers import RandomNormal 
from keras.layers import Dense 
from keras.layers import Activation 
from keras.layers import * 
from keras.optimizers import Nadam 
from keras.optimizers import adam 
from keras.regularizers import l2 
from keras.callbacks import EarlyStopping 
from keras.utils import np_utils 
import h5py

#configure options
from optparse import OptionParser
parser = OptionParser()
parser.add_option('-t','--trainDir', help='Directory for input files')
parser.add_option('-d','--dataFrame', default=None, help='Path to dataframe if it already exists')
parser.add_option('--intLumi',type='float', default=35.9, help='Integrated luminosity')
parser.add_option('--trainParams',default=None, help='Comma-separated list of colon-separated pairs corresponding to parameters for the training')
(opts,args)=parser.parse_args()

def checkDir( dName ):
  if dName.endswith('/'): dName = dName[:-1]
  if not path.isdir(dName): 
    system('mkdir -p %s'%dName)
  return dName

#setup global variables
trainDir = checkDir(opts.trainDir)
frameDir = checkDir(trainDir.replace('trees','frames'))
modelDir = checkDir(trainDir.replace('trees','models'))
plotDir  = checkDir(trainDir.replace('trees','plots') + '/NNCategorisation')
if opts.trainParams: opts.trainParams = opts.trainParams.split(',')

trainFrac   = 0.7
validFrac   = 0.1
sampleFrac = 1.0
nGGHClasses = 9
sampleFrame = False
equaliseWeights = False
weightScale = False

binNames = ['0J low','0J high','1J low','1J med','1J high','2J low low','2J low med','2J low high','BSM'] 

#define the different sets of variables used

allVars   = ['n_jet_30','dijet_Mjj',
              'dijet_leadEta','dijet_subleadEta','dijet_subsubleadEta',
              'dijet_LeadJPt','dijet_SubJPt','dijet_SubsubJPt',
              'dijet_leadPUMVA','dijet_subleadPUMVA','dijet_subsubleadPUMVA',
              'dijet_leadDeltaPhi','dijet_subleadDeltaPhi','dijet_subsubleadDeltaPhi',
              'dijet_leadDeltaEta','dijet_subleadDeltaEta','dijet_subsubleadDeltaEta',
              'dipho_leadIDMVA','dipho_subleadIDMVA','dipho_lead_ptoM','dipho_sublead_ptoM',
              'dipho_leadEta','dipho_subleadEta',
              'CosPhi','vtxprob','sigmarv','sigmawv']


procFileMap = {'ggh':'ggH_powheg_jetinfo.root'}
theProcs = procFileMap.keys()

#either get existing data frame or create it
trainTotal = None
if not opts.dataFrame:
  trainFrames = {}
  #get the trees, turn them into arrays
  for proc,fn in procFileMap.iteritems():
      trainFile   = r.TFile('%s/%s'%(trainDir,fn))
      if proc[-1].count('h') or 'vbf' in proc:
        trainTree = trainFile.Get('vbfTagDumper/trees/%s_125_13TeV_GeneralDipho'%proc)
      else:
        print('Did not get an tree. Exiting')
        sys.exit(1)
      trainTree.SetBranchStatus('nvtx',0)
      trainTree.SetBranchStatus('VBFMVAValue',0)
      trainTree.SetBranchStatus('dZ',0)
      trainTree.SetBranchStatus('centralObjectWeight',0)
      trainTree.SetBranchStatus('rho',0)
      trainTree.SetBranchStatus('nvtx',0)
      trainTree.SetBranchStatus('event',0)
      trainTree.SetBranchStatus('lumi',0)
      trainTree.SetBranchStatus('processIndex',0)
      trainTree.SetBranchStatus('run',0)
      trainTree.SetBranchStatus('npu',0)
      trainTree.SetBranchStatus('puweight',0)
      newFile = r.TFile('/vols/cms/es811/Stage1categorisation/trainTrees/new.root','RECREATE')
      newTree = trainTree.CloneTree()
      trainFrames[proc] = pd.DataFrame( tree2array(newTree) )
      del newTree
      del newFile
      trainFrames[proc]['proc'] = proc
  print 'got trees'
  
  #create one total frame
  trainList = []
  for proc in theProcs:
      trainList.append(trainFrames[proc])
  trainTotal = pd.concat(trainList)
  del trainFrames
  print 'created total frame'
  
  #then filter out the events into only those with the phase space we are interested in
  trainTotal = trainTotal[trainTotal.dipho_mass>100.]
  trainTotal = trainTotal[trainTotal.dipho_mass<180.]
  print 'done mass cuts'
  
  #some extra cuts that are applied for diphoton BDT in the AN
  trainTotal = trainTotal[trainTotal.dipho_leadIDMVA>-0.9]
  trainTotal = trainTotal[trainTotal.dipho_subleadIDMVA>-0.9]
  trainTotal = trainTotal[trainTotal.dipho_lead_ptoM>0.333]
  trainTotal = trainTotal[trainTotal.dipho_sublead_ptoM>0.25]
  trainTotal = trainTotal[trainTotal.HTXSstage1_1_cat!=-100]   
  trainTotal = trainTotal[trainTotal.dijet_Mjj<350]
  print 'done basic preselection cuts' 

  #add extra info to dataframe
  print 'about to add extra columns'
  trainTotal['truthClass'] = trainTotal.apply(truthClass, axis=1)
  #Remove vbf-like classes as don't care about predicting them
  # only do this at gen level in training!
  trainTotal = trainTotal[trainTotal.truthClass<9]
  trainTotal['diphopt'] = trainTotal.apply(addPt, axis=1)
  trainTotal['reco'] = trainTotal.apply(reco, axis=1)
  trainTotal['truthJets'] = trainTotal.apply(truthJets, axis=1)
  trainTotal['procWeight'] = trainTotal.apply(procWeight, axis=1)
  trainTotal['truthDipho'] = trainTotal.apply(truthDipho, axis=1)
  trainTotal['procWeight'] = trainTotal.apply(procWeight, axis=1)
  print 'all columns added'

  #only select processes relevant for nJet training
  trainTotal = trainTotal[trainTotal.truthJets>-1]
  trainTotal = trainTotal[trainTotal.reco!=-1]
  trainTotal = trainTotal[trainTotal.truthClass!=-1]

  #remove vbf_like procs - don't care about predicting these/ dont want to predict these
  print 'done basic preselection cuts'

  #replace missing entries with -10 to avoid bias from -999 
  trainTotal = trainTotal.replace(-999,-10) 

  # do this step if later reading df with python 2 
  #trainTotal.loc[:, 'proc'] = trainTotal['proc'].astype(str)  
 
 #scale weights or normalised weights closer to one if desired

  if weightScale:
    print('MC weights before were:')
    print(trainTotal['weight'].head(10))
    trainTotal.loc[:,'weight'] *= 1000
    print('weights after scaling are:') 
    print(trainTotal['weight'].head(10))                                                                        

  #save as a pickle file
  trainTotal.to_pickle('%s/nClassNNTotal.pkl'%frameDir)
  print 'frame saved as %s/nClassNNTotal.pkl'%frameDir

#read in dataframe if above steps done once before
else:
  trainTotal = pd.read_pickle(opts.dataFrame)
  print 'Successfully loaded the dataframe \n'

#used when setting weights for normalisation, in row functions
procWeightDict = {}
for iProc in range(nGGHClasses):  #from zero to 8 are ggH bins
  sumW = np.sum(trainTotal[trainTotal.truthClass==iProc]['weight'].values)
  sumW_proc = np.sum(trainTotal[trainTotal.truthClass==iProc]['procWeight'].values)
  procWeightDict[iProc] = sumW
  print 'Sum of weights for ggH STXS bin %i is: %.2f' %  (iProc,sumW_proc)  
  print 'Frac is %.6f' % (sumW/ (np.sum(trainTotal['weight'].values)))
  print 'Sum of proc weights for bin %i is: %.5f' % (iProc,sumW_proc)

#shape and shuffle definitions
theShape = trainTotal.shape[0]
classShuffle = np.random.permutation(theShape)
classTrainLimit = int(theShape*trainFrac)
classValidLimit = int(theShape*(trainFrac+validFrac))

#setup the various datasets for multiclass training
classI        = trainTotal[allVars].values
classProcW    = trainTotal['procWeight'].values
classFW       = trainTotal['weight'].values
classR        = trainTotal['reco'].values
classY        = trainTotal['truthClass'].values

#shuffle datasets
classI        = classI[classShuffle]
classFW       = classFW[classShuffle]
classProcW    = classProcW[classShuffle]
classR        = classR[classShuffle]
classY        = classY[classShuffle]

#split datasets
X_train, X_valid, X_test              = np.split( classI,     [classTrainLimit,classValidLimit] )
w_mc_train, w_mc_valid, w_mc_test     = np.split( classFW,    [classTrainLimit,classValidLimit] )
procW_train, procW_valid, procW_test  = np.split( classProcW, [classTrainLimit,classValidLimit] )
classTrainR, classValidR, classTestR  = np.split( classR,     [classTrainLimit,classValidLimit] )
y_train, y_valid, y_test              = np.split( classY,     [classTrainLimit,classValidLimit] )

#one hot encode target column (necessary for keras) and scale training
y_train_onehot  = np_utils.to_categorical(y_train, num_classes=9)
y_valid_onehot  = np_utils.to_categorical(y_valid, num_classes=9)
y_test_onehot   = np_utils.to_categorical(y_test, num_classes=9)
scaler          = StandardScaler()
X_train_scaled  = scaler.fit_transform(X_train)

paramExt = ''
trainParams = {}
if opts.trainParams:
  paramExt = '__'
  for pair in opts.trainParams:
    key  = pair.split(':')[0]
    data = pair.split(':')[1]
    trainParams[key] = data
    paramExt += '%s_%s__'%(key,data)
  paramExt = paramExt[:-2]

print 'training HPs:'
print(trainParams)

numLayers = int(trainParams['hiddenLayers'],10)
nodes     = int(trainParams['nodes'],10)
dropout   = float(trainParams['dropout'])
batchSize = int(trainParams['batchSize'],10)

#build the category classifier 
num_inputs  = X_train_scaled.shape[1]
num_outputs = nGGHClasses

model = Sequential()

#FIXME: overwriting nodes variable!
for i, nodes in enumerate([200] * numLayers):                                                                   
  if i == 0: #first layer
    model.add(
    Dense(
            nodes,
            kernel_initializer='glorot_normal',
            activation='relu',
            kernel_regularizer=l2(1e-5),
            input_dim=num_inputs
            )
    )
    model.add(Dropout(dropout))
  else: #hidden layers
    model.add(
    Dense(
            nodes,
            kernel_initializer='glorot_normal',
            activation='relu',
            kernel_regularizer=l2(1e-5),
            )
    )
    model.add(Dropout(dropout))

#final layer
model.add(
        Dense(
            num_outputs,
            kernel_initializer=RandomNormal(),
            activation='softmax'
            )
        )

model.compile(
        loss='categorical_crossentropy',
        optimizer=Nadam(),
        metrics=['accuracy']
)
callbacks = []
callbacks.append(EarlyStopping(patience=50))
model.summary()


#Fit the model with best n_trees/estimators
print('Fitting on the training data')
history = model.fit(
    X_train_scaled,
    y_train_onehot,
    sample_weight=w_mc_train,
    batch_size=batchSize,
    epochs=1000,
    shuffle=True,
    callbacks=callbacks # add function to print stuff out there
    )
print('Done')


#save model

modelDir = trainDir.replace('trees','models')
if not path.isdir(modelDir):
  system('mkdir -p %s'%modelDir)
model.save('%s/nClassesNN_MCweights__%s.h5'%(modelDir,paramExt))
print 'saved NN as %s/nClassesNNMCWeights__%s.h5'%(modelDir,paramExt)


#plot train and validation acc over time
'''
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()
plt.savefig('NNAccuracyHist.pdf')
plt.savefig('NNAccuracyHist.png')
'''
'''
#save model
modelDir = trainDir.replace('trees','models')
if not path.isdir(modelDir):
  system('mkdir -p %s'%modelDir)
model.save_weights('%s/ggHNeuralNet.h5'%(modelDir))
print 'saved as %s/ggHNeuralNet%s.h5'%(modelDir,paramExt)
'''

'''
#Evaluate performance with priors 
yProb = model.predict(X_test_scaled)
predProbClass = y_prob.reshape(y_test.shape[0],nGGHClasses)
totSumW =  np.sum(trainTotal['weight'].values)
priors = [] #easier to append to list than numpy array. Then just convert after
for i in range(nGGHClasses):
  priors.append(procWeightDict[i]/totSumW)
predProbClass *= np.asarray(priors) #this is part where include class frac, not MC frac
classPredY = np.argmax(predProbClass, axis=1) 
print 'Accuracy score with priors'
print(accuracy_score(y_test, classPredY, sample_weight=w_mc_test))
'''

#Evaluate performance, no priors
y_prob = model.predict(X_test_scaled) 
y_pred = y_prob.argmax(axis=1)
print 'Accuracy score: '
NNaccuracy = accuracy_score(y_test, y_pred, sample_weight=w_mc_test)
print(NNaccuracy)


print
print '                   reco class =  %s' %classTestR
print '          NN predicted class  =  %s'%y_pred
print '                  truth class =  %s'%y_test
print '         Reco accuracy score  =  %.4f' %accuracy_score(y_test,classTestR, sample_weight=w_mc_test) #include orig MC weights here
print 'NN accuracy score (no priors )=  %.4f' %NNaccuracy #include orig MC weights here

mLogLoss = log_loss(y_test, y_prob, sample_weight=procW_test)
print 'NN log-loss=  %.4f' %mLogLoss

