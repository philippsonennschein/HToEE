import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import ROOT as r
r.gROOT.SetBatch(True)
import math
from itertools import product


binNames = ['qqH_Rest',
            'QQ2HQQ_GE2J_MJJ_60_120',
            'QQ2HQQ_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_0_25',
            'QQ2HQQ_GE2J_MJJ_350_700_PTH_0_200_PTHJJ_GT25',
            'QQ2HQQ_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_0_25',
            'QQ2HQQ_GE2J_MJJ_GT700_PTH_0_200_PTHJJ_GT25',
            'QQ2HQQ_GE2J_MJJ_GT350_PTH_GT200']

cm_comp = [
 [0.53, 0.14, 0.13, 0.11, 0.05, 0.02, 0.02],
 [0.07, 0.83, 0.01, 0.07, 0., 0.01, 0.01],
 [0.07, 0.02, 0.67, 0.2, 0.03, 0.02, 0.],
 [0.03, 0.03, 0.3, 0.58, 0.02, 0.04, 0.],
 [0.07, 0.01, 0.09, 0.05, 0.51, 0.28, 0.],
 [0.02, 0.02, 0.04, 0.11, 0.19, 0.61, 0.],
 [0., 0.01, 0.01, 0.06, 0.01, 0.09, 0.81]]

cm_new = [
 [0.54, 0.14, 0.13, 0.11, 0.05, 0.02, 0.02],
 [0.07, 0.80, 0.01, 0.07, 0., 0.01, 0.01],
 [0.07, 0.02, 0.62, 0.2, 0.03, 0.02, 0.],
 [0.03, 0.03, 0.3, 0.57, 0.02, 0.04, 0.],
 [0.07, 0.01, 0.09, 0.05, 0.53, 0.28, 0.],
 [0.02, 0.02, 0.04, 0.11, 0.19, 0.60, 0.],
 [0., 0.01, 0.01, 0.06, 0.01, 0.09, 0.89]]

 #Plot delta cm
def plot_delta_confusion_matrix(cm1,cm2,classes,normalize=True,title='Confusion matrix',cmap=plt.cm.Blues):
    cm1 = np.array(cm1)#, dtype=np.float32)
    cm2 = np.array(cm2)#, dtype=np.float32)
    cm1 = cm1.astype('float')/cm1.sum(axis=1)[:,np.newaxis]
    for i in range(len(cm1[0])):
        for j in range(len(cm1[1])):
            cm1[i][j] = float("{:.3f}".format(cm1[i][j]))
    cm2 = cm2.astype('float')/cm2.sum(axis=1)[:,np.newaxis]
    for i in range(len(cm2[0])):
        for j in range(len(cm2[1])):
            cm2[i][j] = float("{:.3f}".format(cm2[i][j]))
    cm = np.array(cm1)-np.array(cm2)
    for i in range(len(cm[0])):
        for j in range(len(cm[1])):
            cm[i][j] = float("{:.2f}".format(cm[i][j]))
    fig, ax = plt.subplots(figsize = (10,10))
    #plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks,classes,rotation=90)
    plt.yticks(tick_marks,classes)
    #cm = cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
    #for i in range(len(cm[0])):
    #    for j in range(len(cm[1])):
    #        cm[i][j] = float("{:.2f}".format(cm[i][j]))
    thresh = cm.max()/2.
    print(cm1)
    print(cm2)
    print(cm)
    plt.imshow(cm,interpolation='nearest',cmap=cmap)
    plt.title(title)
    for i, j in product(range(cm.shape[0]),range(cm.shape[1])):
        plt.text(j,i,cm[i,j],horizontalalignment='center',color='white' if cm[i,j]>thresh else 'black')
    plt.tight_layout()
    plt.colorbar()
    plt.ylabel('True Label')
    plt.xlabel('Predicted label')
    name = 'plotting/NN_plots/Delta_Confusion_Matrix'
    fig.savefig(name, dpi = 1200)

plot_delta_confusion_matrix(cm_new,cm_comp,binNames,normalize=True)
