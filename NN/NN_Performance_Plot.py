import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


cm = [
    [0.41, 0.05, 0.12, 0.03, 0.03, 0.34, 0.02],
    [0.01, 0.61, 0.07, 0.13, 0.02, 0.05, 0.11],
    [0.03, 0.05, 0.54, 0.3, 0.01, 0.04, 0.03],
    [0.01, 0.1, 0.17, 0.64, 0.01, 0.07, 0.01],
    [0., 0.01, 0.01, 0., 0.95, 0., 0.01],
    [0.12, 0.1, 0.05, 0.05, 0.02, 0.66, 0.01],
    [0.01, 0.11, 0.05, 0.01, 0.02, 0., 0.81]
    ]
binNames = ['MJJ_GT700_PTH_0_200_PTHJJ_GT25', 'Rest', 'MJJ_350_700_PTH_0_200_PTHJJ_GT25', 'MJJ_350_700_PTH_0_200_PTHJJ_0_25', 'MJJ_GT350_PTH_GT200', 'MJJ_GT700_PTH_0_200_PTHJJ_0_25', 'MJJ_60_120']

def plot_performance_plot(cm=cm,labels=binNames):
    fig, ax = plt.subplots(figsize = (10,10))
    plt.rcParams.update({
    'font.size': 9})
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks,labels,rotation=90)
    #ax.hist(output_score_qqh0, bins=50, label='FWDH', histtype='step',weights=qqh0_w)
    #ax.hist(output_score_qqh1, bins=50, label='rest', histtype='step',weights=qqh1_w)
    cm = np.array(cm)
    bottom = np.zeros(len(labels))
    for i in range(len(cm)):
        ax.bar(labels, cm[:,i],label=labels[i],bottom=bottom)
        bottom += np.array(cm[:,i])
    plt.legend()
    current_bottom, current_top = ax.get_ylim()
    ax.set_ylim(bottom=0, top=current_top*1.3)
    #plt.title('Performance Plot')
    plt.ylabel('Fraction of events')
    ax.set_xlabel('Events', ha='right',x=1,size=9) #, x=1, size=13)
    name = 'plotting/NN_plots/NN_Performance_Plot'
    plt.savefig(name, dpi = 500)
    plt.show()

plot_performance_plot()
