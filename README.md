# Categorisation of VBF and ggH HToEE events

## Setup

The first thing to do is: `source setup.sh`. This appends the appropriate pacakages to your `PYTHONPATH`. It also loads a cms environment; you may remove this line if already using a package manager.

## BDT Training

To separate the VBF or ggH HToEE signal from the dominant Drell-Yan background, we can use a Boosted Decision Tree (BDT).
For this, use the `training/train_bdt.py` script. This requires a configuration file `-c bdt_config_<vbf/ggh>.yaml` which specifies: 

* whether we are training on ggH or VBF processes
* the sample options (files dirs and names, tree names, years, ...)
* variables to train our BDT with
* preselection for all samples.

An example config for VBF training can be found here:  `bdt_config_vbf.yaml`

The command to train a simple VBF vs Bkg BDT with a train-test splitting fraction of 0.7, is:

```
python training/train_bdt.py -c configs/bdt_config_vbf.yaml -t 0.7
```

You will notice that the performance training this simple BDT is poor; the classifier predicts only the background class, due to the massive class imbalance. To remedy this, try training with the sum of signal weights increased such that they equal the sum of background weights, by adding the option `-w`.

### optimising model parameters
To optimise the model hyper parameters (HP), use the option `-o`. This submits ~6000 jobs to the IC computing cluster, one for each hyper parameter configuration. If you want to change the HP space, modify the `BDT()` class constructor inside `HToEEML.py`. Each model is cross validated; the number of folds can be specified with the `-k` option. Note that to keep the optimisation fair, you should not run with the `-r` option, since this re-loads and *reshuffles* the dataset; we want each model to use the same sample set.

Once the best model has been found, you may train/test on the full sample by running with option `-b`. This will produce plots of the ROC score for the train and test set, and save it in the working directory in a `plots` folder. The output score for the two classes is also plot.

### background reweighting
To apply a rewighting of a given MC sample (Drell-Yan as default) to data, in bins of pT(ee), use the option `-P`. This is motivated by known disagreement between Drell-Yan simulation and data. The reweighting is derived in a dielectron mass control region around the Z-mass, before being applied in the signal region. An option to reweight double differentially in bins of jet multiplicity may also be configured in the training script. 

An extra word of warning on this - reading in the control region to derive the rewighting requires much more memory than the nominal training. Therefore its best to submit the training to the IC batch, first time of running. Once the re-weighted dataframe has been saved once, it should be fine to run locally again.

## LSTM training

To use an LSTM neural network to perform separate VBF/ggJ signal from background, we use the script `training/train_lstm.py`. This takes a separate config, similar to the BDT, but with a nested list of low-level variables to be used in the LSTM layers. High level variables are also used, interfaced with the outputs of the LSTM units in fully connected layers. For an example config, see: `configs/lstm_config.yaml`.

The training can be run with the same syntax used in the BDT training. For example, to train a simple LSTM with default architecture and equalised class weights:

```
python training/train_lstm.py -c configs/lstm_config_ggh.yaml -t 0.7 -w
```

To optimise network hyper paramters, add the `-o` option. Following the optimisation, if all jobs have finished, add `-b` to train with the best model and produce ROC plots for the classifier.


## Category Optimisation
Once a model has been trained and optimised, you may use the output score to construct categories targeting ggH/VBF production.
This is done by splitting the signal + background events into boundaries defined by the output classifier score, such that the average median significance (AMS) is optimised.
To perform the optimisation e.g. for a ggH BDT, which also optimises the number of categories (defaults to between 1 and 4) the following command can be run:

```
python categoryOpt/bdt_category_opt.py -c configs/bdt_config_ggh.yaml -m models/ggH_BDT_best_clf.pickle.dat -d
```

which runs an optimisation for ggH categories. Note that the `-d` option replaces background simulated samples with data. The `-m` option specified the pre-trained BDT; in this case we use the best model from earlier HP optimisation.If wanting to use the classifier from a pT(ee) reweighted sample, be aware that the model name will be different to the nominal models.

If instead we want to use a DNN output score, we use the script `categoryOpt/dnn_category_opt.py`. An example command to run this is:

```
python categoryOpt/dnn_category_opt.py -c configs/lstm_config_ggh.yaml -a models/ggH_DNN_pt_reweighted_model_architecture.json -m models/ggH_DNN_pt_reweighted_model.hdf5 -d 
```

## Final Tag Producer
To produce the trees with the category information needed for final fits, we use `categoryOpt/make_tag_sequence.py`. This takes a config specifying the BDT boundaries for each set of tags (see `configs/bdt_boundaries_config.yaml` for an example). The script also requires an additional config with sample and training varaible info (see `configs/tag_seq_config.yaml` for info an example)

### systematics
The script which handles systematic variations and their affect on category compisition is `categoryOpt/make_tag_sequence_with_systs.py`. This should be run for each systematic variation being considered, and the Up/Down fluctuation type. An example for running the categorisation for the JEC Up variation is:

```
python categoryOpt/make_tag_sequence_with_systs.py -c configs/tag_seq_config.yaml -M configs/mva_boundaries_config.yaml -d -S JecUp -r
```

Some important notes:
* if the memory gets too high, may need to modify DataHandling.py such that we dont read every systematic in every time, since the script is only run once per systematic
* the output trees need to be hadded over procs, along with the nominal trees

### To do:
* break plotting class up to avoid duplicating code
