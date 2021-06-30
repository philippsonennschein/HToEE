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

Note that for each sample, the effeciency x acceptance, and XS (x BR) should also be specified in the constructor of the `python/DataHandling.py` class.


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

To use an LSTM neural network to perform separate VBF/ggH signal from background, we use the script `training/train_lstm.py`. This takes a separate config, similar to the BDT, but with a nested list of low-level variables to be used in the LSTM layers. High level variables are also used, interfaced with the outputs of the LSTM units in fully connected layers. For an example config, see: `configs/lstm_config.yaml`.

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
To produce the trees with the category information needed for final fits, we use `categoryOpt/make_tag_sequence.py`. This takes a config specifying the BDT boundaries for each set of tags (see `configs/bdt_boundaries_config.yaml` for an example). The script also requires an additional config with sample and training varaible info (see `configs/tag_seq_config.yaml` for info an example). An example command to run the tagger for 2016 without any systematic variations is:

```
python categoryOpt/generic_tagger.py -c configs/tag_seq_config_2016.yaml -M configs/mva_boundaries_config.yaml -d 
```

The option `-d` is important since it tells the script to use data as replacement for simulated background. Note that this step is really just a check - producing trees for workspaces (inlcuding the nominal trees) is all handled below.

### systematics
The same script can handle systematic variations and their affect on category composition. The name of the systematic is specified with the `-S` options. This should be run for each systematic variation being considered, and the Up/Down fluctuation type. An example for running the categorisation for the JEC Up variation is:

```
python categoryOpt/generic_tagger.py -c configs/tag_seq_config_2016.yaml -M configs/mva_boundaries_config.yaml -d -S jecUp -r
```

For variations that only effect the event weight, we do not split by systematic. Instead, we run with the `-W` option, which dumps all weight variations alongside the nominal event weight in the nominal tree. This tree should be hadded as usual with the other systematics trees at the end. Note that the script takes care not to dump these weight variation branches in the Data trees.

Note that the nominal trees are also filled when running the weight variations, so no need to run the tagger with no systematics again. This would actually overwrite the weight variation trees, since they have the same naming convention (same nominal tree names that is, since there are additional weight systematic branches in the latter).

Any additional systematics should be added to the dictionary keys in `python/syst_maps.py`

Finally, since we only dump systematics for signal events, if we run the above scripts then we will be missing our data trees. Hence, to run the data categorisation only at the end you can add the `-D` option:

```
python categoryOpt/generic_tagger.py -c configs/tag_seq_config_2016.yaml -M configs/mva_boundaries_config.yaml -d -D
```

Some other important notes:
* `submissions/sub_complete_tagger/sh` is a nice script that produces all needed ROOT files for the fits, including syst varied files. You may add/subtract systs here as needed.
* the order of the variables in the tag sequence config **must be the same** as the order they are given in the training config
* this script and all the systematics variations should be run once per year, such that the signal models can be split at the fit stage
* if the memory gets too high, you could to modify DataHandling.py such that we dont read every systematic in every time, since the script is only run once per systematic
* the output trees need to be hadded over procs for each year e.g. for 2016 ggH: `hadd ggH_Hee_2016.root output_trees/2016/ggh_125_13TeV_*`
