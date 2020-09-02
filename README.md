# BDTs and DNNs for VBF H->ee searches 

## BDT Training

To separate the VBF HToEE signal from the dominant Drell-Yan background, we may use a Boosted Decision Tree (BDT).
For this, you may use the `train_bdt.py` script. This requires a configuration file `-c bdt_config.yaml` which specifies the sample options (files dirs and names, tree names, years, ...), variables to train with, and a preselection for all samples. An example config can be found here:  `bdt_config.yaml`

The command to train a simple BDT with a train-test splitting fraction of 0.7, is:

```
python -c bdt_config.yaml -t 0.7
```

You will notice that the performance training this simple BDT is poor; the classifier predicts only the background class, due to the massive class imbalance. To remedy this, try training with the sum of signal weights increased such that they equal the sum of background weights, by adding the option `-w`.

To optimise the model hyper parameters, use the option `-o`. This submits ~6000 jobs to the IC computing cluster, one for each hyper parameter configuration. If you want to change the hyper parameter space, modify the `BDT()` class constructor inside `HToEEML.py`. Each model is cross validated; the number of folds can be specified with the `-k` option.
If you want to re-run the optimisation again, delete the generated text file first (`bdt_hp_opt.txt`).

Once the best model has been found, you may train/test on the full sample by running with option `-b`. This will produce plots of the ROC score for the train and test set, and save it in the working directory in a `plots` folder. The output score for the two classes is also plot.

### Other features
* Plot the input features for both classes using: `plot_input_features -c bdt_config.yaml`
* Train on multiple years combined (2016, 2017, and 2018)
* Add additional features based on existing ones, by specifying the formula in the `vars_to_add` section of the config (not yet added)
* Plot data for the inputs as well (not yet added) (data/mc)

### To do:
* fix random seed in the cross validation
* fix adding variables (put it in a function that takes a string equation)
* assert cut string vars are in nominal vars
* tidy plotting class
* tidy variable conversions e.g. str(year) can just be read in as a string probs
* Auto delete the text file `bdt_hp_opt.txt` if it exists (when running initial -o submit script, not inside loop)

## BDT Category Optimisation

Once a BDT has been trained and optimised, you may use the output score to construct categories targeting VBF production.
This is done by splitting the signal+background events into boundaries defined by the output score, such that the average median significance (AMS) is optimised.
To perform the optimisation, which also optimises the number of categories (defaults to between 1 and 4) the following command can be run:

``python bdt_category_opt.py -c bdt_config.yaml -m models/clf.pickle.dat -d``

Note that the `-d` option replaces background simulated samples with data. The `-m` option specified the pre-trained BDT.
