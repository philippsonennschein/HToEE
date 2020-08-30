# Repo to perform training of a BDT and DNN to separate VBF (H->ee) and DYMC

## Implemented features
* ROOT files -> pandas dataframe 
* training BDT (with equalised weights)
* optimisation of BDT hyper parameters (by submitting grid search + CV to IC batch)

To do:

* fix random seed in the cross validation
* fix adding variables (put it in a function that takes a string equation
* add functionality to concat years
* assert cut string vars are in nominal vars
* class for plotting
