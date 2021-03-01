#!/bin/bash

cd /vols/cms/jwd18/Hee/MLCategorisation/CMSSW_10_2_0/src/HToEE
source setup.sh

declare -a systs=("jesTotalUp" "jesTotalDown" "jerUp" "jerDown" "ElPtScaleUp" "ElPtScaleDown")

#2016
#usual systs
for syst in "${systs[@]}"
do
    python categoryOpt/generic_tagger.py -c configs/tag_seq_config_2016.yaml -M configs/mva_boundaries_config.yaml -d -S "$syst"
done
#weight systs (plus nominal branches)
python categoryOpt/generic_tagger.py -c configs/tag_seq_config_2016.yaml -M configs/mva_boundaries_config.yaml -d -W 
#data with no syst variations
python categoryOpt/generic_tagger.py -c configs/tag_seq_config_2016.yaml -M configs/mva_boundaries_config.yaml -d -D

#2017
#usual systs
for syst in "${systs[@]}"
do
    python categoryOpt/generic_tagger.py -c configs/tag_seq_config_2017.yaml -M configs/mva_boundaries_config.yaml -d -S "$syst" 
done
#weight systs (plus nominal branches)
python categoryOpt/generic_tagger.py -c configs/tag_seq_config_2017.yaml -M configs/mva_boundaries_config.yaml -d -W 
#data with no syst variations
python categoryOpt/generic_tagger.py -c configs/tag_seq_config_2017.yaml -M configs/mva_boundaries_config.yaml -d -D

#2018
#usual systs
for syst in "${systs[@]}"
do
    python categoryOpt/generic_tagger.py -c configs/tag_seq_config_2018.yaml -M configs/mva_boundaries_config.yaml -d -S "$syst" 
done
#weight systs (plus nominal branches)
python categoryOpt/generic_tagger.py -c configs/tag_seq_config_2018.yaml -M configs/mva_boundaries_config.yaml -d -W 
#data with no syst variations
python categoryOpt/generic_tagger.py -c configs/tag_seq_config_2018.yaml -M configs/mva_boundaries_config.yaml -d -D
