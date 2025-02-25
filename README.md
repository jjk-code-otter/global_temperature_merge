Global temperature merge
========================

Code for merging estimates of global mean temperature into a single ensemble dataset of global 
mean temperature.

Input datasets are converted into a common format using code in the "translators" directory.

Merging is performed using `make_meta_ensemble.py` based on dataset family trees defined in json files 
such as `FamilyTrees/hierarchy_ur.json` which are organised into experiments 
such as `Experiments/basic.json`.
