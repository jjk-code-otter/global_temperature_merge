Global temperature merge
========================

Code for merging estimates of global mean temperature into a single ensemble dataset of global mean temperature.

Input datasets are converted into a common format using code in the "translators" directory.

Merging is performed using `combinator.py` based on dataset family trees defined in json files such as `hierarchy_ur.json`.

Some other plots are produced by `compare_all_ensembles.py` and `plot_hierarchy.py`.
