#  Global Temperature Merge - a package for merging global temperature datasets.
#  Copyright \(c\) 2025 John Kennedy and Bruce Calvert
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.


import os
from pathlib import Path
from inspect import getmembers, isfunction
import json
from typing import List

import numpy as np

import gmst_merge.family_tree as ft
import gmst_merge.metaensemblefactory as mef
from gmst_merge.dataset import Dataset, Metric

from gmst_merge.metrics import metrics


def find_and_run_all_metrics(meta_ensemble: Dataset) -> List[Metric]:
    """
    Get all functions from the metrics module and run them one at a time using the
    Dataset get_metric_range method.

    :param meta_ensemble: Dataset

    :return: list[Metric]
        Returns list of Metric objects which can be printed out for a neat summary
    """
    functions_in_metrics = getmembers(metrics, isfunction)

    metric_list = []
    for ifunc in functions_in_metrics:
        metric_list.append(meta_ensemble.get_metric_range(ifunc[1]))

    return metric_list


def load_experiments():
    """Load all the experiments in the Experiments directory"""
    experiment_dir = Path("Experiments")
    experiments = []
    for file in experiment_dir.glob("*.json"):
        print(file)
        with open(file, 'r') as f:
            experiments.append(json.load(f))
    print("")
    return experiments


def run_experiment(experiment, data_dir, rng):
    experiment_name = experiment["name"]

    print(f"Running experiment {experiment_name}")

    figure_dir = Path("Figures") / f"{experiment_name}"
    figure_dir.mkdir(exist_ok=True)

    output_dir = Path("Output") / f"{experiment_name}"
    output_dir.mkdir(exist_ok=True)

    for tree in experiment['trees']:
        print(f"Running tree {tree} in experiment {experiment_name}")

        tree_filename = f'FamilyTrees/hierarchy_{tree}.json'

        tails = ft.FamilyTree.read_from_json(tree_filename, data_dir, 'tails')
        heads = ft.FamilyTree.read_from_json(tree_filename, data_dir, 'heads')
        whole = ft.FamilyTree.read_from_json(tree_filename, data_dir, 'master')

        whole.plot_tree(figure_dir / f'{tree}_treeogram.svg')
        tails.plot_tree(figure_dir / f'{tree}_tails_treeogram.svg')
        heads.plot_tree(figure_dir / f'{tree}_heads_treeogram.svg')

        factory = mef.MetaEnsembleFactory(tails, heads)
        factory.set_parameters(experiment)

        meta_ensemble = factory.make_meta_ensemble(rng, end_year=2024)

        # Calculate the desired metrics for this ensemble
        metric_list = find_and_run_all_metrics(meta_ensemble)

        # Plot the ensemble
        meta_ensemble.plot_whole_ensemble(figure_dir / f'{tree}_clusters.png', alpha=1)

        # Calculate the smoothed ensemble
        smoothed = meta_ensemble.lowess_smooth()

        # Write out the files
        meta_ensemble.to_csv(output_dir / f'{tree}.csv')
        meta_ensemble.summary_to_csv(output_dir / f'{tree}_summary.csv')
        smoothed.summary_to_csv(output_dir / f'{tree}_smoothed_summary.csv')

        # Plot various outputs
        meta_ensemble.plot_whole_ensemble(figure_dir / f'{tree}_whole_ensemble.png')
        meta_ensemble.plot_heat_map(figure_dir / f'{tree}_heat_map.png')
        meta_ensemble.plot_joy_division_histogram(figure_dir / f'{tree}_joy_division.png')

        smoothed.plot_heat_map(figure_dir / f'{tree}_smoothed_heat_map.png')
        smoothed.plot_passing_thresholds(figure_dir / f'{tree}_passing.png')
        smoothed.plot_time_series_with_exceedances(figure_dir / f'{tree}_time_series_with_exceedances.png')
        smoothed.plot_joy_division_histogram(figure_dir / f'{tree}_smoothed_joy_division.png')

        # Now 81-10 climatology
        meta_ensemble.anomalize(1981, 2010)
        smoothed = meta_ensemble.lowess_smooth()
        # Write out the files
        meta_ensemble.to_csv(output_dir / f'{tree}_1981-2010.csv')
        meta_ensemble.summary_to_csv(output_dir / f'{tree}_summary_1981-2010.csv')
        smoothed.summary_to_csv(output_dir / f'{tree}_smoothed_summary_1981-2010.csv')

        print(tails)
        print(heads)

        for metric in metric_list:
            print(metric)

        print()
    print()


if __name__ == '__main__':
    data_dir_env = os.getenv('DATADIR')
    data_dir = Path(data_dir_env) / 'ManagedData' / 'Data'

    experiments = load_experiments()

    for experiment in experiments:
        rng = np.random.default_rng(experiment['seed'])
        run_experiment(experiment, data_dir, rng)
        print(rng.integers(20))
