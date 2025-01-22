#  Global Temperature Merge - a package for merging global temperature datasets.
#  Copyright \(c\) 2025 John Kennedy
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

import gmst_merge.family_tree as ft
import gmst_merge.metaensemblefactory as mef


def build_ensemble_and_plot(tree_filename, tag, n_meta_ensemble):
    print(tag)

    if tag in ['sst', 'lsat', 'interp']:
        master = ft.FamilyTree.read_from_json(tree_filename, DATA_DIR, 'master')
        master.plot_tree(f'Figures/{tag}_tree.svg')

    tails = ft.FamilyTree.read_from_json(tree_filename, DATA_DIR, 'tails')
    heads = ft.FamilyTree.read_from_json(tree_filename, DATA_DIR, 'heads')

    factory = mef.MetaEnsembleFactory(tails, heads)

    randomize = (tag == 'random')

    meta_ensemble = factory.make_meta_ensemble(n_meta_ensemble, randomize=randomize)
    meta_ensemble.to_csv(f'Output/{tag}.csv')

    meta_ensemble.anomalize(1850, 1900)
    smoothed = meta_ensemble.lowess_smooth()

    # Plot various outputs
    meta_ensemble.plot_whole_ensemble(f'Figures/{tag}_whole_ensemble.png')
    meta_ensemble.plot_heat_map(f'Figures/{tag}_heat_map.png')
    meta_ensemble.plot_joy_division_histogram(f'Figures/{tag}_joy_division.png')

    smoothed.plot_heat_map(f'Figures/{tag}_smoothed_heat_map.png')
    smoothed.plot_passing_thresholds(f'Figures/{tag}_passing.png')
    meta_ensemble.plot_joy_division_histogram(f'Figures/{tag}_smoothed_joy_division.png')

    smoothed.plot_time_series_with_exceedances(f'Figures/{tag}_time_series_with_exceedances.png')

    print(tails)
    print(heads)
    print(meta_ensemble)


if __name__ == '__main__':
    data_dir_env = os.getenv('DATADIR')
    DATA_DIR = Path(data_dir_env) / 'ManagedData' / 'Data'

    n_meta_ensemble = 10000
    cover_factor = 1.96  # 1.645

    for tree in ['ur_pseudo', 'random', 'ur_ensembles_only', 'ur', 'sst', 'lsat', 'interp',
                 'sst_ensembles_only', 'lsat_ensembles_only', 'equal']:
        build_ensemble_and_plot(f'FamilyTrees/hierarchy_{tree}.json', tree, n_meta_ensemble)
