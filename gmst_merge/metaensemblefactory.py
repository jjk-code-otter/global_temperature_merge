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


from typing import List, Tuple
import numpy as np
import gmst_merge.dataset as ds
import gmst_merge.family_tree as ft
import random


def choose_start_and_end_year(y1: int, y2: int, rng, overlap_period=30) -> Tuple[int, int]:
    """
    Given the start and end years of a particular dataset, choose a start year for the overlap

    :param y1: int
        first possible year for overlap
    :param y2: int
        last possible year for overlap
    :return:
        Tuple[int, int]
    """
    trans_start_year = rng.integers(y1, y2 + 1)
    trans_end_year = trans_start_year + overlap_period - 1
    return trans_start_year, trans_end_year


class MetaEnsembleFactory:
    def __init__(self, tails: ft.FamilyTree, heads: ft.FamilyTree):
        self.tails = tails
        self.heads = heads
        self.latest_join_year = 1981
        self.output_baseline = [1850, 1900]
        self.overlap_period = 30
        self.random_overlap = True
        self.random_tree = False
        self.default_overlap = [1981, 2010]
        self.ensemble_size = 1
        self.modified_ensemble_size = 1
        self.thinning = None
        self.balanced = False

    def set_parameters(self, parameter_dictionary) -> None:
        """
        Set the parameters from a dictionary. Dictionary keys should be class attributes if you want them to
        change anything. The dictionary can only change existing keys.

        :param parameter_dictionary: dict
            Dictionary with keys that match
        :return: None
        """
        for key in parameter_dictionary:
            if hasattr(self, key):
                setattr(self, key, parameter_dictionary[key])
            else:
                if key not in ['name', 'description', 'trees', 'seed']:
                    print(f"Tried to set {key} but {key} is not a class attribute. These are defined in __init__")

    def make_meta_ensemble(self, rng, end_year=2024) -> ds.Dataset:
        """
        Make a meta ensemble

        :param n_meta_ensemble: int
            Number of ensemble members to generate
        :param rng: Random Number Generator
            Numpy random number generator
        :return: ds.Dataset
            Ensemble dataset
        """
        meta_ensemble = np.zeros((end_year - 1850 + 1, self.ensemble_size + 1))

        if self.balanced and not self.random_tree:
            # If balanced is set to True then generate a balanced ensemble; does not work if random_tree is True
            tail_list = ft.balanced_pick_ensemble(self.tails.tree, self.ensemble_size, rng)
            head_list = ft.balanced_pick_ensemble(self.heads.tree, self.ensemble_size, rng)

        for i in range(self.ensemble_size):
            if self.balanced and not self.random_tree:
                tails = ft.FamilyTree([tail_list[i]])
                heads = ft.FamilyTree([head_list[i]])
                tail = tails.sample_from_tree(rng)
                head = heads.sample_from_tree(rng)
            # Else if random_tree is set to True then generate a random tree for each ensemble member
            elif self.random_tree:
                tails = ft.FamilyTree.make_random_tree(self.tails.tree, rng)
                heads = ft.FamilyTree.make_random_tree(self.heads.tree, rng)
                tail = tails.sample_from_tree(rng)
                head = heads.sample_from_tree(rng)
            # else select from the tree
            else:
                tail = self.tails.sample_from_tree(rng)
                head = self.heads.sample_from_tree(rng)

            # If random_overlap is set to True then randomly choose the start year
            if self.random_overlap:
                join_start_year, join_end_year = choose_start_and_end_year(
                    head.get_start_year(), self.latest_join_year, rng, overlap_period=self.overlap_period
                )
            # else use the class defaults
            else:
                join_start_year = self.default_overlap[0]
                join_end_year = self.default_overlap[1]

            merged = ds.Dataset.join(tail, head, join_start_year, join_end_year)
            merged.anomalize(self.output_baseline[0], self.output_baseline[1])

            meta_ensemble[:, i + 1] = merged.data[:, 0]
            if i == 0:
                meta_ensemble[:, 0] = merged.time[:]

        output_dataset = ds.Dataset(meta_ensemble, 'meta_ensemble')

        # If a thinning method is specified then call that method on the ensemble
        print(f"Thinning method: {self.thinning}")
        if self.thinning is None:
            return output_dataset
        else:
            if self.thinning == 'cluster_ensemble':
                output_dataset = output_dataset.cluster_ensemble(self.modified_ensemble_size, rng)
            elif self.thinning == 'thin_ensemble':
                output_dataset = output_dataset.thin_ensemble(self.modified_ensemble_size, rng)

        return output_dataset
