from typing import List, Tuple
import numpy as np
import gmst_merge.dataset as ds
import gmst_merge.family_tree as ft
import random

def choose_start_and_end_year(y1: int, y2: int) -> Tuple[int, int]:
    """
    Given the start and end years of a particular dataset, choose a start year for the overlap

    :param y1: int
        first possible year for overlap
    :param y2: int
        last possible year for overlap
    :return:
        Tuple[int, int]
    """
    trans_start_year = random.randint(y1, y2)
    trans_end_year = trans_start_year + 29
    return trans_start_year, trans_end_year


class MetaEnsembleFactory:

    def __init__(self, tails: ft.FamilyTree, heads: ft.FamilyTree):
        self.tails = tails
        self.heads = heads
        self.latest_join_year = 1981

    def make_meta_ensemble(self, n_meta_ensemble, randomize=True):
        meta_ensemble = np.zeros((2024 - 1850 + 1, n_meta_ensemble+1))

        for i in range(n_meta_ensemble):
            tail = self.tails.sample_from_tree()
            head = self.heads.sample_from_tree()

            join_start_year, join_end_year = choose_start_and_end_year(head.get_start_year(), self.latest_join_year)

            merged = ds.Dataset.join(tail, head, join_start_year, join_end_year)
            merged.anomalize(1981, 2010)

            meta_ensemble[:, i+1] = merged.data[:, 0]
            if i ==0:
                meta_ensemble[:, 0] = merged.time[:]

        return ds.Dataset(meta_ensemble, 'meta_ensemble')