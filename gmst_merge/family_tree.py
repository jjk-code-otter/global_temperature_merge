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

import copy
# import random
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import gmst_merge.dataset as ds


def pick_one(inarr: list, rng):
    """
    Given a list of lists, recursively pick from the entries until you find a non-list item and return that

    :param inarr: list
        List containing lists and/or strings
    :return:
    """
    selection = rng.choice(inarr)

    # Sometimes this version of "choice" returns a list and sometimes it converts the list into a numpy ndarray.
    if isinstance(selection, np.ndarray):
        selection = selection.tolist()

    if isinstance(selection, list):
        selection = pick_one(selection, rng)
    else:
        return selection
    return selection


def split_list(in_lst: list, n_splits: int, rng) -> list:
    """
    Given a list, split it into n_splits random groups

    :param in_lst: list
        List to be split
    :param n_splits: int
        Number of output lists to divide the list into
    :return: list
    """
    new_lst = copy.deepcopy(in_lst)

    # mix the list up
    rng.shuffle(new_lst)

    # partition list randomly
    number_of_items = len(new_lst)
    split_points = rng.choice(number_of_items - 2, n_splits - 1, replace=False) + 1
    split_points.sort()
    result = np.split(new_lst, split_points)

    # convert back to a regular list
    result = [x.tolist() for x in result]

    return result


def label_by_depth(lst, lbl=0):
    """Recursively label each non-list in a list of lists by its depth"""
    gst = copy.deepcopy(lst)
    for i, elem in enumerate(gst):
        if isinstance(elem, list):
            gst[i] = label_by_depth(elem, lbl + 1)
        else:
            gst[i] = lbl
    return gst


def read_tree(lst, data_dir):
    """Recursively read datasets from list of lists"""
    gst = copy.deepcopy(lst)
    for i, elem in enumerate(gst):
        if isinstance(elem, list):
            gst[i] = read_tree(elem, data_dir)
        else:
            gst[i] = ds.Dataset.read_csv(elem, data_dir)
    return gst


def get_all_members(lst):
    """Get a list of all non-list items in a list of lists recursively"""
    members = []
    if isinstance(lst, list):
        for i in lst:
            if isinstance(i, list):
                members = members + get_all_members(i)
            else:
                if isinstance(i, ds.Dataset):
                    members.append(i.name)
                else:
                    members.append(i)
    else:
        members.append(lst)

    return members


def plumb(ax, depth, all_members, inlist):
    """Plot hierarchy recursively"""
    midys = []
    for i, member in enumerate(inlist):
        if isinstance(member, list):
            midys.append(plumb(ax, depth - 1, all_members, member))
        else:
            midys.append(all_members.index(member.name))

    m1 = midys[0]
    m2 = midys[-1]
    ax.plot([depth, depth], [m1, m2], linewidth=3, color='black')
    top_level_midy = (m1 + m2) / 2
    ax.plot([depth, depth + 1], [top_level_midy, top_level_midy], linewidth=3, color='black')

    return top_level_midy


class FamilyTree:
    """
    A FamilyTree is essentially a list of lists, which contains Datasets. Functionality is limited to sampling
    from the tree and plotting the tree. Trees can be created from json files or randomly from a list of datasets.
    """

    def __init__(self, inlist):
        self.tree = inlist

    def __str__(self):
        return f"{self.tree}"

    @staticmethod
    def read_from_json(json_file, data_dir, type):
        """
        Read from a json file containing a family tree. Data are read in from the data_dir. type is one of
        'master', 'head' or 'tail'.

        :param json_file: str
            Path of the json file
        :param data_dir: str
            Path of the data directory
        :param type: str
            One of 'master', 'head', or 'tail'
        :return: FamilyTree
            FamilyTree containing the datasets specified in the json file
        """
        if type not in ['heads', 'tails', 'master']:
            raise ValueError(f'Unknown type {type} must be one of head, tail, master')
        with open(json_file, 'r') as f:
            basic_tree = json.load(f)
        basic_tree = basic_tree[type]
        filled_tree = FamilyTree(FamilyTree.read_from_directory(basic_tree, data_dir))
        return filled_tree

    @staticmethod
    def read_from_directory(basic_tree, data_dir):
        """
        Given a tree defined as a list of lists with strings giving the dataset names, create a list of lists holding
        the actual datasets.

        :param basic_tree: list
            List of lists with strings giving the dataset names
        :param data_dir: str
            Path of the data directory from which the data will be read.
        :return: list
            List of lists containing the Datasets
        """
        return read_tree(basic_tree, data_dir)

    @staticmethod
    def make_random_tree(list_of_datasets, rng):
        """
        Given a list of datasets, generate a list of lists specifying a hierarchical family tree by repeatedly
        grouping elements.

        :param list_of_datasets: List[str]
            List of datasets to be
        :param rng:
            numpy random number generator
        :return:
        """
        new_list = copy.deepcopy(list_of_datasets)

        max_depth = 4

        for i in range(max_depth):
            # choose how many breaks to have
            n_items = len(new_list)
            breaks = rng.integers(0, n_items)  # The largest number of breaks for n items is n-1
            # bail if no breaks selected
            if breaks == 0:
                break
            # split list chosen number of times
            new_list = split_list(new_list, breaks, rng)

        return FamilyTree(new_list)

    def sample_from_tree(self, rng) -> ds.Dataset:
        """
        Choose a single dataset from the tree by sampling branches at random.

        :return: Dataset
            The chosen dataset
        """
        chosen_dataset = pick_one(self.tree, rng)
        chosen_dataset = chosen_dataset.sample_from_ensemble(rng)
        return chosen_dataset

    def plot_tree(self, filename) -> None:
        """
        Plot the FamilyTree

        :param filename: str
            Path of file to which the plot will be written
        :return: None
        """
        # Plot all the interesting hierarchies
        fig, axs = plt.subplots()
        fig.set_size_inches(10, 16)

        all_members = get_all_members(self.tree)
        all_member_depths = get_all_members(label_by_depth(self.tree, 1))

        axs.set_ylim(-0.5, len(all_members))

        # Draw all members
        for i, member in enumerate(all_members):
            axs.text(-0.1, i, member, ha='right', va='center', fontsize=20)
            axs.plot([0, 4 - all_member_depths[i]], [i, i], linewidth=3, color='black')

        final_midy = plumb(axs, 3, all_members, self.tree)

        axs.plot([3, 4], [final_midy, final_midy], linewidth=3, color='black')
        axs.axis('off')
        plt.subplots_adjust(wspace=0.6, hspace=0.6)
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close()
