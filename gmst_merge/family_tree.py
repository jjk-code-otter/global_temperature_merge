import copy
import random
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import gmst_merge.dataset as ds


def pick_one(inarr: list):
    """
    Given a list of lists, recursively pick from the entries until you find a non-list item and return that

    :param inarr: list
        List containing lists and/or strings
    :return:
    """
    selection = random.choice(inarr)
    if isinstance(selection, list):
        selection = pick_one(selection)
    else:
        return selection
    return selection


def split_list(in_lst: list, n_splits: int) -> list:
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
    random.shuffle(new_lst)

    # partition list randomly
    number_of_items = len(new_lst)
    split_points = np.random.choice(number_of_items - 2, n_splits - 1, replace=False) + 1
    split_points.sort()
    result = np.split(new_lst, split_points)

    # convert back to a regular list
    result = [x.tolist() for x in result]

    return result


def label_by_depth(lst, lbl=0):
    """Label each non-list in a list of lists by its depth"""
    gst = copy.deepcopy(lst)
    for i, elem in enumerate(gst):
        if isinstance(elem, list):
            gst[i] = label_by_depth(elem, lbl + 1)
        else:
            gst[i] = lbl
    return gst


def read_tree(lst, data_dir):
    """Label each non-list in a list of lists by its depth"""
    gst = copy.deepcopy(lst)
    for i, elem in enumerate(gst):
        if isinstance(elem, list):
            gst[i] = read_tree(elem, data_dir)
        else:
            filestub = 'ensemble_time_series.csv'
            df = pd.read_csv(data_dir / elem / filestub, header=None)
            df = df.to_numpy()
            gst[i] = ds.Dataset(df, name=elem)
    return gst


def get_all_members(lst):
    """Get a list of all non-list items in a list of lists"""
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

    def __init__(self, inlist):
        self.tree = inlist

    def __str__(self):
        return f"{self.tree}"

    @staticmethod
    def read_from_json(json_file, data_dir, type):
        if type not in ['heads', 'tails', 'master']:
            raise ValueError(f'Unknown type {type} must be one of head, tail, master')
        with open(json_file, 'r') as f:
            basic_tree = json.load(f)
        basic_tree = basic_tree[type]
        filled_tree = FamilyTree(FamilyTree.read_from_directory(basic_tree, data_dir))
        return filled_tree

    @staticmethod
    def read_from_directory(basic_tree, data_dir):
        return read_tree(basic_tree, data_dir)

    @staticmethod
    def make_random_tree(list_of_datasets):
        """
        Given a list of datasets, generate a list of lists specifying a hierarchical family tree by repeatedly
        grouping elements.

        :param list_of_datasets: List[str]
            List of datasets to be
        :return:
        """
        new_list = copy.deepcopy(list_of_datasets)

        for i in range(4):
            # choose how many breaks to have
            n_items = len(new_list)
            breaks = random.randint(0, n_items - 1)
            # bail if no breaks selected
            if breaks == 0:
                break
            # split list chosen number of times
            new_list = split_list(new_list, breaks)

        return FamilyTree(new_list)

    def sample_from_tree(self) -> ds.Dataset:
        chosen_dataset = pick_one(self.tree)
        chosen_dataset = chosen_dataset.sample_from_ensemble()
        return chosen_dataset

    def plot_tree(self, filename):
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
