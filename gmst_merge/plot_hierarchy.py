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
import json

import matplotlib.pyplot as plt


def depth(lst):
    """Calculate the depth of the deepest branch in a list of lists"""
    return isinstance(lst, list) and max(map(depth, lst)) + 1


def label(lst, lbl=0):
    """Label each non-list in a list of lists by its depth"""
    gst = copy.deepcopy(lst)
    for i, elem in enumerate(gst):
        if isinstance(elem, list):
            gst[i] = label(elem, lbl + 1)
        else:
            gst[i] = lbl
    return gst


def count_end_points(lst):
    """Count how many non-list items there are in a list of lists"""
    count = 0
    if isinstance(lst, list):
        for i in lst:
            if isinstance(i, list):
                count = count + count_end_points(i)
            else:
                count += 1
    else:
        count = 1
    return count


def get_all_members(lst):
    """Get a list of all non-list items in a list of lists"""
    members = []
    if isinstance(lst, list):
        for i in lst:
            if isinstance(i, list):
                members = members + get_all_members(i)
            else:
                members.append(i)
    else:
        members.append(lst)

    return members


def n_descendants(lst):
    """Calculate the number of descendants of each item in a list, if some are lists of lists"""
    gst = [count_end_points(x) for x in lst]
    return gst

def plumb(ax, depth, all_members, inlist):
    """Plot hierarchy recursively"""
    midys = []
    for i, member in enumerate(inlist):
        if isinstance(member, list):
            midys.append(plumb(ax, depth - 1, all_members, member))
        else:
            midys.append(all_members.index(member))

    m1 = midys[0]
    m2 = midys[-1]
    ax.plot([depth, depth], [m1, m2], linewidth=3, color='black')
    top_level_midy = (m1 + m2) / 2
    ax.plot([depth, depth+1], [top_level_midy, top_level_midy], linewidth=3, color='black')

    return top_level_midy

# Plot all the interesting hierarchies
fig, axs = plt.subplots(1,3)
fig.set_size_inches(30, 16)

for a, hierarchy_variable in enumerate(['sst', 'lsat', 'interp']):

    with open(f'FamilyTrees/hierarchy_{hierarchy_variable}.json') as f:
        hierarch = json.load(f)

    # Only want to plot the master tree
    hierarch = hierarch['master']

    d = depth(hierarch)
    nd = n_descendants(hierarch)

    all_members = get_all_members(hierarch)
    all_member_depths = get_all_members(label(hierarch, 1))

    axs[a].set_ylim(-0.5, len(all_members))

    # Draw all memebers
    for i, member in enumerate(all_members):
        axs[a].text(-0.1, i, member, ha='right', va='center', fontsize=20)
        axs[a].plot([0, 4 - all_member_depths[i]], [i, i], linewidth=3, color='black')

    final_midy = plumb(axs[a],3, all_members, hierarch)

    axs[a].plot([3, 4], [final_midy, final_midy], linewidth=3, color='black')
    axs[a].axis('off')

plt.subplots_adjust(wspace=0.6, hspace=0.6)

plt.savefig(f'Figures/hierarchy.svg', bbox_inches='tight')
plt.savefig(f'Figures/hierarchy.png', bbox_inches='tight')
plt.close()
