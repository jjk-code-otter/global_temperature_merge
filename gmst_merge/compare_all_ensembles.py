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

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gmst_merge.dataset as ds


def plot_comparisons(filename, to_compare, colours, linestyles, climatology):
    STANDARD_PARAMETER_SET = {
        'axes.axisbelow': False,
        'axes.labelsize': 20,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'axes.edgecolor': 'lightgrey',
        'axes.facecolor': 'None',

        'axes.grid.axis': 'y',
        'grid.color': 'lightgrey',
        'grid.alpha': 0.5,

        'axes.labelcolor': 'dimgrey',
        'axes.labelpad': 4,

        'axes.spines.left': False,
        'axes.spines.right': False,
        'axes.spines.top': False,

        'figure.facecolor': 'white',
        'lines.solid_capstyle': 'round',
        'patch.edgecolor': 'w',
        'patch.force_edgecolor': True,
        'text.color': 'dimgrey',

        'xtick.bottom': True,
        'xtick.color': 'dimgrey',
        'xtick.direction': 'out',
        'xtick.top': False,
        'xtick.labelbottom': True,

        'ytick.major.width': 0.4,
        'ytick.color': 'dimgrey',
        'ytick.direction': 'out',
        'ytick.left': False,
        'ytick.right': False
    }

    insert = ''
    if climatology[0] != 1850 and climatology[1] != 1900:
        insert = f'_{climatology[0]}-{climatology[1]}'

    # Read in the data
    frames = {}
    smooth = {}
    for combos in to_compare:
        frames[f'{combos[0]} {combos[1]}'] = pd.read_csv(f'Output/{combos[0]}/{combos[1]}_summary{insert}.csv')
        smooth[f'{combos[0]} {combos[1]}'] = pd.read_csv(f'Output/{combos[0]}/{combos[1]}_smoothed_summary{insert}.csv')

    sns.set(font='Franklin Gothic Book', rc=STANDARD_PARAMETER_SET)

    fig, axs = plt.subplots(3, 1, sharex=True)
    fig.set_size_inches(16, 16)

    i = 0
    for key in frames:
        annual = frames[key]
        smoothed_annual = smooth[key]

        label = key.upper()

        axs[0].plot(annual.time, annual['mean'], color=colours[i], label=label, linestyle=linestyles[i])
        axs[0].plot(smoothed_annual.time, smoothed_annual['mean'], color=colours[i], linestyle=linestyles[i])

        axs[1].plot(annual.time, annual['std'], color=colours[i], linestyle=linestyles[i])

        axs[2].plot(smoothed_annual.time, smoothed_annual['std'], color=colours[i], linestyle=linestyles[i])

        i += 1

    axs[0].legend(labelcolor='linecolor', ncol=2, frameon=False, prop={'size': 20})

    axs[0].set_ylabel(r'Anomaly ($\!^\circ\!$C)')
    axs[1].set_ylabel(r'Standard deviation ($\!^\circ\!$C)')
    axs[2].set_ylabel(r'Standard deviation ($\!^\circ\!$C)')

    axs[0].set_title('(a) Annual global mean temperatures and smoothed temperatures', loc='left', fontsize=20)
    axs[1].set_title('(b) Standard deviation of annual global mean temperatures', loc='left', fontsize=20)
    axs[2].set_title('(c) Standard deviation of smoothed global mean temperatures', loc='left', fontsize=20)

    if climatology[0] == 1850:
        axs[0].set_ylim(-0.35, 1.7)
    else:
        axs[0].set_ylim(-1.05, 1.0)

    axs[1].set_ylim(0.0, 0.176)
    axs[2].set_ylim(0.0, 0.176)

    plt.savefig(filename, bbox_inches='tight')
    plt.savefig(filename.replace('.png', '.svg'), bbox_inches='tight')
    plt.close()


if __name__ == '__main__':

    climatology = [1850, 1900]
    climatology = [1981, 2010]
    insert = ''
    if climatology[0] != 1850 and climatology[1] != 1900:
        insert = f'_{climatology[0]}-{climatology[1]}'

    # Summary plots
    trees = ['ur', 'sst', 'lsat', 'interp', 'equal', 'sst_ensembles_only', 'lsat_ensembles_only', 'ur_ensembles_only', 'ur_pseudo']
    to_compare = [['basic', x] for x in trees]
    to_compare.append(['random', 'random'])
    to_compare.append(['basic', 'sst_pseudo'])
    to_compare.append(['basic', 'lsat_pseudo'])

    colours = ['#555555', '#7fc97f', '#beaed4', '#fdc086', '#E65656', '#7fc97f', '#beaed4', '#555555', '#5DD0DB', '#386cb0', '#000000', '#ffcc00', '#00ffcc']
    linestyles = ['solid', 'solid', 'solid', 'solid', 'solid', 'dashed', 'dashed', 'dashed', 'dotted', 'solid', 'solid', 'solid', 'solid']

    count = 0
    filename = f'Figures/summary_all{insert}.png'
    while Path(filename).exists():
        count += 1
        filename = f'Figures/summary_all{insert}_{count}.png'
    plot_comparisons(filename, to_compare, colours, linestyles, climatology)


    # Summary plot trees
    trees = ['ur_pseudo', 'sst_pseudo', 'lsat_pseudo', 'interp_pseudo', 'equal_pseudo']
    to_compare = [['basic', x] for x in trees]
    to_compare.append(['random', 'random_pseudo'])
    to_compare = to_compare + [['unbalanced', x] for x in ['unbalanced']]

    colours = ['#555555', '#7fc97f', '#beaed4', '#fdc086', '#E65656', '#7fc97f', '#91a19f', '#555555', '#5DD0DB', '#386cb0', '#000000', '#ffcc00', '#00ffcc']
    linestyles = ['solid', 'solid', 'solid', 'solid', 'solid', 'dashed', 'dashed', 'solid']

    count = 0
    filename = f'Figures/summary_trees{insert}.png'
    while Path(filename).exists():
        count += 1
        filename = f'Figures/summary_trees{insert}_{count}.png'
    plot_comparisons(filename, to_compare, colours, linestyles, climatology)


    # Summary of ensembles vs none vs pseudo plots
    trees = ['ur', 'ur_ensembles_only', 'ur_pseudo']
    to_compare = [['basic', x] for x in trees]

    colours = ['#555555', '#7fc97f', '#beaed4']
    linestyles = ['solid', 'solid', 'solid']

    count = 0
    filename = f'Figures/summary_ensembles_or_not{insert}.png'
    while Path(filename).exists():
        count += 1
        filename = f'Figures/summary_ensembles_or_not{insert}_{count}.png'
    plot_comparisons(filename, to_compare, colours, linestyles, climatology)


    # Compare clustered and unclustered
    to_compare = []
    to_compare.append(['final_small_ensemble', 'sst_pseudo'])
    to_compare.append(['final_thinned', 'sst_pseudo'])
    to_compare.append(['final_clustered', 'sst_pseudo'])
    to_compare.append(['basic', 'sst_pseudo'])

    colours = ['#fdc086', '#beaed4','#7fc97f', '#555555', '#E65656']
    linestyles = ['solid', 'solid', 'solid', 'solid', 'dashed']

    count = 0
    filename = f'Figures/thinning_all{insert}.png'
    while Path(filename).exists():
        count += 1
        filename = f'Figures/thinning_all{insert}_{count}.png'
    plot_comparisons(filename, to_compare, colours, linestyles, climatology)


    # Sensitivity tests summary plots
    to_compare = [[x, 'ur_pseudo'] for x in ['basic', 'short_overlap', 'variable_overlap', 'shortest_overlap']]
    to_compare = to_compare + [['random', x] for x in ['random_pseudo']]
    to_compare = to_compare + [['unbalanced', x] for x in ['unbalanced']]

    colours = ['#555555', '#7fc97f', '#beaed4', '#fdc086', '#E65656', '#91a19f', '#386cb0']
    linestyles = ['solid', 'solid', 'solid', 'solid', 'solid', 'dashed', 'solid']

    count = 0
    filename = f'Figures/sensitivity_all{insert}.png'
    while Path(filename).exists():
        count += 1
        filename = f'Figures/sensitivity_all{insert}_{count}.png'
    plot_comparisons(filename, to_compare, colours, linestyles, climatology)


    # Sensitivity tests reanalysis regrouping summary plots
    to_compare = [['basic', x] for x in ['sst_pseudo', 'sst_pseudo_reanalysis_switch', 'sst_pseudo_no_US']]

    colours = ['#555555', '#7fc97f', '#beaed4', '#fdc086', '#E65656', '#91a19f', '#386cb0']
    linestyles = ['solid', 'solid', 'solid', 'solid', 'solid', 'dashed', 'solid']

    count = 0
    filename = f'Figures/sensitivity_reanalysis{insert}.png'
    while Path(filename).exists():
        count += 1
        filename = f'Figures/sensitivity_reanalysis{insert}_{count}.png'
    plot_comparisons(filename, to_compare, colours, linestyles, climatology)
