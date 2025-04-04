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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gmst_merge.dataset as ds


def plot_ensemble(filename, to_compare, colours, linestyles, size):
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

    columns = ['time']
    columns = columns + [f'{x}' for x in range(size)]

    # Read in the data
    frames = {}
    smooth = {}
    for combos in to_compare:
        frames[f'{combos[0]} {combos[1]}'] = pd.read_csv(f'Output/{combos[0]}/{combos[1]}.csv', header=None,
                                                         names=columns)
        smooth[f'{combos[0]} {combos[1]}'] = pd.read_csv(f'Output/{combos[0]}/{combos[1]}.csv', header=None,
                                                         names=columns)

    sns.set(font='Franklin Gothic Book', rc=STANDARD_PARAMETER_SET)

    fig, axs = plt.subplots(2, 1, sharex=True)
    fig.set_size_inches(16, 16 * 2 / 3)

    i = 0
    for key in frames:
        annual = frames[key]

        for j in range(size):

            values = annual[f'{j}']
            values = values - np.mean(values[(annual.time >= 1981) & (annual.time <= 2010)])

            alf = 0.1
            if size == 10000:
                alf = 0.01

            axs[0].plot(annual.time, annual[f'{j}'], color='black', label=None, linestyle='solid', alpha=alf)
            axs[1].plot(annual.time, values, color='black', label=None, linestyle='solid', alpha=alf)

        i += 1

        annual = annual.to_numpy()

    years = annual[:, 0]
    data = annual[:, 1:]

    print(f'Median = {np.median(data[-1, :]):.2f}')
    print(f'Mean = {np.mean(data[-1, :]):.2f}')
    print(f'2.5% = {np.quantile(data[-1, :], 0.025):.2f}')
    print(f'97.5% = {np.quantile(data[-1, :], 0.975):.2f}')
    print(f'5% = {np.quantile(data[-1, :], 0.05):.2f}')
    print(f'95% = {np.quantile(data[-1, :], 0.95):.2f}')
    print('')

    axs[0].set_ylabel(r'Anomaly ($\!^\circ\!$C)')
    axs[1].set_ylabel(r'Anomaly ($\!^\circ\!$C)')

    axs[0].set_title('(a) Annual global mean temperatures, 1850-1900 baseline', loc='left', fontsize=20)
    axs[1].set_title('(b) Annual global mean temperatures, 1981-2010 baseline', loc='left', fontsize=20)

    axs[0].set_ylim(-0.5, 1.9)
    axs[1].set_ylim(-0.5 - 0.9, 1.9 - 0.9)

    plt.savefig(filename, bbox_inches='tight')
    plt.savefig(filename.replace('.png', '.svg'), bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    # Compare clustered and unclustered
    to_compare = []
    to_compare.append(['final_clustered', 'lsat_pseudo'])

    colours = ['#fdc086', '#beaed4', '#7fc97f', '#555555', '#E65656']
    linestyles = ['solid']

    plot_ensemble('Figures/final.png', to_compare, colours, linestyles, 100)

    # Compare clustered and unclustered
    to_compare = []
    to_compare.append(['basic', 'lsat_pseudo'])

    colours = ['#fdc086', '#beaed4', '#7fc97f', '#555555', '#E65656']
    linestyles = ['solid']

    plot_ensemble('Figures/final_full.png', to_compare, colours, linestyles, 10000)
