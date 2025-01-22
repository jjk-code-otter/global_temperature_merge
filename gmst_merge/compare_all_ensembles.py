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

frames = []

variables = [
    'ur', 'sst', 'lsat', 'interp', 'equal', 'random', 'sst_ensembles_only', 'lsat_ensembles_only',
    'ur_ensembles_only', 'ur_pseudo'
]

for hierarchy_variable in variables:
    df = ds.Dataset.read_csv_from_file(
        f'Output/{hierarchy_variable}.csv',
        hierarchy_variable,
        header=0
    )
    df.anomalize(1850, 1900)
    frames.append(df)

colours = [
    '#555555',
    '#7fc97f',
    '#beaed4',
    '#fdc086',
    '#E65656',
    '#386cb0',
    '#7fc97f',
    '#beaed4',
    '#555555',
    '#5DD0DB'
]
linestyles = [
    'solid', 'solid', 'solid', 'solid', 'solid', 'dashed', 'dashed', 'dashed', 'dashed', 'dotted'
]

sns.set(font='Franklin Gothic Book', rc=STANDARD_PARAMETER_SET)

fig, axs = plt.subplots(3, 1, sharex=True)
fig.set_size_inches(16, 16)

for i, df in enumerate(frames):
    smoo =df.lowess_smooth()

    axs[0].plot(df.time, df.get_ensemble_mean(), color=colours[i], label=variables[i].upper(), linestyle=linestyles[i])
    axs[0].plot(df.time, smoo.get_ensemble_mean(), color=colours[i], linestyle=linestyles[i])

    axs[1].plot(df.time, df.get_ensemble_std(), color=colours[i], linestyle=linestyles[i])
    axs[1].plot(df.time, smoo.get_ensemble_std(), color=colours[i], linestyle=linestyles[i])

    q1, q2 = df.get_quantile_range(95)
    r1, r2 = smoo.get_quantile_range(95)
    axs[2].plot(df.time, q2 - q1, color=colours[i], linestyle=linestyles[i])
    axs[2].plot(df.time, r2 - r1, color=colours[i], linestyle=linestyles[i])

axs[0].legend()
handles, labels = axs[0].get_legend_handles_labels()
# specify order of items in legend
order = [x for x in range(len(handles))]
# add legend to plot
leg = axs[0].legend([handles[idx] for idx in order], [labels[idx] for idx in order],
                    frameon=False, prop={'size': 20}, labelcolor='linecolor', ncol=2,
                    handlelength=0, handletextpad=0.3, loc="upper left", bbox_to_anchor=(0.02, 0.96))
for line in leg.get_lines():
    line.set_linewidth(3.0)
for item in leg.legendHandles:
    item.set_visible(False)

axs[0].set_title('(a) Annual global mean temperatures and smoothed temperatures', loc='left', fontsize=20)
axs[1].set_title('(b) Standard deviation of annual and smoothed global mean temperatures', loc='left', fontsize=20)
axs[2].set_title('(c) 95% range of annual and smoothed global mean temperatures', loc='left', fontsize=20)

plt.savefig('Figures/summary_all.png', bbox_inches='tight')
plt.savefig('Figures/summary_all.svg', bbox_inches='tight')
plt.close()

assert False

fig, axs = plt.subplots(3, 1, sharex=True)
fig.set_size_inches(16, 16)

for i, df in enumerate(frames):
    axs[0].fill_between(df.year, df.q025, df.q975, color=colours[i], alpha=0.1)
    axs[0].plot(df.year, df["mean"], color=colours[i], label=variables[i].upper(), linestyle=linestyles[i])
    axs[1].plot(df.year, df["stdev"], color=colours[i], linestyle=linestyles[i])
    axs[2].plot(df.year, df["q975"] - df["q025"], color=colours[i], linestyle=linestyles[i])

axs[0].legend()
handles, labels = axs[0].get_legend_handles_labels()
# specify order of items in legend
order = [x for x in range(len(handles))]
# add legend to plot
leg = axs[0].legend([handles[idx] for idx in order], [labels[idx] for idx in order],
                    frameon=False, prop={'size': 20}, labelcolor='linecolor', ncol=2,
                    handlelength=0, handletextpad=0.3, loc="upper left", bbox_to_anchor=(0.02, 0.96))
for line in leg.get_lines():
    line.set_linewidth(3.0)
for item in leg.legendHandles:
    item.set_visible(False)

axs[0].set_title('(a) Annual global mean temperatures temperatures', loc='left', fontsize=20)
axs[1].set_title('(b) Standard deviation of annual global mean temperatures', loc='left', fontsize=20)
axs[2].set_title('(c) 95% range of annual global mean temperatures', loc='left', fontsize=20)

plt.savefig('Figures/summary_all_8110.png', bbox_inches='tight')
plt.savefig('Figures/summary_all_8110.svg', bbox_inches='tight')
plt.close()
