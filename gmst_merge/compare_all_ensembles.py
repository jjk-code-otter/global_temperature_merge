import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
    'ur_ensembles_only'
]

for hierarchy_variable in variables:
    df = pd.read_csv(f'Output/full_summary_{hierarchy_variable}.csv')
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
    'solid', 'solid', 'solid', 'solid', 'solid', 'dashed', 'dashed', 'dashed','dashed'
]

sns.set(font='Franklin Gothic Book', rc=STANDARD_PARAMETER_SET)

fig, axs = plt.subplots(3, 1, sharex=True)
fig.set_size_inches(16, 16)

for i, df in enumerate(frames):
    axs[0].plot(df.year, df["mean preindustrial"], color=colours[i], label=variables[i].upper(),
                linestyle=linestyles[i])
    axs[0].plot(df.year, df["mean lowess"], color=colours[i], linestyle=linestyles[i])

    axs[1].plot(df.year, df["stdev preindustrial"], color=colours[i], linestyle=linestyles[i])
    axs[1].plot(df.year, df["stdev lowess"], color=colours[i], linestyle=linestyles[i])

    axs[2].plot(df.year, df["q975 preindustrial"] - df["q025 preindustrial"], color=colours[i], linestyle=linestyles[i])
    axs[2].plot(df.year, df["q975 lowess"] - df["q025 lowess"], color=colours[i], linestyle=linestyles[i])

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
