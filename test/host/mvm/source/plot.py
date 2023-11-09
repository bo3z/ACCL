import os
import json
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import FormatStrFormatter

# Constants
RESULT_FILE = 'results.json'
WS_COMBINATIONS = ['2', '4']
MODES = ['openmpi', 'accl']
MATRIX_SIZES = ['2048 x 2048', '4096 x 4096', '8192 x 8192']
M_N_COMBINATIONS = ['M_2048_N_2048', 'M_4096_N_4096', 'M_8192_N_8192']

# Results
if (os.path.exists(RESULT_FILE)):
        with open(RESULT_FILE, 'r') as j:
            results = json.loads(j.read())

# Extracting relavant data
matrix_combos = {}
for size, key in zip(MATRIX_SIZES, M_N_COMBINATIONS):
    mode_results = {}
    for i, mode in enumerate(MODES):
        mode_latencies_compute = []
        mode_latencies_network = []
        for ws in WS_COMBINATIONS:
            mode_latencies_compute.append(results[f'world_size_{ws}_{key}_mode_{mode}']['computation_mean'])
            mode_latencies_network.append(results[f'world_size_{ws}_{key}_mode_{mode}']['result_available_mean'] - results[f'world_size_{ws}_{key}_mode_{mode}']['computation_mean'])
        mode_results[mode] = [mode_latencies_compute, mode_latencies_network]
    matrix_combos[size] = mode_results

# Plotting 
WIDTH = 0.4
x = np.arange(len(WS_COMBINATIONS))
fig, axs = plt.subplots(1, len(MATRIX_SIZES), sharey=False)
plt.setp(axs, xticks=[0.1, 0.9], xticklabels=WS_COMBINATIONS)
fig.subplots_adjust(wspace=0.33)

for i, size in enumerate(MATRIX_SIZES):
    current_data = matrix_combos[size]
    for j, mode in enumerate(MODES):
        # Plot the series
        bars = axs[i].bar(x + j * WIDTH, current_data[mode][0], WIDTH, label=f'{mode.upper()} MULT')
        _ = axs[i].bar(x + j * WIDTH, current_data[mode][1], WIDTH, label=f'{mode.upper()} COMMS', bottom=current_data[mode][0])
        
        # Add the speed-up
        for k, bar in enumerate(bars):
            speed_up = results[f'world_size_1_{M_N_COMBINATIONS[i]}_mode_single']['result_available_mean'] / results[f'world_size_{WS_COMBINATIONS[k]}_{M_N_COMBINATIONS[i]}_mode_{mode}']['result_available_mean']
            speed_up = round(speed_up, 2)
            axs[i].text(
                x = bar.get_x() + bar.get_width() / 2, y = 1.015 * results[f'world_size_{WS_COMBINATIONS[k]}_{M_N_COMBINATIONS[i]}_mode_{mode}']['result_available_mean'],
                s = f'{speed_up}x',
                ha = 'center',
                fontsize="7"
            )
    
    # Current matrix size
    axs[i].set_title(size, fontsize='10')
   
    # For 8K x 8K, atplotlib adds an extra decimal point
    if size == '8192 x 8192':
        axs[i].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        
# Show leggend in convinient place
handles, labels = axs[-1].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper left', ncols=4)

# x- and y- axis labels
fig.text(0.04, 0.5, 'Latency [ms]', va='center', rotation='vertical', fontsize='10')
fig.text(0.5, 0.04, 'Ranks', va='center', rotation='horizontal', fontsize='10')

plt.savefig('accl-openmpi-comarison.png', format='png', bbox_inches='tight') 
