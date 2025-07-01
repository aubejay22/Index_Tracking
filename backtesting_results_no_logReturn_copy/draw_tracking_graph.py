from matplotlib import pyplot as plt
import seaborn as sns
import pickle
from matplotlib.ticker import MaxNLocator
import pandas as pd
import numpy as np

index_type = "s&p500"   # "s&p500", "kospi100", "kosdaq150", "s&p100"
cardinality = "30"
fontsize = 20

# end dates
with open(f'end_date_list_{index_type}.pkl', 'rb') as f:
    end_date_list = pickle.load(f)

end_date_list = pd.to_datetime(end_date_list)
x_labels = end_date_list.strftime("%Y-%m")

    
# Lagrange forward
with open(f'tracking_indices_{index_type}_lagrange_forward_{cardinality}.pkl', 'rb') as f:
    tracking_indices_lagrange_forward = pickle.load(f)
# Lagrange backward
with open(f'tracking_indices_{index_type}_lagrange_backward_{cardinality}.pkl', 'rb') as f:
    tracking_indices_lagrange_backward = pickle.load(f)
# Lagrange Ours
with open(f'tracking_indices_{index_type}_lagrange_ours_{cardinality}.pkl', 'rb') as f:
    tracking_indices_lagrange_ours = pickle.load(f)
# SNN
with open(f'tracking_indices_{index_type}_SNN_{cardinality}.pkl', 'rb') as f:
    tracking_indices_SNN = pickle.load(f)
# full
with open(f'tracking_indices_{index_type}_lagrange_full_None.pkl', 'rb') as f:
    tracking_indices_full = pickle.load(f)

# Target Index
with open(f'target_indices_{index_type}.pkl', 'rb') as f:
    target_indices = pickle.load(f)
    
# Tracking Graph
tracking_indices_lagrange_ours = np.array([row.iloc[-1] for row in tracking_indices_lagrange_ours]) *100
tracking_indices_lagrange_forward = np.array([row.iloc[-1] for row in tracking_indices_lagrange_forward]) *100
tracking_indices_lagrange_backward = np.array([row.iloc[-1] for row in tracking_indices_lagrange_backward]) *100
tracking_indices_SNN = np.array([row.iloc[-1] for row in tracking_indices_SNN]) *100
tracking_indices_full = np.array([row.iloc[-1] for row in tracking_indices_full]) *100
target_indices = np.array([row.iloc[-1] for row in target_indices]) *100

tracking_indices = [tracking_indices_lagrange_ours, tracking_indices_lagrange_forward, tracking_indices_lagrange_backward, tracking_indices_SNN, tracking_indices_full]
# tracking_errors = [tracking_error_ours, tracking_error_forward, tracking_error_backward, tracking_error_SNN]
# tracking_indices = [tracking_indices_lagrange_ours, tracking_indices_lagrange_forward, tracking_indices_SNN]
# methods = ['ours', 'forward', 'SNN']
# colors = ['red', 'blue', 'yellow']
# tracking_indices = [tracking_indices_lagrange_ours]#, tracking_indices_lagrange_forward, tracking_indices_lagrange_backward, tracking_indices_SNN]

methods = ['ours', 'forward', 'backward', 'SNN', 'full']
# colors = ['red', 'blue', 'green', 'yellow']
palettes = ["deep", "muted", "pastel", "bright", "dark", "colorblind"]
colors = sns.color_palette(palettes[1], 5)  # "muted" 팔레트에서 4가지 색상 선택

# methods = ['ours']

for i in range(len(tracking_indices)):
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1], 'hspace': 0})
    # fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12,10), sharex=True)
        
    
    ax1.plot(end_date_list, target_indices, linestyle='-', color='#FFEF96',label='Target Index')
    # ax1.plot(end_date_list, target_indices, linestyle='-', color='grey',label='Target Index')
    ax1.plot(end_date_list, tracking_indices[i], linestyle='-', color=colors[i], label=methods[i])
    ax1.set_ylabel('Index', fontsize=fontsize)
    ax1.set_xlabel('Date (YYYY-MM)', fontsize=fontsize)
    ax1.set_xlim([end_date_list.min(), end_date_list.max()])
    ax1.legend(fontsize=fontsize)
    ax1.set_ylim([-60, 140])
    
    
    err = tracking_indices[i] - target_indices
    print(f"mean squared error sum of {methods[i]} : {np.sum(abs(err)) / len(target_indices)}")
    # tracking_errors = np.where(err > 1e-4, 0.5*np.sqrt(err), 0.5*np.clip(err, None, 0))
    sqrt_errors = np.where(err >= 0, err, -np.abs(err))
    tracking_errors = np.nan_to_num(sqrt_errors)
    
    ax2.fill_between(end_date_list, tracking_errors, 0, where=(tracking_errors > 0), color=colors[i], alpha=0.3)
    ax2.fill_between(end_date_list, tracking_errors, 0, where=(tracking_errors < 0), color=colors[i], alpha=0.3)
    ax2.set_ylabel('Error', fontsize=fontsize)
    ax2.set_ylim([-40, 40])
    
    ax2.xaxis.set_major_locator(MaxNLocator(nbins=20))
    ax2.set_xticks(end_date_list[::int(len(end_date_list)/20)])
    ax2.set_xticklabels(x_labels[::int(len(x_labels)/20)], rotation=90, fontsize=fontsize)
    ax1.tick_params(axis='y', labelsize=fontsize)
    ax2.tick_params(axis='x', labelsize=fontsize)
    ax2.tick_params(axis='y', labelsize=fontsize)

    plt.tight_layout()
    plt.savefig(f"./figures/{index_type}_{methods[i]}_{cardinality}.png")
    plt.show()
    plt.close()
    