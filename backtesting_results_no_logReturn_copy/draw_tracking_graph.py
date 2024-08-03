from matplotlib import pyplot as plt
import pickle
from matplotlib.ticker import MaxNLocator

index_type = "s&p500"   # "s&p500", "kospi100", "kosdaq150", "s&p100"
cardinality = "60"
# methods = "all"   # "all", "lag", "QP", "SNN"

# rebalancing date
with open(f'rebalancing_date_list_{index_type}.pkl', 'rb') as f:
    rebalancing_date_list = pickle.load(f)

# end dates
with open(f'end_date_list_{index_type}.pkl', 'rb') as f:
    end_date_list = pickle.load(f)

x_labels = [day[:7] for day in end_date_list]


    
# Tracking Error
with open(f'tracking_errors_{index_type}_lagrange_ours_{cardinality}.pkl', 'rb') as f:
    tracking_error = pickle.load(f)

    
# Lagrange forward
# with open(f'tracking_indices_{index_type}_lagrange_forward_{cardinality}.pkl', 'rb') as f:
#     tracking_indices_lagrange_forward = pickle.load(f)
# # Lagrange backward
# with open(f'tracking_indices_{index_type}_lagrange_backward_{cardinality}.pkl', 'rb') as f:
#     tracking_indices_lagrange_backward = pickle.load(f)
# # Lagrange Ours
# print(f"tracking_indices_{index_type}_lagrange_ours_{cardinality}.pkl")
with open(f'tracking_indices_{index_type}_lagrange_ours_{cardinality}.pkl', 'rb') as f:
    tracking_indices_lagrange_ours = pickle.load(f)
# # SNN
# with open(f'tracking_indices_{index_type}_SNN_{cardinality}.pkl', 'rb') as f:
#     tracking_indices_SNN = pickle.load(f)


# Target Index
with open(f'target_indices_{index_type}.pkl', 'rb') as f:
    target_indices = pickle.load(f)
# Tracking Graph
tracking_indices_lagrange_ours = [row.iloc[-1] for row in tracking_indices_lagrange_ours]
# tracking_indices_lagrange_forward = [row.iloc[-1] for row in tracking_indices_lagrange_forward]
# tracking_indices_lagrange_backward = [row.iloc[-1] for row in tracking_indices_lagrange_backward]
# tracking_indices_SNN = [row.iloc[-1] for row in tracking_indices_SNN]
target_indices = [row.iloc[-1] for row in target_indices]

# tracking_indices = [tracking_indices_lagrange_ours, tracking_indices_lagrange_forward, tracking_indices_lagrange_backward, tracking_indices_SNN]
# tracking_indices = [tracking_indices_lagrange_ours, tracking_indices_lagrange_forward, tracking_indices_SNN]
# methods = ['ours', 'forward', 'SNN']
# colors = ['red', 'blue', 'yellow']
tracking_indices = [tracking_indices_lagrange_ours]#, tracking_indices_lagrange_forward, tracking_indices_lagrange_backward, tracking_indices_SNN]

methods = ['ours', 'forward', 'backward', 'SNN']
colors = ['red', 'blue', 'green', 'yellow']
methods = ['ours']

for i in range(len(tracking_indices)):
    plt.figure(figsize=(12,6))
    plt.plot(end_date_list, tracking_indices[i], linestyle='-', color=colors[i], label=methods[i])
    plt.plot(end_date_list, target_indices, linestyle='-', color='grey',label='Target Index')
    # for j in range(0,len(rebalancing_date_list)):
    #     # print(rebalancing_date_list[i])
    #     # print(i)
    #     # if rebalancing_date_list[i] in end_date_list:
    #     #     print('True')
    #     # else: print('False')
    #     plt.axvline(x=rebalancing_date_list[j], color='grey', linestyle='--', linewidth=1)

    plt.ylim(-1.2,1.4)
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(nbins=20))  # 레이블 간격을 최대 10개로 설정
    # ax.set_xticklabels(x_labels, fontsize=12)
    plt.xticks(rotation=90, fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.legend(fontsize=16)
    plt.savefig(f"./figures/{index_type}_{methods[i]}_{cardinality}.png")
    plt.show()
    plt.close()
    
# # Tracking & Target Index
# plt.figure(figsize=(12,6))
# plt.plot(end_date_list, tracking_indices_lagrange_ours, linestyle='-', color='#275317', label='ours')
# plt.plot(end_date_list, target_indices, linestyle='-', color='grey',label='Target Index')
# plt.close()

# plt.figure(figsize=(12,6))
# plt.plot(end_date_list, tracking_indices_lagrange_forward, linestyle='-', color='#B4E5A2', label='lag_forward')
# plt.plot(end_date_list, target_indices, linestyle='-', color='grey',label='Target Index')
# plt.close()

# plt.figure(figsize=(12,6))
# plt.plot(end_date_list, tracking_indices_lagrange_backward, linestyle='-', color='#8ED973', label='lag_backward')
# plt.plot(end_date_list, target_indices, linestyle='-', color='grey',label='Target Index')
# plt.close()

# plt.figure(figsize=(12,6))
# plt.plot(end_date_list, tracking_indices_SNN, linestyle='-', color='pink', label='SNN')
# plt.plot(end_date_list, target_indices, linestyle='-', color='grey',label='Target Index')


