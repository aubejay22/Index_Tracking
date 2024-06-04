from matplotlib import pyplot as plt
import pickle

index_type = "kospi100"
cardinality = 30
methods = "QP"   # "all", "lag", "QP", "SNN"

# end dates
with open('end_date_list.pkl', 'rb') as f:
    end_date_list = pickle.load(f)
    
# QP forward
with open(f'tracking_indices_{index_type}_QP_forward_{cardinality}.pkl', 'rb') as f:
    tracking_indices_QP_forward = pickle.load(f)
# QP backward
with open(f'tracking_indices_{index_type}_QP_backward_{cardinality}.pkl', 'rb') as f:
    tracking_indices_QP_backward = pickle.load(f)
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

# Target Index
with open(f'target_indices.pkl', 'rb') as f:
    target_indices = pickle.load(f)
# Tracking Graph
plt.figure(figsize=(12,6))

if methods == "all":
    plt.plot(end_date_list, tracking_indices_lagrange_ours, linestyle='-', color='#275317', label='ours')
    plt.plot(end_date_list, tracking_indices_lagrange_forward, linestyle='-', color='#B4E5A2', label='lag_forward')
    plt.plot(end_date_list, tracking_indices_lagrange_backward, linestyle='-', color='#8ED973', label='lag_backward')
    plt.plot(end_date_list, tracking_indices_QP_forward, linestyle='-', color='#A6CAEC', label='QP_forward')
    plt.plot(end_date_list, tracking_indices_QP_backward, linestyle='-', color='#4E95D9', label='QP_backward')
    plt.plot(end_date_list, tracking_indices_SNN, linestyle='-', color='pink', label='SNN')
elif methods == "lag":
    plt.plot(end_date_list, tracking_indices_lagrange_ours, linestyle='-', color='#275317', label='ours')
    plt.plot(end_date_list, tracking_indices_lagrange_forward, linestyle='-', color='#B4E5A2', label='lag_forward')
    plt.plot(end_date_list, tracking_indices_lagrange_backward, linestyle='-', color='#8ED973', label='lag_backward')
elif methods == "QP":
    plt.plot(end_date_list, tracking_indices_lagrange_ours, linestyle='-', color='#275317', label='ours')
    plt.plot(end_date_list, tracking_indices_QP_forward, linestyle='-', color='#A6CAEC', label='QP_forward')
    plt.plot(end_date_list, tracking_indices_QP_backward, linestyle='-', color='#4E95D9', label='QP_backward')
elif methods == "SNN":
    plt.plot(end_date_list, tracking_indices_lagrange_ours, linestyle='-', color='#275317', label='ours')
    plt.plot(end_date_list, tracking_indices_SNN, linestyle='-', color='pink', label='SNN')
    
plt.plot(end_date_list, target_indices, linestyle='-', color='r',label='Target Index')
plt.xlabel('Date')
plt.ylabel('Tracking Value')
plt.title('Tracking Index over Time')
plt.legend()
plt.savefig(f"./backtesting_results/{index_type}_{methods}_{cardinality}.png")
plt.show()