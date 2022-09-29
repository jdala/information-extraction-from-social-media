import time
import numpy as np
import pandas as pd
import networkx as nx

import community as community_louvain
import collections

from sklearn.metrics import f1_score, mutual_info_score, normalized_mutual_info_score
from networkx.algorithms.community.quality import modularity, performance, coverage

from utils import *


LOUVAIN_SEED = 537921
np.random.seed(LOUVAIN_SEED)
seeds = np.random.randint(10000, size=10)


# **** IMPORT GRAPH FROM EDGE LIST FILE ( ****
G = nx.read_edgelist('../datasets/gr_politicians/single/graph.edges', nodetype=int)
y_true = np.array([1,1,1,1,3,3,3,1,3,1,1,1,1,1,1,1,2,0,0,0,0,1,1,1,0,1,4,0,3,1,1,0,1,0,1,2,1,3,1,3,4,1,1,1,1,1,2,0,0,0,1,1,1,1,0,1,1,1,0,1,1,1,1,0,0,0,2,1,0,0,0,3,3,1,4,1,1,1,1,4,1,1,0,1,0,0,0,0,1,1,0,0,3,4,3,1,1,1,0,1,1,1,1,0,0,1,0,0,1,1,1,1,1,1,1,0,0,0,1,0,1,2,0,0,0,0,1,3,1,0,1,1,0,1,1,1,1,0,1,1,1,0,1,1,1,1,1,3,2,1,0,0,0,1,0,1,1,3,4,1,0,1,0,1,0,0,1,1,0,3,1,0,1,1,1,1,1,4,1,1,3,1,3,0,1,0,1,3,1,1,0,1,1,0,1,4,1,1,1,1,1,1,1,4,0,1,1,1,1,3,1,0,1,1,1,0,0,1,0,3,2,1,1,1,1,4,1,0,3,1,4,0,1,0,1,0,1,1,0,0,1,1])
# **** ) IMPORT GRAPH FROM EDGE LIST FILE ****

# **** IMPORT GRAPH FROM CSV FILE ( ****
# ground_truth = pd.read_csv('../datasets/gr_politicians/batch/dataset_cleanup/mapping-not_NaN.csv', usecols=['id', 'party'])

# G_df = pd.read_csv(f'../datasets/gr_politicians/batch/edge_lists/32.edges')
# G_df.rename(columns={'overlap': 'weight'}, inplace=True)
# G = nx.from_pandas_edgelist(G_df, source='u', target='v', edge_attr='weight', create_using=nx.Graph())

# ground_truth_temp = ground_truth[ground_truth.id.isin(G.nodes())].sort_values(by=['id'])
# y_true = ground_truth_temp.party.to_numpy()

# id_list = ground_truth_temp.id.to_numpy()
# id_mapping = dict(zip(id_list, list(range(id_list.shape[0]))))
# G = G.subgraph(id_list)
# G = nx.relabel_nodes(G, id_mapping)
# **** ) IMPORT GRAPH FROM CSV FILE ****


# resolution_array = [1]
resolution_array = np.linspace(0.5, 1.5, 11)
# resolution_array = np.linspace(0.8, 1.1, 31)
# resolution_array = np.linspace(0.94, 1.06, 121)

total_acc = list()
total_f1_micro = list()
total_f1_macro = list()
total_mi = list()
total_nmi = list()
total_mdlrt = list()
total_perf = list()
total_covg = list()

for resolution in resolution_array:
    parameters = {'resolution': round(resolution, 3)}

    acc = f1_micro = f1_macro = mi = norm_mi = mdlrt = perf = covg = 0
    start_time = time.perf_counter()

    for seed in seeds:
        partition = community_louvain.best_partition(G, resolution=resolution, random_state=seed)
        partition = collections.OrderedDict(sorted(partition.items()))

        y_pred = np.array(list(partition.values()))

        temp_acc, y_pred = calculate_accuracy(y_true=y_true, y_pred=y_pred, k=5)
        clusters = calculate_clusters(y_pred)

        save_clusters(filename=f'Louvain-seed_{seed}-acc_{temp_acc:.4f}', clusters=clusters)

        acc += temp_acc
        f1_micro += f1_score(y_true, y_pred, average='micro')
        f1_macro += f1_score(y_true, y_pred, average='macro')
        mi += mutual_info_score(y_true, y_pred)
        norm_mi += normalized_mutual_info_score(y_true, y_pred)
        mdlrt += modularity(G, clusters)
        perf += performance(G, clusters)
        covg += coverage(G, clusters)

    cluster_metrics = {'Accuracy': acc / seeds.shape[0],
                       'F1-micro': f1_micro / seeds.shape[0],
                       'F1-macro': f1_macro / seeds.shape[0],
                       'MutualInformation': mi / seeds.shape[0],
                       'NormalizedMutualInformation': norm_mi / seeds.shape[0],
                       'Modularity': mdlrt / seeds.shape[0],
                       'Performance': perf / seeds.shape[0],
                       'Coverage': covg / seeds.shape[0]}

    duration = (time.perf_counter() - start_time)
    
    total_acc.append(cluster_metrics.get('Accuracy'))
    total_f1_micro.append(cluster_metrics.get('F1-micro'))
    total_f1_macro.append(cluster_metrics.get('F1-macro'))
    total_mi.append(cluster_metrics.get('MutualInformation'))
    total_nmi.append(cluster_metrics.get('NormalizedMutualInformation'))
    total_mdlrt.append(cluster_metrics.get('Modularity'))
    total_perf.append(cluster_metrics.get('Performance'))
    total_covg.append(cluster_metrics.get('Coverage'))

    save_results(filename='results-single-louvain', algorithm='Louvain', dimensions='-', cluster_metrics=cluster_metrics, num_of_clusters=len(set(y_pred)), duration=duration, parameters=parameters)

save_metric_plot(filename=f'single-Louvain3-seed_{LOUVAIN_SEED}-accuracy', metric='Accuracy', parameter='Resolution', y_axis=total_acc, x_axis=resolution_array)
save_metric_plot(filename=f'single-Louvain3-seed_{LOUVAIN_SEED}-f1_micro', metric='F1-micro', parameter='Resolution', y_axis=total_f1_micro, x_axis=resolution_array)
save_metric_plot(filename=f'single-Louvain3-seed_{LOUVAIN_SEED}-f1_macro', metric='F1-macro', parameter='Resolution', y_axis=total_f1_macro, x_axis=resolution_array)
save_metric_plot(filename=f'single-Louvain3-seed_{LOUVAIN_SEED}-mi', metric='MutualInformation', parameter='Resolution', y_axis=total_mi, x_axis=resolution_array)
save_metric_plot(filename=f'single-Louvain3-seed_{LOUVAIN_SEED}-nmi', metric='NormalizedMutualInformation', parameter='Resolution', y_axis=total_nmi, x_axis=resolution_array)
save_metric_plot(filename=f'single-Louvain3-seed_{LOUVAIN_SEED}-mdlrt', metric='Modularity', parameter='Resolution', y_axis=total_mdlrt, x_axis=resolution_array)
save_metric_plot(filename=f'single-Louvain3-seed_{LOUVAIN_SEED}-perf', metric='Performance', parameter='Resolution', y_axis=total_perf, x_axis=resolution_array)
save_metric_plot(filename=f'single-Louvain3-seed_{LOUVAIN_SEED}-covg', metric='Coverage', parameter='Resolution', y_axis=total_covg, x_axis=resolution_array)
