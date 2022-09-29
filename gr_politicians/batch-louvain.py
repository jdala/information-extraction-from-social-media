import time
import numpy as np
import pandas as pd
import networkx as nx

import community as community_louvain
import collections

from sklearn.metrics import f1_score, mutual_info_score, normalized_mutual_info_score
from networkx.algorithms.community.quality import modularity, performance, coverage
from sklearn.manifold import TSNE

from utils import *


LOUVAIN_SEED = 537921
np.random.seed(LOUVAIN_SEED)
seeds = np.random.randint(10000, size=10)


ground_truth = pd.read_csv('../datasets/gr_politicians/batch/dataset_cleanup/mapping-not_NaN.csv', usecols=['id', 'party'])


resolution = 1

for i in range(1, 49):
    # **** IMPORT GRAPH ( ****
    G_df = pd.read_csv(f'../datasets/gr_politicians/batch/edge_lists/{i}.edges')
    G_df.rename(columns={'overlap': 'weight'}, inplace=True)
    G = nx.from_pandas_edgelist(G_df, source='u', target='v', edge_attr='weight', create_using=nx.Graph())

    ground_truth_temp = ground_truth[ground_truth.id.isin(G.nodes())].sort_values(by=['id'])
    y_true = ground_truth_temp.party.to_numpy()
    # print(f'Graph {i:2.0f} -> {ground_truth_temp.shape[0]} | {len(set(y_true))}')
    # continue

    id_list = ground_truth_temp.id.to_numpy()
    id_mapping = dict(zip(id_list, list(range(id_list.shape[0]))))
    id_mapping_reversed = dict(zip(list(range(id_list.shape[0])), id_list))

    G = G.subgraph(id_list)
    G = nx.relabel_nodes(G, id_mapping)
    # **** ) IMPORT GRAPH ****

    parameters = {'resolution': resolution}

    acc = f1_micro = f1_macro = mi = norm_mi = mdlrt = perf = covg = 0
    start_time = time.perf_counter()

    for seed in seeds:
        partition = community_louvain.best_partition(G, weight='weight', resolution=resolution, random_state=seed)
        partition = collections.OrderedDict(sorted(partition.items()))
        
        y_pred = np.array(list(partition.values()))
        num_of_clusters = len(set(y_pred))

        temp_acc, y_pred = calculate_accuracy(y_true=y_true, y_pred=y_pred, k=num_of_clusters)
        clusters = calculate_clusters(y_pred)

        save_clusters(filename=f'Louvain-Snapshot_{i}-seed_{seed}-acc_{temp_acc:.4f}', clusters=clusters)

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
    
    save_results(filename=f'results-batch-louvain', algorithm=f'Louvain-Snapshot_{i}', dimensions='-', cluster_metrics=cluster_metrics, num_of_clusters=num_of_clusters, duration=duration, parameters=parameters)
