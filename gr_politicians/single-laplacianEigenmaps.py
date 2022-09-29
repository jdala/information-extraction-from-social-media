import time
import numpy as np
import pandas as pd
import networkx as nx

from karateclub.node_embedding.neighbourhood import LaplacianEigenmaps

from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, mutual_info_score, normalized_mutual_info_score
from networkx.algorithms.community.quality import modularity, performance, coverage

from utils import *


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


num_of_clusters = 5
dimensions_array = [2, 4, 8, 16, 32, 64, 128]

total_acc = list()
total_f1_micro = list()
total_f1_macro = list()
total_mi = list()
total_nmi = list()
total_mdlrt = list()
total_perf = list()
total_covg = list()

for dimensions in dimensions_array:
    start_time = time.perf_counter()

    model = LaplacianEigenmaps(dimensions = dimensions)
    
    model.fit(G.copy())
    embs = model.get_embedding()

    kmeans = KMeans(n_clusters=num_of_clusters, random_state=42).fit(embs)
    y_pred = kmeans.labels_

    temp_acc, y_pred = calculate_accuracy(y_true=y_true, y_pred=y_pred, k=num_of_clusters)
    clusters = calculate_clusters(y_pred)

    save_embeddings(filename=f'LaplacianEigenmaps-dims_{dimensions}-acc_{temp_acc:.4f}', embeddings=embs)
    save_clusters(filename=f'LaplacianEigenmaps-dims_{dimensions}-acc_{temp_acc:.4f}', clusters=clusters)

    cluster_metrics = {'Accuracy': temp_acc,
                       'F1-micro': f1_score(y_true, y_pred, average='micro'),
                       'F1-macro': f1_score(y_true, y_pred, average='macro'),
                       'MutualInformation': mutual_info_score(y_true, y_pred),
                       'NormalizedMutualInformation': normalized_mutual_info_score(y_true, y_pred),
                       'Modularity': modularity(G, clusters),
                       'Performance': performance(G, clusters),
                       'Coverage': coverage(G, clusters)}

    duration = (time.perf_counter() - start_time)

    total_acc.append(cluster_metrics.get('Accuracy'))
    total_f1_micro.append(cluster_metrics.get('F1-micro'))
    total_f1_macro.append(cluster_metrics.get('F1-macro'))
    total_mi.append(cluster_metrics.get('MutualInformation'))
    total_nmi.append(cluster_metrics.get('NormalizedMutualInformation'))
    total_mdlrt.append(cluster_metrics.get('Modularity'))
    total_perf.append(cluster_metrics.get('Performance'))
    total_covg.append(cluster_metrics.get('Coverage'))

    save_results(filename='results-single-laplacianEigenmaps', algorithm='LaplacianEigenmaps', dimensions=dimensions, cluster_metrics=cluster_metrics, num_of_clusters=num_of_clusters, parameters={})

save_metric_plot(filename=f'single-LaplacianEigenmaps-accuracy', metric='Accuracy', parameter='Embedding dimensions', y_axis=total_acc, x_axis=dimensions_array)
save_metric_plot(filename=f'single-LaplacianEigenmaps-f1_micro', metric='F1-micro', parameter='Embedding dimensions', y_axis=total_f1_micro, x_axis=dimensions_array)
save_metric_plot(filename=f'single-LaplacianEigenmaps-f1_macro', metric='F1-macro', parameter='Embedding dimensions', y_axis=total_f1_macro, x_axis=dimensions_array)
save_metric_plot(filename=f'single-LaplacianEigenmaps-mi', metric='MutualInformation', parameter='Embedding dimensions', y_axis=total_mi, x_axis=dimensions_array)
save_metric_plot(filename=f'single-LaplacianEigenmaps-nmi', metric='NormalizedMutualInformation', parameter='Embedding dimensions', y_axis=total_nmi, x_axis=dimensions_array)
save_metric_plot(filename=f'single-LaplacianEigenmaps-mdlrt', metric='Modularity', parameter='Embedding dimensions', y_axis=total_mdlrt, x_axis=dimensions_array)
save_metric_plot(filename=f'single-LaplacianEigenmaps-perf', metric='Performance', parameter='Embedding dimensions', y_axis=total_perf, x_axis=dimensions_array)
save_metric_plot(filename=f'single-LaplacianEigenmaps-covg', metric='Coverage', parameter='Embedding dimensions', y_axis=total_covg, x_axis=dimensions_array)
