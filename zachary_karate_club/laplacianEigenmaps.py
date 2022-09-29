import numpy as np
import networkx as nx

from karateclub.node_embedding.neighbourhood import LaplacianEigenmaps

from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, mutual_info_score, normalized_mutual_info_score, homogeneity_score, completeness_score, f1_score
from networkx.algorithms.community.quality import modularity, performance, coverage

from utils import *


G = nx.karate_club_graph()
y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
colors = ['orangered' if i==0 else 'paleturquoise' for i in y_true]


dimensions_array = [2, 4, 8, 16, 32]

total_acc = list()
total_f1 = list()

for dimensions in dimensions_array:
    model = LaplacianEigenmaps(dimensions = dimensions)

    model.fit(G.copy())
    embs = model.get_embedding()

    kmeans = KMeans(n_clusters=2, random_state=42).fit(embs)
    y_pred = kmeans.labels_
    y_pred_reversed = np.array([int(not(x)) for x in y_pred])

    temp_acc = max(accuracy_score(y_true, y_pred), accuracy_score(y_true, y_pred_reversed))
    temp_f1 = max(f1_score(y_true, y_pred), f1_score(y_true, y_pred_reversed))

    clusters = calculate_clusters(y_pred)
    save_clusters(f'LaplacianEigenmaps-dims_{dimensions}-acc_{temp_acc:.4f}', clusters=clusters)

    if dimensions == 2:
        positions = dict(zip(G.nodes, embs.tolist()))
        save_coloured_graph(filename=f'LaplacianEigenmaps-dims_{dimensions}-acc_{temp_acc:.4f}', G=G, positions=positions, colors=colors)


    cluster_metrics = {'Accuracy': temp_acc,
                       'F1-score': temp_f1,
                       'MutualInformation': mutual_info_score(y_true, y_pred),
                       'NormalizedMutualInformation': normalized_mutual_info_score(y_true, y_pred),
                       'Homogeneity': homogeneity_score(y_true, y_pred),
                       'Completeness': completeness_score(y_true, y_pred),
                       'Modularity': modularity(G, clusters),
                       'Performance': performance(G, clusters),
                       'Coverage': coverage(G, clusters)}


    save_results(filename=f'results-laplacianeigenmaps', algorithm='LaplacianEigenmaps', dimensions=dimensions, cluster_metrics=cluster_metrics, parameters={})

    total_acc.append(cluster_metrics.get('Accuracy'))
    total_f1.append(cluster_metrics.get('F1-score'))

save_metric_plot(filename=f'LaplacianEigenmaps-accuracy', metric='Accuracy', parameter='Embedding dimensions', y_axis=total_acc, x_axis=dimensions_array)
save_metric_plot(filename=f'LaplacianEigenmaps-f1_score', metric='F1-score', parameter='Embedding dimensions', y_axis=total_f1, x_axis=dimensions_array)