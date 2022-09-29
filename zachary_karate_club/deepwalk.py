import numpy as np
import networkx as nx

from karateclub.node_embedding.neighbourhood import DeepWalk

from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, mutual_info_score, normalized_mutual_info_score, homogeneity_score, completeness_score, f1_score
from networkx.algorithms.community.quality import modularity, performance, coverage

from utils import *


DEEPWALK_SEED = 408145
np.random.seed(DEEPWALK_SEED)
seeds = np.random.randint(10000, size=10)


G = nx.karate_club_graph()
y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
colors = ['orangered' if i==0 else 'paleturquoise' for i in y_true]


choice = input('Choose one: \n\t1 Embedding dimensions : 2, Walk length: 40 \n\t2 Embedding dimensions : 2, Walk number: 10 \n\t3 Walk length : 40, Walk number: 10 \n[1 / 2 / 3 / e(exit)]: ')

dimensions_array = [2]
walk_length_array = [40]
walk_number_array = [10]
if choice == '1':
    walk_number_array += [5, 15, 20, 25, 30, 35, 40]
    walk_number_array.sort()
elif choice == '2':
    walk_length_array += [20, 60, 80, 100, 120, 140]
    walk_length_array.sort()
elif choice == '3':
    dimensions_array += [4, 8, 16, 32]
    dimensions_array.sort()
else:
    exit()

total_acc = list()
total_f1 = list()

for dimensions in dimensions_array:
    for walk_length in walk_length_array:
        for walk_number in walk_number_array:
            parameters = {'Walk length': walk_length,
                          'Walk number': walk_number}

            acc = f1 = mdlrt = perf = covg = mi = norm_mi = homogen = complet = 0

            for seed in seeds:
                model = DeepWalk(walk_number = walk_number,
                                 walk_length = walk_length,
                                 dimensions = dimensions,
                                 workers = 1,
                                 window_size = 3,
                                 seed = seed)

                model.fit(G.copy())
                embs = model.get_embedding()

                kmeans = KMeans(n_clusters=2, random_state=42).fit(embs)
                y_pred = kmeans.labels_
                y_pred_reversed = np.array([int(not(x)) for x in y_pred])

                temp_acc = max(accuracy_score(y_true, y_pred), accuracy_score(y_true, y_pred_reversed))
                temp_f1 = max(f1_score(y_true, y_pred), f1_score(y_true, y_pred_reversed))

                clusters = calculate_clusters(y_pred)
                save_clusters(f'DeepWalk-dims_{dimensions}-walk_len_{walk_length}-walk_num{walk_number}-seed_{seed}-acc_{temp_acc:.4f}', clusters=clusters)

                acc += temp_acc
                f1 += temp_f1
                mi += mutual_info_score(y_true, y_pred)
                norm_mi += normalized_mutual_info_score(y_true, y_pred)
                homogen += homogeneity_score(y_true, y_pred)
                complet += completeness_score(y_true, y_pred)
                mdlrt += modularity(G, clusters)
                perf += performance(G, clusters)
                covg += coverage(G, clusters)

                if choice == '3':
                    continue
                positions = dict(zip(G.nodes, embs.tolist()))
                save_coloured_graph(filename=f'DeepWalk-dims_{dimensions}-walk_len_{walk_length}-walk_num{walk_number}-seed_{seed}-acc_{temp_acc:.4f}', G=G, positions=positions, colors=colors)

            cluster_metrics = {'Accuracy': acc / seeds.shape[0],
                               'F1-score': f1 / seeds.shape[0],
                               'MutualInformation': mi / seeds.shape[0],
                               'NormalizedMutualInformation': norm_mi / seeds.shape[0],
                               'Homogeneity': homogen / seeds.shape[0],
                               'Completeness': complet / seeds.shape[0],
                               'Modularity': mdlrt / seeds.shape[0],
                               'Performance': perf / seeds.shape[0],
                               'Coverage': covg / seeds.shape[0]}

            save_results(filename=f'results-deepwalk', algorithm='DeepWalk', dimensions=dimensions, cluster_metrics=cluster_metrics, parameters=parameters)

            total_acc.append(cluster_metrics.get('Accuracy'))
            total_f1.append(cluster_metrics.get('F1-score'))

if choice == '1':
    save_metric_plot(filename=f'DeepWalk-dims_2-walk_len_40-seed_{DEEPWALK_SEED}-accuracy', metric='Accuracy', parameter='Walk number', y_axis=total_acc, x_axis=walk_number_array)
    save_metric_plot(filename=f'DeepWalk-dims_2-walk_len_40-seed_{DEEPWALK_SEED}-f1_score', metric='F1-score', parameter='Walk number', y_axis=total_f1, x_axis=walk_number_array)
elif choice == '2':
    save_metric_plot(filename=f'DeepWalk-dims_2-walk_num_10-seed_{DEEPWALK_SEED}-accuracy', metric='Accuracy', parameter='Walk length', y_axis=total_acc, x_axis=walk_length_array)
    save_metric_plot(filename=f'DeepWalk-dims_2-walk_num_10-seed_{DEEPWALK_SEED}-f1_score', metric='F1-score', parameter='Walk length', y_axis=total_f1, x_axis=walk_length_array)
elif choice == '3':
    save_metric_plot(filename=f'DeepWalk-walk_len_40-walk_num_10-seed_{DEEPWALK_SEED}-accuracy', metric='Accuracy', parameter='Embedding dimensions', y_axis=total_acc, x_axis=dimensions_array)
    save_metric_plot(filename=f'DeepWalk-walk_len_40-walk_num_10-seed_{DEEPWALK_SEED}-f1_score', metric='F1-score', parameter='Embedding dimensions', y_axis=total_f1, x_axis=dimensions_array)