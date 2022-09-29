import time
import numpy as np
import pandas as pd
import networkx as nx

from stellargraph import StellarGraph
from stellargraph.data import BiasedRandomWalk
from gensim.models import Word2Vec

from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, mutual_info_score, normalized_mutual_info_score
from networkx.algorithms.community.quality import modularity, performance, coverage
from stellargraph.random import set_seed

from utils import *


NODE2VEC_SEED = 708450
np.random.seed(NODE2VEC_SEED)
set_seed(NODE2VEC_SEED)
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

G_st = StellarGraph.from_networkx(G)


choice = input('Choose one: \n\t1 Embedding dimensions: 4, Walk length: 100, p: 1, q: 1 \n\t2 Embedding dimensions: 4, Walk number: 20, p: 1, q: 1 \n\t3 Walk length: 100, Walk number: 20, p: 1, q: 1 \n[1 / 2 / 3 / e(exit)]: ')

num_of_clusters = 4
dimensions_array = [4]
walk_length_array = [100]
walk_number_array = [20]
pq_array = [[1,1]]
# pq_array = [[0.5,2], [1,1], [2,0.5]]

if choice == '1':
    walk_number_array += [10, 40, 60]
    walk_number_array.sort()
elif choice == '2':
    walk_length_array += [50, 100, 150, 200]
    walk_length_array.sort()
elif choice == '3':
    dimensions_array += [2, 8, 16, 32, 64, 128]
    dimensions_array.sort()
else:
    exit()

total_acc = list()
total_f1_micro = list()
total_f1_macro = list()
total_mi = list()
total_nmi = list()
total_mdlrt = list()
total_perf = list()
total_covg = list()

for dimensions in dimensions_array:
    for walk_length in walk_length_array:
        for walk_number in walk_number_array:
            for [p,q] in pq_array:
                parameters = {'Walk length': walk_length,
                              'Walk number': walk_number,
                              'p': p,
                              'q': q}

                acc = f1_micro = f1_macro = mi = norm_mi = mdlrt = perf = covg = 0
                start_time = time.perf_counter()

                for seed in seeds:
                    rw = BiasedRandomWalk(G_st)
                    walks = rw.run(
                        nodes=G_st.nodes(),     # root nodes
                        length=walk_length,     # maximum length of a random walk
                        n=walk_number,          # number of random walks per root node
                        p=p,                    # unormalised probability, 1/p, of returning to source node
                        q=q,                    # unormalised probability, 1/q, for moving away from source node
                        weighted=True,
                        seed=int(seed)
                    )
                    str_walks = [[str(n) for n in walk] for walk in walks]

                    model = Word2Vec(
                        sentences=str_walks,
                        vector_size=dimensions,
                        window=5,               # maximum distance between the current and predicted word within a sentence
                        min_count=0,            # ignores all words with total frequency lower than this
                        sg=1,                   # training algorithm: 1 for skip-gram; otherwise CBOW
                        workers=1,              # worker threads to train the model
                        seed=int(seed)
                    )
                    embs = [model.wv[str(n)] for n in range(G.number_of_nodes())]
                    save_embeddings(filename=f'Node2Vec-dims_{dimensions}-walk_len_{walk_length}-walk_num{walk_number}-p_{p}-q_{q}-seed_{seed}', embeddings=embs)

                    kmeans = KMeans(n_clusters=num_of_clusters, random_state=42).fit(embs)
                    y_pred = kmeans.labels_

                    temp_acc, y_pred = calculate_accuracy(y_true=y_true, y_pred=y_pred, k=num_of_clusters)
                    clusters4 = calculate_clusters(y_pred)
                    save_clusters(filename=f'Node2Vec-dims_{dimensions}-clusters_{num_of_clusters}-walk_len_{walk_length}-walk_num{walk_number}-p_{p}-q_{q}-seed_{seed}-acc_{temp_acc:.4f}', clusters=clusters4)

                    acc += temp_acc
                    f1_micro += f1_score(y_true, y_pred, average='micro')
                    f1_macro += f1_score(y_true, y_pred, average='macro')
                    mi += mutual_info_score(y_true, y_pred)
                    norm_mi += normalized_mutual_info_score(y_true, y_pred)
                    mdlrt += modularity(G, clusters4)
                    perf += performance(G, clusters4)
                    covg += coverage(G, clusters4)

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

                save_results(filename=f'results-batch-node2vec', algorithm='Node2Vec', dimensions=dimensions, duration=duration, num_of_clusters=num_of_clusters, cluster_metrics=cluster_metrics, parameters=parameters)

if choice == '1':
    save_metric_plot(filename=f'Node2Vec-dims_4-walk_len_100-p_1-q_1-seed_{NODE2VEC_SEED}-accuracy', metric='Accuracy', parameter='Walk number', y_axis=total_acc, x_axis=walk_number_array)
    save_metric_plot(filename=f'Node2Vec-dims_4-walk_len_100-p_1-q_1-seed_{NODE2VEC_SEED}-f1_micro', metric='F1-micro', parameter='Walk number', y_axis=total_f1_micro, x_axis=walk_number_array)
    save_metric_plot(filename=f'Node2Vec-dims_4-walk_len_100-p_1-q_1-seed_{NODE2VEC_SEED}-f1_macro', metric='F1-macro', parameter='Walk number', y_axis=total_f1_macro, x_axis=walk_number_array)
    save_metric_plot(filename=f'Node2Vec-dims_4-walk_len_100-p_1-q_1-seed_{NODE2VEC_SEED}-mi', metric='MutualInformation', parameter='Walk number', y_axis=total_mi, x_axis=walk_number_array)
    save_metric_plot(filename=f'Node2Vec-dims_4-walk_len_100-p_1-q_1-seed_{NODE2VEC_SEED}-nmi', metric='NormalizedMutualInformation', parameter='Walk number', y_axis=total_nmi, x_axis=walk_number_array)
    save_metric_plot(filename=f'Node2Vec-dims_4-walk_len_100-p_1-q_1-seed_{NODE2VEC_SEED}-mdlrt', metric='Modularity', parameter='Walk number', y_axis=total_mdlrt, x_axis=walk_number_array)
    save_metric_plot(filename=f'Node2Vec-dims_4-walk_len_100-p_1-q_1-seed_{NODE2VEC_SEED}-perf', metric='Performance', parameter='Walk number', y_axis=total_perf, x_axis=walk_number_array)
    save_metric_plot(filename=f'Node2Vec-dims_4-walk_len_100-p_1-q_1-seed_{NODE2VEC_SEED}-covg', metric='Coverage', parameter='Walk number', y_axis=total_covg, x_axis=walk_number_array)
elif choice == '2':
    save_metric_plot(filename=f'Node2Vec-dims_4-walk_num_20-p_1-q_1-seed_{NODE2VEC_SEED}-accuracy', metric='Accuracy', parameter='Walk length', y_axis=total_acc, x_axis=walk_length_array)
    save_metric_plot(filename=f'Node2Vec-dims_4-walk_num_20-p_1-q_1-seed_{NODE2VEC_SEED}-f1_micro', metric='F1-micro', parameter='Walk length', y_axis=total_f1_micro, x_axis=walk_length_array)
    save_metric_plot(filename=f'Node2Vec-dims_4-walk_num_20-p_1-q_1-seed_{NODE2VEC_SEED}-f1_macro', metric='F1-macro', parameter='Walk length', y_axis=total_f1_macro, x_axis=walk_length_array)
    save_metric_plot(filename=f'Node2Vec-dims_4-walk_num_20-p_1-q_1-seed_{NODE2VEC_SEED}-mi', metric='MutualInformation', parameter='Walk length', y_axis=total_mi, x_axis=walk_length_array)
    save_metric_plot(filename=f'Node2Vec-dims_4-walk_num_20-p_1-q_1-seed_{NODE2VEC_SEED}-nmi', metric='NormalizedMutualInformation', parameter='Walk length', y_axis=total_nmi, x_axis=walk_length_array)
    save_metric_plot(filename=f'Node2Vec-dims_4-walk_num_20-p_1-q_1-seed_{NODE2VEC_SEED}-mdlrt', metric='Modularity', parameter='Walk length', y_axis=total_mdlrt, x_axis=walk_length_array)
    save_metric_plot(filename=f'Node2Vec-dims_4-walk_num_20-p_1-q_1-seed_{NODE2VEC_SEED}-perf', metric='Performance', parameter='Walk length', y_axis=total_perf, x_axis=walk_length_array)
    save_metric_plot(filename=f'Node2Vec-dims_4-walk_num_20-p_1-q_1-seed_{NODE2VEC_SEED}-covg', metric='Coverage', parameter='Walk length', y_axis=total_covg, x_axis=walk_length_array)
elif choice == '3':
    save_metric_plot(filename=f'Node2Vec-walk_len_100-walk_num_20-p_1-q_1-seed_{NODE2VEC_SEED}-accuracy', metric='Accuracy', parameter='Embedding dimensions', y_axis=total_acc, x_axis=dimensions_array)
    save_metric_plot(filename=f'Node2Vec-walk_len_100-walk_num_20-p_1-q_1-seed_{NODE2VEC_SEED}-f1_micro', metric='F1-micro', parameter='Embedding dimensions', y_axis=total_f1_micro, x_axis=dimensions_array)
    save_metric_plot(filename=f'Node2Vec-walk_len_100-walk_num_20-p_1-q_1-seed_{NODE2VEC_SEED}-f1_macro', metric='F1-macro', parameter='Embedding dimensions', y_axis=total_f1_macro, x_axis=dimensions_array)
    save_metric_plot(filename=f'Node2Vec-walk_len_100-walk_num_20-p_1-q_1-seed_{NODE2VEC_SEED}-mi', metric='MutualInformation', parameter='Embedding dimensions', y_axis=total_mi, x_axis=dimensions_array)
    save_metric_plot(filename=f'Node2Vec-walk_len_100-walk_num_20-p_1-q_1-seed_{NODE2VEC_SEED}-nmi', metric='NormalizedMutualInformation', parameter='Embedding dimensions', y_axis=total_nmi, x_axis=dimensions_array)
    save_metric_plot(filename=f'Node2Vec-walk_len_100-walk_num_20-p_1-q_1-seed_{NODE2VEC_SEED}-mdlrt', metric='Modularity', parameter='Embedding dimensions', y_axis=total_mdlrt, x_axis=dimensions_array)
    save_metric_plot(filename=f'Node2Vec-walk_len_100-walk_num_20-p_1-q_1-seed_{NODE2VEC_SEED}-perf', metric='Performance', parameter='Embedding dimensions', y_axis=total_perf, x_axis=dimensions_array)
    save_metric_plot(filename=f'Node2Vec-walk_len_100-walk_num_20-p_1-q_1-seed_{NODE2VEC_SEED}-covg', metric='Coverage', parameter='Embedding dimensions', y_axis=total_covg, x_axis=dimensions_array)
