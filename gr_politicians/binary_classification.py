import numpy as np
import pandas as pd
import networkx as nx
from itertools import combinations

import community as community_louvain
import collections

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from utils import *


LOUVAIN_SEED = 537921
np.random.seed(LOUVAIN_SEED)

TRAIN_GRAPHS =10
TEST_GRAPHS = 1
START_GRAPH = 38

ground_truth = pd.read_csv('../datasets/gr_politicians/batch/dataset_cleanup/mapping-not_NaN.csv', usecols=['id', 'party'])

total_nodes = list()
total_predictions = list()

resolution = 1

for i in range(START_GRAPH, START_GRAPH+TRAIN_GRAPHS+TEST_GRAPHS):
    G_df = pd.read_csv(f'../datasets/gr_politicians/batch/edge_lists/{i}.edges')
    G_df.rename(columns={'overlap': 'weight'}, inplace=True)
    G = nx.from_pandas_edgelist(G_df, source='u', target='v', edge_attr='weight', create_using=nx.Graph())

    ground_truth_temp = ground_truth[ground_truth.id.isin(G.nodes())].sort_values(by=['id'])
    y_true = ground_truth_temp.party.to_numpy()

    id_list = ground_truth_temp.id.to_numpy()
    id_mapping = dict(zip(id_list, list(range(id_list.shape[0]))))
    id_mapping_reversed = dict(zip(list(range(id_list.shape[0])), id_list))

    G = G.subgraph(id_list)
    total_nodes.append(list(G.nodes()))
    G = nx.relabel_nodes(G, id_mapping)

    # **** CALCULATE COMMUNITIES ( ****
    partition = community_louvain.best_partition(G, weight='weight', resolution=resolution, random_state=LOUVAIN_SEED)
    partition = collections.OrderedDict(sorted(partition.items()))
    y_pred = np.array(list(partition.values()))
    
    temp_acc, y_pred = calculate_accuracy(y_true=y_true, y_pred=y_pred, k=6)
    # print(y_pred)

    # clusters = calculate_clusters(y_pred)
    total_predictions.append(y_pred)
    # save_clusters(filename=f'Louvain-seed_{seed}-acc_{temp_acc:.4f}', clusters=clusters)

    # **** ) CALCULATE COMMUNITIES ****


# **** CREATE DATASET FOR CLASSIFIER ( ****
common_nodes = list(set.intersection(*map(set, total_nodes)))
combination_of_common_nodes = list(combinations(common_nodes, 2))

x_array = list()
y_array = list()

for pair in combination_of_common_nodes:
    temp_x = list()
    for i in range(TRAIN_GRAPHS):
        if total_predictions[i][id_mapping[pair[0]]] == total_predictions[i][id_mapping[pair[1]]]:
            temp_x.append(1)
        else:
            temp_x.append(0)
    x_array.append(temp_x)

    if total_predictions[-1][id_mapping[pair[0]]] == total_predictions[-1][id_mapping[pair[1]]]:
        y_array.append(1)
    else:
        y_array.append(0)
# **** ) CREATE DATASET FOR CLASSIFIER ****

# **** TRAIN AND TEST PREDICTION MODEL ( ****
x_train, x_test, y_train, y_test = train_test_split(x_array,
                                                    y_array,
                                                    test_size=0.33,
                                                    random_state=42)


model = SVC()
model.fit(x_train, y_train)
preds = model.predict(x_test)

print(f'# of common nodes: \t\t\t{len(common_nodes)}')

print(f'\n# of permutations of common nodes: \t{len(combination_of_common_nodes)}')
print(f'# of permutations for training: \t{len(x_train)}')
print(f'# of permutations for testing: \t\t{len(x_test)}')

num_of_y_zeros = len(y_array) - sum(x for x in y_array)
num_of_y_train_zeros = len(y_train) - sum(x for x in y_train)
num_of_y_test_zeros = len(y_test) - sum(x for x in y_test)

print(f'\n# of zeros in y: \t\t\t{num_of_y_zeros}')
print(f'# of zeros in y_train: \t\t\t{num_of_y_train_zeros}')
print(f'# of zeros in y_test: \t\t\t{num_of_y_test_zeros}')

print(f'\npercentage of zeros in y: \t\t{num_of_y_zeros / len(y_array) * 100 :.2f}%')
print(f'percentage of zeros in y_train: \t{num_of_y_train_zeros / len(y_train) * 100 :.2f}%')
print(f'percentage of zeros in y_test: \t\t{num_of_y_test_zeros / len(y_test) * 100 :.2f}%')

print(f'\nPrediction accuracy: \t\t\t{accuracy_score(y_test, preds):.4f}')
print(f'Prediction f1_score: \t\t\t{f1_score(y_test, preds):.4f}')
# **** ) TRAIN AND TEST PREDICTION MODEL ****
