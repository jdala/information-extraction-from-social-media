import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

import community as community_louvain
import collections

from stellargraph import StellarGraph
from stellargraph.data import BiasedRandomWalk
from gensim.models import Word2Vec

from sklearn.manifold import TSNE
from stellargraph.random import set_seed

from utils import *


LOUVAIN_SEED = 537921
np.random.seed(LOUVAIN_SEED)
set_seed(LOUVAIN_SEED)
seeds = np.random.randint(10000, size=10)


ground_truth = pd.read_csv('../datasets/gr_politicians/batch/dataset_cleanup/mapping-not_NaN.csv', usecols=['id', 'party'])


choice = input('Choose one: \n\t1 Coordinates based on networkx spring_layout \n\t2 Coordinates based on node2vec and t-SNE \n[1 / 2]: ')

resolution = 1

for i in range(1, 49):
    # **** IMPORT GRAPH ( ****
    G_df = pd.read_csv(f'../datasets/gr_politicians/batch/edge_lists/{i}.edges')
    G_df.rename(columns={'overlap': 'weight'}, inplace=True)
    G = nx.from_pandas_edgelist(G_df, source='u', target='v', edge_attr='weight', create_using=nx.Graph())

    ground_truth_temp = ground_truth[ground_truth.id.isin(G.nodes())].sort_values(by=['id'])
    y_true = ground_truth_temp.party.to_numpy()

    id_list = ground_truth_temp.id.to_numpy()
    id_mapping = dict(zip(id_list, list(range(id_list.shape[0]))))
    id_mapping_reversed = dict(zip(list(range(id_list.shape[0])), id_list))

    G = G.subgraph(id_list)
    G = nx.relabel_nodes(G, id_mapping)
    # **** ) IMPORT GRAPH ****


    # **** FIND COMMUNITIES USING THE LOUVAIN METHOD ( ****
    partition = community_louvain.best_partition(G, weight='weight', resolution=resolution, random_state=LOUVAIN_SEED)
    partition = collections.OrderedDict(sorted(partition.items()))
    y_pred = np.array(list(partition.values()))

    acc, y_pred = calculate_accuracy(y_true=y_true, y_pred=y_pred, k=6)    
    # **** ) FIND COMMUNITIES USING THE LOUVAIN METHOD ****


    palette = ['blue', 'red', 'green', 'yellow', 'orange']
    node_colors = [palette[i] for i in y_pred]

    if choice == '1':
        G = nx.relabel_nodes(G, id_mapping_reversed)
        pos = nx.spring_layout(G, weight='weight', seed=LOUVAIN_SEED)

        filename = f'Louvain-spring_layout-Snapshot_{i}'
    elif choice == '2':
        # **** CALCULATE NODE COORDINATED BASED ON NODE2VEC EMBEDDINGS ( ****
        G_st = StellarGraph.from_networkx(G)

        rw = BiasedRandomWalk(G_st)
        walks = rw.run(
            nodes=G_st.nodes(),     # root nodes
            length=100,             # maximum length of a random walk
            n=20,                   # number of random walks per root node
            p=1,                    # unormalised probability, 1/p, of returning to source node
            q=1,                    # unormalised probability, 1/q, for moving away from source node
            weighted=True,
            seed=42
        )
        str_walks = [[str(n) for n in walk] for walk in walks]

        model = Word2Vec(
            sentences=str_walks,
            vector_size=4,
            window=5,               # maximum distance between the current and predicted word within a sentence
            min_count=0,            # ignores all words with total frequency lower than this
            sg=1,                   # training algorithm: 1 for skip-gram; otherwise CBOW
            workers=1,              # worker threads to train the model
            seed=42
        )
        embs = [model.wv[str(n)] for n in range(G.number_of_nodes())]
        tsne = TSNE(n_components=2, random_state=42)
        embs_2d = tsne.fit_transform(embs)
        pos = dict(zip(id_list, embs_2d.tolist()))

        G = nx.relabel_nodes(G, id_mapping_reversed)

        filename = f'Louvain-node2vec-dims_4-tSNE-Snapshot_{i}'
        # **** ) CALCULATE NODE COORDINATED BASED ON NODE2VEC EMBEDDINGS ****
    else:
        print('Invalid input\n')
        exit()


    # **** SAVE FIGURES ( ****
    dir_path='../figures'
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)

    fig = plt.figure(figsize=[9.6, 7.2], dpi=300)
    nx.draw_networkx_nodes(G, pos, node_size=150, node_color=node_colors, alpha=.4)
    nx.draw_networkx_labels(G, pos, font_size=8)

    plt.title(f'accuracy = {acc:.4f}')
    fig.savefig(f'{dir_path}/{filename}.png', bbox_inches='tight')
    plt.close(fig=fig)
    # **** ) SAVE FIGURES ****
