import csv
import networkx as nx
import numpy as np


G = nx.read_edgelist('./graph.edges', nodetype=int)
graph_edges = nx.get_edge_attributes(G, 'weight')


with open(f'graph.csv', mode='w', newline='') as f:
    writer = csv.DictWriter(f, delimiter=',', fieldnames=['node1', 'node2', 'weight'])
    writer.writeheader()

    for pair, weight in graph_edges.items():
        writer.writerow({'node1': pair[0], 'node2': pair[1], 'weight': weight})
        writer.writerow({'node1': pair[1], 'node2': pair[0], 'weight': weight})


with open(f'ground_truth.txt', mode='r') as f:
    ground_truth_file = f.read().splitlines()


with open(f'members-9_zeros_1_random.csv', mode='w', newline='') as f:
    writer = csv.writer(f)

    for memb in ground_truth_file:
        memb = memb.split(' ')
        # writer.writerow([memb[0], memb[1], f'{np.random.rand()} {np.random.rand()} {np.random.rand()} {np.random.rand()} {np.random.rand()} {np.random.rand()} {np.random.rand()} {np.random.rand()} {np.random.rand()} {np.random.rand()} {np.random.rand()} {np.random.rand()} {np.random.rand()} {np.random.rand()} {np.random.rand()}'])
        writer.writerow([memb[0], memb[1], f'{np.random.choice([0, np.random.rand()], p=[.9, .1])} {np.random.choice([0, np.random.rand()], p=[.9, .1])} {np.random.choice([0, np.random.rand()], p=[.9, .1])} {np.random.choice([0, np.random.rand()], p=[.9, .1])} {np.random.choice([0, np.random.rand()], p=[.9, .1])} {np.random.choice([0, np.random.rand()], p=[.9, .1])} {np.random.choice([0, np.random.rand()], p=[.9, .1])} {np.random.choice([0, np.random.rand()], p=[.9, .1])} {np.random.choice([0, np.random.rand()], p=[.9, .1])} {np.random.choice([0, np.random.rand()], p=[.9, .1])} {np.random.choice([0, np.random.rand()], p=[.9, .1])} {np.random.choice([0, np.random.rand()], p=[.9, .1])} {np.random.choice([0, np.random.rand()], p=[.9, .1])} {np.random.choice([0, np.random.rand()], p=[.9, .1])} {np.random.choice([0, np.random.rand()], p=[.9, .1])}'])
