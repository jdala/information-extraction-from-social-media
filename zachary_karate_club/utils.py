import os
import csv
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def calculate_clusters(arr):
    """A function for creating clusters of nodes based on an array
    of predictions.

    Parameters:
        arr (array): Array containing predictions

    Returns:
        res (array): 2d numpy array containing clusters of nodes on 
                     each row
    """

    res = []
    for cluster in range(max(arr) + 1):
        res.append(np.array([ind for ind, val in enumerate(arr) if val==cluster]))

    return res


def save_embeddings(filename, embeddings):
    """A function for creating a text file that contains the calculated
    emeddings.

    Parameters:
        filename (string): Name of the file (without extension)
        embeddings (array): Calculated embeddings
    """

    dir_path = './embeddings'
    file_path = f'{dir_path}/{filename}.txt'

    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)

    with open(file_path, mode='w', newline='') as f:
        for embedding in embeddings:
            for value in embedding:
                f.write(f'{value}, ')
            f.write('\n')


def save_clusters(filename, clusters):
    """A function for creating a text file that contains the calculated
    clusters.

    Parameters:
        filename (string): Name of the file (without extension)
        clusters (array): Calculated clusters
    """

    dir_path = './clusters'
    file_path = f'{dir_path}/{filename}.txt'

    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)

    with open(file_path, mode='w', newline='') as f:
        for cluster in clusters:
            for i in cluster:
                f.write(f'{i}, ')
            f.write('\n')


def save_results(filename, algorithm, dimensions, cluster_metrics, parameters=None):
    """A function for creating a CSV file that contains the calculated
    results along with the parameters of the algorithm.

    Parameters:
        filename (string): Name of the file (without extension)
        algorithm (string): Name of the algorithm that was used
        dimensions (int): Vector size of the final embeddings
        cluster_metrics (dict): Embedding and clustering evaluation scores
        parameters (dict): Parameters of the algorithm. Default is None
    """

    field_names = ['Algorithm', 'Dimensions']
    for key in parameters.keys():
        field_names.append(key)
    for key in cluster_metrics.keys():
        field_names.append(key)

    dir_path = './results'
    file_path = f'{dir_path}/{filename}.csv'

    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)

    if not os.path.isfile(file_path):
        with open(file_path, mode='w', newline='') as f:
            writer = csv.DictWriter(f, delimiter=',', fieldnames=field_names)
            
            writer.writeheader()

    with open(file_path, mode='a', newline='') as f:
        writer = csv.DictWriter(f, delimiter=',', fieldnames=field_names)

        row = {
            'Algorithm' : algorithm,
            'Dimensions' : dimensions,
        }

        for k, v in parameters.items():
            row[k] = v
        for k, v in cluster_metrics.items():
            row[k] = round(v, 4)

        writer.writerow(row)


def save_coloured_graph(filename, G, positions, colors):
    """A function for creating a PNG file that contains the plot of the graph
    colored according to ground truth and positioned according to embeddings.

    Parameters:
        filename (string): Name of the file (without extension)
        G (nx.Graph): Networkx graph
        positions (array): Calculated coordinates of each point (ONLY 2D)
        colors (array): Color of each point
    """

    dir_path = './figures'
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    
    fig = plt.figure()
    nx.draw(G, with_labels=True, node_color=colors, pos=positions)
    fig.savefig(f'{dir_path}/{filename}.png', bbox_inches='tight')
    plt.close(fig=fig)


def save_metric_plot(filename, metric, parameter, y_axis, x_axis):
    """A function for creating a PNG file that contains the plot of the diagram
    showing a metric changes relative to the value of a parameter.

    Parameters:
        filename (string): Name of the file (without extension)
        metric (array): Metric values to be plotted (y-axis)
        parameter (array): Parameter values to be plotted (x-axis)
        x-axis (string): Title for the x-axis
        y-axis (string): Title for the y-axis
    """

    dir_path = './metrics_plots'
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)

    fig = plt.figure()
    plt.plot(x_axis, y_axis)

    plt.xlabel(parameter)
    plt.ylabel(metric)
    plt.ylim([0,1])

    fig.savefig(f'{dir_path}/{filename}.png')
    plt.close(fig=fig)
