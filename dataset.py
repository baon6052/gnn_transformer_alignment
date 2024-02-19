import os

import numpy as np


def load_batch(batch_path: str):
    node_features = np.load(os.path.join(batch_path, "node_features.npy"))
    edge_features = np.load(os.path.join(batch_path, "edge_features.npy"))
    graph_features = np.load(os.path.join(batch_path, "graph_features.npy"))
    adjacency_matrix = np.load(os.path.join(batch_path, "adjacency_matrix.npy"))
    hidden_node_features = np.load(
        os.path.join(batch_path, "hidden_node_features.npy")
    )
    out_node_features = np.load(
        os.path.join(batch_path, "out_node_features.npy")
    )
    return (
        node_features,
        edge_features,
        graph_features,
        adjacency_matrix,
        hidden_node_features,
    ), out_node_features


def data_loader(dataset_path: str):
    batch_dirs = [
        os.path.join(dataset_path, d)
        for d in sorted(os.listdir(dataset_path))
        if os.path.isdir(os.path.join(dataset_path, d))
    ]
    for batch_dir in batch_dirs:
        yield load_batch(batch_dir)
