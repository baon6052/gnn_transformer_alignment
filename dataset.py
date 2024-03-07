import os
from enum import Enum

import numpy as np


class DatasetPath(Enum):
    TRAIN_PATH = "dataset/train"
    VALIDATION_PATH = "dataset/validation"
    TEST_PATH = "dataset/test"


def load_batch(batch_path: str):
    input_node_features = np.load(
        os.path.join(batch_path, "input_node_features.npy")
    )
    input_edge_features = np.load(
        os.path.join(batch_path, "input_edge_features.npy")
    )
    input_graph_features = np.load(
        os.path.join(batch_path, "input_graph_features.npy")
    )
    input_adjacency_matrix = np.load(
        os.path.join(batch_path, "input_adjacency_matrix.npy")
    )
    input_hidden_node_features = np.load(
        os.path.join(batch_path, "input_hidden_node_features.npy")
    )
    input_hidden_edge_features = np.load(
        os.path.join(batch_path, "input_hidden_edge_features.npy")
    )

    node_features_all_layers = []
    edge_features_all_layers = []

    for i in range(4):
        node_features = np.load(
            os.path.join(batch_path, f"out_node_features_{i}.npy")
        )
        node_features_all_layers.append(node_features)

        edge_features = np.load(
            os.path.join(batch_path, f"out_edge_features_{i}.npy")
        )
        edge_features_all_layers.append(edge_features)

    return (
        (
            input_node_features,
            input_edge_features,
            input_graph_features,
            input_adjacency_matrix,
            input_hidden_node_features,
            input_hidden_edge_features,
        ),
        node_features_all_layers,
        edge_features_all_layers,
    )


def dataloader(dataset_path: DatasetPath):
    dataset_path = dataset_path.value
    batch_dirs = [
        os.path.join(dataset_path, d)
        for d in sorted(os.listdir(dataset_path))
        if os.path.isdir(os.path.join(dataset_path, d))
    ]
    for batch_dir in batch_dirs:
        yield load_batch(batch_dir)
