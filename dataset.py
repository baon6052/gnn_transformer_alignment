import os
from enum import StrEnum

import numpy as np


class DatasetPath(StrEnum):
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

    node_features_all_layers = []

    for i in range(3):
        node_features = np.load(
            os.path.join(batch_path, f"out_node_features_{i}.npy")
        )
        node_features_all_layers.append(node_features)

    out_edge_features = np.load(
        os.path.join(batch_path, "out_edge_features.npy")
    )

    return (
        (
            input_node_features,
            input_edge_features,
            input_graph_features,
            input_adjacency_matrix,
            input_hidden_node_features,
        ),
        node_features_all_layers,
        out_edge_features,
    )


def dataloader(dataset_path: DatasetPath):
    batch_dirs = [
        os.path.join(dataset_path, d)
        for d in sorted(os.listdir(dataset_path))
        if os.path.isdir(os.path.join(dataset_path, d))
    ]
    for batch_dir in batch_dirs:
        yield load_batch(batch_dir)
