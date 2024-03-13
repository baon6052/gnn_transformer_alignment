import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from checkpointer import Checkpointer
from dataset import DatasetPath, dataloader
from models.mpnn import AlignedMPNN
import jax
import jax.numpy as jnp
import haiku as hk
from pathlib import Path
import glob

MODEL_DIR = Path(Path.cwd(), "visualise_models")
MODEL_DIR.mkdir(exist_ok=True, parents=True)

trained_models = glob.glob(f"{MODEL_DIR}/*.pkl")
rt_c = "#8ceda6"
mpnn_c = "#e8806b"


# sns.set(style='whitegrid')
# sns.set_context('notebook')


model = None
parameters = None
optimizer = None
optimizer_state = None


def model_fn(
    node_fts,
    edge_fts,
    graph_fts,
    adj_mat,
    hidden,
    edge_em,
    mid_size=192,
    reduction_func=jnp.mean,
    use_layer_norm: bool = True,
    add_virtual_node: bool = True,
    disable_edge_updates: bool = True,
    apply_attention: bool = False,
    number_of_attention_heads: int = 1,
):
    model = AlignedMPNN(
        nb_layers=3,
        out_size=192,
        mid_size=mid_size,
        activation=None,
        reduction=reduction_func,
        use_ln=use_layer_norm,
        add_virtual_node=add_virtual_node,
        disable_edge_updates=disable_edge_updates,
        apply_attention=apply_attention,
        number_of_attention_heads=number_of_attention_heads,
    )

    return model(node_fts, edge_fts, graph_fts, adj_mat, hidden, edge_em)


def convert_bool(a):
    if a == "False":
        b = False
    else:
        b = True
    return b


def plot(trained_models, train_batch, val_batch, test_batch):
    global model, parameters, optimizer, optimizer_state

    for i, path in enumerate(trained_models):

        fn = path.split("/")[-1].split(".")[0].split("-")
        add_virtual_node = convert_bool(fn[1])
        use_layer_norm = convert_bool(fn[3])
        mid_dim = int(fn[5])
        reduction_str = fn[7]
        disable_edge_updates = convert_bool(fn[9])
        apply_attention = convert_bool(fn[11])
        number_of_attention_heads = 0

        if apply_attention:
            number_of_attention_heads = int(fn[13])

        if reduction_str == "mean":
            reduction_func = jnp.mean

        elif reduction_str == "sum":
            reduction_func = jnp.sum

        elif reduction_str == "max":
            reduction_func = jnp.max

        fig_train, ax_train = plt.subplots(2, 4, figsize=(15, 7))
        fig_val, ax_val = plt.subplots(2, 4, figsize=(15, 7))
        fig_test, ax_test = plt.subplots(2, 4, figsize=(15, 7))

        # INITIALISE MODEL
        (
            (
                input_node_fts,
                input_edge_fts,
                input_graph_fts,
                input_adj_mat,
                input_hidden,
                input_hidden_edge,
            ),
            transformer_node_features_all_layers,
            transformer_edge_features_all_layers,
        ) = test_batch

        def model_wrapper(node_fts, edge_fts, graph_fts, adj_mat, hidden, edge_em):
            return model_fn(
                node_fts,
                edge_fts,
                graph_fts,
                adj_mat,
                hidden,
                edge_em,
                mid_size=mid_dim,
                reduction_func=reduction_func,
                use_layer_norm=use_layer_norm,
                add_virtual_node=add_virtual_node,
                disable_edge_updates=disable_edge_updates,
                apply_attention=apply_attention,
                number_of_attention_heads=number_of_attention_heads,
            )

        model = hk.without_apply_rng(hk.transform(model_wrapper))

        parameters = model.init(
            jax.random.PRNGKey(42),
            node_fts=input_node_fts,
            edge_fts=input_edge_fts,
            graph_fts=input_graph_fts,
            adj_mat=input_adj_mat,
            hidden=input_hidden,
            edge_em=input_hidden_edge,
        )
        ckpt = Checkpointer(f"{path}")
        parameters = ckpt.load()
        print("A")
        for batch, axes in zip(
            [train_batch, val_batch, test_batch], [ax_train, ax_val, ax_test]
        ):
            # GET TRANSFORMER LAYER and EDGE EMBEDDINGS
            (
                (
                    input_node_fts,
                    input_edge_fts,
                    input_graph_fts,
                    input_adj_mat,
                    input_hidden,
                    input_hidden_edge,
                ),
                transformer_node_features_all_layers,
                transformer_edge_features_all_layers,
            ) = batch

            # GET MPNN LAYER and EDGE EMBEDDINGS
            mpnn_node_features_all_layers, mpnn_edge_features_all_layers = model.apply(
                parameters,
                input_node_fts,
                input_edge_fts,
                input_graph_fts,
                input_adj_mat,
                input_hidden,
                input_hidden_edge,
            )

            for visualise, transformer_fts, mpnn_fts in zip(
                ["Edges", "Nodes"],
                [
                    transformer_edge_features_all_layers,
                    transformer_node_features_all_layers,
                ],
                [mpnn_edge_features_all_layers, mpnn_node_features_all_layers],
            ):
                if visualise == "Edges":
                    ax_row = 0

                else:
                    ax_row = 1
                for i, (transformer_ft, mpnn_ft) in enumerate(
                    zip(transformer_fts, mpnn_fts)
                ):
                    rt = jax.numpy.asarray(transformer_ft).ravel()
                    mpnn = jax.numpy.asarray(mpnn_ft).ravel()
                    ax = axes[ax_row, i]
                    ax.set_title(f"{visualise} Layer {i}")

                    ax.hist(mpnn, color=mpnn_c, alpha=0.5, label="MPNN", bins=50)
                    ax.hist(rt, color=rt_c, alpha=0.5, label="Transformer", bins=50)
                    ax.legend()
                    ax.set_xlabel("Feature Value")
                    ax.set_ylabel("Frequency")
                    ax.grid(True)

        fig_train.suptitle(f"Train - {path.split('/')[-1]}")
        fig_val.suptitle(f"Validation - {path.split('/')[-1]}")
        fig_test.suptitle(f"Test - {path.split('/')[-1]}")
        plt.tight_layout()
        fig_train.tight_layout()
        output_dir = Path(f"visualise_embeddings/{path.split('/')[-1]}")
        output_dir.mkdir(exist_ok=True, parents=True)
        fig_train.savefig(f"{output_dir}/train.png")

        fig_val.tight_layout()
        fig_val.savefig(f"{output_dir}/validation.png")

        fig_test.tight_layout()
        fig_test.savefig(f"{output_dir}/test.png")


if __name__ == "__main__":
    for train_batch in dataloader(DatasetPath.TRAIN_PATH):
        inputs, targets = train_batch[:-1], train_batch[-1]
        train_batch = (
            *inputs,
            targets,
        )
        break

    for val_batch in dataloader(DatasetPath.VALIDATION_PATH):
        inputs, targets = val_batch[:-1], val_batch[-1]
        val_batch = (
            *inputs,
            targets,
        )
        break

    for test_batch in dataloader(DatasetPath.TEST_PATH):
        inputs, targets = test_batch[:-1], test_batch[-1]
        test_batch = (
            *inputs,
            targets,
        )
        break

    plot(trained_models, train_batch, val_batch, test_batch)
