from enum import Enum, auto
from pathlib import Path

import click
import haiku as hk
import jax
import jax.numpy as jnp
import optax

import wandb
from checkpointer import Checkpointer
from dataset import DatasetPath, dataloader
from models.att_mpnn import AttMPNN
from models.mpnn import AlignedMPNN
from models.qkv_att_mpnn import QKVMPNN

MODEL_DIR = Path(Path.cwd(), "trained_models")
MODEL_DIR.mkdir(exist_ok=True, parents=True)


class ValidationMode(Enum):
    VALIDATION = auto()
    TEST = auto()


train_dataloader = dataloader(DatasetPath.TRAIN_PATH)
(
    (
        input_node_features,
        input_edge_features,
        input_graph_features,
        input_adjacency_matrix,
        input_hidden_node_features,
        input_hidden_edge_features,
    ),
    transformer_node_features_all_layers,
    transformer_edge_embedding,
) = next(train_dataloader)

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
):
    model = AttMPNN(
        nb_layers=3,
        out_size=192,
        mid_size=mid_size,
        activation=None,
        reduction=reduction_func,
        use_ln=use_layer_norm,
        add_virtual_node=add_virtual_node,
    )

    return model(node_fts, edge_fts, graph_fts, adj_mat, hidden, edge_em)


def l2_loss_function(parameters, batch):
    (
        (
            input_node_fts,
            input_edge_fts,
            input_graph_fts,
            input_adj_mat,
            input_hidden,
            input_edge_em,
        ),
        transformer_node_features_all_layers,
        transformer_edge_embedding,
    ) = batch

    mpnn_node_features_all_layers, mpnn_edge_embeddings = model.apply(
        parameters,
        input_node_fts,
        input_edge_fts,
        input_graph_fts,
        input_adj_mat,
        input_hidden,
        input_edge_em,
    )

    loss = jnp.mean(
        optax.l2_loss(mpnn_edge_embeddings, transformer_edge_embedding)
    )

    for mpnn_node_embedding, transformer_node_embedding in zip(
        mpnn_node_features_all_layers,
        transformer_node_features_all_layers,
        strict=True,
    ):
        loss += jnp.mean(
            optax.l2_loss(mpnn_node_embedding, transformer_node_embedding)
        )

    return loss


@jax.jit
def train_step(parameters, optimizer_state, batch):
    loss, grads = jax.value_and_grad(l2_loss_function)(parameters, batch)
    updates, optimizer_state = optimizer.update(grads, optimizer_state)
    new_parameters = optax.apply_updates(parameters, updates)
    return new_parameters, optimizer_state, loss


def train_model(
    parameters, optimizer_state, use_wandb, checkpointer, epochs=25
):
    best_validation_loss = float("inf")

    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0

        for batch in dataloader(DatasetPath.TRAIN_PATH):
            inputs, targets = batch[:-1], batch[-1]
            batch = (
                *inputs,
                targets,
            )
            parameters, optimizer_state, loss = train_step(
                parameters, optimizer_state, batch
            )
            total_loss += loss
            num_batches += 1

        train_loss = total_loss / num_batches

        validation_loss = validate_model(parameters, ValidationMode.VALIDATION)

        print(f"Epoch {epoch}, Loss: {train_loss.item()}")

        if use_wandb:
            wandb.log(
                {
                    "train_loss": train_loss,
                    "validation_loss": validation_loss,
                }
            )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            checkpointer.save(parameters)

    return parameters


def validate_model(parameters, mode: ValidationMode):
    loss = 0
    num_batches = 0

    dataset_path = (
        DatasetPath.VALIDATION_PATH
        if mode == ValidationMode.VALIDATION
        else DatasetPath.TEST_PATH
    )

    for batch in dataloader(dataset_path):
        inputs, targets = batch[:-1], batch[-1]
        batch = (
            *inputs,
            targets,
        )
        loss += l2_loss_function(parameters, batch)
        num_batches += 1

    average_loss = loss / num_batches
    print(f"{mode} Loss: {average_loss}")

    return average_loss


@click.command()
@click.option("--model_save_name", type=str, default=None)
@click.option("--use_layer_norm", type=bool, default=True)
@click.option("--mid_dim", type=int, default=192)
@click.option("--add_virtual_node", type=bool, default=True)
@click.option("--reduction", type=str, default="max")
@click.option("--use_wandb", type=bool, default=False)
def main(
    model_save_name: str | None,
    use_layer_norm: bool,
    mid_dim: int,
    add_virtual_node: bool,
    reduction: str,
    use_wandb: bool,
) -> None:
    global model, parameters, optimizer, optimizer_state

    if model_save_name is None:
        model_save_name = f"vn_{add_virtual_node}_ln_{use_layer_norm}_mid_dim_{mid_dim}_reduction_{reduction}"

    reduction_func = jnp.mean

    if reduction == "sum":
        reduction_func = jnp.sum
    else:
        reduction_func = jnp.max

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
        )

    model = hk.without_apply_rng(hk.transform(model_wrapper))

    parameters = model.init(
        jax.random.PRNGKey(42),
        node_fts=input_node_features,
        edge_fts=input_edge_features,
        graph_fts=input_graph_features,
        adj_mat=input_adjacency_matrix,
        hidden=input_hidden_node_features,
        edge_em=input_hidden_edge_features,
    )

    optimizer = optax.adam(0.001)
    optimizer_state = optimizer.init(parameters)

    if use_wandb:
        run = wandb.init(
            project="gnn_alignment",
            entity="monoids",
            name=model_save_name,
            group="experiment_1",
        )

    checkpointer = Checkpointer(f"{MODEL_DIR}/{model_save_name}.pkl")
    parameters = train_model(
        parameters,
        optimizer_state,
        use_wandb=use_wandb,
        checkpointer=checkpointer,
    )

    parameters = checkpointer.load()
    test_loss = validate_model(parameters, ValidationMode.TEST)

    if use_wandb:
        wandb.log({"test_loss": test_loss})


if __name__ == "__main__":
    main()
