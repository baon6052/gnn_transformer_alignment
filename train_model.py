from enum import StrEnum, auto

import haiku as hk
import jax
import jax.numpy as jnp
import optax

import wandb
from checkpointer import Checkpointer
from dataset import DatasetPath, dataloader
from models.mpnn import AlignedMPNN
from pathlib import Path

run = wandb.init(project="gnn_alignment", entity="monoids")
MODEL_DIR = Path(Path.cwd(), "trained_models")
MODEL_DIR.mkdir(exist_ok=True, parents=True)


class ValidationMode(StrEnum):
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
    ),
    transformer_node_features_all_layers,
    transformer_edge_embedding,
) = next(train_dataloader)


def model_fn(node_fts, edge_fts, graph_fts, adj_mat, hidden):
    model = AlignedMPNN(
        nb_layers=3,
        out_size=192,
        mid_size=192,
        activation=None,
        reduction=jnp.max,
        num_layers=3,
    )
    return model(node_fts, edge_fts, graph_fts, adj_mat, hidden)


model = hk.without_apply_rng(hk.transform(model_fn))


parameters = model.init(
    jax.random.PRNGKey(42),
    node_fts=input_node_features,
    edge_fts=input_edge_features,
    graph_fts=input_graph_features,
    adj_mat=input_adjacency_matrix,
    hidden=input_hidden_node_features,
)

optimizer = optax.adam(0.001)
optimizer_state = optimizer.init(parameters)


def l2_loss_function(parameters, batch):
    (
        (
            input_node_fts,
            input_edge_fts,
            input_graph_fts,
            input_adj_mat,
            input_hidden,
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
    )

    loss = jnp.mean(optax.l2_loss(mpnn_edge_embeddings, transformer_edge_embedding))

    # for mpnn_node_embedding, transformer_node_embedding in zip(
    #     mpnn_node_features_all_layers,
    #     transformer_node_features_all_layers,
    #     strict=True,
    # ):
    #     loss += jnp.mean(optax.l2_loss(mpnn_node_embedding, transformer_node_embedding))

    loss += jnp.mean(  # NB +
        optax.l2_loss(
            mpnn_node_features_all_layers[-1], transformer_node_features_all_layers[-1]
        )
    )

    return loss


@jax.jit
def train_step(parameters, optimizer_state, batch):
    loss, grads = jax.value_and_grad(l2_loss_function)(parameters, batch)
    updates, optimizer_state = optimizer.update(grads, optimizer_state)
    new_parameters = optax.apply_updates(parameters, updates)
    return new_parameters, optimizer_state, loss


def train_model(parameters, optimizer_state, epochs=50):
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
        print(f"Epoch {epoch}, Loss: {train_loss.item()}")

        validation_loss = validate_model(parameters, ValidationMode.VALIDATION)

        wandb.log(
            {
                "train_loss": train_loss.item(),
                "validation_loss": validation_loss.item(),
            }
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            checkpointer = Checkpointer(f"{MODEL_DIR}/mpnn.pkl")
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


if __name__ == "__main__":
    parameters = train_model(parameters, optimizer_state)
    test_loss = validate_model(parameters, ValidationMode.TEST)

    wandb.log({"test_loss": test_loss})
