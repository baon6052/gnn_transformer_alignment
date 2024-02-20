from enum import StrEnum, auto

import haiku as hk
import jax
import jax.numpy as jnp
import optax

from checkpointer import Checkpointer
from dataset import DatasetPath, dataloader
from models.mpnn import Basic_MPNN


class ValidationMode(StrEnum):
    VALIDATION = auto()
    TEST = auto()


train_dataloader = dataloader(DatasetPath.TRAIN_PATH)
(
    node_features,
    edge_features,
    graph_features,
    adjacency_matrix,
    hidden_node_features,
), out_node_features = next(train_dataloader)


def model_fn(node_fts, edge_fts, graph_fts, adj_mat, hidden):
    model = Basic_MPNN(
        nb_layers=3,
        out_size=192,
        mid_size=64,
        activation=jax.nn.relu,
        reduction=jnp.max,
    )
    return model(node_fts, edge_fts, graph_fts, adj_mat, hidden)


model = hk.without_apply_rng(hk.transform(model_fn))


parameters = model.init(
    jax.random.PRNGKey(42),
    node_fts=node_features,
    edge_fts=edge_features,
    graph_fts=graph_features,
    adj_mat=adjacency_matrix,
    hidden=hidden_node_features,
)

optimizer = optax.adam(0.001)
optimizer_state = optimizer.init(parameters)


def loss_function(parameters, batch):
    (
        node_fts,
        edge_fts,
        graph_fts,
        adj_mat,
        hidden,
    ), transformer_embedding = batch
    mpnn_embedding = model.apply(
        parameters, node_fts, edge_fts, graph_fts, adj_mat, hidden
    )
    return jnp.mean(optax.l2_loss(mpnn_embedding, transformer_embedding))


@jax.jit
def train_step(parameters, optimizer_state, batch):
    loss, grads = jax.value_and_grad(loss_function)(parameters, batch)
    updates, optimizer_state = optimizer.update(grads, optimizer_state)
    new_parameters = optax.apply_updates(parameters, updates)
    return new_parameters, optimizer_state, loss


def train_model(parameters, optimizer_state, epochs=50):
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

        loss = total_loss / num_batches
        print(f"Epoch {epoch}, Loss: {loss}")

        validate_model(parameters, ValidationMode.VALIDATION)

    checkpointer = Checkpointer("./trained_models/mpnn.pkl")
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
        loss += loss_function(parameters, batch)
        num_batches += 1

    average_loss = loss / num_batches
    print(f"{mode} Loss: {average_loss}")


if __name__ == "__main__":
    parameters = train_model(parameters, optimizer_state)
    validate_model(parameters, ValidationMode.TEST)
