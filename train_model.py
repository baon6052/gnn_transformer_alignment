from typing import Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import optax

from dataset import data_loader
from models.mpnn import Basic_MPNN

# Assuming you have a dataset ready
# X_train, y_train, X_val, y_val = your_dataset_loader()
dataset_loader = data_loader("dataset")
(
    node_features,
    edge_features,
    graph_features,
    adjacency_matrix,
    hidden_node_features,
), out_node_features = next(dataset_loader)

# Assuming the dataset path is 'dataset', replace with the actual path
dataset_path = "dataset"


def model_fn(node_fts, edge_fts, graph_fts, adj_mat, hidden):
    model = Basic_MPNN(
        nb_layers=3,
        out_size=192,
        mid_size=64,
        activation=jax.nn.relu,
        reduction=jnp.max,
    )
    return model(node_fts, edge_fts, graph_fts, adj_mat, hidden)


# Transform the model function to a Haiku model
model = hk.without_apply_rng(hk.transform(model_fn))


# Example loss function (modify as needed)
def loss_fn(params, batch):
    (node_fts, edge_fts, graph_fts, adj_mat, hidden), targets = batch
    predictions = model.apply(
        params, node_fts, edge_fts, graph_fts, adj_mat, hidden
    )
    return jnp.mean(optax.l2_loss(predictions, targets))


# Initialize parameters
params = model.init(
    jax.random.PRNGKey(42),
    node_fts=node_features,
    edge_fts=edge_features,
    graph_fts=graph_features,
    adj_mat=adjacency_matrix,
    hidden=hidden_node_features,
)

# Define optimizer
optimizer = optax.adam(1e-3)
opt_state = optimizer.init(params)


# Training step
@jax.jit
def train_step(params, opt_state, batch):
    loss, grads = jax.value_and_grad(loss_fn)(params, batch)
    updates, opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, loss


# Training loop
def train(model, params, opt_state, epochs=10):
    dataset_path = "dataset"
    for epoch in range(epochs):
        for batch in data_loader(dataset_path):
            inputs, targets = batch[:-1], batch[-1]
            batch = (
                *inputs,
                targets,
            )  # Combine inputs and targets in the required format
            params, opt_state, loss = train_step(params, opt_state, batch)
        # Validation logic here
        print(f"Epoch {epoch}, Loss: {loss}")


# Example usage
train(model, params, opt_state)
