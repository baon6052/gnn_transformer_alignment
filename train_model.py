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
from models.mpnn import AlignedMPNN

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
        transformer_edge_features_all_layers,
    ) = batch

    mpnn_node_features_all_layers, mpnn_edge_features_all_layers = model.apply(
        parameters,
        input_node_fts,
        input_edge_fts,
        input_graph_fts,
        input_adj_mat,
        input_hidden,
        input_edge_em,
    )

    loss = 0.0

    for mpnn_node_embedding, transformer_node_embedding in zip(
        mpnn_node_features_all_layers,
        transformer_node_features_all_layers[1:],
        strict=True,
    ):
        loss += jnp.mean(optax.l2_loss(mpnn_node_embedding, transformer_node_embedding))

    for mpnn_edge_embedding, transformer_edge_embedding in zip(
        mpnn_edge_features_all_layers,
        transformer_edge_features_all_layers[1:],
        strict=True,
    ):
        loss += jnp.mean(optax.l2_loss(mpnn_edge_embedding, transformer_edge_embedding))

    return loss


@jax.jit
def train_step(parameters, optimizer_state, batch):
    loss, grads = jax.value_and_grad(l2_loss_function)(parameters, batch)
    updates, optimizer_state = optimizer.update(grads, optimizer_state)
    new_parameters = optax.apply_updates(parameters, updates)
    return new_parameters, optimizer_state, loss


def train_model(parameters, optimizer_state, use_wandb, checkpointer, epochs=25):
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
@click.option("--disable_edge_updates", type=bool, default=True)
@click.option("--apply_attention", type=bool, default=False)
@click.option("--number_of_attention_heads", type=int, default=1)
@click.option("--use_wandb", type=bool, default=True)
def main(
    model_save_name: str | None,
    use_layer_norm: bool,
    mid_dim: int,
    add_virtual_node: bool,
    reduction: str,
    disable_edge_updates: bool,
    apply_attention: bool,
    number_of_attention_heads: int,
    use_wandb: bool,
) -> None:
    global model, parameters, optimizer, optimizer_state

    if model_save_name is None:
        if apply_attention:
            model_save_name = f"vn-{add_virtual_node}-ln-{use_layer_norm}-mid_dim-{mid_dim}-reduction-{reduction}-disable_edge_updates-{disable_edge_updates}-apply_attention-{apply_attention}-number_of_attention_heads-{number_of_attention_heads}"
        else:
            model_save_name = f"vn-{add_virtual_node}-ln-{use_layer_norm}-mid_dim-{mid_dim}-reduction-{reduction}-disable_edge_updates-{disable_edge_updates}-apply_attention-{apply_attention}"
    model_save_name = "test"
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
            disable_edge_updates=disable_edge_updates,
            apply_attention=apply_attention,
            number_of_attention_heads=number_of_attention_heads,
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
            group="experiment_2",
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
