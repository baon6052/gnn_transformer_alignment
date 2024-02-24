import jax.numpy as jnp
import optax


def l2_loss_function(parameters, batch):
    (
        (
            node_fts,
            edge_fts,
            graph_fts,
            adj_mat,
            hidden,
        ),
        transformer_node_embedding,
        transformer_edge_embedding,
    ) = batch

    mpnn_node_embedding, mpnn_edge_embeddings = model.apply(
        parameters, node_fts, edge_fts, graph_fts, adj_mat, hidden
    )
    return jnp.mean(
        optax.l2_loss(mpnn_embedding, transformer_embedding)
    ) + jnp.mean(
        optax.l2_loss(mpnn_edge_embeddings, transformer_edge_embedding)
    )


def l1_loss_function(parameters, batch):
    (
        (
            node_fts,
            edge_fts,
            graph_fts,
            adj_mat,
            hidden,
        ),
        transformer_node_embedding,
        transformer_edge_embedding,
    ) = batch

    mpnn_node_embedding, mpnn_edge_embeddings = model.apply(
        parameters, node_fts, edge_fts, graph_fts, adj_mat, hidden
    )
    return jnp.mean(
        jnp.abs(mpnn_node_embedding - transformer_node_embedding)
    ) + jnp.mean(jnp.abs(mpnn_edge_embeddings - transformer_edge_embedding))
