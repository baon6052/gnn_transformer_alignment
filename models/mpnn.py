from copy import deepcopy
from typing import Any, Callable, List, Optional

import chex
import haiku as hk
import jax
import jax.numpy as jnp

_Array = chex.Array
_Fn = Callable[..., Any]
BIG_NUMBER = 1e6


class MPNNLayer(hk.Module):
    def __init__(
        self,
        nb_layers: int,
        out_size: int,
        mid_size: Optional[int] = None,
        mid_act: Optional[_Fn] = None,
        activation: Optional[_Fn] = jax.nn.relu,
        reduction: _Fn = jnp.mean,
        msgs_mlp_sizes: Optional[List[int]] = None,
        use_ln: bool = False,
        name: str = "mpnn_mp",
        edge_hid_size_1: int = 16,
        edge_hid_size_2: int = 8,
        disable_edge_updates: bool = True,
        dropout_rate: float = 0.0,
        graph_vec: str = "att",
        apply_attention: bool = False,
        number_of_attention_heads: int = 1,
    ):
        super().__init__(name=name)
        self.nb_layers = nb_layers
        if mid_size is None:
            self.mid_size = out_size
        else:
            self.mid_size = mid_size
        self.out_size = out_size
        self.mid_act = mid_act
        self.activation = activation
        self.reduction = reduction
        self._msgs_mlp_sizes = msgs_mlp_sizes
        self.use_ln = use_ln
        self.edge_hidden_size_1 = edge_hid_size_1
        self.edge_hidden_size_2 = edge_hid_size_2
        self.edge_vec_size = mid_size  # =vec_size = out_size
        self.disable_edge_updates = disable_edge_updates
        self.dropout_rate = dropout_rate
        self.graph_vec = graph_vec

        self.apply_attention = apply_attention
        self.number_of_attention_heads = number_of_attention_heads

    def __call__(
        self,
        node_fts: _Array,
        edge_fts: _Array,
        graph_fts: _Array,
        adj_mat: _Array,
        hidden: _Array,
        edge_em: _Array,
    ) -> _Array:
        N = node_fts.shape[1]
        node_tensors = node_fts
        edge_tensors = edge_fts
        graph_tensors = graph_fts

        m_1 = hk.Linear(self.mid_size)
        m_2 = hk.Linear(self.mid_size)
        m_e = hk.Linear(self.mid_size)
        m_g = hk.Linear(self.mid_size)

        o1 = hk.Linear(self.out_size)
        o2 = hk.Linear(self.out_size)
        o3 = hk.Linear(self.out_size)

        msg_1 = m_1(node_tensors)
        msg_2 = m_2(node_tensors)
        msg_e = m_e(edge_tensors)
        msg_g = m_g(graph_tensors)

        msgs = (
            jnp.expand_dims(msg_1, axis=1)
            + jnp.expand_dims(msg_2, axis=2)
            + msg_e
            + jnp.expand_dims(msg_g, axis=(1, 2))
        )

        if self.apply_attention:
            a_1 = hk.Linear(self.number_of_attention_heads)
            a_2 = hk.Linear(self.number_of_attention_heads)
            a_e = hk.Linear(self.number_of_attention_heads)
            a_g = hk.Linear(self.number_of_attention_heads)
            attention_score_layer = hk.Linear(1)

            bias_mat = (adj_mat - 1.0) * 1e9
            bias_mat = jnp.tile(
                bias_mat[..., None], (1, 1, 1, self.number_of_attention_heads)
            )  # [B, N, N, H]
            bias_mat = jnp.transpose(bias_mat, (0, 3, 1, 2))  # [B, H, N, N]

            att_1 = jnp.expand_dims(a_1(node_tensors), axis=-1)
            att_2 = jnp.expand_dims(a_2(node_tensors), axis=-1)
            att_e = a_e(edge_fts)
            att_g = jnp.expand_dims(a_g(graph_fts), axis=-1)

            logits = (
                jnp.transpose(att_1, (0, 2, 1, 3))
                + jnp.transpose(att_2, (0, 2, 3, 1))  # + [B, H, N, 1]
                + jnp.transpose(att_e, (0, 3, 1, 2))  # + [B, H, 1, N]
                + jnp.expand_dims(  # + [B, H, N, N]
                    att_g, axis=-1
                )  # + [B, H, 1, 1]
            )  # = [B, H, N, N]

            # Calculate coefficients and reduce to a single logit
            logits = jax.nn.leaky_relu(logits) + bias_mat  # [B, H. N. N]
            logits = jnp.transpose(logits, (0, 2, 3, 1))  # [B, N, N, H]
            logits = attention_score_layer(logits)  # [B, N, N, 1]
            coefs = jax.nn.softmax(logits, axis=-1)  # [B, N, N, 1]

            msgs = coefs * msgs

        if self._msgs_mlp_sizes is not None:
            msgs = hk.nets.MLP(self._msgs_mlp_sizes)(jax.nn.relu(msgs))

        if self.mid_act is not None:
            msgs = self.mid_act(msgs)

        if self.reduction == jnp.mean:
            msgs = jnp.sum(msgs * jnp.expand_dims(adj_mat, -1), axis=1)
            msgs = msgs / jnp.sum(adj_mat, axis=-1, keepdims=True)
        elif self.reduction == jnp.max:
            maxarg = jnp.where(jnp.expand_dims(adj_mat, -1), msgs, -BIG_NUMBER)
            msgs = jnp.max(maxarg, axis=1)
        else:
            msgs = self.reduction(msgs * jnp.expand_dims(adj_mat, -1), axis=1)

        h_1 = o1(node_tensors)
        h_2 = o2(msgs)

        ret = h_1 + h_2

        if self.activation is not None:
            ret = self.activation(ret)

        if self.use_ln:
            ln = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
            ret = ln(ret)

        if not self.disable_edge_updates:
            source_nodes = jnp.expand_dims(ret, 1)
            expanded_source_nodes = jnp.tile(source_nodes, (1, N, 1, 1))
            target_nodes = jnp.expand_dims(ret, 2)
            expanded_target_nodes = jnp.tile(target_nodes, (1, 1, N, 1))
            reversed_edge_tensors = jnp.swapaxes(msg_e, -2, -3)
            input_tensors = (
                msg_e,
                reversed_edge_tensors,
                expanded_source_nodes,
                expanded_target_nodes,
            )
            if self.graph_vec == "att":
                global_tensors = jnp.expand_dims(graph_tensors, (1, 2))
                expanded_global_tensors = jnp.tile(global_tensors, (1, N, N, 1))
                input_tensors += (expanded_global_tensors,)

            concatenated_inputs = jnp.concatenate(input_tensors, axis=-1)

            EL1 = hk.Linear(self.edge_hidden_size_1)
            EL2 = hk.Linear(self.edge_vec_size)
            ELN1 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)

            EL3 = hk.Linear(self.edge_hidden_size_2)
            EL4 = hk.Linear(self.edge_vec_size)
            ELN2 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)

            residuals = EL2(jax.nn.relu(EL1(concatenated_inputs)))
            # residuals = hk.dropout(hk.next_rng_key(), self.dropout_rate, residuals)
            msg_e = ELN1(msg_e + residuals)

            residuals = EL4(jax.nn.relu(EL3(msg_e)))
            # residuals = hk.dropout(hk.next_rng_key(), self.dropout_rate, residuals)
            msg_e = ELN2(msg_e + residuals)

        msg_e = o3(msg_e)

        return ret, msg_e


class AlignedMPNN(hk.Module):
    def __init__(
        self,
        nb_layers: int,
        out_size: int,
        mid_size: Optional[int] = None,
        mid_act: Optional[_Fn] = None,
        activation: Optional[_Fn] = jax.nn.relu,
        reduction: _Fn = jnp.mean,
        msgs_mlp_sizes: Optional[List[int]] = None,
        use_ln: bool = False,
        add_virtual_node: bool = True,
        disable_edge_updates: bool = True,
        name: str = "mpnn_mp",
        apply_attention: bool = False,
        number_of_attention_heads: int = 1,
    ):
        super().__init__(name=name)
        self.nb_layers = nb_layers
        if mid_size is None:
            self.mid_size = out_size
        else:
            self.mid_size = mid_size
        self.out_size = out_size
        self.mid_act = mid_act
        self.activation = activation
        self.reduction = reduction
        self._msgs_mlp_sizes = msgs_mlp_sizes
        self.use_ln = use_ln
        self.add_virtual_node = add_virtual_node
        self.disable_edge_updates = disable_edge_updates

        self.apply_attention = apply_attention
        self.number_of_attention_heads = number_of_attention_heads

    def __call__(
        self,
        node_fts: _Array,
        edge_fts: _Array,
        graph_fts: _Array,
        adj_mat: _Array,
        hidden: _Array,
        edge_em: _Array,
        num_layers: int = 3,
    ) -> tuple[list[_Array], _Array]:
        node_features_all_layers = []
        edge_features_all_layers = []

        node_tensors = jnp.concatenate([node_fts, hidden], axis=-1)
        edge_tensors = jnp.concatenate([edge_fts, edge_em], axis=-1)
        graph_tensors = graph_fts

        node_enc = hk.Linear(self.out_size)
        edge_enc = hk.Linear(self.out_size)

        node_tensors = node_enc(node_tensors)
        edge_tensors = edge_enc(edge_tensors)

        node_features_all_layers.append(deepcopy(node_tensors))
        edge_features_all_layers.append(deepcopy(edge_tensors))

        if self.add_virtual_node:
            # NODE FEATURES
            # add features of 0
            virtual_node_features = jnp.zeros(
                (node_tensors.shape[0], 1, node_tensors.shape[-1])
            )
            node_tensors = jnp.concatenate(
                [node_tensors, virtual_node_features], axis=1
            )

            # EDGE FEATURES
            # add features of 0
            # column
            virtual_node_edge_features_col = jnp.zeros(
                (
                    edge_tensors.shape[0],
                    edge_tensors.shape[1],
                    1,
                    edge_tensors.shape[-1],
                )
            )
            edge_tensors = jnp.concatenate(
                [edge_tensors, virtual_node_edge_features_col], axis=2
            )

            # Row
            virtual_node_edge_features_row = jnp.zeros(
                (
                    edge_tensors.shape[0],
                    1,
                    edge_tensors.shape[2],
                    edge_tensors.shape[-1],
                )
            )
            edge_tensors = jnp.concatenate(
                [edge_tensors, virtual_node_edge_features_row], axis=1
            )

            # ADJ MATRIX
            # add connection between VN and all other nodes
            virtual_node_adj_mat_row = jnp.ones(
                (adj_mat.shape[0], 1, adj_mat.shape[-1])
            )
            adj_mat = jnp.concatenate(
                [adj_mat, virtual_node_adj_mat_row], axis=1
            )
            virtual_node_adj_mat_col = jnp.ones(
                (adj_mat.shape[0], adj_mat.shape[1], 1)
            )
            adj_mat = jnp.concatenate(
                [adj_mat, virtual_node_adj_mat_col], axis=2
            )

        layers = []

        for _ in range(num_layers):
            layers.append(
                MPNNLayer(
                    nb_layers=self.nb_layers,
                    out_size=self.out_size,
                    mid_size=self.mid_size,
                    activation=self.activation,
                    reduction=self.reduction,
                    use_ln=self.use_ln,
                    disable_edge_updates=self.disable_edge_updates,
                    apply_attention=self.apply_attention,
                    number_of_attention_heads=self.number_of_attention_heads,
                )
            )

        for layer in layers:
            node_tensors, edge_tensors = layer(
                node_tensors,
                edge_tensors,
                graph_tensors,
                adj_mat,
                hidden,
                edge_em,
            )

            if self.add_virtual_node:
                node_features_all_layers.append(
                    deepcopy(node_tensors[:, : node_tensors.shape[1] - 1, :])
                )
                edge_features_all_layers.append(
                    deepcopy(
                        edge_tensors[
                            :,
                            : edge_tensors.shape[1] - 1,
                            : edge_tensors.shape[2] - 1,
                            :,
                        ]
                    )
                )
            else:
                node_features_all_layers.append(deepcopy(node_tensors))
                edge_features_all_layers.append(deepcopy(edge_tensors))

        if self.add_virtual_node:
            return (node_features_all_layers, edge_features_all_layers)

        return node_features_all_layers, edge_features_all_layers
