import math
from copy import deepcopy
from typing import Any, Callable, List, Optional

import chex
import haiku as hk
import jax
import jax.numpy as jnp

_Array = chex.Array
_Fn = Callable[..., Any]
BIG_NUMBER = 1e6


class QKVMPNNLayer(hk.Module):
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
        number_of_attention_heads: int = 1,
        node_hid_size=32,
        edge_hid_size_1=16,
        edge_hid_size_2=8,
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

        self.node_feature_size = out_size
        self.number_of_attention_heads = number_of_attention_heads
        self.out_attention_head_size = self.node_feature_size // self.number_of_attention_heads
        self.node_hid_size = node_hid_size
        self.edge_hid_size_1 = edge_hid_size_1
        self.edge_hid_size_2 = edge_hid_size_2

        self.scale = 1.0 / math.sqrt(self.out_attention_head_size)

    def separate_node_heads(self, x):
        new_shape = x.shape[:-1] + (self.number_of_attention_heads, self.out_attention_head_size)
        x = jnp.reshape(x, new_shape)
        return jnp.transpose(x, (0, 2, 1, 3))

    def separate_edge_heads(self, x):
        new_shape = x.shape[:-1] + (self.number_of_attention_heads, self.out_attention_head_size)
        x = jnp.reshape(x, new_shape)
        return jnp.transpose(x, (0, 3, 1, 2, 4))

    def separate_graph_heads(self, x):
        x = jnp.expand_dims(x, -2)
        new_shape = x.shape[:-1] + (self.number_of_attention_heads, self.out_attention_head_size)
        x = jnp.reshape(x, new_shape)
        return jnp.transpose(x, (0, 2, 1, 3))

    def concatenate_heads(self, x):
        x = jnp.transpose(x, (0, 2, 1, 3))
        new_shape = x.shape[:-2] + (self.out_attention_head_size,)
        return jnp.reshape(x, new_shape)

    def apply_attention(self, node_tensors, edge_tensors, graph_tensors):
        Wnq = hk.Linear(self.out_size)
        Wnk = hk.Linear(self.out_size)
        Wnv = hk.Linear(self.out_size)

        Weq = hk.Linear(self.out_size)
        Wek = hk.Linear(self.out_size)
        Wev = hk.Linear(self.out_size)

        Wgq = hk.Linear(self.out_size)
        Wgk = hk.Linear(self.out_size)
        Wgv = hk.Linear(self.out_size)

        B = node_tensors.shape[0]
        N = node_tensors.shape[1]

        eQ = Weq(edge_tensors)
        eK = Wek(edge_tensors)
        eV = Wev(edge_tensors)

        nQ = Wnq(node_tensors)
        nK = Wnk(node_tensors)
        nV = Wnv(node_tensors)

        gQ = Wgq(graph_tensors)
        gK = Wgk(graph_tensors)
        gV = Wgv(graph_tensors)

        eQ = self.separate_edge_heads(eQ)
        eK = self.separate_edge_heads(eK)
        eV = self.separate_edge_heads(eV)

        nQ = self.separate_node_heads(nQ)
        nK = self.separate_node_heads(nK)
        nV = self.separate_node_heads(nV)

        gQ = self.separate_graph_heads(gQ)
        gK = self.separate_graph_heads(gK)
        gV = self.separate_graph_heads(gV)

        Q = (
                eQ
                + jnp.reshape(nQ, (B, self.number_of_attention_heads, N, 1, self.out_attention_head_size))
                + jnp.reshape(gQ, (B, self.number_of_attention_heads, 1, 1, self.out_attention_head_size))
        )
        K = (
                eK
                + jnp.reshape(nK, (B, self.number_of_attention_heads, 1, N, self.out_attention_head_size))
                + jnp.reshape(gK, (B, self.number_of_attention_heads, 1, 1, self.out_attention_head_size))
        )

        Q = jnp.reshape(Q, (B, self.number_of_attention_heads, N, N, 1, self.out_attention_head_size))
        K = jnp.reshape(K, (B, self.number_of_attention_heads, N, N, self.out_attention_head_size, 1))
        QK = jnp.matmul(Q, K)
        QK = jnp.reshape(QK, (B, self.number_of_attention_heads, N, N))

        QK = QK * self.scale
        att_dist = jax.nn.softmax(QK, axis=-1)

        att_dist = jnp.reshape(att_dist, (B, self.number_of_attention_heads, N, 1, N))

        v2 = (
                eV
                + jnp.reshape(nV, (B, self.number_of_attention_heads, 1, N, self.out_attention_head_size))
                + jnp.reshape(gV, (B, self.number_of_attention_heads, 1, 1, self.out_attention_head_size))
        )

        new_nodes = jnp.matmul(att_dist, v2)
        new_nodes = jnp.reshape(new_nodes, (B, self.number_of_attention_heads, N, self.out_attention_head_size))

        return self.concatenate_heads(new_nodes)


    def __call__(
        self,
        node_fts: _Array,
        edge_fts: _Array,
        graph_fts: _Array,
        adj_mat: _Array,
        hidden: _Array,
        edge_em: _Array,
    ) -> _Array:
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

        attention_linear_layer = hk.Linear(self.mid_size)

        attended_node_tensors = self.apply_attention(node_tensors, edge_tensors, graph_tensors)
        attended_node_tensors = attention_linear_layer(attended_node_tensors)

        msg_1 = m_1(attended_node_tensors)
        msg_2 = m_2(attended_node_tensors)
        msg_e = m_e(edge_tensors)
        msg_g = m_g(graph_tensors)

        msgs = (
            jnp.expand_dims(msg_1, axis=1)
            + jnp.expand_dims(msg_2, axis=2)
            + msg_e
            + jnp.expand_dims(msg_g, axis=(1, 2))
        )

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

        h_e = o3(msg_e)

        ret = h_1 + h_2

        if self.activation is not None:
            ret = self.activation(ret)

        if self.use_ln:
            ln = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
            ret = ln(ret)

        return ret, h_e


class QKVMPNN(hk.Module):
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
        name: str = "mpnn_mp",
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
        node_tensors = jnp.concatenate([node_fts, hidden], axis=-1)
        edge_tensors = jnp.concatenate([edge_fts, edge_em], axis=-1)
        graph_tensors = graph_fts

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
        node_features_all_layers = []

        for _ in range(num_layers):
            layers.append(
                QKVMPNNLayer(
                    nb_layers=self.nb_layers,
                    out_size=self.out_size,
                    mid_size=self.mid_size,
                    activation=self.activation,
                    reduction=self.reduction,
                    use_ln=self.use_ln,
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
            else:
                node_features_all_layers.append(deepcopy(node_tensors))

        if self.add_virtual_node:
            return (
                node_features_all_layers,
                edge_tensors[
                    :,
                    : edge_tensors.shape[1] - 1,
                    : edge_tensors.shape[2] - 1,
                    :,
                ],
            )

        return node_features_all_layers, edge_tensors
