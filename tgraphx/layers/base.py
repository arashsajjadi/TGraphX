import torch
import torch.nn as nn


class TensorMessagePassingLayer(nn.Module):
    r"""Base class for tensor-based message passing layers.

    Given a graph (node features, edge index, and optionally edge features),
    the layer performs:

        Message:   m_ij = f( node_i, node_j, edge_ij )
        Aggregation: m_i = aggr( { m_ij }_{j in N(i)} )
        Update:    x_i' = g( x_i, m_i )

    Subclasses should override the ``message()`` and ``update()`` methods.
    The default aggregation is implemented using index_add.

    Naming note:
        The public ``forward()`` signature accepts ``edge_features`` (matching
        the ``Graph`` data class).  The abstract ``message()`` method receives
        the same tensor as ``edge_attr``.  Both names refer to identical data;
        ``edge_attr`` is the per-edge name used in the internal message step.
    """

    def __init__(self, in_shape, out_shape, aggr='sum', dropout_prob=0.0, residual=False, use_batchnorm=False):
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.aggr = aggr
        self.dropout_prob = dropout_prob
        self.residual = residual
        self.use_batchnorm = use_batchnorm
        if use_batchnorm:
            ndim = len(out_shape)
            if ndim == 1:
                self.bn = nn.BatchNorm1d(out_shape[0])
            elif ndim == 3:
                self.bn = nn.BatchNorm2d(out_shape[0])
            elif ndim == 4:
                self.bn = nn.BatchNorm3d(out_shape[0])
            else:
                raise ValueError(
                    f"use_batchnorm requires out_shape of 1 (vector), 3 (2-D spatial), "
                    f"or 4 (3-D volumetric) dimensions; got {ndim}."
                )
        self.dropout = nn.Dropout(p=dropout_prob) if dropout_prob > 0 else None


    def message(self, src, dest, edge_attr):
        r"""Compute message for each edge.

        Args:
            src (Tensor): Source node features for each edge.
            dest (Tensor): Destination node features for each edge.
            edge_attr (Tensor or None): Edge features for each edge.

        Returns:
            Tensor: Message for each edge.
        """
        raise NotImplementedError("Subclasses must implement message()")

    def aggregate(self, messages, edge_index, num_nodes):
        r"""Aggregate messages from incoming edges for each node.

        Args:
            messages (Tensor): Messages for each edge, shape [E, ...].
            edge_index (LongTensor): Edge indices with shape [2, E].
            num_nodes (int): Number of nodes.

        Returns:
            Tensor: Aggregated messages for each node.
        """
        target = edge_index[1]  # destination indices for each edge
        aggregated = torch.zeros(num_nodes, *messages.shape[1:], device=messages.device)
        aggregated = aggregated.index_add(0, target, messages)
        if self.aggr == 'mean':
            counts = torch.zeros(num_nodes, device=messages.device)
            ones = torch.ones(messages.shape[0], device=messages.device)
            counts = counts.index_add(0, target, ones)
            # Reshape to broadcast over all trailing feature dimensions.
            view_shape = (num_nodes,) + (1,) * (aggregated.dim() - 1)
            counts = counts.view(view_shape).clamp(min=1)
            aggregated = aggregated / counts
        elif self.aggr == 'max':
            raise NotImplementedError(
                "aggr='max' is not implemented in TensorMessagePassingLayer. "
                "Use aggr='sum' or aggr='mean', or subclass and override aggregate()."
            )
        return aggregated

    def update(self, node_feature, aggregated_message):
        out = aggregated_message
        if self.use_batchnorm:
            out = self.bn(out)
        if self.dropout:
            out = self.dropout(out)
        if self.residual:
            # Only add skip connection if dimensions match
            if node_feature.shape == out.shape:
                out = node_feature + out
        return out

    def forward(self, node_features, edge_index, edge_features=None):
        src_idx = edge_index[0]
        dest_idx = edge_index[1]
        src_features = node_features[src_idx]
        dest_features = node_features[dest_idx]
        messages = self.message(src_features, dest_features, edge_features)
        num_nodes = node_features.size(0)
        aggregated = self.aggregate(messages, edge_index, num_nodes)
        updated_nodes = self.update(node_features, aggregated)
        return updated_nodes


class LinearMessagePassing(TensorMessagePassingLayer):
    r"""Message passing layer based on linear transformations (suitable for vector inputs).

    This layer concatenates the source and destination node features (and optionally
    edge features) then passes the result through a learnable linear module.
    """

    def __init__(self, in_shape, out_shape, aggr='sum', use_edge_features=False,
                 dropout_prob=0.0, residual=False, use_batchnorm=False):
        super().__init__(in_shape, out_shape, aggr, dropout_prob=dropout_prob, residual=residual, use_batchnorm=use_batchnorm)
        self.use_edge_features = use_edge_features
        in_dim = in_shape[0]
        linear_in_dim = in_dim * 3 if use_edge_features else in_dim * 2
        self.message_linear = nn.Linear(linear_in_dim, out_shape[0])

    def message(self, src, dest, edge_attr):
        if self.use_edge_features and edge_attr is not None:
            msg_input = torch.cat([src, dest, edge_attr], dim=-1)
        else:
            msg_input = torch.cat([src, dest], dim=-1)
        return self.message_linear(msg_input)

    def update(self, node_feature, aggregated_message):
        # In this simple example, we directly return the aggregated message.
        return aggregated_message
