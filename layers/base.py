import torch
import torch.nn as nn


class TensorMessagePassingLayer(nn.Module):
    r"""Base class for tensor-based message passing layers.

    Given a graph (node features, edge index, and optionally edge features),
    the layer performs:

        Message:   m_ij = f( node_i, node_j, edge_ij )
        Aggregation: m_i = aggr( { m_ij }_{j in N(i)} )
        Update:    x_i' = g( x_i, m_i )

    Subclasses should override the `message()` and `update()` methods.
    The default aggregation is implemented using index_add.
    """

    def __init__(self, in_shape, out_shape, aggr='sum'):
        super().__init__()
        self.in_shape = in_shape  # e.g., (C,) or (C, H, W) etc.
        self.out_shape = out_shape  # similar structure as in_shape but possibly different channels
        self.aggr = aggr

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
            # Compute counts per node and take mean
            counts = torch.zeros(num_nodes, device=messages.device)
            ones = torch.ones(messages.shape[0], device=messages.device)
            counts = counts.index_add(0, target, ones)
            counts = counts.unsqueeze(-1).clamp(min=1)
            aggregated = aggregated / counts
        # For 'max', a more complex (and less vectorized) implementation is required.
        return aggregated

    def update(self, node_feature, aggregated_message):
        r"""Update node features given the aggregated messages.

        Default implementation simply returns the aggregated message.
        Subclasses can override this (e.g., adding residual connections).

        Args:
            node_feature (Tensor): Original node features.
            aggregated_message (Tensor): Aggregated message for each node.

        Returns:
            Tensor: Updated node features.
        """
        return aggregated_message

    def forward(self, node_features, edge_index, edge_features=None):
        r"""Forward pass for message passing.

        Args:
            node_features (Tensor): Node features of shape [N, ...].
            edge_index (LongTensor): Edge indices of shape [2, E].
            edge_features (Tensor, optional): Edge features of shape [E, ...].

        Returns:
            Tensor: Updated node features.
        """
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

    def __init__(self, in_shape, out_shape, aggr='sum', use_edge_features=False):
        super().__init__(in_shape, out_shape, aggr)
        self.use_edge_features = use_edge_features
        # in_shape is expected to be a tuple like (C,)
        in_dim = in_shape[0]
        if self.use_edge_features:
            linear_in_dim = in_dim * 3  # source, destination, edge features
        else:
            linear_in_dim = in_dim * 2  # source and destination only
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
