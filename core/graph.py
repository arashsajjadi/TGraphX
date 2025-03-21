import torch


class Graph:
    r"""Graph data structure for GNNs.

    Attributes:
        node_features (torch.Tensor): Tensor of node features of shape [N, ...].
        edge_index (torch.LongTensor): Tensor of edge indices with shape [2, E].
        edge_features (torch.Tensor, optional): Tensor of edge features of shape [E, ...].
    """
    def __init__(self, node_features, edge_index, edge_features=None):
        self.node_features = node_features
        self.edge_index = edge_index
        self.edge_features = edge_features

    def to(self, device):
        """Move all tensors to the specified device."""
        self.node_features = self.node_features.to(device)
        self.edge_index = self.edge_index.to(device)
        if self.edge_features is not None:
            self.edge_features = self.edge_features.to(device)
        return self


class GraphBatch:
    r"""Batch of Graph objects.

    This class concatenates a list of Graphs into a single (batched) graph,
    adjusting the edge indices and returning a batch vector indicating the
    graph membership for each node.

    Attributes:
        node_features (torch.Tensor): Batched node features.
        edge_index (torch.LongTensor): Batched edge indices.
        edge_features (torch.Tensor or None): Batched edge features.
        batch (torch.LongTensor): For each node, the index of its corresponding graph.
    """
    def __init__(self, graphs):
        self.graphs = graphs
        (self.node_features, self.edge_index,
         self.edge_features, self.batch) = self.batch_graphs(graphs)

    def batch_graphs(self, graphs):
        node_features_list = []
        edge_index_list = []
        edge_features_list = []
        batch = []
        node_offset = 0
        for i, g in enumerate(graphs):
            N = g.node_features.size(0)
            node_features_list.append(g.node_features)
            batch.append(torch.full((N,), i, dtype=torch.long, device=g.node_features.device))
            if g.edge_index is not None:
                edge_index_list.append(g.edge_index + node_offset)
            if g.edge_features is not None:
                edge_features_list.append(g.edge_features)
            node_offset += N
        node_features = torch.cat(node_features_list, dim=0)
        edge_index = torch.cat(edge_index_list, dim=1) if edge_index_list else None
        edge_features = torch.cat(edge_features_list, dim=0) if edge_features_list else None
        batch = torch.cat(batch, dim=0)
        return node_features, edge_index, edge_features, batch

    def to(self, device):
        """Move all batched tensors to the specified device."""
        self.node_features = self.node_features.to(device)
        if self.edge_index is not None:
            self.edge_index = self.edge_index.to(device)
        if self.edge_features is not None:
            self.edge_features = self.edge_features.to(device)
        self.batch = self.batch.to(device)
        return self
