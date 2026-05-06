import torch


class Graph:
    r"""Graph data structure for GNNs.

    Attributes:
        node_features (torch.Tensor): Node feature tensor of shape [N, ...].
        edge_index (torch.LongTensor | None): Edge indices with shape [2, E].
        edge_features (torch.Tensor | None): Edge feature tensor of shape [E, ...].
    """
    def __init__(self, node_features, edge_index, edge_features=None):
        # --- node_features validation ---
        if not isinstance(node_features, torch.Tensor):
            raise TypeError(
                f"node_features must be a torch.Tensor, got {type(node_features).__name__}"
            )

        num_nodes = node_features.size(0)

        # --- edge_index validation ---
        if edge_index is not None:
            if not isinstance(edge_index, torch.Tensor):
                raise TypeError(
                    f"edge_index must be a torch.Tensor or None, "
                    f"got {type(edge_index).__name__}"
                )
            if edge_index.dim() != 2 or edge_index.size(0) != 2:
                raise ValueError(
                    f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}"
                )
            if edge_index.dtype != torch.long:
                raise TypeError(
                    f"edge_index must have dtype torch.long, got {edge_index.dtype}"
                )
            if edge_index.numel() > 0:
                lo, hi = int(edge_index.min()), int(edge_index.max())
                if lo < 0 or hi >= num_nodes:
                    raise ValueError(
                        f"edge_index contains out-of-range indices for {num_nodes} nodes "
                        f"(found min={lo}, max={hi}; valid range [0, {num_nodes - 1}])"
                    )
            if edge_index.device != node_features.device:
                raise ValueError(
                    f"edge_index device ({edge_index.device}) must match "
                    f"node_features device ({node_features.device})"
                )

        # --- edge_features validation ---
        if edge_features is not None:
            if not isinstance(edge_features, torch.Tensor):
                raise TypeError(
                    f"edge_features must be a torch.Tensor or None, "
                    f"got {type(edge_features).__name__}"
                )
            if edge_index is None:
                raise ValueError(
                    "edge_features were provided but edge_index is None"
                )
            if edge_features.size(0) != edge_index.size(1):
                raise ValueError(
                    f"edge_features has {edge_features.size(0)} entries but "
                    f"edge_index has {edge_index.size(1)} edges"
                )
            if edge_features.device != node_features.device:
                raise ValueError(
                    f"edge_features device ({edge_features.device}) must match "
                    f"node_features device ({node_features.device})"
                )

        self.node_features = node_features
        self.edge_index = edge_index
        self.edge_features = edge_features

    def to(self, device):
        """Move all tensors to the specified device."""
        self.node_features = self.node_features.to(device)
        if self.edge_index is not None:
            self.edge_index = self.edge_index.to(device)
        if self.edge_features is not None:
            self.edge_features = self.edge_features.to(device)
        return self


class GraphBatch:
    r"""Batch of Graph objects.

    Concatenates a list of Graphs into a single batched graph, adjusting edge
    indices by the cumulative node count so that indices remain globally unique.

    All graphs in the list must have the same per-node feature shape
    (i.e. ``node_features.shape[1:]`` must be identical for every graph).

    Attributes:
        node_features (torch.Tensor): Batched node features [N_total, ...].
        edge_index (torch.LongTensor | None): Batched edge indices [2, E_total].
        edge_features (torch.Tensor | None): Batched edge features [E_total, ...].
        batch (torch.LongTensor): Graph membership vector [N_total].
    """
    def __init__(self, graphs):
        self.graphs = graphs
        (self.node_features, self.edge_index,
         self.edge_features, self.batch) = self._batch_graphs(graphs)

    def _batch_graphs(self, graphs):
        if not graphs:
            raise ValueError("Cannot create GraphBatch from an empty list of graphs")

        # Verify that all graphs share the same per-node feature shape.
        ref_shape = graphs[0].node_features.shape[1:]
        for i, g in enumerate(graphs):
            actual = g.node_features.shape[1:]
            if actual != ref_shape:
                raise ValueError(
                    f"Cannot batch graph {i} (per-node feature shape {tuple(actual)}) "
                    f"with graph 0 (per-node feature shape {tuple(ref_shape)}). "
                    f"All graphs in a batch must share the same per-node feature shape. "
                    f"Consider resizing or padding node features before batching."
                )

        node_features_list = []
        edge_index_list = []
        edge_features_list = []
        batch_list = []
        node_offset = 0

        for i, g in enumerate(graphs):
            N = g.node_features.size(0)
            node_features_list.append(g.node_features)
            batch_list.append(
                torch.full((N,), i, dtype=torch.long, device=g.node_features.device)
            )
            if g.edge_index is not None:
                edge_index_list.append(g.edge_index + node_offset)
            if g.edge_features is not None:
                edge_features_list.append(g.edge_features)
            node_offset += N

        node_features = torch.cat(node_features_list, dim=0)
        edge_index = torch.cat(edge_index_list, dim=1) if edge_index_list else None
        edge_features = torch.cat(edge_features_list, dim=0) if edge_features_list else None
        batch = torch.cat(batch_list, dim=0)
        return node_features, edge_index, edge_features, batch

    # Keep old name as alias for backward compatibility.
    batch_graphs = _batch_graphs

    def to(self, device):
        """Move all batched tensors to the specified device."""
        self.node_features = self.node_features.to(device)
        if self.edge_index is not None:
            self.edge_index = self.edge_index.to(device)
        if self.edge_features is not None:
            self.edge_features = self.edge_features.to(device)
        self.batch = self.batch.to(device)
        return self
