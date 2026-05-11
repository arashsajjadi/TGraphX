"""Graph and GraphBatch data structures.

These are the user-facing containers for tensor-aware GNN inputs. They keep
multi-dimensional node and edge features intact (e.g. ``[N, C, H, W]`` for
nodes, ``[E, C_e, H, W]`` for edges) and offer a uniform device/dtype API
on top of clear, eager input validation.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from .graph_utils import (
    add_self_loops as _add_self_loops,
    coalesce_edges as _coalesce_edges,
    is_undirected as _is_undirected,
    make_undirected as _make_undirected,
    remove_self_loops as _remove_self_loops,
    validate_edge_features,
    validate_edge_index,
    validate_edge_weight,
)


# --------------------------------------------------------------------------- #
# Internal helpers                                                             #
# --------------------------------------------------------------------------- #

def _check_label_tensor(
    name: str,
    tensor: torch.Tensor,
    expected_first_dim: int,
    device: torch.device,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(
            f"{name} must be a torch.Tensor or None, got {type(tensor).__name__}"
        )
    if tensor.size(0) != expected_first_dim:
        raise ValueError(
            f"{name} has {tensor.size(0)} entries but expected {expected_first_dim}"
        )
    if tensor.device != device:
        raise ValueError(
            f"{name} device ({tensor.device}) must match "
            f"node_features device ({device})"
        )


def _move(
    tensor: Optional[torch.Tensor],
    *,
    device: Optional[torch.device],
    dtype: Optional[torch.dtype],
    allow_dtype: bool = True,
) -> Optional[torch.Tensor]:
    """Move ``tensor`` to ``device``/``dtype`` if provided.

    ``allow_dtype=False`` skips the dtype change (used for index/label tensors
    whose dtype must not be coerced to a float).
    """
    if tensor is None:
        return None
    kwargs: Dict[str, Any] = {}
    if device is not None:
        kwargs["device"] = device
    if dtype is not None and allow_dtype and tensor.is_floating_point():
        kwargs["dtype"] = dtype
    if not kwargs:
        return tensor
    return tensor.to(**kwargs)


# --------------------------------------------------------------------------- #
# Graph                                                                        #
# --------------------------------------------------------------------------- #

class Graph:
    r"""A single graph with tensor-shaped node and edge features.

    Args:
        node_features (Tensor): ``[N, ...]`` tensor. Common shapes are
            ``[N, D]`` (vector features), ``[N, C, H, W]`` (image-like) and
            ``[N, C, D, H, W]`` (volumetric — storage only, layers may not
            consume it).
        edge_index (LongTensor, optional): ``[2, E]`` source/destination indices.
        edge_weight (Tensor, optional): ``[E]`` per-edge scalar weight.
        edge_features (Tensor, optional): ``[E, ...]`` per-edge feature tensor
            (vector ``[E, D_e]``, image-like ``[E, C_e, H, W]``, or volumetric
            ``[E, C_e, D, H, W]``).
        edge_attr (Tensor, optional): Alias for ``edge_features``.  If both are
            provided, they must be the same tensor; providing both raises an error.
        node_labels (Tensor, optional): ``[N, ...]`` per-node labels.
        y (Tensor, optional): Alias for ``node_labels`` (PyG-style).  If both
            ``y`` and ``node_labels`` are provided they must be identical.
        labels (Tensor, optional): Alias for ``node_labels``.
        edge_labels (Tensor, optional): ``[E, ...]`` per-edge labels.
        graph_label (Tensor, optional): graph-level **target label** tensor used for
            graph classification / regression tasks.  Stored in ``self.graph_label``.
        graph_features (Tensor, optional): graph-level **input feature** tensor (e.g.
            a global context vector for the whole graph).  Stored in
            ``self.graph_features``.  This is *distinct* from ``graph_label``; do not
            confuse graph-level input features with graph-level target labels.
        train_mask (Tensor, optional): ``BoolTensor[N]`` training mask stored
            in ``metadata['masks']['train']``.
        val_mask (Tensor, optional): ``BoolTensor[N]`` validation mask.
        test_mask (Tensor, optional): ``BoolTensor[N]`` test mask.
        metadata (dict, optional): arbitrary user data carried alongside the
            graph. Preserved verbatim by ``GraphBatch``.

    The constructor validates every input eagerly and raises ``TypeError`` /
    ``ValueError`` with descriptive messages on mismatched shape, dtype, or
    device. No silent coercion is performed.

    PyG/DGL users:
        ``graph.x`` → ``node_features``
        ``graph.y`` → ``node_labels``
        ``graph.edge_attr`` → ``edge_features``
        ``graph.num_node_features`` → ``node_features.shape[1:]`` item count
    """

    def __init__(
        self,
        node_features: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
        edge_weight: Optional[torch.Tensor] = None,
        edge_features: Optional[torch.Tensor] = None,
        node_labels: Optional[torch.Tensor] = None,
        edge_labels: Optional[torch.Tensor] = None,
        graph_label: Optional[torch.Tensor] = None,
        metadata: Optional[Dict[str, Any]] = None,
        *,
        x: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        graph_features: Optional[torch.Tensor] = None,
        train_mask: Optional[torch.Tensor] = None,
        val_mask: Optional[torch.Tensor] = None,
        test_mask: Optional[torch.Tensor] = None,
    ) -> None:
        # --- alias resolution (v1.4.0+: x=, edge_attr=, y=, labels=) ---
        # x → node_features
        if x is not None:
            if node_features is not None and x is not node_features:
                raise ValueError(
                    "Provide node_features or x, not both. "
                    "They are aliases for the same field."
                )
            node_features = x
        if node_features is None:
            raise TypeError(
                "Graph requires `node_features` (or PyG-style alias `x`)."
            )
        # edge_attr → edge_features
        if edge_attr is not None:
            if edge_features is not None and edge_attr is not edge_features:
                raise ValueError(
                    "Provide edge_features or edge_attr, not both. "
                    "They are aliases for the same field."
                )
            edge_features = edge_attr

        # y / labels → node_labels
        label_sources = [(v, n) for v, n in [(y, "y"), (labels, "labels"), (node_labels, "node_labels")] if v is not None]
        if len(label_sources) > 1:
            # Check they all point to the same tensor; if not, error.
            vals = [v for v, _ in label_sources]
            if not all(v is vals[0] for v in vals[1:]):
                names = [n for _, n in label_sources]
                raise ValueError(
                    f"Provide at most one of y / labels / node_labels. "
                    f"Got multiple: {names}."
                )
        if label_sources:
            node_labels = label_sources[0][0]

        # graph_features is a separate graph-level INPUT feature field.
        # It is stored in self.graph_features, NOT aliased to graph_label.
        # graph_label stores the graph-level TARGET (for graph classification).
        if graph_features is not None:
            if not isinstance(graph_features, torch.Tensor):
                raise TypeError(
                    f"graph_features must be a torch.Tensor or None, "
                    f"got {type(graph_features).__name__}"
                )
        # --- node_features ---
        if not isinstance(node_features, torch.Tensor):
            raise TypeError(
                f"node_features must be a torch.Tensor, "
                f"got {type(node_features).__name__}"
            )
        if node_features.dim() < 2:
            raise ValueError(
                f"node_features must have at least 2 dimensions [N, ...] "
                f"(vector or spatial per-node feature), "
                f"got shape {tuple(node_features.shape)}"
            )

        device = node_features.device
        num_nodes = node_features.size(0)

        # --- edge_index ---
        edge_index = validate_edge_index(edge_index, num_nodes, device=device)
        num_edges = 0 if edge_index is None else edge_index.size(1)

        # --- edge_weight ---
        if edge_weight is not None and edge_index is None:
            raise ValueError("edge_weight was provided but edge_index is None")
        edge_weight = validate_edge_weight(edge_weight, num_edges, device=device)

        # --- edge_features ---
        if edge_features is not None and edge_index is None:
            raise ValueError("edge_features were provided but edge_index is None")
        edge_features = validate_edge_features(edge_features, num_edges, device=device)

        # --- node_labels ---
        if node_labels is not None:
            _check_label_tensor("node_labels", node_labels, num_nodes, device)

        # --- edge_labels ---
        if edge_labels is not None:
            if edge_index is None:
                raise ValueError("edge_labels were provided but edge_index is None")
            _check_label_tensor("edge_labels", edge_labels, num_edges, device)

        # --- graph_label ---
        if graph_label is not None:
            if not isinstance(graph_label, torch.Tensor):
                raise TypeError(
                    f"graph_label must be a torch.Tensor or None, "
                    f"got {type(graph_label).__name__}"
                )
            if graph_label.device != device:
                raise ValueError(
                    f"graph_label device ({graph_label.device}) must match "
                    f"node_features device ({device})"
                )

        # --- graph_features device check ---
        if graph_features is not None and graph_features.device != device:
            raise ValueError(
                f"graph_features device ({graph_features.device}) must match "
                f"node_features device ({device})"
            )

        # --- metadata ---
        if metadata is not None and not isinstance(metadata, dict):
            raise TypeError(
                f"metadata must be a dict or None, got {type(metadata).__name__}"
            )

        # Store masks inside metadata so they travel with the graph.
        if train_mask is not None or val_mask is not None or test_mask is not None:
            metadata = dict(metadata) if metadata else {}
            masks: Dict[str, torch.Tensor] = {}
            for mask_tensor, mask_name in [
                (train_mask, "train"),
                (val_mask, "val"),
                (test_mask, "test"),
            ]:
                if mask_tensor is not None:
                    if not isinstance(mask_tensor, torch.Tensor):
                        raise TypeError(
                            f"{mask_name}_mask must be a torch.Tensor or None, "
                            f"got {type(mask_tensor).__name__}"
                        )
                    if mask_tensor.size(0) != num_nodes:
                        raise ValueError(
                            f"{mask_name}_mask has {mask_tensor.size(0)} entries "
                            f"but graph has {num_nodes} nodes."
                        )
                    masks[mask_name] = mask_tensor.to(device)
            metadata["masks"] = masks

        self.node_features = node_features
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.edge_features = edge_features
        self.node_labels = node_labels
        self.edge_labels = edge_labels
        self.graph_label = graph_label
        self.graph_features = graph_features  # graph-level INPUT features (not the label)
        self.metadata = metadata

    # ----- properties ----------------------------------------------------- #

    @property
    def num_nodes(self) -> int:
        return self.node_features.size(0)

    @property
    def num_edges(self) -> int:
        return 0 if self.edge_index is None else self.edge_index.size(1)

    @property
    def feature_shape(self) -> Tuple[int, ...]:
        return tuple(self.node_features.shape[1:])

    @property
    def edge_feature_shape(self) -> Optional[Tuple[int, ...]]:
        if self.edge_features is None:
            return None
        return tuple(self.edge_features.shape[1:])

    @property
    def has_edges(self) -> bool:
        return self.edge_index is not None and self.edge_index.size(1) > 0

    @property
    def has_edge_weight(self) -> bool:
        return self.edge_weight is not None

    @property
    def has_edge_features(self) -> bool:
        return self.edge_features is not None

    @property
    def device(self) -> torch.device:
        return self.node_features.device

    @property
    def dtype(self) -> torch.dtype:
        return self.node_features.dtype

    # ----- PyG-compatible aliases ---------------------------------------- #

    @property
    def x(self) -> torch.Tensor:
        """Alias for ``node_features`` (PyG/DGL style)."""
        return self.node_features

    @x.setter
    def x(self, value: torch.Tensor) -> None:
        self.node_features = value

    @property
    def y(self) -> Optional[torch.Tensor]:
        """Alias for ``node_labels`` (PyG/DGL style)."""
        return self.node_labels

    @y.setter
    def y(self, value: Optional[torch.Tensor]) -> None:
        self.node_labels = value

    @property
    def labels(self) -> Optional[torch.Tensor]:
        """Alias for ``node_labels``."""
        return self.node_labels

    @labels.setter
    def labels(self, value: Optional[torch.Tensor]) -> None:
        self.node_labels = value

    @property
    def edge_attr(self) -> Optional[torch.Tensor]:
        """Alias for ``edge_features`` (PyG/DGL style)."""
        return self.edge_features

    @edge_attr.setter
    def edge_attr(self, value: Optional[torch.Tensor]) -> None:
        self.edge_features = value

    @property
    def num_node_features(self) -> int:
        """Total number of scalar node features (product of per-node tensor dims)."""
        shape = self.node_features.shape[1:]
        result = 1
        for s in shape:
            result *= s
        return result

    @property
    def num_classes(self) -> Optional[int]:
        """Inferred number of classes from integer ``node_labels``, or ``None``."""
        if self.node_labels is None:
            return None
        if not (self.node_labels.dtype in (torch.long, torch.int, torch.int8, torch.int16, torch.int32)):
            return None
        if self.node_labels.dim() != 1:
            return None
        return int(self.node_labels.max().item()) + 1

    # ----- NetworkX-style read-only methods (v1.4.0+) -------------------- #

    def number_of_nodes(self) -> int:
        """NetworkX-style alias for :attr:`num_nodes`."""
        return self.num_nodes

    def number_of_edges(self) -> int:
        """NetworkX-style alias for :attr:`num_edges`."""
        return self.num_edges

    def summary(self) -> Dict[str, Any]:
        """JSON-serializable summary of this graph (v1.4.0+).

        Equivalent to :func:`tgraphx.describe`. Returns shapes, dtypes, devices,
        mask counts, and label info.
        """
        from ..ux.describe import describe as _describe
        return _describe(self)

    @property
    def train_mask(self) -> Optional[torch.Tensor]:
        """Training mask stored in ``metadata['masks']['train']``."""
        if isinstance(self.metadata, dict):
            return self.metadata.get("masks", {}).get("train")
        return None

    @property
    def val_mask(self) -> Optional[torch.Tensor]:
        """Validation mask stored in ``metadata['masks']['val']``."""
        if isinstance(self.metadata, dict):
            return self.metadata.get("masks", {}).get("val")
        return None

    @property
    def test_mask(self) -> Optional[torch.Tensor]:
        """Test mask stored in ``metadata['masks']['test']``."""
        if isinstance(self.metadata, dict):
            return self.metadata.get("masks", {}).get("test")
        return None

    # ----- label helpers ------------------------------------------------- #

    def has_labels(self) -> bool:
        """Return ``True`` if this graph has per-node labels (``node_labels`` / ``y``)."""
        return self.node_labels is not None

    def get_labels(self) -> torch.Tensor:
        """Return ``node_labels``, raising a helpful error if absent.

        Raises:
            ValueError: If ``node_labels`` is ``None``.  The error message
                tells the user how to add labels.
        """
        if self.node_labels is None:
            raise ValueError(
                "Graph labels are missing.  Create the graph with "
                "Graph(..., y=labels) or assign graph.y = labels.\n"
                "See docs/graph_basics.md#labels for more information."
            )
        return self.node_labels

    def with_labels(self, labels: torch.Tensor) -> "Graph":
        """Return a shallow copy with ``node_labels`` set to ``labels``.

        The original graph is not modified.
        """
        import copy as _copy
        g = _copy.copy(self)
        g.node_labels = labels
        return g

    # ----- copy / device / dtype ----------------------------------------- #

    def clone(self) -> "Graph":
        """Deep-copy: every tensor is cloned and metadata is deep-copied."""
        g = Graph(
            node_features=self.node_features.clone(),
            edge_index=self.edge_index.clone() if self.edge_index is not None else None,
            edge_weight=self.edge_weight.clone() if self.edge_weight is not None else None,
            edge_features=self.edge_features.clone() if self.edge_features is not None else None,
            node_labels=self.node_labels.clone() if self.node_labels is not None else None,
            edge_labels=self.edge_labels.clone() if self.edge_labels is not None else None,
            graph_label=self.graph_label.clone() if self.graph_label is not None else None,
            metadata=copy.deepcopy(self.metadata) if self.metadata is not None else None,
        )
        if self.graph_features is not None:
            g.graph_features = self.graph_features.clone()
        return g

    def to(
        self,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "Graph":
        """Move all tensor fields. ``dtype`` only applies to floating tensors."""
        if isinstance(device, str):
            device = torch.device(device)
        self.node_features = _move(self.node_features, device=device, dtype=dtype)
        self.edge_index = _move(self.edge_index, device=device, dtype=dtype, allow_dtype=False)
        self.edge_weight = _move(self.edge_weight, device=device, dtype=dtype)
        self.edge_features = _move(self.edge_features, device=device, dtype=dtype)
        self.node_labels = _move(self.node_labels, device=device, dtype=dtype, allow_dtype=False)
        self.edge_labels = _move(self.edge_labels, device=device, dtype=dtype, allow_dtype=False)
        self.graph_label = _move(self.graph_label, device=device, dtype=dtype, allow_dtype=False)
        self.graph_features = _move(self.graph_features, device=device, dtype=dtype)
        return self

    def cpu(self) -> "Graph":
        return self.to(device=torch.device("cpu"))

    def cuda(self, device: Optional[Union[int, str, torch.device]] = None) -> "Graph":
        if device is None:
            return self.to(device=torch.device("cuda"))
        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        return self.to(device=device)

    # ----- validation re-run --------------------------------------------- #

    def validate(self) -> "Graph":
        """Re-run input validation against the current state.

        Useful after manual mutation of attributes. Raises with a descriptive
        message on the first inconsistency found.
        """
        if not isinstance(self.node_features, torch.Tensor):
            raise TypeError("node_features is no longer a torch.Tensor")
        if self.node_features.dim() < 2:
            raise ValueError(
                f"node_features must have at least 2 dimensions, "
                f"got shape {tuple(self.node_features.shape)}"
            )
        device = self.node_features.device
        validate_edge_index(self.edge_index, self.num_nodes, device=device)
        validate_edge_weight(self.edge_weight, self.num_edges, device=device)
        validate_edge_features(self.edge_features, self.num_edges, device=device)
        if self.node_labels is not None:
            _check_label_tensor("node_labels", self.node_labels, self.num_nodes, device)
        if self.edge_labels is not None:
            _check_label_tensor("edge_labels", self.edge_labels, self.num_edges, device)
        if self.graph_label is not None and self.graph_label.device != device:
            raise ValueError(
                f"graph_label device ({self.graph_label.device}) must match "
                f"node_features device ({device})"
            )
        if self.graph_features is not None and self.graph_features.device != device:
            raise ValueError(
                f"graph_features device ({self.graph_features.device}) must match "
                f"node_features device ({device})"
            )
        return self

    # ----- topology ------------------------------------------------------ #

    def is_undirected(self) -> bool:
        return _is_undirected(self.edge_index, num_nodes=self.num_nodes)

    def is_directed(self) -> bool:
        return not self.is_undirected()

    def add_self_loops(self, fill_value: float = 1.0) -> "Graph":
        """Add ``i->i`` self-loops in place (skipping nodes that already have one).

        Raises if ``edge_labels`` is set, since label assignment for new
        loops is ambiguous — drop or detach ``edge_labels`` first.
        """
        if self.edge_labels is not None:
            raise ValueError(
                "Graph.add_self_loops() is unsupported when edge_labels is set. "
                "Clear edge_labels (set to None) and re-attach them after the "
                "operation if needed."
            )
        new_ei, new_w, new_ef = _add_self_loops(
            self.edge_index,
            self.edge_weight,
            self.edge_features,
            num_nodes=self.num_nodes,
            fill_value=fill_value,
            device=self.device,
        )
        self.edge_index = new_ei
        self.edge_weight = new_w
        self.edge_features = new_ef
        return self

    def remove_self_loops(self) -> "Graph":
        """Remove self-loops in place; per-edge tensors are filtered by the same mask."""
        new_ei, new_w, new_ef, new_el = _remove_self_loops(
            self.edge_index,
            self.edge_weight,
            self.edge_features,
            self.edge_labels,
        )
        self.edge_index = new_ei
        self.edge_weight = new_w
        self.edge_features = new_ef
        self.edge_labels = new_el
        return self

    def make_undirected(self, reduce: str = "mean") -> "Graph":
        """Symmetrize edges in place. Coalesces duplicates with ``reduce``.

        Raises if ``edge_labels`` is set — symmetric duplicates would have
        ambiguous labels under arithmetic reduction.
        """
        if self.edge_labels is not None:
            raise ValueError(
                "Graph.make_undirected() is unsupported when edge_labels is set "
                "because reverse-edge label assignment is ambiguous. "
                "Clear edge_labels (set to None) and re-attach them after the "
                "operation if needed."
            )
        new_ei, new_w, new_ef = _make_undirected(
            self.edge_index,
            self.edge_weight,
            self.edge_features,
            num_nodes=self.num_nodes,
            reduce=reduce,
        )
        self.edge_index = new_ei
        self.edge_weight = new_w
        self.edge_features = new_ef
        return self

    def coalesce(self, reduce: str = "mean") -> "Graph":
        """Sort edges by ``(src, dst)`` and merge duplicates in place."""
        new_ei, new_w, new_ef = _coalesce_edges(
            self.edge_index,
            self.edge_weight,
            self.edge_features,
            num_nodes=self.num_nodes,
            reduce=reduce,
        )
        self.edge_index = new_ei
        self.edge_weight = new_w
        self.edge_features = new_ef
        return self

    # ----- classmethod constructors (v1.4.0+) ---------------------------- #

    @classmethod
    def from_edges(
        cls,
        edge_list: Any,
        num_nodes: Optional[int] = None,
        node_features: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> "Graph":
        """Build a Graph from a Python edge list, list of tuples, or [E, 2] tensor.

        Args:
            edge_list: One of:
              - list of (src, dst) tuples / lists,
              - numpy array of shape [E, 2],
              - torch.Tensor of shape [E, 2] or [2, E].
            num_nodes: Required if ``node_features`` is None. Inferred from max ID otherwise.
            node_features: Optional `[N, ...]` node features. If None, identity features
              of shape `[N, 1]` are created (zero-initialized).
            **kwargs: Forwarded to :class:`Graph` constructor.

        Returns:
            A new :class:`Graph`.
        """
        # Normalize to [2, E] LongTensor
        if isinstance(edge_list, torch.Tensor):
            if edge_list.dim() != 2 or edge_list.shape[0] not in (2,) and edge_list.shape[1] not in (2,):
                raise ValueError(
                    f"edge_list tensor must have shape [E, 2] or [2, E]; got {tuple(edge_list.shape)}"
                )
            ei = edge_list if edge_list.shape[0] == 2 else edge_list.t().contiguous()
            ei = ei.long()
        else:
            try:
                pairs = list(edge_list)
                if len(pairs) == 0:
                    ei = torch.zeros(2, 0, dtype=torch.long)
                else:
                    arr = torch.as_tensor(pairs, dtype=torch.long)
                    if arr.dim() != 2 or arr.size(1) != 2:
                        raise ValueError(
                            f"edge_list must be iterable of (src, dst) pairs; "
                            f"got tensor shape {tuple(arr.shape)}"
                        )
                    ei = arr.t().contiguous()
            except Exception as exc:
                raise ValueError(
                    f"Could not parse edge_list: {type(exc).__name__}: {exc}. "
                    "Pass a list of (src, dst) tuples or a [E, 2] / [2, E] tensor."
                ) from exc

        if node_features is None:
            if num_nodes is None:
                if ei.numel() == 0:
                    raise ValueError(
                        "Graph.from_edges: provide num_nodes or node_features "
                        "when edge_list is empty."
                    )
                num_nodes = int(ei.max().item()) + 1
            node_features = torch.zeros(num_nodes, 1)
        elif num_nodes is None:
            num_nodes = int(node_features.size(0))
        return cls(node_features=node_features, edge_index=ei, **kwargs)

    @classmethod
    def from_adjacency(
        cls,
        adj: Any,
        node_features: Optional[torch.Tensor] = None,
        directed: bool = True,
        **kwargs: Any,
    ) -> "Graph":
        """Build a Graph from a dense or sparse adjacency.

        Args:
            adj: One of:
              - torch.Tensor of shape [N, N] (dense),
              - scipy.sparse matrix [N, N] (if scipy installed).
            node_features: Optional `[N, ...]` node features.
            directed: If False and adj is non-symmetric, raise a warning.
            **kwargs: Forwarded to :class:`Graph`.
        """
        if isinstance(adj, torch.Tensor):
            if adj.dim() != 2 or adj.size(0) != adj.size(1):
                raise ValueError(
                    f"Dense adjacency must be square [N, N]; got {tuple(adj.shape)}"
                )
            N = adj.size(0)
            src, dst = torch.nonzero(adj, as_tuple=True)
            ei = torch.stack([src, dst], dim=0).long()
        else:
            # Try scipy sparse
            try:
                import scipy.sparse as _sp
            except ImportError as exc:
                raise ImportError(
                    "Sparse adjacency requires scipy. Install with: pip install scipy"
                ) from exc
            if not _sp.issparse(adj):
                raise TypeError(
                    f"Graph.from_adjacency expects torch.Tensor or scipy.sparse; "
                    f"got {type(adj).__name__}"
                )
            coo = adj.tocoo()
            N = coo.shape[0]
            ei = torch.tensor([coo.row.tolist(), coo.col.tolist()], dtype=torch.long)
        if node_features is None:
            node_features = torch.zeros(N, 1)
        return cls(node_features=node_features, edge_index=ei, **kwargs)

    @classmethod
    def from_networkx(
        cls,
        G: Any,
        node_feature_key: Optional[str] = None,
        **kwargs: Any,
    ) -> "Graph":
        """Build a Graph from a NetworkX graph.

        Node attributes named ``node_feature_key`` are stacked as node features
        when provided. Edge attributes are NOT carried by default; use
        ``edge_features=`` after construction if needed.
        """
        try:
            import networkx as nx
        except ImportError as exc:
            raise ImportError(
                "Graph.from_networkx requires networkx. Install with: pip install networkx"
            ) from exc
        if not isinstance(G, (nx.Graph, nx.DiGraph)):
            raise TypeError(
                f"Graph.from_networkx expects a networkx.Graph/DiGraph; "
                f"got {type(G).__name__}"
            )
        # Relabel nodes to 0..N-1
        mapping = {node: i for i, node in enumerate(G.nodes())}
        N = len(mapping)
        # Edges
        if G.number_of_edges() > 0:
            src = [mapping[u] for u, v in G.edges()]
            dst = [mapping[v] for u, v in G.edges()]
            ei = torch.tensor([src, dst], dtype=torch.long)
            if not isinstance(G, nx.DiGraph):
                # Make undirected explicit
                ei = torch.cat([ei, ei.flip(0)], dim=1)
        else:
            ei = torch.zeros(2, 0, dtype=torch.long)
        # Node features
        if node_feature_key is not None:
            feats = []
            for node in G.nodes():
                v = G.nodes[node].get(node_feature_key)
                if v is None:
                    raise ValueError(
                        f"Node {node!r} missing attribute {node_feature_key!r}"
                    )
                feats.append(v)
            node_features = torch.as_tensor(feats).float()
            if node_features.dim() == 1:
                node_features = node_features.unsqueeze(-1)
        else:
            node_features = torch.zeros(N, 1)
        return cls(node_features=node_features, edge_index=ei, **kwargs)

    def to_networkx(self, directed: bool = True) -> Any:
        """Convert this Graph to a NetworkX graph (optional dependency)."""
        try:
            import networkx as nx
        except ImportError as exc:
            raise ImportError(
                "Graph.to_networkx requires networkx. Install with: pip install networkx"
            ) from exc
        G = nx.DiGraph() if directed else nx.Graph()
        G.add_nodes_from(range(self.num_nodes))
        if self.edge_index is not None and self.edge_index.numel() > 0:
            edges = self.edge_index.t().tolist()
            G.add_edges_from(edges)
        return G

    def save(self, path: Any) -> str:
        """Save this graph to a `.tgx` native bundle (preserves tensor features)."""
        from ..ux.serialization import save_tgraphx
        return save_tgraphx(self, path)

    @classmethod
    def load(cls, path: Any) -> "Graph":
        """Load a Graph from a `.tgx` native bundle."""
        from ..ux.serialization import load_tgraphx
        obj = load_tgraphx(path)
        if not isinstance(obj, cls):
            raise TypeError(
                f"File {path} contains a {type(obj).__name__}, not a Graph."
            )
        return obj

    # ----- repr ---------------------------------------------------------- #

    def __repr__(self) -> str:
        parts = [
            f"num_nodes={self.num_nodes}",
            f"num_edges={self.num_edges}",
            f"node_features_shape={tuple(self.node_features.shape)}",
        ]
        if self.has_edge_weight:
            parts.append("edge_weight=True")
        if self.has_edge_features:
            parts.append(f"edge_feature_shape={self.edge_feature_shape}")
        if self.node_labels is not None:
            parts.append(f"y_shape={tuple(self.node_labels.shape)}")
        if self.edge_labels is not None:
            parts.append("edge_labels=True")
        if self.graph_label is not None:
            parts.append("graph_label=True")
        if self.graph_features is not None:
            parts.append(f"graph_features_shape={tuple(self.graph_features.shape)}")
        parts.append(f"device={self.device}")
        return f"Graph({', '.join(parts)})"


# --------------------------------------------------------------------------- #
# GraphBatch                                                                   #
# --------------------------------------------------------------------------- #

class GraphBatch:
    r"""Batch of :class:`Graph` objects concatenated into a single super-graph.

    All graphs in the batch must share the same per-node feature shape
    (``node_features.shape[1:]``). If any graph has an optional per-edge or
    per-node tensor (``edge_weight``, ``edge_features``, ``edge_labels``,
    ``node_labels``, ``graph_label``), every graph that has edges (or nodes,
    respectively) must provide that tensor with a matching trailing shape —
    otherwise a ``ValueError`` is raised. Mixing-some-with-none is rejected
    rather than silently dropped, because silently dropping per-edge data
    is a footgun.

    Attributes:
        graphs: the list of source graphs (kept for traceability).
        node_features: ``[N_total, ...]``.
        edge_index: ``[2, E_total]`` with per-graph index offsets applied.
        edge_weight: ``[E_total]`` or ``None``.
        edge_features: ``[E_total, ...]`` or ``None``.
        node_labels: ``[N_total, ...]`` or ``None``.
        edge_labels: ``[E_total, ...]`` or ``None``.
        graph_labels: ``[B, ...]`` stacked from per-graph ``graph_label`` or ``None``.
        metadata: ``list[Any]`` of length ``B``, one entry per graph (the original
            ``Graph.metadata``, possibly ``None``).
        batch: ``[N_total]`` LongTensor mapping each node to its graph index.
    """

    def __init__(self, graphs: List[Graph]) -> None:
        if not graphs:
            raise ValueError("Cannot create GraphBatch from an empty list of graphs")
        self._validate_compatibility(graphs)
        self.graphs = graphs
        (
            self.node_features,
            self.edge_index,
            self.edge_weight,
            self.edge_features,
            self.node_labels,
            self.edge_labels,
            self.graph_labels,
            self.metadata,
            self.batch,
        ) = self._batch_graphs(graphs)

    # ----- validation ---------------------------------------------------- #

    @staticmethod
    def _validate_compatibility(graphs: List[Graph]) -> None:
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

        ef_ref: Optional[Tuple[int, ...]] = None
        for g in graphs:
            if g.edge_features is not None:
                ef_ref = tuple(g.edge_features.shape[1:])
                break
        if ef_ref is not None:
            for i, g in enumerate(graphs):
                if g.edge_features is not None and tuple(g.edge_features.shape[1:]) != ef_ref:
                    raise ValueError(
                        f"Cannot batch graph {i} (per-edge feature shape "
                        f"{tuple(g.edge_features.shape[1:])}) with reference "
                        f"per-edge feature shape {ef_ref}."
                    )

    # ----- batching ------------------------------------------------------ #

    def _batch_graphs(
        self, graphs: List[Graph]
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        List[Any],
        torch.Tensor,
    ]:
        needs_w = any(g.edge_weight is not None for g in graphs)
        needs_ef = any(g.edge_features is not None for g in graphs)
        needs_el = any(g.edge_labels is not None for g in graphs)
        needs_nl = any(g.node_labels is not None for g in graphs)
        needs_gl = any(g.graph_label is not None for g in graphs)

        node_features_list: List[torch.Tensor] = []
        edge_index_list: List[torch.Tensor] = []
        edge_weight_list: List[torch.Tensor] = []
        edge_features_list: List[torch.Tensor] = []
        node_labels_list: List[torch.Tensor] = []
        edge_labels_list: List[torch.Tensor] = []
        graph_labels_list: List[torch.Tensor] = []
        metadata_list: List[Any] = []
        batch_list: List[torch.Tensor] = []
        node_offset = 0

        for i, g in enumerate(graphs):
            N = g.node_features.size(0)
            node_features_list.append(g.node_features)
            batch_list.append(
                torch.full((N,), i, dtype=torch.long, device=g.node_features.device)
            )

            if needs_nl:
                if g.node_labels is None:
                    raise ValueError(
                        f"Graph {i} is missing node_labels but other graphs in "
                        f"the batch have them. Provide node_labels on every "
                        f"graph or none."
                    )
                node_labels_list.append(g.node_labels)

            if needs_gl:
                if g.graph_label is None:
                    raise ValueError(
                        f"Graph {i} is missing graph_label but other graphs in "
                        f"the batch have one. Provide graph_label on every "
                        f"graph or none."
                    )
                graph_labels_list.append(g.graph_label)

            metadata_list.append(g.metadata)

            if g.edge_index is None:
                if needs_w or needs_ef or needs_el:
                    # A graph with no edges contributes nothing per-edge — no
                    # check required against per-edge optional fields.
                    pass
                node_offset += N
                continue

            edge_index_list.append(g.edge_index + node_offset)
            E_g = g.edge_index.size(1)

            if needs_w:
                if g.edge_weight is None:
                    raise ValueError(
                        f"Graph {i} has edges but no edge_weight, while other "
                        f"graphs in the batch provide edge_weight. Provide "
                        f"edge_weight on every graph with edges or none."
                    )
                edge_weight_list.append(g.edge_weight)
            if needs_ef:
                if g.edge_features is None:
                    raise ValueError(
                        f"Graph {i} has edges but no edge_features, while other "
                        f"graphs in the batch provide edge_features."
                    )
                edge_features_list.append(g.edge_features)
            if needs_el:
                if g.edge_labels is None:
                    raise ValueError(
                        f"Graph {i} has edges but no edge_labels, while other "
                        f"graphs in the batch provide edge_labels."
                    )
                edge_labels_list.append(g.edge_labels)

            # Consistency: optional per-edge tensors must match E_g (caught
            # already by Graph.__init__, but assert here for defensive batching).
            if g.edge_weight is not None and g.edge_weight.size(0) != E_g:
                raise ValueError(
                    f"Graph {i}: edge_weight length {g.edge_weight.size(0)} != "
                    f"num_edges {E_g}"
                )

            node_offset += N

        node_features = torch.cat(node_features_list, dim=0)
        edge_index = torch.cat(edge_index_list, dim=1) if edge_index_list else None
        edge_weight = torch.cat(edge_weight_list, dim=0) if edge_weight_list else None
        edge_features = torch.cat(edge_features_list, dim=0) if edge_features_list else None
        edge_labels = torch.cat(edge_labels_list, dim=0) if edge_labels_list else None
        node_labels = torch.cat(node_labels_list, dim=0) if node_labels_list else None

        graph_labels: Optional[torch.Tensor] = None
        if graph_labels_list:
            try:
                graph_labels = torch.stack(graph_labels_list, dim=0)
            except RuntimeError as e:
                raise ValueError(
                    "Failed to stack graph_label across the batch — every graph "
                    f"must have the same graph_label shape. Underlying error: {e}"
                ) from e

        batch = torch.cat(batch_list, dim=0)
        return (
            node_features,
            edge_index,
            edge_weight,
            edge_features,
            node_labels,
            edge_labels,
            graph_labels,
            metadata_list,
            batch,
        )

    # Keep old name as alias for backward compatibility.
    batch_graphs = _batch_graphs

    # ----- properties ---------------------------------------------------- #

    @property
    def num_graphs(self) -> int:
        return len(self.graphs)

    @property
    def num_nodes(self) -> int:
        return self.node_features.size(0)

    @property
    def num_edges(self) -> int:
        return 0 if self.edge_index is None else self.edge_index.size(1)

    @property
    def feature_shape(self) -> Tuple[int, ...]:
        return tuple(self.node_features.shape[1:])

    @property
    def edge_feature_shape(self) -> Optional[Tuple[int, ...]]:
        if self.edge_features is None:
            return None
        return tuple(self.edge_features.shape[1:])

    @property
    def has_edges(self) -> bool:
        return self.edge_index is not None and self.edge_index.size(1) > 0

    @property
    def has_edge_weight(self) -> bool:
        return self.edge_weight is not None

    @property
    def has_edge_features(self) -> bool:
        return self.edge_features is not None

    @property
    def device(self) -> torch.device:
        return self.node_features.device

    # ----- device / dtype ------------------------------------------------ #

    def to(
        self,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "GraphBatch":
        if isinstance(device, str):
            device = torch.device(device)
        self.node_features = _move(self.node_features, device=device, dtype=dtype)
        self.edge_index = _move(self.edge_index, device=device, dtype=dtype, allow_dtype=False)
        self.edge_weight = _move(self.edge_weight, device=device, dtype=dtype)
        self.edge_features = _move(self.edge_features, device=device, dtype=dtype)
        self.node_labels = _move(self.node_labels, device=device, dtype=dtype, allow_dtype=False)
        self.edge_labels = _move(self.edge_labels, device=device, dtype=dtype, allow_dtype=False)
        self.graph_labels = _move(self.graph_labels, device=device, dtype=dtype, allow_dtype=False)
        self.batch = _move(self.batch, device=device, dtype=dtype, allow_dtype=False)
        return self

    def cpu(self) -> "GraphBatch":
        return self.to(device=torch.device("cpu"))

    def cuda(self, device: Optional[Union[int, str, torch.device]] = None) -> "GraphBatch":
        if device is None:
            return self.to(device=torch.device("cuda"))
        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        return self.to(device=device)

    # ----- repr ---------------------------------------------------------- #

    def __repr__(self) -> str:
        parts = [
            f"num_graphs={self.num_graphs}",
            f"num_nodes={self.num_nodes}",
            f"num_edges={self.num_edges}",
            f"feature_shape={self.feature_shape}",
        ]
        if self.has_edge_weight:
            parts.append("edge_weight=True")
        if self.has_edge_features:
            parts.append(f"edge_feature_shape={self.edge_feature_shape}")
        return f"GraphBatch({', '.join(parts)})"
