"""Tests for tgraphx.graph_builders and patch helpers."""
import pytest
import torch

from tgraphx.graph_builders import (
    build_fully_connected_graph,
    build_grid_graph,
    build_grid_graph_3d,
    build_iou_graph,
    build_knn_graph,
    build_radius_graph,
    build_random_graph,
    image_to_patches,
    patch_grid_shape,
    volume_patch_grid_shape,
    volume_to_patches,
)
from tgraphx.layers.conv_message import ConvMessagePassing
from tgraphx.layers.gat import TensorGATLayer
from tgraphx.layers.gin import TensorGINLayer
from tgraphx.layers.sage import TensorGraphSAGELayer


# =========================================================================== #
# Helpers                                                                       #
# =========================================================================== #

def _has_no_duplicate_self_loops(edge_index: torch.Tensor) -> bool:
    """Return True iff each node appears at most once as a self-loop."""
    mask = edge_index[0] == edge_index[1]
    self_src = edge_index[0][mask]
    return self_src.numel() == torch.unique(self_src).numel()


def _count_self_loops(edge_index: torch.Tensor) -> int:
    return int((edge_index[0] == edge_index[1]).sum())


def _edge_set(edge_index: torch.Tensor):
    """Return a Python set of (src, dst) tuples."""
    return set(zip(edge_index[0].tolist(), edge_index[1].tolist()))


# =========================================================================== #
# Grid graph 2-D                                                                #
# =========================================================================== #

class TestBuildGridGraph:

    def test_edge_count_undirected_no_self_loops(self):
        # rows=3, cols=3: h_pairs=6, v_pairs=6 → undirected=24
        ei = build_grid_graph(3, 3, directed=False, self_loops=False)
        assert ei.shape[0] == 2
        assert ei.shape[1] == 24

    def test_edge_count_directed_no_self_loops(self):
        ei = build_grid_graph(3, 3, directed=True, self_loops=False)
        assert ei.shape[1] == 12

    def test_edge_count_undirected_with_self_loops(self):
        ei = build_grid_graph(3, 3, directed=False, self_loops=True)
        assert ei.shape[1] == 24 + 9

    def test_edge_count_directed_with_self_loops(self):
        ei = build_grid_graph(3, 3, directed=True, self_loops=True)
        assert ei.shape[1] == 12 + 9

    def test_edge_count_2x2(self):
        # h_pairs=2, v_pairs=2 → undirected=8, +4 self-loops = 12
        ei = build_grid_graph(2, 2, directed=False, self_loops=True)
        assert ei.shape[1] == 12

    def test_undirected_contains_reciprocal_edges(self):
        ei = build_grid_graph(3, 3, directed=False, self_loops=False)
        edges = _edge_set(ei)
        # Every (u, v) must have (v, u)
        for u, v in edges:
            assert (v, u) in edges

    def test_directed_fewer_than_undirected(self):
        d = build_grid_graph(4, 4, directed=True, self_loops=False)
        u = build_grid_graph(4, 4, directed=False, self_loops=False)
        assert d.shape[1] * 2 == u.shape[1]

    def test_no_duplicate_self_loops_undirected(self):
        ei = build_grid_graph(3, 3, directed=False, self_loops=True)
        assert _has_no_duplicate_self_loops(ei)

    def test_no_duplicate_self_loops_directed(self):
        ei = build_grid_graph(3, 3, directed=True, self_loops=True)
        assert _has_no_duplicate_self_loops(ei)

    def test_no_self_loops_when_disabled(self):
        ei = build_grid_graph(3, 3, directed=False, self_loops=False)
        assert _count_self_loops(ei) == 0

    def test_dtype_and_shape(self):
        ei = build_grid_graph(2, 3)
        assert ei.dtype == torch.long
        assert ei.dim() == 2
        assert ei.shape[0] == 2

    def test_single_node(self):
        ei = build_grid_graph(1, 1, self_loops=True)
        assert ei.shape[1] == 1  # one self-loop

    def test_single_node_no_self_loops(self):
        ei = build_grid_graph(1, 1, self_loops=False)
        assert ei.shape[1] == 0

    def test_invalid_rows(self):
        with pytest.raises(ValueError):
            build_grid_graph(0, 3)

    def test_device(self):
        ei = build_grid_graph(2, 2, device=torch.device("cpu"))
        assert ei.device.type == "cpu"


# =========================================================================== #
# Grid graph 3-D                                                                #
# =========================================================================== #

class TestBuildGridGraph3d:

    def test_edge_count_directed_no_self_loops(self):
        # depth=2, rows=2, cols=2:
        # dep_pairs=4, row_pairs=4, col_pairs=4 → directed=12
        ei = build_grid_graph_3d(2, 2, 2, directed=True, self_loops=False)
        assert ei.shape[1] == 12

    def test_edge_count_undirected_no_self_loops(self):
        ei = build_grid_graph_3d(2, 2, 2, directed=False, self_loops=False)
        assert ei.shape[1] == 24

    def test_edge_count_with_self_loops(self):
        ei = build_grid_graph_3d(2, 2, 2, directed=False, self_loops=True)
        assert ei.shape[1] == 24 + 8

    def test_undirected_reciprocal(self):
        ei = build_grid_graph_3d(2, 3, 3, directed=False, self_loops=False)
        edges = _edge_set(ei)
        for u, v in edges:
            assert (v, u) in edges

    def test_no_duplicate_self_loops(self):
        ei = build_grid_graph_3d(2, 2, 2, directed=False, self_loops=True)
        assert _has_no_duplicate_self_loops(ei)

    def test_dtype(self):
        ei = build_grid_graph_3d(2, 2, 2)
        assert ei.dtype == torch.long

    def test_invalid_depth(self):
        with pytest.raises(ValueError):
            build_grid_graph_3d(0, 2, 2)

    def test_3x3x3_directed_count(self):
        # dep=(2*9)=18, row=(3*2*3)=18, col=(3*3*2)=18 → directed=54
        ei = build_grid_graph_3d(3, 3, 3, directed=True, self_loops=False)
        dep = 2 * 3 * 3
        row = 3 * 2 * 3
        col = 3 * 3 * 2
        assert ei.shape[1] == dep + row + col


# =========================================================================== #
# Directed vs undirected (shared behaviour)                                     #
# =========================================================================== #

class TestDirectedVsUndirected:

    def test_grid_directed_no_reciprocal(self):
        ei = build_grid_graph(3, 3, directed=True, self_loops=False)
        edges = _edge_set(ei)
        # For directed grid, not all pairs have both directions
        found_asymmetric = False
        for u, v in edges:
            if u != v and (v, u) not in edges:
                found_asymmetric = True
                break
        assert found_asymmetric

    def test_undirected_self_loops_count(self):
        ei = build_grid_graph(4, 4, directed=False, self_loops=True)
        assert _count_self_loops(ei) == 16

    def test_directed_self_loops_count(self):
        ei = build_grid_graph(4, 4, directed=True, self_loops=True)
        assert _count_self_loops(ei) == 16


# =========================================================================== #
# Fully connected graph                                                         #
# =========================================================================== #

class TestBuildFullyConnectedGraph:

    def test_edge_count_no_self_loops(self):
        N = 5
        ei = build_fully_connected_graph(N, self_loops=False)
        assert ei.shape[1] == N * (N - 1)

    def test_edge_count_with_self_loops(self):
        N = 4
        ei = build_fully_connected_graph(N, self_loops=True)
        assert ei.shape[1] == N * N

    def test_dtype(self):
        ei = build_fully_connected_graph(4)
        assert ei.dtype == torch.long

    def test_no_self_loops(self):
        ei = build_fully_connected_graph(4, self_loops=False)
        assert _count_self_loops(ei) == 0

    def test_all_pairs_present(self):
        N = 4
        ei = build_fully_connected_graph(N, self_loops=False)
        edges = _edge_set(ei)
        for i in range(N):
            for j in range(N):
                if i != j:
                    assert (i, j) in edges

    def test_invalid_num_nodes(self):
        with pytest.raises(ValueError):
            build_fully_connected_graph(0)

    def test_device(self):
        ei = build_fully_connected_graph(3, device=torch.device("cpu"))
        assert ei.device.type == "cpu"


# =========================================================================== #
# kNN graph                                                                     #
# =========================================================================== #

class TestBuildKnnGraph:

    @pytest.fixture
    def coords_4(self):
        return torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])

    def test_output_shape(self, coords_4):
        ei = build_knn_graph(coords_4, k=2, directed=True, self_loops=False)
        assert ei.shape[0] == 2
        assert ei.dtype == torch.long

    def test_no_self_loops_when_disabled(self, coords_4):
        ei = build_knn_graph(coords_4, k=2, directed=True, self_loops=False)
        assert _count_self_loops(ei) == 0

    def test_self_loops_count(self, coords_4):
        ei = build_knn_graph(coords_4, k=2, directed=True, self_loops=True)
        assert _count_self_loops(ei) == 4

    def test_no_duplicate_self_loops(self, coords_4):
        ei = build_knn_graph(coords_4, k=2, self_loops=True)
        assert _has_no_duplicate_self_loops(ei)

    def test_undirected_has_reciprocal(self, coords_4):
        ei = build_knn_graph(coords_4, k=2, directed=False, self_loops=False)
        edges = _edge_set(ei)
        for u, v in edges:
            if u != v:
                assert (v, u) in edges

    def test_k_too_large_raises(self, coords_4):
        with pytest.raises(ValueError):
            build_knn_graph(coords_4, k=4)

    def test_invalid_coords_rank(self):
        with pytest.raises(ValueError):
            build_knn_graph(torch.randn(4), k=1)

    def test_known_neighbors(self):
        # Line of 5 points spaced 1 apart; each should connect to its immediate neighbours
        coords = torch.arange(5).float().unsqueeze(1)
        ei = build_knn_graph(coords, k=1, directed=True, self_loops=False)
        edges = _edge_set(ei)
        # Node 0 → node 1 (closest non-self)
        assert (0, 1) in edges
        # Node 4 → node 3
        assert (4, 3) in edges


# =========================================================================== #
# Radius graph                                                                  #
# =========================================================================== #

class TestBuildRadiusGraph:

    @pytest.fixture
    def coords_grid(self):
        # 3x3 grid of 2-D points spaced 1 apart
        pts = [(float(r), float(c)) for r in range(3) for c in range(3)]
        return torch.tensor(pts)

    def test_output_shape(self, coords_grid):
        ei = build_radius_graph(coords_grid, radius=1.5, self_loops=False)
        assert ei.shape[0] == 2
        assert ei.dtype == torch.long

    def test_no_self_loops_when_disabled(self, coords_grid):
        ei = build_radius_graph(coords_grid, radius=1.5, self_loops=False)
        assert _count_self_loops(ei) == 0

    def test_self_loops_count(self, coords_grid):
        ei = build_radius_graph(coords_grid, radius=1.5, self_loops=True)
        assert _count_self_loops(ei) == 9

    def test_no_duplicate_self_loops(self, coords_grid):
        ei = build_radius_graph(coords_grid, radius=1.5, self_loops=True)
        assert _has_no_duplicate_self_loops(ei)

    def test_undirected_reciprocal(self, coords_grid):
        ei = build_radius_graph(coords_grid, radius=1.5, directed=False, self_loops=False)
        edges = _edge_set(ei)
        for u, v in edges:
            if u != v:
                assert (v, u) in edges

    def test_radius_zero_raises(self):
        with pytest.raises(ValueError):
            build_radius_graph(torch.randn(4, 2), radius=0.0)

    def test_invalid_coords_rank(self):
        with pytest.raises(ValueError):
            build_radius_graph(torch.randn(4), radius=1.0)

    def test_small_radius_no_edges(self):
        coords = torch.tensor([[0.0, 0.0], [100.0, 100.0]])
        ei = build_radius_graph(coords, radius=1.0, self_loops=False)
        assert ei.shape[1] == 0


# =========================================================================== #
# IoU graph                                                                     #
# =========================================================================== #

class TestBuildIouGraph:

    @pytest.fixture
    def boxes(self):
        # box_a overlaps box_b (IoU ≈ 0.143); box_c is isolated
        return torch.tensor([
            [0.0, 0.0, 2.0, 2.0],   # a: 2×2 box
            [1.0, 1.0, 3.0, 3.0],   # b: overlaps a
            [5.0, 5.0, 7.0, 7.0],   # c: no overlap
        ])

    def test_known_iou_connections(self, boxes):
        # threshold=0.1 → a-b connected; c isolated (besides self-loops)
        ei = build_iou_graph(boxes, threshold=0.1, directed=False, self_loops=False)
        edges = _edge_set(ei)
        assert (0, 1) in edges
        assert (1, 0) in edges
        assert (0, 2) not in edges

    def test_high_threshold_no_edges(self, boxes):
        ei = build_iou_graph(boxes, threshold=0.9, self_loops=False)
        assert ei.shape[1] == 0

    def test_self_loops_present_when_enabled(self, boxes):
        ei = build_iou_graph(boxes, threshold=0.5, self_loops=True)
        # IoU(i,i)=1.0 ≥ 0.5 → all self-loops
        assert _count_self_loops(ei) == 3

    def test_no_self_loops_when_disabled(self, boxes):
        ei = build_iou_graph(boxes, threshold=0.1, self_loops=False)
        assert _count_self_loops(ei) == 0

    def test_no_duplicate_self_loops(self, boxes):
        ei = build_iou_graph(boxes, threshold=0.0, self_loops=True)
        assert _has_no_duplicate_self_loops(ei)

    def test_invalid_boxes_shape(self):
        with pytest.raises(ValueError):
            build_iou_graph(torch.randn(3, 5), threshold=0.5)

    def test_invalid_threshold(self, boxes):
        with pytest.raises(ValueError):
            build_iou_graph(boxes, threshold=1.5)

    def test_dtype(self, boxes):
        ei = build_iou_graph(boxes, threshold=0.1)
        assert ei.dtype == torch.long


# =========================================================================== #
# Random graph                                                                  #
# =========================================================================== #

class TestBuildRandomGraph:

    def test_deterministic_with_seed(self):
        ei1 = build_random_graph(10, 15, seed=42)
        ei2 = build_random_graph(10, 15, seed=42)
        assert torch.equal(ei1, ei2)

    def test_different_seeds_differ(self):
        ei1 = build_random_graph(10, 15, seed=0)
        ei2 = build_random_graph(10, 15, seed=1)
        assert not torch.equal(ei1, ei2)

    def test_directed_edge_count(self):
        ei = build_random_graph(8, 10, directed=True, self_loops=False, seed=0)
        assert ei.shape[1] == 10

    def test_no_self_loops_when_disabled(self):
        ei = build_random_graph(8, 20, directed=True, self_loops=False, seed=0)
        assert _count_self_loops(ei) == 0

    def test_dtype(self):
        ei = build_random_graph(5, 5, seed=0)
        assert ei.dtype == torch.long

    def test_device(self):
        ei = build_random_graph(5, 5, seed=0, device=torch.device("cpu"))
        assert ei.device.type == "cpu"

    def test_too_many_edges_raises(self):
        with pytest.raises(ValueError):
            # N=3, no self-loops, directed → max 6; asking for 10
            build_random_graph(3, 10, directed=True, self_loops=False)

    def test_zero_edges(self):
        ei = build_random_graph(5, 0, seed=0)
        assert ei.shape[1] == 0

    def test_undirected_has_reciprocal(self):
        ei = build_random_graph(6, 4, directed=False, self_loops=False, seed=7)
        edges = _edge_set(ei)
        for u, v in edges:
            if u != v:
                assert (v, u) in edges


# =========================================================================== #
# Device and dtype correctness                                                  #
# =========================================================================== #

class TestDeviceDtype:

    def test_grid_dtype_long(self):
        assert build_grid_graph(3, 3).dtype == torch.long

    def test_grid_3d_dtype_long(self):
        assert build_grid_graph_3d(2, 2, 2).dtype == torch.long

    def test_fc_dtype_long(self):
        assert build_fully_connected_graph(4).dtype == torch.long

    def test_knn_dtype_long(self):
        coords = torch.randn(5, 2)
        assert build_knn_graph(coords, k=2).dtype == torch.long

    def test_radius_dtype_long(self):
        coords = torch.randn(5, 2)
        assert build_radius_graph(coords, radius=2.0).dtype == torch.long

    def test_random_dtype_long(self):
        assert build_random_graph(5, 4, seed=0).dtype == torch.long

    def test_grid_cpu_device(self):
        ei = build_grid_graph(2, 2, device=torch.device("cpu"))
        assert ei.device.type == "cpu"


# =========================================================================== #
# Patch helpers — 2-D                                                           #
# =========================================================================== #

class TestPatchGridShape:

    def test_basic_non_overlapping(self):
        assert patch_grid_shape(8, 8, 4) == (2, 2)

    def test_non_square(self):
        assert patch_grid_shape(12, 8, (4, 4)) == (3, 2)

    def test_strided(self):
        assert patch_grid_shape(8, 8, 4, stride=2) == (3, 3)

    def test_single_patch(self):
        assert patch_grid_shape(4, 4, 4) == (1, 1)

    def test_bad_dimension_raises(self):
        with pytest.raises(ValueError, match="not exactly covered"):
            patch_grid_shape(7, 8, 4)


class TestImageToPatches:

    def test_output_shape_non_overlapping(self):
        images = torch.randn(2, 3, 8, 8)
        patches = image_to_patches(images, patch_size=4)
        # P = (8//4) * (8//4) = 4, each patch is [C, ph, pw] = [3, 4, 4]
        assert patches.shape == (2, 4, 3, 4, 4)

    def test_output_shape_strided(self):
        images = torch.randn(1, 1, 6, 6)
        patches = image_to_patches(images, patch_size=4, stride=2)
        # n_h = n_w = (6-4)//2+1 = 2
        assert patches.shape == (1, 4, 1, 4, 4)

    def test_invalid_rank_raises(self):
        with pytest.raises(ValueError, match="4-D"):
            image_to_patches(torch.randn(3, 8, 8), patch_size=4)

    def test_bad_stride_raises(self):
        with pytest.raises(ValueError):
            image_to_patches(torch.randn(1, 1, 7, 8), patch_size=4)

    def test_patch_values_correct(self):
        # Each 4×4 block of a known image should match the patch
        images = torch.arange(64, dtype=torch.float).view(1, 1, 8, 8)
        patches = image_to_patches(images, patch_size=4)
        # patch (0, 0) = top-left 4×4 block
        assert torch.equal(patches[0, 0, 0], images[0, 0, :4, :4])
        # patch (0, 1) = top-right 4×4 block
        assert torch.equal(patches[0, 1, 0], images[0, 0, :4, 4:])
        # patch (0, 2) = bottom-left 4×4 block
        assert torch.equal(patches[0, 2, 0], images[0, 0, 4:, :4])

    def test_patch_order_matches_grid(self):
        # Patch index p = r * n_w + c should match build_grid_graph node order
        images = torch.randn(1, 3, 8, 8)
        patches = image_to_patches(images, patch_size=4)
        # n_h=2, n_w=2 → grid node 0=(r=0,c=0), 1=(r=0,c=1), 2=(r=1,c=0), 3=(r=1,c=1)
        assert torch.equal(patches[0, 0, :, :, :], images[0, :, :4, :4])   # node 0
        assert torch.equal(patches[0, 1, :, :, :], images[0, :, :4, 4:])   # node 1
        assert torch.equal(patches[0, 2, :, :, :], images[0, :, 4:, :4])   # node 2
        assert torch.equal(patches[0, 3, :, :, :], images[0, :, 4:, 4:])   # node 3


# =========================================================================== #
# Patch helpers — 3-D                                                           #
# =========================================================================== #

class TestVolumePatchGridShape:

    def test_basic(self):
        assert volume_patch_grid_shape(8, 8, 8, 4) == (2, 2, 2)

    def test_non_cubic(self):
        assert volume_patch_grid_shape(4, 8, 6, (2, 4, 3)) == (2, 2, 2)

    def test_bad_dimension_raises(self):
        with pytest.raises(ValueError, match="not exactly covered"):
            volume_patch_grid_shape(7, 8, 8, 4)


class TestVolumeToPatches:

    def test_output_shape(self):
        volumes = torch.randn(2, 3, 8, 8, 8)
        patches = volume_to_patches(volumes, patch_size=4)
        # P = 2*2*2 = 8
        assert patches.shape == (2, 8, 3, 4, 4, 4)

    def test_non_cubic(self):
        volumes = torch.randn(1, 1, 4, 8, 6)
        patches = volume_to_patches(volumes, patch_size=(2, 4, 3))
        # n_d=2, n_h=2, n_w=2 → P=8
        assert patches.shape == (1, 8, 1, 2, 4, 3)

    def test_invalid_rank_raises(self):
        with pytest.raises(ValueError, match="5-D"):
            volume_to_patches(torch.randn(1, 3, 8, 8), patch_size=4)

    def test_bad_stride_raises(self):
        with pytest.raises(ValueError):
            volume_to_patches(torch.randn(1, 1, 7, 8, 8), patch_size=4)

    def test_patch_order_matches_3d_grid(self):
        # Depth-row-col order should match build_grid_graph_3d node order
        volumes = torch.randn(1, 2, 4, 4, 4)
        patches = volume_to_patches(volumes, patch_size=2)
        # n_d=n_h=n_w=2 → P=8
        # Node (d, r, c) = d*4 + r*2 + c
        # Node 0: (0,0,0) → depth slice 0:2, rows 0:2, cols 0:2
        assert torch.equal(patches[0, 0, :, :, :, :], volumes[0, :, :2, :2, :2])
        # Node 1: (0,0,1) → depth 0:2, rows 0:2, cols 2:4
        assert torch.equal(patches[0, 1, :, :, :, :], volumes[0, :, :2, :2, 2:4])
        # Node 2: (0,1,0) → depth 0:2, rows 2:4, cols 0:2
        assert torch.equal(patches[0, 2, :, :, :, :], volumes[0, :, :2, 2:4, :2])
        # Node 4: (1,0,0) → depth 2:4, rows 0:2, cols 0:2
        assert torch.equal(patches[0, 4, :, :, :, :], volumes[0, :, 2:4, :2, :2])


# =========================================================================== #
# Layer integration — 2-D                                                       #
# =========================================================================== #

class TestLayerIntegration2D:
    """Run each GNN family on a small 2-D patch graph."""

    # Tiny 2-D image: [B=1, C=2, H=4, W=4], patch_size=2 → 2×2=4 patches
    @pytest.fixture
    def setup_2d(self):
        images = torch.randn(1, 2, 4, 4)
        patches = image_to_patches(images, patch_size=2)  # [1, 4, 2, 2, 2]
        x = patches[0]  # [4, 2, 2, 2] — 4 nodes with [C=2, H=2, W=2] features
        ei = build_grid_graph(2, 2, directed=False, self_loops=True)
        return x, ei

    def test_conv_message_passing_2d(self, setup_2d):
        x, ei = setup_2d
        layer = ConvMessagePassing(in_shape=(2, 2, 2), out_shape=(4, 2, 2), aggr="sum")
        out = layer(x, ei)
        assert out.shape == (4, 4, 2, 2)

    def test_gat_2d(self, setup_2d):
        x, ei = setup_2d
        layer = TensorGATLayer(
            in_channels=2, out_channels=4, num_heads=2, concat_heads=True,
            spatial_rank=2
        )
        out = layer(x, ei)
        assert out.shape == (4, 4, 2, 2)

    def test_sage_2d(self, setup_2d):
        x, ei = setup_2d
        layer = TensorGraphSAGELayer(in_channels=2, out_channels=4, spatial_rank=2)
        out = layer(x, ei)
        assert out.shape == (4, 4, 2, 2)

    def test_gin_2d(self, setup_2d):
        x, ei = setup_2d
        layer = TensorGINLayer(in_channels=2, out_channels=4, spatial_rank=2)
        out = layer(x, ei)
        assert out.shape == (4, 4, 2, 2)


# =========================================================================== #
# Layer integration — 3-D                                                       #
# =========================================================================== #

class TestLayerIntegration3D:
    """Run each GNN family on a small 3-D volume patch graph."""

    # Volume: [B=1, C=2, D=4, H=4, W=4], patch_size=2 → 2×2×2=8 patches
    @pytest.fixture
    def setup_3d(self):
        volumes = torch.randn(1, 2, 4, 4, 4)
        patches = volume_to_patches(volumes, patch_size=2)  # [1, 8, 2, 2, 2, 2]
        x = patches[0]  # [8, 2, 2, 2, 2] — 8 nodes with [C=2, D=2, H=2, W=2] features
        ei = build_grid_graph_3d(2, 2, 2, directed=False, self_loops=True)
        return x, ei

    def test_conv_message_passing_3d(self, setup_3d):
        x, ei = setup_3d
        layer = ConvMessagePassing(
            in_shape=(2, 2, 2, 2), out_shape=(4, 2, 2, 2), aggr="sum"
        )
        out = layer(x, ei)
        assert out.shape == (8, 4, 2, 2, 2)

    def test_gat_3d(self, setup_3d):
        x, ei = setup_3d
        layer = TensorGATLayer(
            in_channels=2, out_channels=4, num_heads=2, concat_heads=True,
            spatial_rank=3
        )
        out = layer(x, ei)
        assert out.shape == (8, 4, 2, 2, 2)

    def test_sage_3d(self, setup_3d):
        x, ei = setup_3d
        layer = TensorGraphSAGELayer(in_channels=2, out_channels=4, spatial_rank=3)
        out = layer(x, ei)
        assert out.shape == (8, 4, 2, 2, 2)

    def test_gin_3d(self, setup_3d):
        x, ei = setup_3d
        layer = TensorGINLayer(in_channels=2, out_channels=4, spatial_rank=3)
        out = layer(x, ei)
        assert out.shape == (8, 4, 2, 2, 2)
