"""Reproducibility: same seed → same dataset splits."""
import torch
from od_graph_fusion.reproducibility import set_global_seed
from od_graph_fusion.datasets import load_dataset


def _tiny_cfg():
    return {
        "seed": 42,
        "dataset": {
            "name": "synthetic_voc_like",
            "num_images": 8, "num_classes": 3,
            "image_size": [64, 64],
            "class_names": ["a", "b", "c"],
        },
    }


def test_same_seed_same_dataset():
    set_global_seed(42)
    r1 = load_dataset(_tiny_cfg())
    set_global_seed(42)
    r2 = load_dataset(_tiny_cfg())
    assert len(r1) == len(r2)
    for a, b in zip(r1, r2):
        assert a.image_id == b.image_id
        assert torch.allclose(a.gt_boxes, b.gt_boxes)
        assert torch.equal(a.gt_labels, b.gt_labels)


def test_global_seed_returns_dict():
    state = set_global_seed(7)
    assert state["seed"] == 7
    assert "torch_version" in state
