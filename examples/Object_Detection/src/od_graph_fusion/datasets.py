"""Dataset loaders: synthetic, VOC2007, custom folder, COCO mini.

Each loader returns a list of ``ImageRecord``:

    @dataclass
    class ImageRecord:
        image_id: str
        image: torch.Tensor          # [C, H, W] in [0, 1]
        image_size: (H, W)
        gt_boxes: Tensor [N_gt, 4]   # xyxy in pixel coords
        gt_labels: Tensor [N_gt]     # int class ids
        class_names: List[str]       # idx → name
        split: "train" | "val" | "test"
        source: str
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch


@dataclass
class ImageRecord:
    image_id: str
    image: torch.Tensor              # [C, H, W] float in [0, 1]
    image_size: Tuple[int, int]      # (H, W)
    gt_boxes: torch.Tensor           # [N, 4] xyxy
    gt_labels: torch.Tensor          # [N] long
    class_names: List[str]
    split: str
    source: str = "synthetic"


# Default class set used by synthetic FAST_SMOKE — a 5-class subset.
SYNTHETIC_CLASS_NAMES = ["person", "car", "dog", "cat", "bicycle"]

# VOC 2007 20-class list
VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]


# ── Synthetic VOC-like ─────────────────────────────────────────────────────


def _draw_synthetic_image(
    H: int, W: int, n_objects: int, num_classes: int, rng: random.Random,
    generator: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate a synthetic image with rectangle "objects"."""
    img = torch.zeros(3, H, W, dtype=torch.float32)
    img += torch.rand(3, 1, 1, generator=generator) * 0.3 + 0.1
    gt_boxes = []
    gt_labels = []
    for _ in range(n_objects):
        w = rng.randint(20, max(21, W // 3))
        h = rng.randint(20, max(21, H // 3))
        x1 = rng.randint(0, W - w - 1)
        y1 = rng.randint(0, H - h - 1)
        x2, y2 = x1 + w, y1 + h
        cls = rng.randint(0, num_classes - 1)
        # Draw a colored rectangle whose color is tied to class id
        color = torch.tensor([
            ((cls * 67) % 200 + 55) / 255,
            ((cls * 113) % 200 + 55) / 255,
            ((cls * 199) % 200 + 55) / 255,
        ], dtype=torch.float32)
        img[:, y1:y2, x1:x2] = color[:, None, None]
        # Add a bit of noise
        img[:, y1:y2, x1:x2] += torch.randn(3, y2 - y1, x2 - x1,
                                            generator=generator) * 0.05
        gt_boxes.append([x1, y1, x2, y2])
        gt_labels.append(cls)
    img = img.clamp(0.0, 1.0)
    return (img,
            torch.tensor(gt_boxes, dtype=torch.float32).reshape(-1, 4),
            torch.tensor(gt_labels, dtype=torch.long))


def _synthetic_dataset(
    num_images: int,
    num_classes: int,
    image_size: Tuple[int, int],
    class_names: Sequence[str],
    seed: int = 42,
) -> List[ImageRecord]:
    rng = random.Random(seed)
    gen = torch.Generator().manual_seed(seed)
    records = []
    for i in range(num_images):
        n_objs = rng.randint(1, 4)
        H, W = image_size
        img, boxes, labels = _draw_synthetic_image(H, W, n_objs, num_classes,
                                                    rng, gen)
        # split deterministically
        if i < int(0.7 * num_images):
            split = "train"
        elif i < int(0.85 * num_images):
            split = "val"
        else:
            split = "test"
        records.append(ImageRecord(
            image_id=f"synth_{i:04d}",
            image=img,
            image_size=(H, W),
            gt_boxes=boxes,
            gt_labels=labels,
            class_names=list(class_names),
            split=split,
            source="synthetic_voc_like",
        ))
    return records


# ── VOC 2007 loader (uses existing VOCdevkit/ if present) ─────────────────


def _parse_voc_xml(xml_path: Path, class_to_idx: Dict[str, int]) -> Tuple[List[List[float]], List[int]]:
    import xml.etree.ElementTree as ET
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes, labels = [], []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        if name not in class_to_idx:
            continue
        bnd = obj.find("bndbox")
        x1 = float(bnd.find("xmin").text)
        y1 = float(bnd.find("ymin").text)
        x2 = float(bnd.find("xmax").text)
        y2 = float(bnd.find("ymax").text)
        boxes.append([x1, y1, x2, y2])
        labels.append(class_to_idx[name])
    return boxes, labels


def _voc2007_dataset(
    voc_root: str | Path,
    num_images: int,
    image_size: Tuple[int, int],
    seed: int = 42,
    class_filter: Optional[List[str]] = None,
) -> Optional[List[ImageRecord]]:
    """Load VOC 2007. Returns None if the data folder is missing."""
    from PIL import Image
    voc_root = Path(voc_root)
    candidates = [voc_root, voc_root / "VOC2007", voc_root / "VOCdevkit" / "VOC2007"]
    voc_dir = next((p for p in candidates if (p / "JPEGImages").exists()), None)
    if voc_dir is None:
        return None
    jpegs = voc_dir / "JPEGImages"
    anns = voc_dir / "Annotations"
    all_imgs = sorted(p.name for p in jpegs.glob("*.jpg"))
    if not all_imgs:
        return None

    rng = random.Random(seed)
    rng.shuffle(all_imgs)
    # When class_filter is active, pre-scan annotations to find images
    # that actually contain the target class, then cap at num_images.
    if class_filter is not None:
        import xml.etree.ElementTree as ET
        filtered = []
        for img_name in all_imgs:
            ann_path = anns / img_name.replace(".jpg", ".xml")
            if not ann_path.exists():
                continue
            try:
                tree = ET.parse(ann_path)
            except Exception:
                continue
            names_in_img = {obj.findtext("name", "") for obj in tree.getroot().findall("object")}
            if any(c in names_in_img for c in class_filter):
                filtered.append(img_name)
            if len(filtered) >= num_images:
                break
        all_imgs = filtered
    else:
        all_imgs = all_imgs[:num_images]

    class_to_idx = {c: i for i, c in enumerate(VOC_CLASSES)}
    records = []
    H, W = image_size
    for i, img_name in enumerate(all_imgs):
        img_path = jpegs / img_name
        ann_path = anns / img_name.replace(".jpg", ".xml")
        try:
            pil = Image.open(img_path).convert("RGB")
        except Exception:
            continue
        orig_w, orig_h = pil.size
        pil_resized = pil.resize((W, H))
        import numpy as np
        arr = np.asarray(pil_resized, dtype=np.float32) / 255.0
        img = torch.from_numpy(arr).permute(2, 0, 1).contiguous()

        if ann_path.exists():
            boxes_orig, labels = _parse_voc_xml(ann_path, class_to_idx)
            scale_x = W / orig_w; scale_y = H / orig_h
            boxes = []
            for (x1, y1, x2, y2), lbl in zip(boxes_orig, labels):
                # class_filter: keep only boxes matching the target classes
                if class_filter is not None:
                    lbl_name = VOC_CLASSES[lbl] if 0 <= lbl < len(VOC_CLASSES) else ""
                    if lbl_name not in class_filter:
                        continue
                boxes.append([x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y])
            gt_boxes = (torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
                        if boxes else torch.zeros(0, 4, dtype=torch.float32))
            # Remap labels to filtered class index when class_filter is active
            if class_filter is not None:
                filt_idx = {name: i for i, name in enumerate(class_filter)}
                filt_labels = [filt_idx[VOC_CLASSES[lbl]] for (_, lbl) in
                               zip(boxes_orig, labels)
                               if VOC_CLASSES[lbl] in class_filter]
                gt_labels = torch.tensor(filt_labels, dtype=torch.long)
            else:
                gt_labels = torch.tensor(labels, dtype=torch.long)
        else:
            gt_boxes = torch.zeros(0, 4, dtype=torch.float32)
            gt_labels = torch.zeros(0, dtype=torch.long)

        # Skip images with no GT boxes when class_filter is active
        if class_filter is not None and gt_boxes.numel() == 0:
            continue

        if i < int(0.7 * len(all_imgs)):
            split = "train"
        elif i < int(0.85 * len(all_imgs)):
            split = "val"
        else:
            split = "test"
        record_class_names = list(class_filter) if class_filter else list(VOC_CLASSES)
        records.append(ImageRecord(
            image_id=img_path.stem,
            image=img,
            image_size=(H, W),
            gt_boxes=gt_boxes,
            gt_labels=gt_labels,
            class_names=record_class_names,
            split=split,
            source="voc2007",
        ))
    return records


# ── Top-level dispatcher ──────────────────────────────────────────────────


def load_dataset(config: Dict[str, Any]) -> List[ImageRecord]:
    """Load a dataset based on the config dict.

    Falls back to ``synthetic_voc_like`` if the requested dataset is missing.
    """
    ds = config.get("dataset", {})
    name = ds.get("name", "synthetic_voc_like")
    num_images = int(ds.get("num_images", 16))
    image_size = tuple(ds.get("image_size", [256, 256]))
    seed = int(config.get("seed", 42))
    class_names = ds.get("class_names") or SYNTHETIC_CLASS_NAMES

    if name == "voc2007":
        from .config import project_root
        voc_root = Path(ds.get("voc_root", "data/VOCdevkit"))
        if not voc_root.is_absolute():
            voc_root = project_root() / voc_root
        class_filter = ds.get("class_filter", None)
        records = _voc2007_dataset(voc_root, num_images, image_size, seed,
                                    class_filter=class_filter)
        if records:
            return records
        print(f"[datasets] VOC2007 not found at {voc_root}; falling back to synthetic.")
    elif name == "custom_folder":
        return _custom_folder_dataset(
            ds.get("root"), num_images, image_size, seed,
            class_names=class_names,
        )

    # default / fallback
    num_classes = int(ds.get("num_classes", len(class_names)))
    return _synthetic_dataset(num_images, num_classes, image_size,
                              class_names, seed)


def _custom_folder_dataset(
    root: Optional[str], num_images: int,
    image_size: Tuple[int, int], seed: int,
    class_names: Sequence[str],
) -> List[ImageRecord]:
    """Load images from a folder, no GT (purely for inference)."""
    from PIL import Image
    if root is None:
        raise ValueError("custom_folder dataset requires 'dataset.root'")
    p = Path(root)
    imgs = (sorted(p.glob("*.jpg")) + sorted(p.glob("*.png")))[:num_images]
    H, W = image_size
    records = []
    import numpy as np
    for i, ip in enumerate(imgs[:num_images]):
        pil = Image.open(ip).convert("RGB").resize((W, H))
        arr = np.asarray(pil, dtype=np.float32) / 255.0
        img = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        records.append(ImageRecord(
            image_id=ip.stem, image=img, image_size=(H, W),
            gt_boxes=torch.zeros(0, 4), gt_labels=torch.zeros(0, dtype=torch.long),
            class_names=list(class_names),
            split="test", source="custom_folder",
        ))
    return records


def split_records(records: List[ImageRecord]) -> Dict[str, List[ImageRecord]]:
    """Group records by split label."""
    out: Dict[str, List[ImageRecord]] = {"train": [], "val": [], "test": []}
    for r in records:
        out.setdefault(r.split, []).append(r)
    return out


def dataset_summary(records: List[ImageRecord]) -> Dict[str, Any]:
    by_split = split_records(records)
    return {
        "num_images": len(records),
        "splits": {k: len(v) for k, v in by_split.items()},
        "num_classes": len(records[0].class_names) if records else 0,
        "class_names": records[0].class_names if records else [],
        "source": records[0].source if records else "unknown",
        "image_size": list(records[0].image_size) if records else None,
    }
