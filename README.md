

# TGraphX

TGraphX is a **PyTorch**-based framework for building Graph Neural Networks (GNNs) that work with node and edge features of any dimension while retaining their **spatial layout**. The code is designed for flexibility, easy GPU-acceleration, and rapid prototyping of new GNN ideas, **especially** those that need to preserve local spatial details (e.g., image or volumetric patches).  

📄 **Preprint**: [TGraphX: Tensor-Aware Graph Neural Network for Multi-Dimensional Feature Learning](https://arxiv.org/abs/2504.03953)  
✏️ *Authors: Arash Sajjadi, Mark Eramian*  
🗓️ *Published on arXiv, April 2025*


> **Note:** TGraphX includes optional skip connections that help with 
> stable gradient flow in deeper GNN stacks. The overall design is rooted 
> in rigorous theoretical and practical foundations, aiming to unify 
> convolutional neural networks (CNNs) with GNN-based relational reasoning.

## Installation

You may try installing TGraphX using:

```bash
pip install tgraphx
```

> ⚠️ **Note:** This command is expected to work once the package is fully published on PyPI.  
> It might work now, but we haven’t tested it yet.  
> Full support and testing for PyPI installation is planned in a future release.

Alternatively, you can install it manually by cloning the repository:

```bash
git clone https://github.com/arashsajjadi/TGraphX.git
cd TGraphX
pip install -e .
```


---  
## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture Highlights](#architecture-highlights)
  - [Preserving Spatial Fidelity](#preserving-spatial-fidelity)
  - [Convolution-Based Message Passing](#convolution-based-message-passing)
  - [Deep CNN Aggregator with Residuals](#deep-cnn-aggregator-with-residuals)
  - [End-to-End Differentiability](#end-to-end-differentiability)
- [Future Works](#future-works)
- [Installation](#installation)
- [Folder Structure](#folder-structure)
- [Core Components](#core-components)
- [Layers](#layers)
- [Models](#models)
- [Configuration Options](#configuration-options)
- [Advanced Topics](#advanced-topics)
- [Novelties and Contributions](#novelties-and-contributions)
- [Conclusion](#conclusion)
- [Citations](#citations)
- [License](#license)


---

## Overview

TGraphX provides a modular way to create GNNs by combining several components:

1. **Graph Representation**  
   A `Graph` class holds node features, edge indices, and optional edge features. Unlike traditional GNNs where node features are vectors, TGraphX supports multi-dimensional features such as `[C, H, W]` tensors—making it particularly effective for vision tasks.

2. **Message Passing Layers**  
   Customizable layers process messages between nodes *while preserving the spatial layout of features*. In TGraphX:
   - **ConvMessagePassing** uses `1×1` convolutions on concatenated spatial features (e.g., `Conv1×1(Concat(Xi, Xj, Eij))`).
   - **DeepCNNAggregator** is a deep CNN (default 4 layers) that refines aggregated messages, keeping their spatial structure intact (i.e., `[C, H, W]` shape).

3. **Models**  
   Pre-built models combine a CNN encoder with GNN layers:
   - **CNN Encoder** processes raw image patches into spatial feature maps (e.g., `[C, H, W]`).
   - **Optional Pre-Encoder** (e.g., ResNet-like) can be enabled to further refine raw patches before the main CNN encoder.
   - **Unified CNN‑GNN Model** uses CNN encoders for local features and GNN layers for global relational reasoning, then pools the result for final classification.
   - An extra *skip connection* (if enabled) merges the raw CNN patch output with the GNN output for better gradient flow and more stable learning.

---  
## Key Features

- **Support for Arbitrary Dimensions**  
  Handle vectors, 2D images, or even volumetric 3D patches as node features.  

- **Spatial Message Passing**  
  Messages preserve spatial dimensions (e.g., `[C, H, W]`), letting convolutional filters capture local patterns and avoid destructive flattening of features.

- **Deep Aggregation**  
  A deep CNN aggregator (with multiple `3×3` convolutions, batch normalization, dropout, and ReLU) refines messages across multiple hops, enabling better local–global fusion.

- **Optional Pre‑Encoder**  
  Pre-process raw patches with a ResNet-like module (or even load a pretrained ResNet-18) to enrich features before the main GNN pipeline.

- **Flexible Data Loading**  
  TGraphX includes custom dataset and data loader classes (`GraphDataset` and `GraphDataLoader`) for direct graph-based batching.

- **Configurable Skip Connections**  
  Enable or disable skip connections that pass CNN outputs directly into the final stages, improving gradient flow.

---

## Architecture Highlights

### Preserving Spatial Fidelity
Unlike conventional GNNs that flatten node features into vectors, TGraphX retains the full spatial layout `[C, H, W]` at each node. This ensures that local pixel-level (or voxel-level) structure, which is crucial for vision tasks, remains intact throughout the message passing process.

### Convolution-Based Message Passing
TGraphX implements message passing via `Conv1×1(Concat(Xi, Xj, Eij))`. This approach:
- Respects spatial alignment (i.e., each spatial location in one node’s feature map can directly interact with the same location in its neighbors’ feature maps).
- Preserves the dimension `[C, H, W]`, avoiding vector flattening.
- Optionally incorporates edge features `Eij` for more advanced relational cues (e.g., distances, bounding-box overlaps).

### Deep CNN Aggregator with Residuals
Messages from neighbors are aggregated (summed or averaged) and then passed to a **deep CNN aggregator** that uses multiple `3×3` convolutions with *residual skips*. This design:
- Prevents the overwriting of original features by always adding `Aggregator(mj)` to the old node state `Xj`.
- Facilitates stable gradient flow in deep GNN stacks.
- Broadens the effective receptive field in feature space, capturing both local patches and more distant interactions.

### End-to-End Differentiability
Every stage of TGraphX—patch extraction, optional pre-encoder, CNN encoder, graph construction, message passing, aggregation, and classification—remains **fully differentiable** in PyTorch. This end-to-end design simplifies model development, parameter tuning, and experimentation with novel GNN layers.

---

## Future Works

- **Scalability and Data Requirements**  
  Adapting TGraphX to higher-resolution inputs or massive datasets (e.g., MS COCO) may require further optimizations, including efficient graph construction or pruning strategies.

- **Domain-Specific Customization**  
  Some tasks might not need full spatial fidelity at every message-passing step. Researchers could explore ways to selectively reduce resolution or apply specialized convolutions to different node subsets.

- **Alternative Edge Definitions**  
  Learned adjacency or richer spatial features (e.g., IoU or geometric cues) can further improve performance in complex scenes.

- **Multimodal and Real-Time Extensions**  
  Integrating TGraphX with sensor data or text embeddings could enable richer reasoning for applications like autonomous driving or real-time video surveillance.

---
## Installation

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/YourUsername/TGraphX.git
   cd TGraphX
   ```

2. **Set Up the Environment**  
   Use the provided `environment.yml` to create a conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate tgraphx
   ```

3. **Install PyTorch**  
   Install a recent version of [PyTorch](https://pytorch.org/) (GPU version if possible).  

4. **Install Additional Dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

5. **Editable Mode (Optional)**  
   ```bash
   pip install -e .
   ```  

---

## Folder Structure

```
TGraphX/
├── __init__.py
├── core/
│   ├── dataloader.py
│   ├── graph.py
│   └── utils.py
├── layers/
│   ├── aggregator.py
│   ├── attention_message.py
│   ├── base.py
│   ├── conv_message.py
│   └── safe_pool.py
├── models/
│   ├── cnn_encoder.py
│   ├── cnn_gnn_model.py
│   ├── graph_classifier.py
│   ├── node_classifier.py
│   └── pre_encoder.py
├── environment.yml
└── README.md
```

---

## Core Components

### Graph and Data Loading

- **`Graph` & `GraphBatch`**  
  Represent individual graphs (nodes, edges) and batches of graphs. The batch version offsets node indices to avoid collisions, allowing parallel processing in PyTorch.

- **`GraphDataset` & `GraphDataLoader`**  
  Custom dataset and data loader classes that streamline the creation of graph batches from a set of images, patches, or other structured data.

### Utility Functions

- **`load_config`**  
  Load YAML/JSON configuration files to keep hyperparameters consistent across experiments.

- **`get_device`**  
  Utility to automatically detect and return the correct device (GPU or CPU).

---

## Layers

### Base Layer

- **`TensorMessagePassingLayer`**  
  An abstract base class that defines the interface (message, aggregate, update steps) for all message passing. Crucially, it handles multi-dimensional node features (e.g., `[C, H, W]`).

### Convolution-Based Message Passing

- **`ConvMessagePassing`**  
  Concatenates source and target node feature maps (plus optional edge features) along the channel dimension and applies a `1×1` convolution:
  ```python
  Mij = Conv1×1(Concat(Xi, Xj, Eij))
  ```
  - **Message Phase**: Each pair `(i, j)` of nodes exchanges messages computed by a `1×1` conv.
  - **Aggregation + Residual Update**: After summing messages from neighbors, a deep CNN aggregator processes the sum, and the original node features are updated via a **residual skip**.

### Deep CNN Aggregator

- **`DeepCNNAggregator`**  
  A stack of `3×3` convolutional layers with batch normalization, ReLU, and dropout. It refines the aggregated messages:
  ```python
  X'_j = X_j + A( m_j )
  ```
  where `m_j = sum of messages to node j`. Residual connections ensure stable gradient flow.

### Attention-Based Message Passing

- **`AttentionMessagePassing`**  
  An alternative that uses `1×1` convolutions to compute query, key, and value maps for each node. Spatial alignment is preserved while attention weights scale incoming messages. Useful for tasks needing dynamic connectivity or weighting.

### Safe Pooling

- **`SafeMaxPool2d`**  
  A robust pooling module that checks if spatial dimensions `[H, W]` are large enough before applying max pooling. Prevents dimension mismatch errors in deeper aggregator stacks.

---

## Models

### CNN Encoder and Pre-Encoder

- **`CNNEncoder`**  
  Converts raw patches (`[C_in, patch_H, patch_W]`) into *spatial feature maps* (e.g., `[C_out, H2, W2]`). Includes:
  - Multiple 3×3 conv blocks with BN, ReLU, and dropout.
  - Optional residual connections.
  - Safe max pooling if the spatial size remains large.

- **Optional Pre‑Encoder**  
  - If `use_preencoder` is `True`, a **ResNet‑like** (or fully custom) module first processes each patch, returning refined features with the same spatial structure.  
  - `pretrained_resnet` can load weights from a standard ResNet‑18 for transfer learning.

### Unified CNN‑GNN Model

- **`CNN_GNN_Model`**  
  A single pipeline that:
  1. Splits the image into patches, optionally uses `PreEncoder`.
  2. Feeds patches into `CNNEncoder` to get `[C, H, W]` maps.
  3. Builds a graph where each node holds a `[C, H, W]` map.
  4. Applies multiple GNN layers (like `ConvMessagePassing` + `DeepCNNAggregator`).
  5. Optionally uses a skip connection to combine CNN outputs with GNN outputs.
  6. Performs final spatial pooling before classification.

### Graph & Node Classification Models

- **`GraphClassifier`**  
  Intended for graph-level tasks (e.g., classification of an entire image or object ensemble). Combines message passing with a final pooling layer (mean, max, or attention) over nodes, then feeds the result into a classifier.

- **`NodeClassifier`**  
  Suitable for node-level tasks (e.g., labeling each patch or region). Stacks simpler message passing layers for classification on each node separately.

---

## Configuration Options

TGraphX is highly configurable. Some key parameters include:

```python
config = {
    "cnn_params": {
         "in_channels": 3,
         "out_features": 64,
         "num_layers": 2,
         "hidden_channels": 64,
         "dropout_prob": 0.3,
         "use_batchnorm": True,
         "use_residual": True,
         "pool_layers": 2,
         "debug": False,
         "return_feature_map": True
    },
    "use_preencoder": False,
    "pretrained_resnet": False,
    "preencoder_params": {
         "in_channels": 3,
         "out_channels": 32,
         "hidden_channels": 32
    },
    "gnn_in_dim": (64, 5, 5),
    "gnn_hidden_dim": (128, 5, 5),
    "num_classes": 10,
    "num_gnn_layers": 4,
    "gnn_dropout": 0.3,
    "residual": True,
    "aggregator_params": {
         "num_layers": 4,
         "dropout_prob": 0.3,
         "use_batchnorm": True
    }
}
```

- **`cnn_params`**: Controls the CNN encoder architecture (e.g., channels, dropout, pooling).
- **`use_preencoder`**: Boolean indicating whether to preprocess patches with a custom or pretrained module.  
- **`pretrained_resnet`**: If `True`, loads pretrained ResNet-18 weights in the pre-encoder.  
- **`gnn_in_dim`, `gnn_hidden_dim`**: Shapes of the node features in GNN layers. Each dimension can be `[C, H, W]`.  
- **`num_gnn_layers`**: How many message passing layers to stack.  
- **`aggregator_params`**: Depth, dropout, and BN usage in the aggregator.  
- **`residual`**: Enables skip connections in the GNN layers.  

---

## Advanced Topics

### Theoretical Insights

1. **Universal Approximation via Deep CNN**  
   Stacking multiple convolutional layers with residual skips (in both the CNN encoder and the aggregator) enhances the effective receptive field and helps approximate complex local feature maps.

2. **Residual Learning for Gradient Flow**  
   Residual connections in both the CNN encoder and aggregator mitigate vanishing gradients, allowing deeper structures to train effectively end-to-end.

3. **Spatial vs. Flattened Features**  
   Preserving the `[C, H, W]` layout at each node addresses a key limitation in conventional GNNs—loss of local spatial semantics. TGraphX’s design is grounded in the observation that many vision tasks require capturing fine-grained local details alongside global relational structures.

### Possible Extensions

- **Adaptive Edge Construction**  
  Dynamically compute adjacency based on patch similarity or learned attention, rather than fixed proximity thresholds.

- **Mixed Modalities**  
  Combine image data with textual or numerical features by storing them as separate channels or separate GNN streams.

- **Task-Specific Losses**  
  Add auxiliary losses (e.g., bounding-box IoU or segmentation overlap) for detection or segmentation tasks, integrated into the GNN training loop.

- **Performance Optimizations**  
  Use group convolutions or low-rank factorization in the aggregator to reduce memory and computational overhead.


---

## Novelties and Contributions

TGraphX departs from traditional GNN designs in several ways:

1. **Full Spatial Fidelity**  
   Each node in the graph remains a *multi-dimensional* feature map rather than a flattened vector, preserving local spatial relationships crucial for vision tasks.

2. **Convolution-Based Message Passing**  
   Employing `1×1` convolutions on `[C, H, W]` feature maps lets neighboring patches exchange information at *every pixel location*, ensuring alignment and detail retention.

3. **Deep Residual Aggregation**  
   Multiple `3×3` CNN layers in the aggregator—complete with batch normalization, ReLU, dropout, and skip connections—allow the model to fuse multi-hop messages in a stable, expressive manner.

4. **End-to-End Differentiable**  
   From raw image patches to final classification or detection outputs, **all** steps—CNN feature extraction, graph construction, message passing, and aggregator updates—are trained jointly, strengthening synergy between local feature extraction and relational reasoning.

5. **Modular & Extensible**  
   - Allows easy substitution of the aggregator or attention-based message passing layers.
   - Accommodates multiple data modalities (image, volumetric, or otherwise).
   - Scales from small graphs (few patches) to larger patch partitions for high-resolution images.

These innovations build on earlier GNN research while pushing further to **retain** all the valuable local details that are typically lost in flattened GNN nodes.

---

## Conclusion

We have presented **TGraphX**, an architecture aimed at integrating convolutional neural 
networks (CNNs) and graph neural networks (GNNs) in a way that preserves spatial fidelity. 
By retaining multi-dimensional CNN feature maps as node representations and employing 
convolution-based message passing, TGraphX captures both local and global spatial context. 
Our experiments—particularly those involving detection refinement—demonstrate its potential 
to resolve detection discrepancies and refine localization accuracy in challenging vision tasks.

While we do not claim it to be universally optimal for all computer vision scenarios, TGraphX 
offers a flexible framework that other researchers can adapt or extend. This integration of 
CNN-based feature extraction with GNN-based relational reasoning is a promising direction 
for future AI and vision research.

---
## Citations

```bibtex
@misc{sajjadi2025tgraphxtensorawaregraphneural,
      title={TGraphX: Tensor-Aware Graph Neural Network for Multi-Dimensional Feature Learning}, 
      author={Arash Sajjadi and Mark Eramian},
      year={2025},
      eprint={2504.03953},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2504.03953}, 
}
```
---

## License

TGraphX is released under the [MIT License](https://opensource.org/licenses/MIT). See the `LICENSE` file for more details.

---

**Enjoy exploring and developing your spatially-aware graph neural networks with TGraphX!**  
If you have any questions, suggestions, or want to contribute, feel free to open an issue or submit a pull request.