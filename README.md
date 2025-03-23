# TGraphX

TGraphX is a **PyTorch**-based framework for building Graph Neural Networks (GNNs) that work with node and edge features of any dimension. The code is designed for flexibility, easy GPU-acceleration, and rapid prototyping of new GNN ideas.

> **Note:** The current architecture is new and under active development. Some features (like the skip connection and spatial message passing via CNN aggregators) are experimental.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Folder Structure](#folder-structure)
- [Core Components](#core-components)
- [Layers](#layers)
- [Models](#models)
- [Configuration Options](#configuration-options)
- [Examples](#examples)
- [License](#license)

---

## Overview

TGraphX provides a modular way to create GNNs by combining several components:

1. **Graph Representation:**  
   A `Graph` class holds node features, edge indices, and optional edge features.

2. **Message Passing Layers:**  
   Customizable layers that process messages between nodes while preserving the spatial layout of features. New updates allow the message passing to treat node features as spatial maps (e.g., image patches) rather than flattening them into vectors. This is achieved with:
   - **ConvMessagePassing:** Uses 1×1 convolutions on concatenated spatial features.
   - **DeepCNNAggregator:** A deep CNN (by default 4 layers) that aggregates spatial messages, keeping full spatial structure intact.

3. **Models:**  
   Pre-built models combine a CNN encoder with GNN layers:
   - **CNN Encoder:** Processes raw image patches into spatial feature maps.
   - **Optional Pre‑Encoder:** Pre-processes patches using a ResNet‑like network (with an option to use pretrained ResNet‑18).
   - **Unified CNN‑GNN Model:** Passes spatial features through GNN message passing layers and then pools them for final classification.
   - An extra skip connection (if enabled) can merge the raw CNN patch output with the GNN output for better gradient flow.

---

## Key Features

- **Support for Arbitrary Dimensions:**  
  Work with vectors, images, or even volumetric data.
- **Spatial Message Passing:**  
  Processes each node’s feature map as an image. Pairwise messages preserve the spatial dimensions, so convolutional filters can capture local patterns.
- **Deep Aggregation:**  
  Uses a deep CNN aggregator that applies multiple 3×3 convolutions (with batch normalization, ReLU, and dropout) on the spatial messages.
- **Optional Pre‑Encoder:**  
  Pre-processes raw image patches with a ResNet‑like module. You can load pretrained ResNet‑18 weights or use a custom module.
- **Flexible Data Loading:**  
  Includes custom dataset and data loader classes for graphs.
- **Configurable Skip Connections:**  
  Optionally pass raw CNN features directly to the classifier, enhancing gradient flow and preserving fine spatial details.

---

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/YourUsername/TGraphX.git
   cd TGraphX
   ```

2. **Set Up the Environment:**  
   Use the provided `environment.yml` to create a conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate tgraphx
   ```

3. **Install PyTorch:**  
   Install a recent version of [PyTorch](https://pytorch.org/) (use the GPU version if available).

4. **Install Additional Dependencies:**  
   For example, run:
   ```bash
   pip install -r requirements.txt
   ```

5. **Editable Mode (Optional):**
   ```bash
   pip install -e .
   ```

---

## Folder Structure

```
TGraphX/
├── __init__.py
├── core/
│   ├── dataloader.py      # Dataset and DataLoader classes for graph data.
│   ├── graph.py           # Graph and GraphBatch data structures.
│   └── utils.py           # Utility functions (e.g., load_config, get_device).
├── layers/
│   ├── aggregator.py      # Deep CNN aggregator to combine spatial messages.
│   ├── attention_message.py  # Attention-based message passing layer.
│   ├── base.py           # Base class for all message passing layers.
│   ├── conv_message.py    # Convolution-based message passing that keeps spatial maps.
│   └── safe_pool.py       # Safe max pooling module for small spatial sizes.
├── models/
│   ├── cnn_encoder.py     # CNN encoder with optional residual connections and safe pooling.
│   ├── cnn_gnn_model.py   # Unified CNN‑GNN model that integrates the CNN encoder and GNN layers.
│   ├── graph_classifier.py  # Model for graph-level classification.
│   ├── node_classifier.py   # Model for node-level classification.
│   └── pre_encoder.py     # Optional pre‑encoder module (ResNet‑18 or custom).
├── examples/              # Jupyter notebooks demonstrating TGraphX usage.
│   ├── node_classification_tensor.ipynb
│   ├── graph_classification_volumetric.ipynb
│   ├── graph_to_image_ssim.ipynb
│   └── comparison.png
├── environment.yml
└── README.md
```

---

## Core Components

### Graph and Data Loading
- **`Graph` & `GraphBatch`:**  
  Represent individual graphs and batches of graphs.
- **`GraphDataset` & `GraphDataLoader`:**  
  Simplify data loading and batching of graph data.

### Utility Functions
- **`load_config`:** Load YAML/JSON configuration files.
- **`get_device`:** Returns the available device (GPU or CPU).

---

## Layers

### Base Layer
- **`TensorMessagePassingLayer`:**  
  The abstract base class for message passing layers. It defines the interface for the message, aggregation, and update steps while preserving spatial dimensions.

### Convolution-Based Message Passing
- **`ConvMessagePassing`:**  
  Concatenates source and destination node feature maps (and optional edge features) along the channel dimension and applies a 1×1 convolution.  
  - **Message Phase:**  
    Processes each pair of node features as a spatial map.
  - **Update Phase:**  
    Uses the **DeepCNNAggregator** to aggregate these spatial messages with multiple convolutional layers.

### Deep CNN Aggregator
- **`DeepCNNAggregator`:**  
  A series of convolutional layers (default 4 layers) that operate on spatial feature maps. Each layer uses a 3×3 kernel, batch normalization, ReLU activation, and dropout to preserve spatial details and ensure good gradient flow.

### Attention-Based Message Passing
- **`AttentionMessagePassing`:**  
  Uses attention mechanisms on spatial maps by applying 1×1 convolutions to compute query, key, and value maps while preserving the spatial structure.

### Safe Pooling
- **`SafeMaxPool2d`:**  
  A pooling layer that checks if the input’s spatial dimensions are large enough before applying max pooling.

---

## Models

### CNN Encoder and Pre-Encoder
- **`CNNEncoder`:**  
  Converts raw node image patches into spatial feature maps using several convolutional layers.  
  - **Optional Pre‑Encoder:**  
    If provided, the pre‑encoder (from `pre_encoder.py`) pre-processes raw image patches. This module can load a pretrained ResNet‑18 or use a custom architecture.
  
### Unified CNN‑GNN Model
- **`CNN_GNN_Model`:**  
  Combines a CNN encoder (with optional pre‑encoder) with GNN message passing layers.  
  - The CNN encoder produces spatial feature maps from image patches.
  - The GNN layers (using `ConvMessagePassing`) process these spatial maps without flattening.
  - An optional skip connection can merge raw CNN outputs with GNN outputs for better gradient flow.
  - Final spatial pooling is applied before classification.

### Graph & Node Classification Models
- **`GraphClassifier`:**  
  Uses GNN message passing and pooling to perform graph-level classification.
- **`NodeClassifier`:**  
  Stacks simple message passing layers (using linear transformations) for node-level classification.

---

## Configuration Options

The framework is highly configurable. Key options include:

- **CNN Encoder Settings (`cnn_params`):**  
  Control the number of layers, channels, dropout, and whether to use batch normalization or residual connections.

- **Pre‑Encoder Settings:**  
  - `use_preencoder`: Enable or disable the pre‑encoder stage.
  - `pretrained_resnet`: Choose to load pretrained ResNet‑18 weights.
  - `preencoder_params`: Parameters for the pre‑encoder (e.g., number of channels).

- **GNN Settings:**  
  - `gnn_in_dim` and `gnn_hidden_dim`: Define the input and hidden dimensions for GNN layers.
  - `num_gnn_layers`: How many GNN message passing layers to stack.
  - `gnn_dropout` and `residual`: Dropout rate and whether to use skip connections in GNN layers.

- **Aggregator Settings (`aggregator_params`):**  
  Define the number of layers, dropout, and batch normalization options for the deep CNN aggregator.

---

## Sample Configuration

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

To disable the pre‑encoder or ResNet‑18, set `"use_preencoder": False` or `"pretrained_resnet": False` accordingly.

---
## Visual Summary (Just an Example – e.g., Conceptual Diagram)

```
+-------------------------------+
| Raw Node Image Patch (3x8x8)   |  <-- Each patch is an image-like tensor.
+-------------------------------+
             │
             ▼
+-------------------------------+
|       Pre-Encoder             |  <-- Optional: Preprocess patch using ResNet-like (or custom) module.
| (if enabled, maintains spatial|
|  structure; output shape:      |
|    [N, 32, H, W])              |
+-------------------------------+
             │
             ▼
+-------------------------------+
|         CNN Encoder           |  <-- Processes the (preprocessed) patch into a spatial feature map.
| (Multiple conv layers, with    |
|  residual connections; output: |
|   [N, 64, 5, 5])               |
+-------------------------------+
             │
             ▼
+-------------------------------+
|    GNN Layers (Message       |  <-- For each pair (edge) of nodes, features are concatenated 
|    Passing using ConvMessage  |      spatially (keeping [C, H, W] format) and processed via 1×1 conv.
|    Passing)                   |      Then aggregated using a deep CNN aggregator.
| (Output: [N, 128, 5, 5])       |
+-------------------------------+
             │
             ▼
+------------------------------------------+
| Optional Skip Connection:                |  <-- If enabled, add original CNN encoder output (patch features)
| Combine (sum or concatenate) CNN output    |      to GNN output for richer gradient flow.
| with GNN output.                         |
+------------------------------------------+
             │
             ▼
+-------------------------------+
|   Spatial Pooling             |  <-- Average over spatial dimensions (H, W) to get a vector.
|   (Output: [N, 128])           |
+-------------------------------+
             │
             ▼
+-------------------------------+
|       Classifier              |  <-- Fully connected layer maps pooled features to class logits.
+-------------------------------+
             │
             ▼
+-------------------------------+
|         Final Decision        |  <-- Predicted class labels.
+-------------------------------+
```

![mermaid-ai-diagram-2025-03-23-002525.png](examples/mermaid-ai-diagram-2025-03-23-002525.png)
---

## Examples

Check the `examples/` folder for Jupyter notebooks that demonstrate:
- Node classification using image patches as graph nodes.
- Graph classification with volumetric data.
- Image reconstruction from graphs using SSIM loss.

---

## License

TGraphX is released under the [MIT License](https://opensource.org/licenses/MIT). See the LICENSE file for more details.

---

**Enjoy exploring and developing your graph neural networks with TGraphX!**  
If you have any questions or suggestions, please feel free to open an issue or submit a pull request.

