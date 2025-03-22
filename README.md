

# TGraphX

A **PyTorch**-based Graph Neural Network (GNN) framework that supports arbitrary-dimensional node and edge features. TGraphX is designed for flexibility, GPU acceleration, and an easy-to-use, extensible codebase for rapid GNN prototyping.

## Table of Contents

- [Overview](#overview)  
- [Key Features](#key-features)  
- [Installation](#installation)  
- [Folder Structure](#folder-structure)  
- [Core Components](#core-components)  
- [Layers](#layers)  
- [Models](#models)  
- [Examples](#examples)  
- [Configuration Options](#configuration-options)  
- [License](#license)

---

## Overview

TGraphX provides a modular approach to designing graph neural networks. The library revolves around three core concepts:

1. **Graph Representation**  
   A custom `Graph` class that encapsulates node features, edge indices, and optional edge features.

2. **Message Passing Layers**  
   Flexible message passing layers that work on arbitrary-dimensional data. Choose from simple linear transformations, convolution-based layers, or attention mechanisms. A deep CNN aggregator is now available to perform sophisticated aggregation of multi-dimensional features.

3. **Models**  
   Ready-to-use or easily extensible GNN models for node and graph classification. The unified CNN‑GNN model integrates a CNN encoder with GNN message passing layers. Additionally, an optional Pre‑Encoder stage is available to pre-process raw node image patches with a ResNet‑like architecture (or a custom variant).

---

## Key Features

- **Arbitrary-Dimensional Features**  
  Supports vectors (`[N, C]`), images (`[N, C, H, W]`), volumetric data (`[N, C, D, H, W]`), etc.

- **Flexible Message Passing**  
  Offers multiple message passing strategies:  
  - **LinearMessagePassing**: Uses concatenation and linear layers for vector inputs.  
  - **ConvMessagePassing**: Applies 2D/3D convolutions on spatial node features and leverages a deep CNN aggregator (configurable in depth, dropout, and batch normalization).  
  - **AttentionMessagePassing**: Uses attention mechanisms to weight messages.

- **Pre‑Encoder Stage**  
  An optional module (`PreEncoder`) can pre‑process each node’s image patch before the CNN encoder.  
  - **ResNet‑18 Integration**: Optionally load a pretrained ResNet‑18 (via torchvision’s weights API) or use a custom, randomly initialized variant.
  
- **Graph Abstractions**  
  - `Graph` for single-graph data  
  - `GraphBatch` for batching multiple graphs with automatic index adjustments.

- **Batched Data Loading**  
  `GraphDataset` and `GraphDataLoader` mimic PyTorch’s Dataset/DataLoader pattern for seamless batching.

- **Pooling and Classification**  
  Built-in pooling methods (`mean`, `sum`, `max`) for both node-level and graph-level readouts.

---

## Installation

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/YourUsername/TGraphX.git
   cd TGraphX
   ```

2. **Set Up the Environment** (Optional)  
   Use the provided `environment.yml` to create a conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate tgraphx
   ```

3. **Install PyTorch**  
   Make sure to install a recent version of [PyTorch](https://pytorch.org/) (GPU-compatible if desired).

4. **Install Additional Dependencies**  
   For example, if you add a `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

5. **Editable Mode (Optional)**  
   To install TGraphX in editable mode:
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
│   ├── aggregator.py         # Deep CNN aggregator for message passing
│   ├── attention_message.py  # Attention-based message passing layer
│   ├── base.py               # Base class for message passing layers
│   ├── conv_message.py       # Convolution-based message passing layer
│   └── safe_pool.py          # Safe pooling layer to handle small spatial dimensions
├── models/
│   ├── cnn_encoder.py        # CNN encoder with optional safe pooling and residual blocks
│   ├── cnn_gnn_model.py      # Unified CNN‑GNN model that integrates the CNN encoder with GNN layers and optional pre‑encoder
│   ├── graph_classifier.py   # Graph classification model using message passing and pooling
│   ├── node_classifier.py    # Node classification model using stacked message passing layers
│   └── pre_encoder.py        # Pre‑encoder module (ResNet‑18 integration or custom architecture)
├── examples/                 # Example notebooks demonstrating various tasks
│   ├── node_classification_tensor.ipynb
│   ├── graph_classification_volumetric.ipynb
│   ├── graph_to_image_ssim.ipynb
│   └── comparison.png
├── environment.yml
└── README.md
```

- **`__init__.py`**: Package initialization.
- **`core/`**: Contains graph data structures, data loading, and utility functions.
- **`layers/`**: Houses message passing layers and the new deep CNN aggregator along with a safe pooling module.
- **`models/`**: Pre-built models including the unified CNN‑GNN model with the optional pre‑encoder.
- **`examples/`**: Jupyter notebooks showing how to use TGraphX for node/graph classification and other tasks.

---

## Core Components

### `core.graph`
- **`Graph`**: Represents an individual graph with node features, edge indices, and optional edge features.
- **`GraphBatch`**: Batches multiple graphs into a single structure with adjusted indices and a `batch` vector.

### `core.dataloader`
- **`GraphDataset`**: Wraps a list of `Graph` objects.
- **`GraphDataLoader`**: DataLoader tailored for batching graph data.

### `core.utils`
- **`load_config`**: Loads configuration files (YAML/JSON).
- **`get_device`**: Returns `'cuda'` if available, otherwise `'cpu'`.

---

## Layers

### `layers.base`
- **`TensorMessagePassingLayer`** (abstract): Base template for implementing message passing layers.
- **`LinearMessagePassing`**: Implements message passing using linear transformations.

### `layers.conv_message`
- **`ConvMessagePassing`**: Implements message passing using convolutions.  
  - Concatenates source and destination node features (and optionally edge features) and processes them via a 1×1 convolution.
  - Uses the **DeepCNNAggregator** (in `layers/aggregator.py`) to perform complex aggregation with a deep CNN (default 4 layers, configurable dropout and batch normalization).

### `layers.attention_message`
- **`AttentionMessagePassing`**: Implements attention-based message passing.

### `layers.aggregator`
- **`DeepCNNAggregator`**: A configurable deep CNN that aggregates messages over spatial dimensions, providing more expressive aggregation than simple pooling.

### `layers.safe_pool`
- **`SafeMaxPool2d`**: A variant of max pooling that only applies pooling if the spatial dimensions are large enough.

---

## Models

### `models.cnn_encoder`
- **`CNNEncoder`**: Converts raw node data (such as image patches) into a feature map using multiple convolutional layers with optional residual connections and safe pooling.
  - **Optional Pre‑Encoder**: If provided, raw inputs are pre‑processed by the pre‑encoder before the CNN encoder.

### `models.pre_encoder`
- **`PreEncoder`**: An optional module that pre-processes raw node image patches.  
  - **Pretrained ResNet‑18**: Can load pretrained ResNet‑18 weights using the new torchvision API.  
  - **Custom Architecture**: Alternatively, builds a simple custom ResNet‑like network if pretrained weights are not desired.

### `models.cnn_gnn_model`
- **`CNN_GNN_Model`**: A unified model that combines a CNN encoder (with optional pre‑encoder) and GNN layers.  
  - The GNN layers use convolution-based message passing with the deep CNN aggregator.
  - The model aggregates spatial dimensions and node-level features for final classification.

### `models.graph_classifier` and `models.node_classifier`
- **`GraphClassifier`**: For graph-level classification tasks.  
- **`NodeClassifier`**: For node-level classification tasks.

---

## Configuration Options

TGraphX now supports additional configuration keys for advanced model customization:
- **`use_preencoder`**: Boolean flag to enable or disable the pre‑encoder stage.
- **`pretrained_resnet`**: Boolean flag to load pretrained ResNet‑18 weights in the pre‑encoder.
- **`preencoder_params`**: Dictionary for configuring the pre‑encoder (e.g., number of channels, hidden channels).
- **`aggregator_params`**: Dictionary for configuring the deep CNN aggregator (e.g., number of layers, dropout probability, use of batch normalization).

A sample configuration dictionary:

```python
config = {
    # Configuration for the CNN encoder used in the unified model
    "cnn_params": {
         "in_channels": 3,              # Number of input channels for the CNN encoder (e.g., 3 for RGB images)
         "out_features": 64,            # Number of output channels for the CNN encoder (defines the feature map depth)
         "num_layers": 5,               # Total number of convolutional layers in the CNN encoder
         "hidden_channels": 64,         # Number of channels used in the intermediate layers of the CNN encoder
         "dropout_prob": 0.3,           # Dropout probability to apply within the CNN encoder for regularization
         "use_batchnorm": True,         # Enable Batch Normalization in the CNN encoder layers
         "use_residual": True,          # Enable residual (skip) connections within the CNN encoder
         "pool_layers": 2,              # Number of early layers on which max pooling is applied
         "debug": False,                # Debug flag to print internal shapes and statistics during encoding
         "return_feature_map": True     # If True, the encoder returns a spatial feature map (not flattened)
    },
    "use_preencoder": True,              # Flag to enable the optional pre-encoder stage (pre-process raw input)
    "pretrained_resnet": True,           # Flag to load pretrained ResNet-18 weights for the pre-encoder if enabled
    # Configuration for the pre-encoder module (ResNet-like architecture)
    "preencoder_params": {
         "in_channels": 3,             # Number of input channels for the pre-encoder (e.g., 3 for RGB images)
         "out_channels": 32,           # Number of output channels from the pre-encoder
         "hidden_channels": 32         # Number of hidden channels in the custom pre-encoder (if not using pretrained)
    },
    "gnn_in_dim": (64, 5, 5),            # Input dimensions for the GNN layers; must match the CNN encoder's output shape
    "gnn_hidden_dim": (128, 5, 5),         # Hidden state dimensions for the GNN layers (multi-dimensional: channels, H, W)
    "num_classes": 10,                   # Number of classes for the final classification task (e.g., 10 for CIFAR-10)
    "num_gnn_layers": 4,                 # Total number of GNN message passing layers to stack in the model
    "gnn_dropout": 0.3,                  # Dropout probability to use in the GNN layers for regularization
    "residual": True,                    # Enable residual (skip) connections within the GNN layers
    # Parameters for the deep CNN aggregator used in the GNN layers' aggregation step
    "aggregator_params": {
         "num_layers": 4,               # Number of convolutional layers in the deep CNN aggregator
         "dropout_prob": 0.3,           # Dropout probability within the aggregator CNN
         "use_batchnorm": True          # Enable Batch Normalization in the aggregator CNN layers
    }
}

```

---

## License

This repository is available under the [MIT License](https://opensource.org/licenses/MIT). Please refer to the [LICENSE](LICENSE) file for details.

---

**Enjoy building your graph neural networks with TGraphX!**  
If you have any questions, suggestions, or issues, please open an issue or submit a pull request.

---

### Note on Pretrained ResNet‑18  
If you wish to disable the pretrained ResNet‑18 in the pre‑encoder, simply set `"pretrained_resnet": False` in your configuration or set `"use_preencoder": False` if you want to skip the pre‑encoder stage entirely.

lects the latest changes in TGraphX including the deep CNN aggregator, the optional pre‑encoder (with pretrained ResNet‑18 integration), and enhanced configuration flexibility.