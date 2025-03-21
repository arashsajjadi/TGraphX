# TGraphX

A **PyTorch**-based Graph Neural Network (GNN) framework that supports arbitrary-dimensional node and edge features. TGraphX is designed for flexibility, GPU-acceleration, and an easy-to-use, extensible codebase.

## Table of Contents

- [Overview](#overview)  
- [Key Features](#key-features)  
- [Installation](#installation)  
- [Folder Structure](#folder-structure)  
- [Core Components](#core-components)  
- [Layers](#layers)  
- [Models](#models)  
- [Examples](#examples)  
- [License](#license)

---

## Overview

TGraphX provides a clean, modular approach to GNN development. The library revolves around three core concepts:

1. **Graph Representation**: A custom `Graph` class that holds node features, edge indices, and optional edge features.  
2. **Message Passing Layers**: Multiple message passing strategies (linear, convolution-based, attention) for processing node and edge data of various dimensionalities.  
3. **Models**: Ready-to-use or easily extensible GNN models for tasks like node-level and graph-level classification.

TGraphX leverages standard PyTorch modules (`nn.Module`) and supports GPU acceleration. It also introduces a simplified data loading mechanism with `GraphDataset` and `GraphDataLoader` for batching graphs.

---

## Key Features

- **Arbitrary-Dimensional Features**  
  - Works with vector features (`[N, C]`), 2D images (`[N, C, H, W]`), 3D volumetric data (`[N, C, D, H, W]`), and more.
- **Flexible Message Passing**  
  - Choose between `LinearMessagePassing`, `ConvMessagePassing`, or `AttentionMessagePassing`.
  - Easily integrate custom message, aggregation, and update functions.
- **Graph Abstractions**  
  - `Graph` class for single-graph data, `GraphBatch` for batching multiple graphs with appropriate indexing.
- **Batched Data Loading**  
  - `GraphDataset` and `GraphDataLoader` mirror PyTorch’s `Dataset`/`DataLoader` pattern for GNN data.
- **Pooling and Classification**  
  - Built-in pooling methods (`mean`, `sum`, `max`) at the graph level.
  - Examples for node classification (`NodeClassifier`) and graph classification (`GraphClassifier`).

---

## Installation

1. **Clone or Download the Repository**  
   ```bash
   git clone https://github.com/YourUsername/TGraphX.git
   cd TGraphX
   ```

2. **Set Up the Environment** (Optional)  
   - A sample `environment.yml` is provided. You can create and activate a conda environment via:
     ```bash
     conda env create -f environment.yml
     conda activate tgraphx
     ```

3. **Install PyTorch**  
   - Make sure you have a recent version of [PyTorch](https://pytorch.org/) installed (GPU-compatible if desired).

4. **Install Any Additional Dependencies**  
   - For example, `pip install -r requirements.txt` if you add a `requirements.txt`.  

5. **(Optional) Editable Mode**  
   - To install TGraphX in editable mode, run:
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
│   ├── base.py
│   ├── conv_message.py
│   └── attention_message.py
├── models/
│   ├── node_classifier.py
│   └── graph_classifier.py
├── examples/
│   ├── graph_classification_volumetric.ipynb
│   ├── graph_to_image_ssim.ipynb
│   ├── node_classification_tensor.ipynb
│   └── comparison.png
├── environment.yml
└── README.md
```

- **`__init__.py`**: Initializes the TGraphX package.
- **`core/`**: Core classes for `Graph`, `GraphBatch`, data loading, and utility functions.
- **`layers/`**: Contains base `TensorMessagePassingLayer` classes and specific message-passing implementations (`ConvMessagePassing`, `AttentionMessagePassing`, etc.).
- **`models/`**: Pre-built GNN models for node-level and graph-level tasks.
- **`examples/`**: Jupyter notebooks illustrating how to use TGraphX for various tasks like node classification, graph classification, and image reconstruction.

---

## Core Components

### `core.graph`
- **`Graph`**  
  Represents a single graph with:
  - `node_features` ([N, ...]): Node-level data (arbitrary shape).
  - `edge_index` ([2, E]): Defines connections among nodes.
  - `edge_features` ([E, ...], optional): Extra features for edges.
  - A `to(device)` method to move data to GPU or CPU.
  
- **`GraphBatch`**  
  Batches multiple `Graph` objects into a single set of tensors. Adjusts edge indices to keep track of node offsets, and creates a `batch` vector indicating which sub-graph each node belongs to.

### `core.dataloader`
- **`GraphDataset`**  
  Wraps a list of Graph objects into a PyTorch-style Dataset.
- **`GraphDataLoader`**  
  A specialized DataLoader that can shuffle, batch, and (optionally) apply a custom collate function to a `GraphDataset`.

### `core.utils`
- **`load_config`**  
  Loads YAML or JSON configuration files for experimentation or model setup.
- **`get_device`**  
  Convenience function returning `cuda` if available, else `cpu`.

---

## Layers

### `layers.base`
- **`TensorMessagePassingLayer`** (abstract)  
  - Provides a template for message, aggregation, and update phases in GNN layers.
  - Subclasses must implement `message()` and optionally `update()`.
- **`LinearMessagePassing`**  
  - Simple layer that concatenates source/destination (and optional edge) features, then applies a linear transformation.

### `layers.conv_message`
- **`ConvMessagePassing`**  
  - Uses 2D or 3D convolutions on node (and optional edge) features.  
  - Automatically selects between `Conv2d` and `Conv3d` based on the dimensionality of the features (e.g., `(C,H,W)` vs `(C,D,H,W)`).

### `layers.attention_message`
- **`AttentionMessagePassing`**  
  - Computes attention coefficients based on flattened node features (and optionally edge features).  
  - Normalizes and aggregates messages per node using an attention mechanism.

---

## Models

### `models.node_classifier`
- **`NodeClassifier`**  
  - A stack of `LinearMessagePassing` layers for node classification tasks (e.g., predicting labels for each node).

### `models.graph_classifier`
- **`GraphClassifier`**  
  - A stack of message-passing layers plus a graph-level pooling (`mean`, `sum`, or `max`).  
  - Performs final classification on the pooled graph representation (for tasks like graph classification or multi-graph data).

---

## Examples

The `examples/` folder contains Jupyter notebooks demonstrating how to use TGraphX in various settings:

1. **`node_classification_tensor.ipynb`**  
   - Simple node classification using vector features. Demonstrates how to instantiate `NodeClassifier`, set up a training loop, and track performance.
   
2. **`graph_classification_volumetric.ipynb`**  
   - Shows how to apply the `GraphClassifier` model to volumetric (3D) data, such as `[C, D, H, W]` node features.
   
3. **`graph_to_image_ssim.ipynb`**  
   - Uses a GNN to transform graph-structured data into an image output, training with SSIM-based loss.  
   - Illustrates advanced usage (arbitrary shaping, custom losses, and generative tasks).
   
4. **`comparison.png`**  
   - An example side-by-side image showcasing target vs. generated output from a GNN (produced in one of the notebooks).

---

## License

This repository is available under the [MIT License](https://opensource.org/licenses/MIT). Please see the [LICENSE](LICENSE) file for more details.

---

**Enjoy building graph neural networks with TGraphX!** If you have any questions, suggestions, or issues, feel free to open an issue or make a pull request.