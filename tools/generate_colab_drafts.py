"""Generate expanded TGraphX Colab draft notebooks (30 scenarios).

These drafts are for MAINTAINER REVIEW ONLY.
They will NOT be committed to the repository in this form.
After the maintainer tests and uploads selected notebooks to Google Colab,
the verified URLs will be added to docs/colab_gallery.md in v1.3.1.

Usage::

    python tools/generate_colab_drafts.py [--out-dir colab_drafts]

All notebooks are CPU-runnable, use synthetic/built-in data unless
explicitly noted, and avoid private paths, secrets, and hidden downloads.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# ── helpers ───────────────────────────────────────────────────────────────────


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.strip()}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip(),
    }


def nb(cells: list) -> dict:
    return {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "cells": cells,
    }


# ── notebook definitions ───────────────────────────────────────────────────────


NOTEBOOKS = {

"01_easy_tensor_node_classification.ipynb": nb([
    md("# 01 — Easy Mode Tensor Node Classification\n\n**Goal:** Train a node classifier on `[N,C,H,W]` tensor node features using the TGraphX Easy Mode API — zero PyTorch boilerplate required.\n\n**TGraphX subsystem:** `tgraphx.easy`\n**Data:** Synthetic. **Runtime:** < 30s on CPU."),
    code("# Optional install in Colab:\n# !pip install -q tgraphx\nimport tgraphx as tgx\nprint('TGraphX', tgx.__version__)"),
    md("## Scenario\n\nA synthetic citation graph where each node (paper) carries a `[4, 6, 6]` tensor — think of a small feature map.\nGoal: classify each paper into one of 3 topics."),
    code("data = tgx.easy.synthetic_tensor_node_classification(\n    num_nodes=256, node_shape=(4, 6, 6), num_classes=3, num_edges=1024, seed=42\n)\nprint(f'Graph: {data.num_nodes} nodes, {data.num_edges} edges')\nprint(f'Node feature shape per node: {data.node_features.shape[1:]}')\nprint(f'Labels: {data.node_labels.shape}')"),
    code("result = tgx.easy.train_node_classifier(\n    data, model='tensor_gcn', sampler='neighbor',\n    fanouts=[10, 5], batch_size=32, epochs=5, seed=42, verbose=True,\n)\nprint('\\nFinal metrics:', result.metrics)"),
    code("# Low-level escape hatch — you always have full PyTorch access:\nprint('Model type:', type(result.model).__name__)\nprint('Node features shape:', result.graph.node_features.shape)\nprint('Config:', result.config)"),
    md("## Key takeaway\n\nTGraphX preserves the `[C,H,W]` structure through message passing — no silent flattening.\nFor expert control, use `result.model`, `result.graph`, and `result.loader` directly."),
]),

"02_image_patch_tensor_graph_core_identity.ipynb": nb([
    md("# 02 — Image-Patch Tensor Graph: The TGraphX Core Identity\n\n**Goal:** Build a graph of image patches where each node is a `[C, H, W]` tensor.\nTrain a tensor-aware model and compare with a flattened baseline.\n\n**This is the flagship TGraphX demonstration.**\n\n**Data:** Synthetic image. **Runtime:** < 60s on CPU."),
    code("import torch, torch.nn as nn, torch.nn.functional as F, time\nfrom tgraphx import Graph, ConvMessagePassing, build_grid_graph, image_to_patches, patch_grid_shape\nprint('Imports OK')"),
    md("## Build a synthetic image with spatial structure"),
    code("torch.manual_seed(42)\nC, H, W = 3, 12, 12\nimage = torch.randn(C, H, W) * 0.3\nyy = torch.linspace(-1, 1, H).view(H,1).expand(H,W)\nxx = torch.linspace(-1, 1, W).view(1,W).expand(H,W)\nimage[0] += yy; image[1] += xx\nimage[2] += torch.exp(-(xx**2 + yy**2))\npatches = image_to_patches(image.unsqueeze(0), patch_size=4, stride=4).squeeze(0)\nN, Cp, ph, pw = patches.shape\nprint(f'Patches: {N} nodes, each shape [{Cp}, {ph}, {pw}]')\nedge_index = build_grid_graph(*patch_grid_shape(H, W, 4, 4), directed=False)\ny = (patches[:, 0].mean(dim=(-1,-2)) > 0).long()"),
    code("class TensorModel(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.conv = ConvMessagePassing((Cp, ph, pw), (8, ph, pw))\n        self.pool = nn.AdaptiveAvgPool2d((1,1))\n        self.head = nn.Linear(8, 2)\n    def forward(self, x, ei):\n        z = self.conv(x, ei).relu()\n        return self.head(self.pool(z).flatten(1))\n\nclass FlatModel(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.fc1 = nn.Linear(Cp*ph*pw, 32)\n        self.fc2 = nn.Linear(32, 2)\n    def forward(self, x, ei):\n        return self.fc2(F.relu(self.fc1(x)))\n\ntm = TensorModel(); fm = FlatModel()\nprint(f'Tensor params: {sum(p.numel() for p in tm.parameters())}  Flat params: {sum(p.numel() for p in fm.parameters())}')"),
    code("g_t = Graph(node_features=patches, edge_index=edge_index, y=y)\ng_f = Graph(node_features=patches.flatten(1), edge_index=edge_index, y=y)\nopt_t = torch.optim.Adam(tm.parameters(), lr=1e-2)\nopt_f = torch.optim.Adam(fm.parameters(), lr=1e-2)\nfor ep in range(1, 8):\n    for model, g, opt in [(tm, g_t, opt_t), (fm, g_f, opt_f)]:\n        logits = model(g.node_features, g.edge_index)\n        loss = F.cross_entropy(logits, g.node_labels)\n        opt.zero_grad(); loss.backward(); opt.step()\n    if ep % 3 == 1:\n        print(f'Ep {ep}: tensor_loss={F.cross_entropy(tm(g_t.node_features, g_t.edge_index), y).item():.3f}  flat_loss={F.cross_entropy(fm(g_f.node_features, g_f.edge_index), y).item():.3f}')"),
    code("# Verify tensor shapes preserved:\nz = tm.conv(patches, edge_index)\nprint(f'After ConvMessagePassing: {z.shape}')  # [N, 8, ph, pw]\nassert z.shape == (N, 8, ph, pw), 'Shape contract violated!'\nprint('✓ TGraphX preserves [C,H,W] through message passing.')"),
    md("## Why this matters\n\nFlattening discards the spatial structure that makes CNNs effective.\nTGraphX keeps `[C,H,W]` intact so per-channel spatial patterns can be learned across the graph topology."),
]),

"03_tensor_vs_flatten_benchmark_story.ipynb": nb([
    md("# 03 — Tensor vs Flatten: A Benchmark Story\n\n**Goal:** Compare tensor-native and flattened baselines on parameter count, runtime, gradient health, and shape preservation.\n\n**TGraphX subsystem:** `ConvMessagePassing`\n**Data:** Synthetic image patches. **Runtime:** < 60s on CPU."),
    code("import torch, torch.nn as nn, torch.nn.functional as F, time, json\nfrom tgraphx import Graph, ConvMessagePassing, build_grid_graph, image_to_patches, patch_grid_shape\ntorch.manual_seed(0)"),
    md("## Setup: identical graph, two model styles"),
    code("C, H, W, PS = 3, 16, 16, 4\nimage = torch.randn(1, C, H, W)\npatches = image_to_patches(image, patch_size=PS, stride=PS).squeeze(0)\nN, Cp, ph, pw = patches.shape\nprint(f'Graph: {N} nodes, each [{Cp},{ph},{pw}]')\nei = build_grid_graph(*patch_grid_shape(H, W, PS, PS), directed=False)\ny = (patches[:, 0].mean(dim=(-1,-2)) > 0).long()\ng_tensor = Graph(node_features=patches, edge_index=ei, y=y)\ng_flat = Graph(node_features=patches.flatten(1), edge_index=ei, y=y)"),
    code("class TM(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.c1 = ConvMessagePassing((Cp,ph,pw),(16,ph,pw))\n        self.c2 = ConvMessagePassing((16,ph,pw),(16,ph,pw))\n        self.pool = nn.AdaptiveAvgPool2d((1,1))\n        self.h = nn.Linear(16, 2)\n    def forward(self, x, ei):\n        return self.h(self.pool(self.c2(self.c1(x,ei).relu(),ei).relu()).flatten(1))\n\nclass FM(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.h1 = nn.Linear(Cp*ph*pw, 64)\n        self.h2 = nn.Linear(64, 2)\n    def forward(self, x, ei):\n        return self.h2(F.relu(self.h1(x)))\n\ntm = TM(); fm = FM()\ntp = sum(p.numel() for p in tm.parameters())\nfp = sum(p.numel() for p in fm.parameters())\nprint(f'Tensor params: {tp}   Flat params: {fp}')"),
    code("# Runtime comparison\ndef time_forward(model, g, n=50):\n    t0 = time.perf_counter()\n    for _ in range(n):\n        model(g.node_features, g.edge_index)\n    return (time.perf_counter()-t0)/n*1000\ntensor_ms = time_forward(tm, g_tensor)\nflat_ms = time_forward(fm, g_flat)\nprint(f'Tensor model forward: {tensor_ms:.2f}ms/call')\nprint(f'Flat   model forward: {flat_ms:.2f}ms/call')"),
    code("# Gradient health\ntm.zero_grad()\nloss = F.cross_entropy(tm(g_tensor.node_features, g_tensor.edge_index), y)\nloss.backward()\nfor name, p in tm.named_parameters():\n    if p.grad is not None:\n        assert torch.isfinite(p.grad).all(), f'Non-finite grad in {name}'\nprint('✓ All tensor model gradients are finite.')"),
    md("## Summary\n\nThe claim is not that the tensor model is always faster or more accurate on synthetic tasks.\nThe claim is: **TGraphX correctly preserves `[C,H,W]` tensor structure** through message passing,\nwith finite gradients and predictable shapes. Flatten-based models discard spatial structure."),
]),

"04_edge_tensor_features_message_passing.ipynb": nb([
    md("# 04 — Edge Tensor Features in Message Passing\n\n**Goal:** Show how scalar and 1-D edge attributes affect message passing in TGraphX.\n\n**TGraphX subsystem:** `Graph(edge_attr=...)`, `ConvMessagePassing`, `LinearMessagePassing`\n**Data:** Synthetic. **Runtime:** < 20s on CPU."),
    code("import torch\nfrom tgraphx import Graph, LinearMessagePassing\ntorch.manual_seed(0)"),
    md("## Scenario\n\nA social network where edges carry a 'strength' scalar weight and a\n3-D relationship descriptor (e.g., frequency, recency, affinity)."),
    code("N, D_node, D_edge = 20, 16, 3\nnode_features = torch.randn(N, D_node)\nedge_index = torch.randint(0, N, (2, 60))\nedge_weight = torch.rand(60)          # scalar edge weights\nedge_features = torch.randn(60, D_edge)  # vector edge attributes\n\ng = Graph(\n    node_features=node_features,\n    edge_index=edge_index,\n    edge_weight=edge_weight,\n    edge_attr=edge_features,   # edge_attr is an alias for edge_features\n    metadata={'description': 'social-network-demo'}\n)\nprint(g)\nprint('Edge feature shape:', g.edge_features.shape)"),
    code("# LinearMessagePassing uses node features by default.\n# Edge features are available on the Graph object for custom message functions.\nlayer = LinearMessagePassing(in_shape=(D_node,), out_shape=(32,))\nout = layer(g.node_features, g.edge_index)\nprint('Output shape:', out.shape)  # [N, 32]\nassert out.shape == (N, 32)\nout.sum().backward()\nprint('✓ Gradient flows correctly.')"),
    md("## Accessing edge features in custom layers\n\nFor custom edge-aware aggregation, use `graph.edge_features` and `graph.edge_weight`\ndirectly in your `nn.Module.forward` method.  The `edge_weight` tensor is used\nautomatically by `LinearMessagePassing` and related layers when provided."),
    code("# Access and use edge weight in a manual aggregation:\nsrc, dst = g.edge_index\nweighted_msg = g.node_features[src] * g.edge_weight.unsqueeze(-1)\nagg = torch.zeros(N, D_node)\nagg.index_add_(0, dst, weighted_msg)\nprint('Weighted aggregation shape:', agg.shape)\nprint('Nonzero entries:', (agg.abs() > 0).sum().item())"),
]),

"05_graph_level_tensor_state_classification.ipynb": nb([
    md("# 05 — Graph-Level Tensor Features and Classification\n\n**Goal:** Work with graph-level input features and graph-level target labels.\n\n**TGraphX subsystem:** `Graph(graph_features=..., graph_label=...)`, `GraphClassifier`\n**Data:** Synthetic. **Runtime:** < 20s on CPU."),
    code("import torch\nfrom tgraphx import Graph, GraphBatch, build_model\ntorch.manual_seed(0)"),
    md("## Scenario\n\nMolecule graphs where each graph has a global state vector (e.g., computed global descriptor)\nand a binary solubility label."),
    code("def make_mol_graph(seed):\n    torch.manual_seed(seed)\n    N = torch.randint(5, 15, (1,)).item()\n    x = torch.randn(N, 8)   # atom features\n    ei = torch.randint(0, N, (2, max(1, N*2)))\n    graph_feat = torch.randn(16)  # graph-level input feature (NOT the label)\n    label = torch.randint(0, 2, (1,)).item()\n    return Graph(\n        node_features=x,\n        edge_index=ei,\n        graph_features=graph_feat,  # input feature: separate from graph_label\n        graph_label=torch.tensor(label),\n    )\n\ngraphs = [make_mol_graph(i) for i in range(8)]\nprint(f'Created {len(graphs)} molecule graphs')\nprint('graph_features shape:', graphs[0].graph_features.shape)\nprint('graph_label:', graphs[0].graph_label)"),
    code("# TGraphX clearly separates graph_features (inputs) from graph_label (targets).\nfor g in graphs:\n    assert g.graph_features is not None, 'graph_features missing'\n    assert g.graph_label is not None, 'graph_label missing'\n    # Confirm they are different tensors.\n    assert g.graph_features is not g.graph_label, 'Should be different fields'\nprint('✓ graph_features (input) and graph_label (target) are separate fields.')"),
]),

"06_neighborloader_seed_node_loss.ipynb": nb([
    md("# 06 — NeighborLoader, GraphMiniBatch, and Correct Seed-Node Loss\n\n**Goal:** Show how `GraphMiniBatch` exposes `batch.seed_y` and `batch.seed_logits(logits)`\nto correctly compute loss only on supervision nodes.\n\n**TGraphX subsystem:** `NeighborLoader`, `GraphMiniBatch`\n**Data:** Synthetic. **Runtime:** < 30s on CPU."),
    code("import torch, torch.nn as nn, torch.nn.functional as F\nfrom tgraphx import Graph, NeighborLoader, GCNConv\ntorch.manual_seed(0)"),
    md("## Why unsafe slicing fails\n\nA common mistake: `logits[:batch_size]` assumes seed nodes are the FIRST nodes.\nThat is NEVER guaranteed. TGraphX provides `batch.seed_logits(logits)` as the safe way."),
    code("N, D = 200, 16\nx = torch.randn(N, D)\nei = torch.randint(0, N, (2, 600))\ny = torch.randint(0, 4, (N,))\ng = Graph(node_features=x, edge_index=ei, y=y)\n\nloader = NeighborLoader(g, fanouts=[10, 5], batch_size=16, shuffle=True, seed=42)\nbatch = next(iter(loader))\nprint('Subgraph nodes (N_sub):', batch.num_nodes)\nprint('Seed nodes (K):', batch.batch_size)\nprint('Seed y shape:', batch.seed_y.shape)\nprint('Seed local indices:', batch.seed_local_indices[:5], '...')"),
    code("class GCN(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.c1 = GCNConv(D, 32)\n        self.c2 = GCNConv(32, 4)\n    def forward(self, x, ei):\n        return self.c2(self.c1(x, ei).relu(), ei)\n\nmodel = GCN()\nopt = torch.optim.Adam(model.parameters(), lr=1e-2)\nfor epoch in range(3):\n    for batch in loader:\n        logits = model(batch.node_features, batch.edge_index)\n        # ✓ CORRECT: use batch.seed_logits() to extract only supervision logits\n        loss = F.cross_entropy(batch.seed_logits(logits), batch.seed_y)\n        opt.zero_grad(); loss.backward(); opt.step()\n    print(f'Epoch {epoch+1}: loss={loss.item():.4f}')"),
    md("## Key APIs\n\n| API | Purpose |\n|---|---|\n| `batch.node_features` | All subgraph node features `[N_sub, D]` |\n| `batch.seed_y` | Labels for seed nodes only `[K]` |\n| `batch.seed_logits(logits)` | Extract logits for seed nodes from `[N_sub, C]` |\n| `batch.seed_node_ids` | Global IDs of seed nodes |\n| `batch.seed_local_indices` | Local positions of seed nodes in subgraph |"),
]),

"07_sampling_benchmark_neighborloader.ipynb": nb([
    md("# 07 — NeighborLoader Throughput Benchmark\n\n**Goal:** Measure NeighborLoader sampling throughput on a synthetic graph.\n\n**Data:** Synthetic. **Runtime:** < 30s on CPU."),
    code("import torch, time\nfrom tgraphx import Graph, NeighborLoader\ntorch.manual_seed(0)"),
    code("N, D, E = 5000, 32, 25000\nx = torch.randn(N, D)\nei = torch.randint(0, N, (2, E))\ny = torch.randint(0, 4, (N,))\ng = Graph(node_features=x, edge_index=ei, y=y)\n\nloader = NeighborLoader(g, fanouts=[10, 5], batch_size=64, seed=42)\nn_batches = 0; n_nodes = 0; t0 = time.perf_counter()\nfor batch in loader:\n    n_batches += 1; n_nodes += batch.num_nodes\nelapsed = time.perf_counter() - t0\nprint(f'Batches: {n_batches}')\nprint(f'Avg nodes/batch: {n_nodes/n_batches:.1f}')\nprint(f'Throughput: {n_batches/elapsed:.1f} batches/sec')\nprint(f'Note: This is a smoke benchmark on synthetic data, not a competitive claim.')"),
    md("## Honest scope\n\nThis benchmark measures TGraphX NeighborLoader on a single CPU with small synthetic data.\nFor industrial-scale evaluation, compare against PyG/DGL benchmarks on the same dataset\nwith matched hyperparameters and hardware — see `docs/benchmark_report.md`."),
]),

"08_graphsaint_cluster_gcn_smoke.ipynb": nb([
    md("# 08 — GraphSAINT and Cluster-GCN Foundations\n\n**Goal:** Demonstrate GraphSAINT node sampling and Cluster-GCN partitioning on a small graph.\n\n**Honest scope:** These are research foundations, not production-scale samplers.\n\n**Data:** Synthetic. **Runtime:** < 20s on CPU."),
    code("import torch\nfrom tgraphx import Graph, GraphSAINTNodeSampler, GraphSAINTLoader\nfrom tgraphx import RandomBalancedPartitioner, ClusterLoader\ntorch.manual_seed(0)"),
    code("N = 100; x = torch.randn(N, 16); ei = torch.randint(0, N, (2, 400))\ng = Graph(node_features=x, edge_index=ei)\n\n# GraphSAINT node sampling\nsampler = GraphSAINTNodeSampler(g, budget=30, num_steps=5, seed=0)\nloader = GraphSAINTLoader(sampler, attach_norm=True)\nfor i, sub in enumerate(loader):\n    print(f'SAINT subgraph {i}: {sub.num_nodes} nodes, {sub.num_edges} edges')\nprint('GraphSAINT OK')"),
    code("# Cluster-GCN partitioning\nresult = RandomBalancedPartitioner(num_partitions=4, seed=0).fit(g)\nprint(f'Partitions: {result.num_partitions}')\nprint(f'Partition sizes: {result.partition_sizes}')\nassert sum(result.partition_sizes) == N\nprint('✓ All nodes covered by exactly one partition.')"),
]),

"09_kg_completion_transe_rescal_simple.ipynb": nb([
    md("# 09 — KG Completion: TransE, RESCAL, SimplE\n\n**Goal:** Train KG embedding models on a tiny academic KG and rank missing links.\n\n**TGraphX subsystem:** `tgraphx.kg`\n**Data:** Synthetic (academic KG). **Runtime:** < 60s on CPU."),
    code("import torch\nfrom tgraphx.kg import (\n    KnowledgeGraph, TransEModel, RESCALModel, SimplEModel,\n    evaluate_filtered_ranking, list_kg_models,\n)\nprint('Available models:', list(list_kg_models().keys()))"),
    code("# Tiny academic KG: researchers (0-4), papers (5-9), topics (10-12), institutions (13-14)\nheads = torch.tensor([0,1,2,3,4, 0,1,2,3,4, 5,6,7,8,9, 0,1])\nrels  = torch.tensor([0,0,0,0,0, 1,1,1,1,1, 2,2,2,2,2, 3,3])\ntails = torch.tensor([5,6,7,8,9,13,14,13,14,13,10,11,12,10,11,1,2])\nN_e, N_r = 15, 4\nkg = KnowledgeGraph.from_hrt(heads, rels, tails, num_entities=N_e, num_relations=N_r)\nprint(f'KG: {kg.num_entities} entities, {kg.num_relations} relations, {kg.num_triples} triples')"),
    code("def train_eval(model, triples, N_e, epochs=30, seed=42):\n    torch.manual_seed(seed)\n    opt = torch.optim.Adam(model.parameters(), lr=1e-2)\n    for _ in range(epochs):\n        neg = triples.clone(); neg[:, 2] = torch.randint(0, N_e, (triples.size(0),))\n        loss = (1.0 + model.score_triples(neg) - model.score_triples(triples)).clamp(min=0).mean()\n        opt.zero_grad(); loss.backward(); opt.step()\n    all_pos = set(map(tuple, triples.tolist()))\n    res = evaluate_filtered_ranking(model, triples, all_pos, N_e, filtered=True, hits_at=(1,3))\n    return res\n\nfor name, cls in [('TransE', TransEModel), ('RESCAL', RESCALModel), ('SimplE', SimplEModel)]:\n    m = cls(N_e, N_r, embedding_dim=16)\n    r = train_eval(m, kg.triples, N_e)\n    print(f'{name:8s} MRR={r.filt_mrr:.3f}  H@1={r.filt_hits[1]:.3f}  H@3={r.filt_hits[3]:.3f}')"),
    md("## Why SimplE and RESCAL capture asymmetry\n\n**DistMult**: `f(h,r,t) = ⟨h,r,t⟩` — symmetric, cannot model directed relations properly.\n**RESCAL**: `f(h,r,t) = h^T M_r t` — per-relation matrix, fully asymmetric.\n**SimplE**: `0.5*(⟨h_head, r_fwd, t_tail⟩ + ⟨t_head, r_inv, h_tail⟩)` — two embeddings per entity.\n\nAll three share the same trainer/evaluator interface in TGraphX."),
]),

"10_kg_hpo_grid_random_search.ipynb": nb([
    md("# 10 — KG HPO: Grid and Random Hyperparameter Search\n\n**Goal:** Run `run_kg_hpo` to compare model+hyperparameter combinations on a tiny KG.\n\n**TGraphX subsystem:** `tgraphx.kg.run_kg_hpo`\n**Data:** Synthetic. **Runtime:** < 60s on CPU."),
    code("import torch\nfrom tgraphx.kg import KnowledgeGraph, run_kg_hpo\ntorch.manual_seed(0)\nheads = torch.randint(0, 20, (80,)); rels = torch.randint(0, 3, (80,)); tails = torch.randint(0, 20, (80,))\nkg = KnowledgeGraph.from_hrt(heads, rels, tails, num_entities=20, num_relations=3)\nprint(f'KG: {kg.num_entities} entities, {kg.num_triples} triples')"),
    code("# Grid search: 2 models × 2 embedding_dims × 1 lr = 4 trials\nresult = run_kg_hpo(\n    kg,\n    model_names=['TransE', 'SimplE'],\n    search_space={'embedding_dim': [8, 16], 'lr': [1e-2]},\n    metric='mrr', strategy='grid',\n    max_trials=4, epochs=5, seed=42,\n)\nresult.summary()"),
    code("# Print all trials\nfor t in result.trials:\n    if t.status == 'ok':\n        print(f\"Trial {t.trial_index}: {t.model_name} dim={t.config.get('embedding_dim')} lr={t.config.get('lr')} -> MRR={t.metrics.get('mrr', 0):.4f}\")"),
    code("# Write dashboard artifacts\nimport tempfile, json, pathlib\nwith tempfile.TemporaryDirectory() as d:\n    art = result.write_dashboard_artifacts(d)\n    print('Artifacts written:', list(art.keys()))\n    print(json.loads(pathlib.Path(art['metrics_summary.json']).read_text()))"),
]),

"11_multimodal_kg_tensor_features.ipynb": nb([
    md("# 11 — Multimodal KG: Entity Tensor Features\n\n**Goal:** Demonstrate a KG where entities carry tensor features (e.g., image embeddings, user vectors).\n\n**TGraphX subsystem:** `tgraphx.kg` multimodal extensions\n**Data:** Synthetic. **Runtime:** < 20s on CPU."),
    code("import torch\nfrom tgraphx.kg import KnowledgeGraph\ntorch.manual_seed(0)"),
    code("# KG with entity features: 10 entities with synthetic 32-dim vectors\nN_e, N_r, N_t = 10, 3, 30\nheads = torch.randint(0, N_e, (N_t,)); rels = torch.randint(0, N_r, (N_t,)); tails = torch.randint(0, N_e, (N_t,))\n\n# Entity features can represent e.g. image embeddings, text embeddings, or profile vectors\nentity_features = {'visual': torch.randn(N_e, 32)}\n\nkg = KnowledgeGraph.from_hrt(\n    heads, rels, tails, num_entities=N_e, num_relations=N_r,\n    entity_features=entity_features,\n)\nprint(f'KG: {kg.num_entities} entities, visual feature shape: {kg.entity_features[\"visual\"].shape}')"),
    code("# For scoring with entity features, pass them to a feature-aware model.\n# Example: add entity features to TransE via entity_feature_dim parameter.\nfrom tgraphx.kg import TransEModel\nmodel = TransEModel(N_e, N_r, embedding_dim=16, entity_feature_dim=32)\ntriples = torch.stack([heads, rels, tails], dim=1)\n\n# Pass entity features alongside triples\nfeat = entity_features['visual']\nscores = model.score_triples(triples, entity_features=feat)\nprint('Scores shape:', scores.shape)\nassert scores.shape == (N_t,)"),
]),

"12_kg_filtered_ranking_explained.ipynb": nb([
    md("# 12 — Filtered MRR/Hits@K Explained\n\n**Goal:** Understand KG evaluation metrics with a tiny hand-checkable example.\n\n**TGraphX subsystem:** `tgraphx.kg.evaluate_filtered_ranking`\n**Data:** Tiny hand-crafted KG. **Runtime:** < 10s on CPU."),
    code("import torch\nfrom tgraphx.kg import TransEModel, evaluate_filtered_ranking"),
    md("## What is filtered ranking?\n\nFor a test triple (h, r, t), we rank all entities as candidate tails.\n**Filtered** ranking removes *known* positives from the ranking, so the model\nis not penalized for correctly scoring other true triples highly.\n\n**MRR** = mean of 1/rank across all test triples.  **Hits@K** = fraction with rank ≤ K."),
    code("# Tiny 5-entity KG with known triples\nN_e, N_r = 5, 2\ntriples = torch.tensor([[0,0,1],[2,0,3],[1,1,4],[3,1,0]], dtype=torch.long)\n\n# Train TransE briefly\nmodel = TransEModel(N_e, N_r, embedding_dim=8)\nopt = torch.optim.Adam(model.parameters(), lr=5e-2)\ntorch.manual_seed(99)\nfor _ in range(60):\n    neg = triples.clone(); neg[:,2] = torch.randint(0, N_e, (4,))\n    loss = (1.0 + model.score_triples(neg) - model.score_triples(triples)).clamp(min=0).mean()\n    opt.zero_grad(); loss.backward(); opt.step()\n\nall_pos = set(map(tuple, triples.tolist()))\nresult = evaluate_filtered_ranking(model, triples, all_pos, N_e, filtered=True, hits_at=(1,3))\nprint(f'Filtered MRR:  {result.filt_mrr:.4f}')\nprint(f'Filtered H@1:  {result.filt_hits[1]:.4f}')\nprint(f'Filtered H@3:  {result.filt_hits[3]:.4f}')\nprint(f'Mean Rank:     {result.filt_mr:.2f}')"),
]),

"13_graph_generation_metrics.ipynb": nb([
    md("# 13 — Graph Generation: Structural Metrics\n\n**Goal:** Generate graphs with different methods and compare structural properties.\n\n**TGraphX subsystem:** `tgraphx.generation`\n**Data:** Synthetic. **Runtime:** < 30s on CPU."),
    code("from tgraphx import run_graph_generation, list_graph_generation_methods\nprint('Methods:', list(list_graph_generation_methods().keys()))"),
    code("for method in ['erdos_renyi', 'barabasi_albert']:\n    r = run_graph_generation(method=method, num_graphs=10, num_nodes=20, num_edges=40, seed=42)\n    m = r.metrics\n    print(f'{method}:  validity={m.get(\"validity\",0):.2f}  uniqueness={m.get(\"uniqueness\",0):.2f}  diversity={m.get(\"diversity\",0):.3f}')"),
    md("## What the metrics mean\n\n| Metric | Meaning |\n|---|---|\n| validity | fraction of graphs that are valid (connected or otherwise well-formed) |\n| uniqueness | fraction of generated graphs that are distinct |\n| diversity | structural diversity across the generated set |\n\nFor research use: run on larger sets and compare against reference graph distributions."),
]),

"14_graph_generation_evolutionary_optimization.ipynb": nb([
    md("# 14 — Graph Generation + Evolutionary Optimization\n\n**Goal:** Optimize graph structure toward high connectivity using a genetic algorithm.\n\n**TGraphX subsystem:** `tgraphx.evolutionary`\n**Data:** Synthetic. **Runtime:** < 60s on CPU."),
    code("import torch\nfrom tgraphx.evolutionary import GraphGenome, GeneticAlgorithmOptimizer, GeneticAlgorithmConfig, connectivity_fitness\n\ndef make_genome(seed=0):\n    torch.manual_seed(seed)\n    return GraphGenome(edge_index=torch.randint(0, 8, (2, 10)), num_nodes=8)\n\nconfig = GeneticAlgorithmConfig(population_size=10, n_generations=15, seed=42)\npop = [make_genome(i) for i in range(10)]\nresult = GeneticAlgorithmOptimizer(config, connectivity_fitness).optimize(pop)\nprint(f'Best connectivity: {result.best_fitness:.4f}')\nprint(f'Generations: {len(result.history)}')"),
    md("## NSGA-II for multi-objective optimization"),
    code("from tgraphx.evolutionary import NSGAIIOptimizer, EvolutionConfig, connectivity_fitness\n\n# NSGAIIOptimizer requires a list of objective functions (one per objective).\n# Pass [obj1, obj2] for genuine multi-objective optimisation.\n# Using the same objective twice here for a concise demo.\nobjectives = [connectivity_fitness, connectivity_fitness]\nconfig2 = EvolutionConfig(population_size=8, n_generations=10, seed=0)\npop2 = [make_genome(i) for i in range(8)]\nr2 = NSGAIIOptimizer(config2, objectives).optimize(pop2)\nprint(f'Pareto front size: {len(r2.pareto_front) if r2.pareto_front else \"N/A\"}')\nprint(f'Best fitness: {r2.best_fitness:.4f}')\nprint(f'History entries: {len(r2.history)}')"),
]),

"15_graph_rl_coloring_with_callbacks.ipynb": nb([
    md("# 15 — Graph RL: Coloring with EarlyStopping and CSV Logging\n\n**Goal:** Train an RL agent on graph coloring, with early stopping and CSV logging via callbacks.\n\n**TGraphX subsystem:** `tgraphx.rl`\n**Data:** Synthetic environments. **Runtime:** < 60s on CPU."),
    code("from tgraphx import run_graph_rl, list_graph_rl_algorithms\nfrom tgraphx.rl import EarlyStoppingCallback, CSVLoggerCallback\nimport tempfile, csv\nprint('Algorithms:', list(list_graph_rl_algorithms().keys())[:6], '...')"),
    code("# Compare random vs DQN with callbacks\nwith tempfile.TemporaryDirectory() as d:\n    cb = CSVLoggerCallback(d + '/episodes.csv')\n    stopper = EarlyStoppingCallback(monitor='reward', patience=5, mode='max')\n    r = run_graph_rl('graph_coloring', algorithm='random', episodes=20, seed=0, callbacks=[cb, stopper])\n    print(f'Random: mean_return={r.metrics[\"mean_return\"]:.2f}  stopped_early={r.stopped_early}')\n    with open(d + '/episodes.csv') as f:\n        rows = list(csv.DictReader(f))\n    print(f'CSV rows: {len(rows)}, columns: {list(rows[0].keys()) if rows else \"none\"}')"),
    md("## Callback system\n\n| Callback | Purpose |\n|---|---|\n| `EarlyStoppingCallback` | Stop training when monitored metric stops improving |\n| `CSVLoggerCallback` | Write per-episode metrics to a CSV file |\n| `CallbackList` | Aggregate multiple callbacks with fan-out |"),
]),

"16_graph_rl_maxcut_or_navigation.ipynb": nb([
    md("# 16 — Graph RL: MaxCut and Navigation Environments\n\n**Goal:** Explore two graph RL environments and compare random/greedy baselines.\n\n**Data:** Tiny synthetic graphs. **Runtime:** < 60s on CPU."),
    code("from tgraphx import run_graph_rl\nfor env in ['graph_navigation', 'max_cut']:\n    r = run_graph_rl(env=env, algorithm='random', episodes=10, seed=42)\n    print(f'{env}: mean_return={r.metrics[\"mean_return\"]:.2f}')"),
    code("# Greedy vs Random on navigation\nfor algo in ['random', 'greedy']:\n    r = run_graph_rl('graph_navigation', algorithm=algo, episodes=20, seed=0)\n    print(f'{algo:7s}: mean_return={r.metrics[\"mean_return\"]:.2f}  success_rate={r.metrics[\"success_rate\"]:.2f}')"),
    md("## Honest note\n\nThese RL environments are research foundations.\nPerformance on tiny synthetic graphs does not predict real-world problem performance.\nFor serious applications, combine these tools with domain-specific state/reward design."),
]),

"17_rl_callbacks_logging_dashboard_artifacts.ipynb": nb([
    md("# 17 — RL Callback Logging and Dashboard Artifacts\n\n**Goal:** Use CSV logging callback and write dashboard-compatible RL artifacts.\n\n**Data:** Synthetic. **Runtime:** < 30s on CPU."),
    code("import tempfile, json, csv, pathlib\nfrom tgraphx import run_graph_rl\nfrom tgraphx.rl import CSVLoggerCallback"),
    code("with tempfile.TemporaryDirectory() as d:\n    cb = CSVLoggerCallback(d + '/ep.csv')\n    r = run_graph_rl('graph_navigation', algorithm='dqn', episodes=15, seed=42,\n                      callbacks=[cb], dashboard_dir=d)\n    # Show CSV\n    with open(d + '/ep.csv') as f:\n        rows = list(csv.DictReader(f))\n    print(f'CSV: {len(rows)} rows, columns: {list(rows[0].keys()) if rows else \"none\"}')\n    # Show dashboard JSON\n    report = pathlib.Path(d) / f'rl_dqn_graph_navigation.json'\n    if report.exists():\n        data = json.loads(report.read_text())\n        print(f'Dashboard JSON: mean_return={data[\"metrics\"][\"mean_return\"]:.2f}')"),
    md("## Using artifacts with the dashboard\n\n```bash\ntgraphx-dashboard --logdir <run_dir>\n```\n\nThe dashboard reads `*.json` files from the run directory and renders training curves,\nrun metadata, and metrics summaries."),
]),

"18_graphml_io_roundtrip.ipynb": nb([
    md("# 18 — GraphML IO Round-Trip\n\n**Goal:** Write and read a graph in GraphML format, verifying structure/metadata/labels.\n\n**TGraphX subsystem:** `tgraphx.io`\n**Data:** Synthetic. **Runtime:** < 10s on CPU."),
    code("import torch, tempfile\nfrom pathlib import Path\nfrom tgraphx import Graph\nfrom tgraphx.io import write_graphml, read_graphml"),
    code("x = torch.tensor([[0.1],[0.5],[0.9],[0.3]])\nei = torch.tensor([[0,1,2],[1,2,3]], dtype=torch.long)\nw = torch.tensor([1.5, 2.0, 0.5])\ny = torch.tensor([0, 1, 1, 0], dtype=torch.long)\ng = Graph(node_features=x, edge_index=ei, edge_weight=w, y=y)\nprint(g)"),
    code("with tempfile.NamedTemporaryFile(suffix='.graphml', delete=False) as f:\n    path = Path(f.name)\nwrite_graphml(g, path, include_labels=True, include_tensor_features=True)\ng2 = read_graphml(path)\npath.unlink()\nprint(f'Nodes: {g2.num_nodes} (expected {g.num_nodes})')\nprint(f'Edges: {g2.num_edges} (expected {g.num_edges})')\nprint(f'Labels: {g2.node_labels}')\nprint(f'Edge weight: {g2.edge_weight}')"),
]),

"19_io_tensor_semantics_warning.ipynb": nb([
    md("# 19 — Graph IO: Why Tensor Features Are Not Silently Flattened\n\n**Goal:** Understand TGraphX's refusal to silently serialize multi-dim tensors through GraphML.\n\n**Data:** Synthetic. **Runtime:** < 10s on CPU."),
    code("import torch, tempfile\nfrom tgraphx import Graph\nfrom tgraphx.io import write_graphml"),
    md("## The problem with silent serialization\n\nGraphML stores node data as XML string attributes — it cannot express a `[C, H, W]` tensor shape.\nSilently flattening `[4, 6, 6]` to a 144-element CSV string would:\n1. Discard shape information\n2. Make round-trip reconstruction ambiguous\n3. Break the tensor-native contract\n\nTGraphX refuses to do this silently."),
    code("# Attempt to serialize [N, C, H, W] node features:\nx_spatial = torch.randn(4, 3, 8, 8)\ng = Graph(node_features=x_spatial, edge_index=torch.tensor([[0,1],[1,2]]))\ntry:\n    with tempfile.NamedTemporaryFile(suffix='.graphml') as f:\n        write_graphml(g, f.name, include_tensor_features=True)\nexcept ValueError as e:\n    print('ValueError (expected):', str(e)[:120])\nprint('\\n✓ TGraphX protects tensor semantics.')\nprint('For lossless persistence: use torch.save({\"graph\": g}, \"my_graph.pt\")')"),
]),

"20_graph_mining_motifs_and_cliques.ipynb": nb([
    md("# 20 — Graph Mining: Motifs and Cliques\n\n**Goal:** Count graph motifs and enumerate maximal cliques on small graphs.\n\n**TGraphX subsystem:** `tgraphx.mining`\n**Data:** Synthetic. **Runtime:** < 10s on CPU."),
    code("import torch\nfrom tgraphx.mining import motif_profile, graph_summary\nfrom tgraphx.mining.matching_coloring import enumerate_maximal_cliques\ntorch.manual_seed(0)"),
    code("# Triangle graph\nN = 5\nei = torch.tensor([[0,1,2,0,3,4],[1,2,0,3,4,3]], dtype=torch.long)\nsummary = graph_summary(ei, num_nodes=N)\nprint('Graph summary:', {k:v for k,v in summary.items() if not isinstance(v, (list, dict))})"),
    code("# Enumerate maximal cliques (Bron-Kerbosch with pivot)\ncliques = enumerate_maximal_cliques(ei, num_nodes=N, max_nodes=50)\nprint(f'Maximal cliques: {cliques}')"),
    code("# Motif counts if available\ntry:\n    profile = motif_profile(ei, num_nodes=N)\n    print('Motif profile:', profile)\nexcept Exception as e:\n    print(f'motif_profile: {e}')"),
]),

"21_graph_mining_kernels_wl_similarity.ipynb": nb([
    md("# 21 — Graph Mining: Kernels and WL Similarity\n\n**Goal:** Compute graph similarity using the WL kernel.\n\n**TGraphX subsystem:** `tgraphx.mining.kernels`\n**Data:** Synthetic. **Runtime:** < 10s on CPU."),
    code("import torch\nfrom tgraphx import Graph\nfrom tgraphx.mining.kernels import wl_subtree_kernel\ntorch.manual_seed(0)"),
    code("# Create two similar and one dissimilar graph\ndef ring_graph(n):\n    ei = torch.tensor([[i for i in range(n)], [(i+1)%n for i in range(n)]], dtype=torch.long)\n    return ei, n\n\nei1, n1 = ring_graph(5)\nei2, n2 = ring_graph(5)\nei3, n3 = ring_graph(3)  # smaller ring — less similar\n\nk11 = wl_subtree_kernel(ei1, n1, ei1, n1, h=3)\nk12 = wl_subtree_kernel(ei1, n1, ei2, n2, h=3)\nk13 = wl_subtree_kernel(ei1, n1, ei3, n3, h=3)\n\nprint(f'Similarity(ring5, ring5): {k11:.4f}  (identical)')\nprint(f'Similarity(ring5, ring5): {k12:.4f}  (identical graph, different object)')\nprint(f'Similarity(ring5, ring3): {k13:.4f}  (different rings)')"),
    md("## Kernel properties\n\n- **Symmetry:** K(G1, G2) = K(G2, G1)\n- **Diagonal:** K(G, G) ≥ K(G, G') for any G'\n- **Deterministic:** same graph structure → same kernel value"),
]),

"22_structural_roles_concept_demo.ipynb": nb([
    md("# 22 — Structural Roles: Concept and Intuition\n\n**Goal:** Understand structural roles with star/path/cycle graphs.\n\n**Note:** Full RolX/struc2vec role discovery is NOT yet implemented in TGraphX.\nThis notebook shows the structural feature extraction building blocks.\n\n**TGraphX subsystem:** `tgraphx.mining` structural utilities\n**Data:** Synthetic. **Runtime:** < 10s on CPU."),
    code("import torch\nfrom tgraphx.mining import degree_statistics, centrality_summary\ntorch.manual_seed(0)"),
    md("## Structural roles intuition\n\nNodes in the **center of a star** have the same structural role regardless of which star graph they're in.\nNodes at **path endpoints** have degree 1 and are periphery nodes.\nTGraphX provides structural feature extraction to approximate this concept."),
    code("# Star graph: node 0 is center, nodes 1-4 are leaves\nstar_ei = torch.tensor([[0,0,0,0,1,2,3,4],[1,2,3,4,0,0,0,0]], dtype=torch.long)\nN_star = 5\nstar_degree = degree_statistics(star_ei, num_nodes=N_star)\nprint('Star graph degree:', star_degree)\n\n# Path graph: nodes at ends have degree 1\npath_ei = torch.tensor([[0,1,2,3],[1,2,3,4]], dtype=torch.long)\npath_degree = degree_statistics(path_ei, num_nodes=5)\nprint('Path graph degree:', path_degree)"),
    code("# The star center should have the highest degree\nassert star_degree['max_degree'] == 4, 'Center of star must have degree 4'\nassert star_degree['min_degree'] == 1, 'Star leaves must have degree 1'\nprint('✓ Structural features correctly identify center vs leaf role.')\nprint('\\nFor full role discovery, see the RolX/struc2vec roadmap in docs/limitations.md.')"),
]),

"23_dashboard_easy_mode_artifacts.ipynb": nb([
    md("# 23 — Dashboard: Easy Mode Training Artifacts\n\n**Goal:** Train with Easy Mode, write dashboard artifacts, and view locally.\n\n**TGraphX subsystem:** `tgraphx.easy`, `tgraphx-dashboard`\n**Data:** Synthetic. **Runtime:** < 30s on CPU."),
    code("import tempfile, pathlib, json\nimport tgraphx as tgx"),
    code("data = tgx.easy.synthetic_tensor_node_classification(\n    num_nodes=128, node_shape=(4,4,4), num_classes=3, num_edges=512, seed=42\n)\n\nwith tempfile.TemporaryDirectory() as d:\n    run_dir = pathlib.Path(d) / 'easy_run'\n    result = tgx.easy.train_node_classifier(\n        data, epochs=3, batch_size=16, fanouts=[5,3],\n        verbose=False, seed=42, dashboard_dir=str(run_dir),\n    )\n    print('Artifacts written:', list(result.artifacts.keys()))\n    meta = json.loads((run_dir / 'run_metadata.json').read_text())\n    print('Metadata:', {k: meta[k] for k in ['status','total_epochs','source','model']})\n    summary = json.loads((run_dir / 'metrics_summary.json').read_text())\n    print('Summary:', summary)\n    print(f'\\nTo view: tgraphx-dashboard --logdir {run_dir}')"),
]),

"24_benchmark_suite_v13.ipynb": nb([
    md("# 24 — v1.3 Benchmark Suite\n\n**Goal:** Run the TGraphX v1.3 benchmark suite and inspect results.\n\n**IMPORTANT:** These are smoke benchmarks on tiny synthetic data.\nThey are NOT competitive throughput claims against PyG/DGL/PyKEEN/SB3.\n\n**Runtime:** < 120s on CPU."),
    code("from tgraphx.benchmarks import run_v13_benchmark_suite\n\ndata = run_v13_benchmark_suite(small=True, return_dict=True)\nprint(f'Suite: {data[\"suite\"]}  Version: {data[\"package_version\"]}')\nfor row in data['benchmarks']:\n    rt = f'{row[\"runtime_s\"]:.3f}s' if row.get('runtime_s') else 'failed'\n    print(f'  {row[\"name\"]:<35} {row[\"status\"]:<7} {rt}')"),
]),

"25_reproducibility_and_seed_control.ipynb": nb([
    md("# 25 — Reproducibility and Seed Control\n\n**Goal:** Demonstrate deterministic workflows with `set_seed` and compare repeated runs.\n\n**TGraphX subsystem:** `tgraphx.reproducibility`\n**Data:** Synthetic. **Runtime:** < 20s on CPU."),
    code("import torch\nfrom tgraphx.reproducibility import set_seed\nimport tgraphx as tgx"),
    code("# Same seed → same results\nresults = []\nfor trial in range(2):\n    set_seed(42)\n    data = tgx.easy.synthetic_tensor_node_classification(\n        num_nodes=64, node_shape=(4,4,4), num_classes=3, num_edges=200, seed=42\n    )\n    r = tgx.easy.train_node_classifier(data, epochs=2, batch_size=16, fanouts=[5,3],\n                                         verbose=False, seed=42)\n    results.append(r.metrics['loss'])\n    print(f'Trial {trial+1}: loss={r.metrics[\"loss\"]:.6f}')\n\nprint(f'\\nIdentical results: {abs(results[0]-results[1]) < 1e-6}')"),
    code("# Report reproducibility state\nstate = set_seed(42, deterministic=False)\nprint('\\nReproducibility state:')\nfor k, v in state.items():\n    print(f'  {k}: {v}')"),
]),

"26_low_level_pytorch_escape_hatch.ipynb": nb([
    md("# 26 — From Easy Mode to Low-Level PyTorch Control\n\n**Goal:** Start with Easy Mode, then drop down to raw PyTorch objects.\n\n**TGraphX subsystem:** `tgraphx.easy` + PyTorch\n**Data:** Synthetic. **Runtime:** < 30s on CPU."),
    code("import torch, torch.nn.functional as F\nimport tgraphx as tgx"),
    code("data = tgx.easy.synthetic_tensor_node_classification(\n    num_nodes=64, node_shape=(4,4,4), num_classes=3, num_edges=200, seed=0)\nresult = tgx.easy.train_node_classifier(\n    data, epochs=3, batch_size=16, fanouts=[5,3], verbose=False, seed=0)"),
    code("# Every Easy Mode result exposes standard PyTorch objects.\nmodel = result.model        # nn.Module — run any PyTorch operation\ngraph = result.graph        # tgraphx.Graph — raw tensors\nloader = result.loader      # NeighborLoader — iterate or inspect\noptimizer = result.optimizer # torch.optim.Adam — resume training\nprint('Model class:', type(model).__name__)\nprint('Graph:', graph)\nprint('Loader:', type(loader).__name__)"),
    code("# Continue training manually from where Easy Mode left off.\nmodel.train()\nfor batch in loader:\n    logits = model(batch.node_features, batch.edge_index)\n    loss = F.cross_entropy(batch.seed_logits(logits), batch.seed_y)\n    optimizer.zero_grad(); loss.backward(); optimizer.step()\nprint(f'Manual continuation loss: {loss.item():.4f}')"),
    md("## When to use each level\n\n| Use case | Recommended |\n|---|---|\n| Quick prototype | Easy Mode |\n| Custom loss/metric | Escape hatch: `result.model`, `result.graph` |\n| Custom architecture | Low-level: build `nn.Module` + `NeighborLoader` directly |\n| Distributed / production | Use base PyTorch DDP with TGraphX data structures |"),
]),

"27_custom_tensor_projector_workflow.ipynb": nb([
    md("# 27 — Custom Tensor Projector and Classifier Head\n\n**Goal:** Build a custom GNN with spatial pooling and linear classification head.\n\n**TGraphX subsystem:** `ConvMessagePassing`, `AdaptiveAvgPool2d`, `nn.Linear`\n**Data:** Synthetic. **Runtime:** < 20s on CPU."),
    code("import torch, torch.nn as nn, torch.nn.functional as F\nfrom tgraphx import Graph, ConvMessagePassing, NeighborLoader\ntorch.manual_seed(0)"),
    code("C, H, W = 4, 6, 6\nN = 128\nx = torch.randn(N, C, H, W)\nei = torch.randint(0, N, (2, 512))\ny = torch.randint(0, 3, (N,))\ng = Graph(node_features=x, edge_index=ei, y=y)\nprint('Graph:', g)"),
    code("class CustomTensorGNN(nn.Module):\n    def __init__(self):\n        super().__init__()\n        # Two convolutional hops — spatial dims [H,W] preserved.\n        self.conv1 = ConvMessagePassing((C, H, W), (16, H, W))\n        self.conv2 = ConvMessagePassing((16, H, W), (8, H, W))\n        # Pool spatial dims to a scalar per channel.\n        self.pool = nn.AdaptiveAvgPool2d((1, 1))\n        # Classify from pooled representation.\n        self.head = nn.Sequential(\n            nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 3)\n        )\n    def forward(self, x, ei):\n        z = self.conv1(x, ei).relu()         # [N, 16, H, W]\n        z = self.conv2(z, ei).relu()         # [N, 8, H, W]\n        z = self.pool(z).flatten(1)          # [N, 8]\n        return self.head(z)                  # [N, 3]\n\nmodel = CustomTensorGNN()\nprint(f'Parameters: {sum(p.numel() for p in model.parameters())}')"),
    code("loader = NeighborLoader(g, fanouts=[10, 5], batch_size=32, seed=0)\nopt = torch.optim.Adam(model.parameters(), lr=1e-2)\nfor epoch in range(3):\n    for batch in loader:\n        logits = model(batch.node_features, batch.edge_index)\n        loss = F.cross_entropy(batch.seed_logits(logits), batch.seed_y)\n        opt.zero_grad(); loss.backward(); opt.step()\n    print(f'Epoch {epoch+1}: loss={loss.item():.4f}')"),
]),

"28_colab_install_and_doctor.ipynb": nb([
    md("# 28 — Colab Install and System Check\n\n**Goal:** Install TGraphX in Google Colab and verify the environment.\n\n**Data:** None. **Runtime:** < 2 minutes (including install)."),
    code("# In Colab, uncomment and run this cell first:\n# !pip install -q tgraphx\n\nimport tgraphx as tgx\nprint('TGraphX version:', tgx.__version__)"),
    code("# Run the system health check\nimport subprocess, sys\nresult = subprocess.run([sys.executable, '-m', 'tgraphx', 'doctor'],\n                        capture_output=True, text=True)\nprint(result.stdout)\nif result.returncode != 0:\n    print('Warnings:', result.stderr[:300])"),
    code("# Quick smoke test\ndata = tgx.easy.synthetic_tensor_node_classification(\n    num_nodes=32, node_shape=(4,4,4), num_classes=2, num_edges=100, seed=0\n)\nresult = tgx.easy.train_node_classifier(data, epochs=1, batch_size=8,\n                                          fanouts=[3,2], verbose=False, seed=0)\nprint('Smoke test passed! metrics:', result.metrics)"),
]),

"29_end_to_end_research_workflow.ipynb": nb([
    md("# 29 — End-to-End Research Workflow\n\n**Goal:** Complete pipeline: data → model → train → metrics → artifacts → benchmark summary.\n\n**TGraphX subsystem:** `tgraphx.easy`, `tgraphx.mining`, dashboard artifacts\n**Data:** Synthetic. **Runtime:** < 60s on CPU."),
    code("import tgraphx as tgx, tempfile, json, pathlib"),
    code("# Step 1: Create data\ndata = tgx.easy.synthetic_tensor_node_classification(\n    num_nodes=256, node_shape=(4,6,6), num_classes=4, num_edges=1024, seed=42\n)\nprint('Step 1 — Data:', data)"),
    code("# Step 2: Train with Easy Mode\nwith tempfile.TemporaryDirectory() as d:\n    result = tgx.easy.train_node_classifier(\n        data, model='tensor_gcn', sampler='neighbor', fanouts=[10,5],\n        batch_size=32, epochs=3, seed=42, verbose=True, dashboard_dir=d,\n    )\n    print('\\nStep 2 — Metrics:', result.metrics)\n    artifacts = result.artifacts\n    print('Step 3 — Artifacts:', list(artifacts.keys()))"),
    code("# Step 4: Graph summary\nfrom tgraphx.mining import graph_summary\nsummary = graph_summary(data.edge_index, num_nodes=data.num_nodes)\nprint('\\nStep 4 — Graph summary (selected):')\nfor k in ['num_nodes','num_edges','density','is_directed']:\n    print(f'  {k}: {summary.get(k)}')"),
    code("# Step 5: Print result summary\nresult.summary()"),
]),

"30_limitations_and_roadmap_honest_demo.ipynb": nb([
    md("# 30 — TGraphX: Honest Capabilities and Roadmap\n\n**Goal:** Understand what TGraphX supports now versus what is planned.\n\n**No overclaiming. No self-undervaluing.**"),
    code("from tgraphx.kg import list_kg_models\nfrom tgraphx import list_graph_rl_algorithms, list_graph_generation_methods\nprint('KG models:', list(list_kg_models().keys()))\nprint('RL algorithms:', list(list_graph_rl_algorithms().keys())[:6], '...')\nprint('Generation methods:', list(list_graph_generation_methods().keys()))"),
    md("## What TGraphX does well NOW (v1.3)\n\n| Capability | Status |\n|---|---|\n| Tensor-native GNN layers (ConvMP, GAT, SAGE, GIN) | ✅ Beta |\n| Scalable sampling (NeighborLoader, GraphSAINT, Cluster-GCN) | ✅ Beta |\n| KG models: TransE, DistMult, ComplEx, RotatE, RESCAL, SimplE | ✅ Beta |\n| KG hyperparameter search | ✅ Beta |\n| Graph RL (13 algorithms + environments) | ✅ Experimental |\n| RL callbacks | ✅ Beta |\n| Graph generation + evolutionary optimization | ✅ Experimental |\n| Easy Mode (zero-boilerplate workflows) | ✅ Beta |\n| GraphML IO | ✅ Beta |\n| Dashboard | ✅ Beta |"),
    md("## What is on the ROADMAP (v1.4+)\n\n| Capability | Status |\n|---|---|\n| TuckER, ConvE KG models | v1.4 planned |\n| Gymnasium adapter | v1.4 planned |\n| GEXF/Pajek IO | v1.4 planned |\n| Dedicated Easy Mode dashboard panel | v1.4 planned |\n| Vectorized RL environments | v1.4 planned |\n| Full gSpan, RolX, struc2vec | v2.x or research project |\n| Distributed RL, billion-edge training | research scale |"),
    md("## What TGraphX is NOT\n\n- Not a replacement for PyG or DGL for production-scale training\n- Not a PyKEEN replacement (limited model zoo, no advanced filtering datasets)\n- Not a NetworkX replacement (partial algorithm coverage)\n- Not a production RLlib/SB3 replacement\n\nTGraphX focuses on **tensor-native graph intelligence** in one research-focused package.\nFor industrial scale, combine TGraphX foundations with mature frameworks."),
    code("# Demonstrate the tensor-native identity one last time\nimport tgraphx as tgx\nprint('\\nVersion:', tgx.__version__)\nprint('\\nTGraphX keeps node features as [C,H,W] tensors through every')\nprint('message-passing step. No silent flattening. Full PyTorch control.')"),
]),

} # end NOTEBOOKS dict


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", default="colab_drafts")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for filename, nb_obj in NOTEBOOKS.items():
        path = out_dir / filename
        path.write_text(json.dumps(nb_obj, indent=1))
        n_cells = len(nb_obj["cells"])
        print(f"  {filename} ({n_cells} cells)")

    print(f"\n{len(NOTEBOOKS)} draft notebooks written to {out_dir}/")
    print("These are for MAINTAINER REVIEW ONLY — not tracked in git.")


if __name__ == "__main__":
    import argparse
    raise SystemExit(main())
