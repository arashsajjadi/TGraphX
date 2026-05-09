"""Easy KG training — minimal imports required.

This example trains a TransE model on a synthetic knowledge graph using
the TGraphX easy-mode API.

Usage::

    python examples/easy_kg_training_no_torch.py
"""

import tgraphx as tgx

# Discovery: what KG models are available?
print("Available KG models:")
from tgraphx.kg import list_kg_models
for name, desc in list_kg_models().items():
    print(f"  {name}: {desc}")

# Create a synthetic KG.
from tgraphx.kg import KnowledgeGraph, TransEModel, KGTrainer, KGTrainingConfig
import torch

torch.manual_seed(42)
N_e, N_r, N_t = 20, 5, 100
triples = torch.stack([
    torch.randint(0, N_e, (N_t,)),
    torch.randint(0, N_r, (N_t,)),
    torch.randint(0, N_e, (N_t,)),
], dim=1)
kg = KnowledgeGraph(triples, num_entities=N_e, num_relations=N_r)
print(f"\nKG: {kg.num_entities} entities, {kg.num_relations} relations, {kg.num_triples} triples")

# Train TransE.
model = TransEModel(num_entities=N_e, num_relations=N_r, embedding_dim=32)
config = KGTrainingConfig(num_epochs=3, batch_size=32, lr=1e-3, seed=42)
trainer = KGTrainer(model, config, kg.triples)
trainer.train()

print("\nKG training PASSED")
