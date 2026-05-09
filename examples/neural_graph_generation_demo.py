"""Neural Graph Generation Demo.

Shows VGAE, autoregressive, and transformer-based graph generators with one
tiny training pass each to confirm the forward/backward passes are live.

Stability: Experimental (v0.7.0+)
"""
import torch
import torch.nn.functional as F

from tgraphx.generation.neural import (
    VGAEGraphGenerator,
    AutoregressiveEdgeGenerator,
    GraphTransformerGenerator,
)
from tgraphx.mining.vgae import GCNEncoder

torch.manual_seed(42)

N = 8       # nodes per graph
FD = 16     # node feature dim
LD = 8      # latent dim
STEPS = 3   # gradient steps

print("=== Neural Graph Generation Demo ===")
print()

# ---------------------------------------------------------------------------
# 1. VGAE
# ---------------------------------------------------------------------------
print("--- VGAEGraphGenerator ---")
encoder = GCNEncoder(in_dim=FD, hidden_dim=32, out_dim=LD)
vgae = VGAEGraphGenerator(encoder=encoder, latent_dim=LD, max_nodes=N)
optimizer = torch.optim.Adam(vgae.parameters(), lr=1e-3)
nf = torch.randn(N, FD)
ei = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
adj_target = torch.zeros(N, N)
for s, d in zip(ei[0].tolist(), ei[1].tolist()):
    adj_target[s, d] = 1.0

for step in range(STEPS):
    vgae.train()
    optimizer.zero_grad()
    z, mu, logvar, adj_logits, _ = vgae(nf, ei)
    loss = vgae.reconstruction_loss(z, adj_target, mu, logvar)
    loss.backward()
    optimizer.step()
    print(f"  step {step+1}: loss={loss.item():.4f}")

g_vgae = vgae.sample_graph(n_nodes=N)
print(f"  Generated: {g_vgae.num_nodes} nodes, {int(g_vgae.edge_index.shape[1])} edge endpoints")
assert torch.isfinite(torch.tensor(float(loss.item())))

# ---------------------------------------------------------------------------
# 2. Autoregressive Edge Generator
# ---------------------------------------------------------------------------
print()
print("--- AutoregressiveEdgeGenerator ---")
ar = AutoregressiveEdgeGenerator(num_nodes=N, hidden_dim=32)
optimizer_ar = torch.optim.Adam(ar.parameters(), lr=1e-3)

# Build a training sequence: upper-triangular binary edge decisions
seq_len = N * (N - 1) // 2
edge_seq = torch.zeros(1, seq_len)  # [B=1, seq_len]
for step in range(STEPS):
    ar.train()
    optimizer_ar.zero_grad()
    logits = ar(edge_seq)  # [1, seq_len]
    loss_ar = F.binary_cross_entropy_with_logits(logits, edge_seq)
    loss_ar.backward()
    optimizer_ar.step()
    print(f"  step {step+1}: loss={loss_ar.item():.4f}")

g_ar = ar.sample(n_nodes=N)
print(f"  Generated: {g_ar.num_nodes} nodes, {int(g_ar.edge_index.shape[1])} edge endpoints")

# ---------------------------------------------------------------------------
# 3. Graph Transformer Generator
# ---------------------------------------------------------------------------
print()
print("--- GraphTransformerGenerator ---")
tfm = GraphTransformerGenerator(max_nodes=N, hidden_dim=32, num_heads=2)
optimizer_t = torch.optim.Adam(tfm.parameters(), lr=1e-3)

# Action tokens: [B, T] LongTensor of token IDs (1=ADD_EDGE, 2=STOP, 3+=edge slots)
T = min(seq_len, 8)
action_tokens = torch.ones(1, T, dtype=torch.long)  # all ADD_EDGE tokens

for step in range(STEPS):
    tfm.train()
    optimizer_t.zero_grad()
    logits_t = tfm(action_tokens)  # [1, T, vocab_size]
    # Supervise: predict next token = STOP (token 2)
    targets = torch.full((1, T), fill_value=2, dtype=torch.long)
    loss_t = F.cross_entropy(logits_t.view(-1, logits_t.shape[-1]), targets.view(-1))
    loss_t.backward()
    optimizer_t.step()
    print(f"  step {step+1}: loss={loss_t.item():.4f}")

g_tfm = tfm.sample(n_nodes=N)
print(f"  Generated: {g_tfm.num_nodes} nodes, {int(g_tfm.edge_index.shape[1])} edge endpoints")

print()
print("All neural generators: loss backward OK, generate/sample returns GeneratedGraph.")
print()
print("=== Done ===")
