"""Easy graph generation — minimal imports.

This example generates graphs using the TGraphX high-level generation API.

Usage::

    python examples/easy_graph_generation_no_torch.py
"""

import tgraphx as tgx

# Discovery: what generation methods are available?
methods = tgx.list_graph_generation_methods()
print("Available graph generation methods:")
for name, desc in (methods.items() if hasattr(methods, 'items') else enumerate(methods)):
    print(f"  {name}: {desc}")

# Run a classical graph generation workflow.
result = tgx.run_graph_generation(
    method="erdos_renyi",
    num_graphs=5,
    num_nodes=20,
    num_edges=40,
    seed=42,
)
print(f"\nGenerated {len(result.graphs)} graphs")
print(f"Metrics: {result.metrics}")
print(f"\nGraph generation PASSED")
