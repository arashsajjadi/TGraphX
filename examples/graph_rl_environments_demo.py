"""Demonstrate all graph RL environments.

Shows: reset, step, action_mask, episodic run for all 5+ environments.
"""
import torch
from tgraphx.rl.environments import (
    GraphEnvConfig,
    GraphNavigationEnv,
    GraphColoringEnv,
    MaxCutEnv,
    VertexCoverEnv,
    GraphGenerationEnv,
    KGPathReasoningEnv,
)
from tgraphx.generation.actions import GraphActionSpace


def run_episode(env, max_steps=20, seed=42):
    """Run a random-policy episode and return total return."""
    env.reset(seed=seed)
    total_reward = 0.0
    for _ in range(max_steps):
        mask = env.valid_action_mask()
        valid = mask.nonzero(as_tuple=False).squeeze(1)
        if len(valid) == 0:
            break
        action = int(valid[torch.randint(len(valid), (1,)).item()].item())
        _, reward, done, _, _ = env.step(action)
        total_reward += reward
        if done:
            break
    return total_reward


def main():
    print("=== Graph RL Environments Demo ===\n")

    # Path graph: 0-1-2-3-4
    ei = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    n = 5
    nf = torch.randn(n, 4)

    # Navigation
    print("--- GraphNavigationEnv ---")
    config = GraphEnvConfig(max_steps=20, seed=42)
    nav_env = GraphNavigationEnv(ei, n, node_features=nf, target_node=4, config=config)
    obs = nav_env.reset()
    print(f"  Reset: current_node={obs['current_node']}, target={obs['target_node']}")
    print(f"  action_mask shape: {obs['action_mask'].shape}, dtype: {obs['action_mask'].dtype}")
    total = run_episode(nav_env)
    print(f"  Episode return (random policy): {total:.2f}")

    # Coloring
    print("\n--- GraphColoringEnv ---")
    tri_ei = torch.tensor([[0, 1, 1, 2, 0, 2], [1, 0, 2, 1, 2, 0]], dtype=torch.long)
    config = GraphEnvConfig(max_steps=10, seed=0)
    col_env = GraphColoringEnv(tri_ei, 3, num_colors=3, config=config)
    obs = col_env.reset()
    print(f"  Reset: current_node={obs['current_node']}")
    print(f"  action_mask (all valid): {obs['action_mask'].tolist()}")
    total = run_episode(col_env)
    print(f"  Episode return (random policy): {total:.2f}")

    # MaxCut
    print("\n--- MaxCutEnv ---")
    config = GraphEnvConfig(max_steps=10, seed=0)
    mc_env = MaxCutEnv(tri_ei, 3, config=config)
    obs = mc_env.reset()
    print(f"  Reset: action_mask={obs['action_mask'].tolist()}")
    total = run_episode(mc_env)
    print(f"  Episode return (random policy): {total:.2f}")

    # VertexCover
    print("\n--- VertexCoverEnv ---")
    config = GraphEnvConfig(max_steps=10, seed=0)
    vc_env = VertexCoverEnv(tri_ei, 3, config=config)
    obs = vc_env.reset()
    print(f"  Reset: cover={obs['cover'].tolist()}")
    total = run_episode(vc_env)
    print(f"  Episode return (random policy): {total:.2f}")

    # GraphGeneration
    print("\n--- GraphGenerationEnv ---")
    config = GraphEnvConfig(max_steps=20, seed=0)
    space = GraphActionSpace(max_nodes=5, max_edges=10)
    gen_env = GraphGenerationEnv(action_space_config=space, config=config)
    obs = gen_env.reset()
    print(f"  Reset: step={obs['step']}, action_mask sum={obs['action_mask'].sum().item()}")
    total = run_episode(gen_env)
    print(f"  Episode return (random policy): {total:.2f}")

    # KG Reasoning
    print("\n--- KGPathReasoningEnv ---")
    kg_ei = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    kg_rt = torch.tensor([0, 1, 0], dtype=torch.long)
    config = GraphEnvConfig(max_steps=10, seed=0)
    kg_env = KGPathReasoningEnv(
        kg_edge_index=kg_ei,
        relation_types=kg_rt,
        num_entities=4,
        num_relations=2,
        query_pairs=[(0, 3)],
        config=config,
    )
    obs = kg_env.reset()
    print(f"  Reset: current_entity={obs['current_entity']}, target={obs['target_entity']}")
    print(f"  action_mask: {obs['action_mask'].tolist()}")
    total = run_episode(kg_env)
    print(f"  Episode return (random policy): {total:.2f}")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
