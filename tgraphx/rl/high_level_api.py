"""High-level one-line API for graph RL.

Stability: Beta (v0.7.0+) — stable API contract.

Usage:
    from tgraphx.rl import run_graph_rl, list_graph_rl_algorithms
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

__all__ = [
    "run_graph_rl",
    "make_graph_env",
    "make_graph_policy",
    "list_graph_rl_algorithms",
    "RLResult",
]


# ---------------------------------------------------------------------------
# Algorithm registry
# ---------------------------------------------------------------------------

_DISCRETE_ALGORITHMS: Dict[str, Dict[str, str]] = {
    "random": {
        "action_type": "discrete",
        "stability": "Beta",
        "description": "Uniform random sampling of valid actions.",
    },
    "greedy": {
        "action_type": "discrete",
        "stability": "Beta",
        "description": "Select highest Q-value action (no learning).",
    },
    "reinforce": {
        "action_type": "discrete",
        "stability": "Experimental",
        "description": "Monte Carlo policy gradient with optional baseline.",
    },
    "actor_critic": {
        "action_type": "discrete",
        "stability": "Experimental",
        "description": "Synchronous actor-critic with GAE.",
    },
    "a2c": {
        "action_type": "discrete",
        "stability": "Experimental",
        "description": "Advantage Actor-Critic (A2C) with GAE.",
    },
    "dqn": {
        "action_type": "discrete",
        "stability": "Experimental",
        "description": "Deep Q-Network with epsilon-greedy exploration.",
    },
    "double_dqn": {
        "action_type": "discrete",
        "stability": "Experimental",
        "description": "Double DQN with decoupled action selection/evaluation.",
    },
    "dueling_dqn": {
        "action_type": "discrete",
        "stability": "Experimental",
        "description": "Dueling DQN with V(s)+A(s,a) decomposition.",
    },
    "ppo": {
        "action_type": "discrete",
        "stability": "Experimental",
        "description": "Proximal Policy Optimization with clipped objective.",
    },
}

_CONTINUOUS_ALGORITHMS: Dict[str, Dict[str, str]] = {
    "ddpg": {
        "action_type": "continuous",
        "stability": "Experimental",
        "description": "Deep Deterministic Policy Gradient.",
    },
    "delayed_ddpg": {
        "action_type": "continuous",
        "stability": "Experimental",
        "description": "DDPG with delayed actor updates.",
    },
    "td3": {
        "action_type": "continuous",
        "stability": "Experimental",
        "description": "Twin Delayed DDPG with target policy smoothing.",
    },
    "sac": {
        "action_type": "continuous",
        "stability": "Experimental",
        "description": "Soft Actor-Critic with entropy regularization.",
    },
}

_ALL_ALGORITHMS = {**_DISCRETE_ALGORITHMS, **_CONTINUOUS_ALGORITHMS}

_VALID_ENVS = [
    "graph_navigation",
    "shortest_path",
    "graph_coloring",
    "max_cut",
    "vertex_cover",
    "graph_generation",
    "kg_reasoning",
    "continuous_navigation",
    "continuous_graph_edit",
]


def list_graph_rl_algorithms() -> Dict[str, Dict[str, str]]:
    """Return dict mapping algorithm name -> info dict.

    Returns:
        Dict with keys: algorithm name -> {action_type, stability, description}.
    """
    return dict(_ALL_ALGORITHMS)


def make_graph_env(env_name: str, **kwargs) -> Any:
    """Create a graph environment by name.

    Args:
        env_name: One of the supported environment names.
        **kwargs: Extra kwargs forwarded to the environment constructor.

    Returns:
        Graph environment instance.

    Raises:
        ValueError: If env_name is not recognized.
    """
    from tgraphx.rl.environments.navigation import GraphNavigationEnv
    from tgraphx.rl.environments.shortest_path import ShortestPathEnv
    from tgraphx.rl.environments.coloring import GraphColoringEnv
    from tgraphx.rl.environments.max_cut import MaxCutEnv
    from tgraphx.rl.environments.vertex_cover import VertexCoverEnv
    from tgraphx.rl.environments.generation import GraphGenerationEnv
    from tgraphx.rl.environments.kg_reasoning import KGPathReasoningEnv
    from tgraphx.rl.environments.continuous import ContinuousNavigationEnv, ContinuousGraphEditEnv
    from tgraphx.rl.environments.base import GraphEnvConfig

    n = kwargs.pop("num_nodes", 8)
    seed = kwargs.pop("seed", None)
    device = kwargs.pop("device", "cpu")
    config = kwargs.pop("config", None) or GraphEnvConfig(seed=seed, device=device)

    if env_name == "graph_navigation":
        edge_index = kwargs.pop("edge_index", _make_default_edge_index(n))
        nf = kwargs.pop("node_features", torch.ones(n, 4))
        target_node = kwargs.pop("target_node", min(n - 1, 3))
        return GraphNavigationEnv(
            edge_index=edge_index, num_nodes=n, node_features=nf,
            target_node=target_node, config=config, **kwargs
        )

    elif env_name == "shortest_path":
        edge_index = kwargs.pop("edge_index", _make_default_edge_index(n))
        nf = kwargs.pop("node_features", torch.ones(n, 4))
        target_node = kwargs.pop("target_node", min(n - 1, 3))
        return ShortestPathEnv(
            edge_index=edge_index, num_nodes=n, node_features=nf,
            target_node=target_node, config=config, **kwargs
        )

    elif env_name == "graph_coloring":
        edge_index = kwargs.pop("edge_index", _make_default_edge_index(n))
        num_colors = kwargs.pop("num_colors", 3)
        return GraphColoringEnv(
            edge_index=edge_index, num_nodes=n, num_colors=num_colors,
            config=config, **kwargs
        )

    elif env_name == "max_cut":
        edge_index = kwargs.pop("edge_index", _make_default_edge_index(n))
        return MaxCutEnv(
            edge_index=edge_index, num_nodes=n, config=config, **kwargs
        )

    elif env_name == "vertex_cover":
        edge_index = kwargs.pop("edge_index", _make_default_edge_index(n))
        return VertexCoverEnv(
            edge_index=edge_index, num_nodes=n, config=config, **kwargs
        )

    elif env_name == "graph_generation":
        return GraphGenerationEnv(config=config, **kwargs)

    elif env_name == "kg_reasoning":
        num_relations = kwargs.pop("num_relations", 3)
        edge_index = kwargs.pop("edge_index", _make_default_edge_index(n))
        edge_types = kwargs.pop("edge_types", torch.zeros(edge_index.shape[1], dtype=torch.long))
        return KGPathReasoningEnv(
            edge_index=edge_index, num_nodes=n, edge_types=edge_types,
            num_relations=num_relations, config=config, **kwargs
        )

    elif env_name == "continuous_navigation":
        edge_index = kwargs.pop("edge_index", _make_default_edge_index(n))
        nf = kwargs.pop("node_features", torch.randn(n, 8))
        action_dim = kwargs.pop("action_dim", 8)
        target_node = kwargs.pop("target_node", min(n - 1, 3))
        return ContinuousNavigationEnv(
            edge_index=edge_index, num_nodes=n, node_features=nf,
            action_dim=action_dim, target_node=target_node,
            config=config, **kwargs
        )

    elif env_name == "continuous_graph_edit":
        from tgraphx.generation.data_model import GeneratedGraph
        initial_graph = kwargs.pop("initial_graph", None)
        if initial_graph is None:
            ei = _make_default_edge_index(n)
            nf = torch.randn(n, 4)
            initial_graph = GeneratedGraph(edge_index=ei, num_nodes=n, node_features=nf)
        action_dim = kwargs.pop("action_dim", 8)
        return ContinuousGraphEditEnv(
            initial_graph=initial_graph, action_dim=action_dim, config=config, **kwargs
        )

    else:
        raise ValueError(
            f"Unknown environment '{env_name}'. Choose from: {_VALID_ENVS}"
        )


def _make_default_edge_index(n: int) -> torch.Tensor:
    """Create a simple path graph edge_index for n nodes."""
    if n < 2:
        return torch.zeros((2, 0), dtype=torch.long)
    src = list(range(n - 1))
    dst = list(range(1, n))
    return torch.tensor([src + dst, dst + src], dtype=torch.long)


def make_graph_policy(algorithm: str, env: Any, hidden_dim: int = 64, **kwargs) -> nn.Module:
    """Create appropriate policy/Q-network for the given algorithm and env.

    Args:
        algorithm: Algorithm name.
        env: Graph environment instance.
        hidden_dim: Hidden layer dimension.
        **kwargs: Extra kwargs.

    Returns:
        nn.Module policy/Q-network.
    """
    from tgraphx.rl.networks.policy import GraphPolicyNetwork
    from tgraphx.rl.networks.qnetwork import GraphQNetwork, GraphDuelingQNetwork
    from tgraphx.rl.networks.actor_critic import GraphActorCriticNetwork
    from tgraphx.rl.algorithms.continuous import ContinuousGraphActor, StochasticGraphActor

    # Determine node_in_dim from env
    obs = env.reset()
    node_features = obs.get("node_features")
    if node_features is not None:
        node_in_dim = node_features.shape[-1]
    else:
        node_in_dim = 4

    if algorithm in ("dqn", "double_dqn", "greedy"):
        num_actions = getattr(env, "action_space", 10)
        if callable(num_actions):
            num_actions = num_actions()
        return GraphQNetwork(node_in_dim, 0, hidden_dim, int(num_actions))

    elif algorithm == "dueling_dqn":
        num_actions = getattr(env, "action_space", 10)
        if callable(num_actions):
            num_actions = num_actions()
        return GraphDuelingQNetwork(node_in_dim, 0, hidden_dim, int(num_actions))

    elif algorithm in ("reinforce",):
        num_actions = getattr(env, "action_space", 10)
        if callable(num_actions):
            num_actions = num_actions()
        return GraphPolicyNetwork(node_in_dim, 0, hidden_dim, int(num_actions))

    elif algorithm in ("actor_critic", "a2c", "ppo"):
        num_actions = getattr(env, "action_space", 10)
        if callable(num_actions):
            num_actions = num_actions()
        return GraphActorCriticNetwork(node_in_dim, 0, hidden_dim, int(num_actions))

    elif algorithm in ("ddpg", "delayed_ddpg", "td3"):
        action_dim = kwargs.get("action_dim", 8)
        return ContinuousGraphActor(node_in_dim, 0, hidden_dim, action_dim)

    elif algorithm == "sac":
        action_dim = kwargs.get("action_dim", 8)
        return StochasticGraphActor(node_in_dim, 0, hidden_dim, action_dim)

    else:
        num_actions = getattr(env, "action_space", 10)
        if callable(num_actions):
            num_actions = num_actions()
        return GraphPolicyNetwork(node_in_dim, 0, hidden_dim, int(num_actions))


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class RLResult:
    """Result of a run_graph_rl call.

    Attributes:
        metrics: Dict with episode_returns, success_rate, mean_return, algorithm, environment.
        config: Serializable config dict.
        report_path: Path to JSON report if dashboard_dir was set.
    """
    metrics: Dict[str, Any]
    config: Dict[str, Any]
    report_path: Optional[str] = None


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def run_graph_rl(
    env: Union[str, Any],
    algorithm: str = "dqn",
    episodes: int = 50,
    seed: int = 42,
    device: str = "cpu",
    hidden_dim: int = 64,
    gamma: float = 0.99,
    lr: float = 1e-3,
    dashboard_dir: Optional[str] = None,
    verbose: bool = False,
    callbacks=None,
    **env_kwargs,
) -> RLResult:
    """Run graph RL training end-to-end.

    Args:
        env: Environment name string or GraphEnv instance.
        algorithm: Algorithm name from list_graph_rl_algorithms().
        episodes: Number of training episodes.
        seed: Random seed for reproducibility.
        device: 'cpu', 'cuda', or 'auto'.
        hidden_dim: Hidden dimension for networks.
        gamma: Discount factor.
        lr: Learning rate.
        dashboard_dir: If set, writes training report JSON here.
        verbose: Print episode returns if True.
        **env_kwargs: Extra kwargs forwarded to make_graph_env if env is a string.

    Returns:
        RLResult with metrics, config, and optional report_path.

    Raises:
        ValueError: If algorithm not recognized or incompatible with env action space.
    """
    if algorithm not in _ALL_ALGORITHMS:
        known = sorted(_ALL_ALGORITHMS.keys())
        raise ValueError(
            f"Unknown algorithm '{algorithm}'. Choose from: {known}"
        )

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Set seed
    torch.manual_seed(seed)
    gen = torch.Generator()
    gen.manual_seed(seed)

    env_name = env if isinstance(env, str) else "custom"

    # Create environment
    if isinstance(env, str):
        env_kwargs["seed"] = env_kwargs.get("seed", seed)
        env_kwargs["device"] = env_kwargs.get("device", device)
        env_obj = make_graph_env(env, **env_kwargs)
    else:
        env_obj = env

    is_continuous = _ALL_ALGORITHMS[algorithm]["action_type"] == "continuous"

    # Auto-route: continuous algorithms need a continuous env.
    # If the user provided a discrete env name, upgrade to continuous_navigation.
    _DISCRETE_ENV_NAMES = {
        "graph_navigation", "shortest_path", "graph_coloring",
        "max_cut", "vertex_cover", "graph_generation", "kg_reasoning",
    }
    if is_continuous and isinstance(env, str) and env in _DISCRETE_ENV_NAMES:
        env_name = "continuous_navigation"
        env_kwargs.pop("seed", None)
        env_kwargs.pop("device", None)
        env_obj = make_graph_env(
            "continuous_navigation",
            seed=seed,
            device=device,
            **{k: v for k, v in env_kwargs.items()},
        )

    # Normalise callbacks into a CallbackList.
    from tgraphx.rl.callbacks import Callback, CallbackList
    if callbacks is None:
        cb_list = CallbackList()
    elif isinstance(callbacks, CallbackList):
        cb_list = callbacks
    elif isinstance(callbacks, list):
        cb_list = CallbackList(callbacks)
    elif isinstance(callbacks, Callback):
        cb_list = CallbackList([callbacks])
    else:
        raise ValueError(
            f"callbacks must be None, a Callback, a list of Callbacks, or a "
            f"CallbackList; got {type(callbacks).__name__}."
        )

    if is_continuous:
        return _run_continuous(
            env_obj, algorithm, episodes, seed, device, hidden_dim, gamma, lr,
            dashboard_dir, verbose, env_name, gen,
        )
    else:
        return _run_discrete(
            env_obj, algorithm, episodes, seed, device, hidden_dim, gamma, lr,
            dashboard_dir, verbose, env_name, gen,
            cb_list=cb_list,
        )


def _build_discrete_agent(
    algorithm: str,
    node_in_dim: int,
    hidden_dim: int,
    num_actions: int,
    lr: float,
    gamma: float,
) -> Any:
    """Instantiate the correct agent for a discrete-action algorithm."""
    from tgraphx.rl.algorithms.baselines import RandomPolicy, GreedyPolicy
    from tgraphx.rl.algorithms.reinforce import REINFORCEAgent
    from tgraphx.rl.algorithms.actor_critic import ActorCriticAgent, A2CAgent
    from tgraphx.rl.algorithms.dqn import DQNAgent, DoubleDQNAgent
    from tgraphx.rl.algorithms.ppo import PPOAgent
    from tgraphx.rl.algorithms.replay_buffer import ReplayBuffer
    from tgraphx.rl.networks.policy import GraphPolicyNetwork
    from tgraphx.rl.networks.qnetwork import GraphQNetwork, GraphDuelingQNetwork
    from tgraphx.rl.networks.actor_critic import GraphActorCriticNetwork

    if algorithm == "random":
        return RandomPolicy(n_actions=num_actions)

    if algorithm == "greedy":
        q_net = GraphQNetwork(node_in_dim, 0, hidden_dim, num_actions)
        return GreedyPolicy(
            scoring_fn=lambda obs, a: float(
                q_net(obs["node_features"], obs["edge_index"]).squeeze(0)[
                    min(a, num_actions - 1)
                ].item()
            ),
            n_actions=num_actions,
        )

    if algorithm == "reinforce":
        policy = GraphPolicyNetwork(node_in_dim, 0, hidden_dim, num_actions)
        return REINFORCEAgent(
            policy=policy,
            optimizer=torch.optim.Adam(policy.parameters(), lr=lr),
            gamma=gamma,
        )

    if algorithm == "actor_critic":
        ac_net = GraphActorCriticNetwork(node_in_dim, 0, hidden_dim, num_actions)
        return ActorCriticAgent(
            actor_critic_net=ac_net,
            optimizer=torch.optim.Adam(ac_net.parameters(), lr=lr),
            gamma=gamma,
        )

    if algorithm == "a2c":
        ac_net = GraphActorCriticNetwork(node_in_dim, 0, hidden_dim, num_actions)
        return A2CAgent(
            actor_critic_net=ac_net,
            optimizer=torch.optim.Adam(ac_net.parameters(), lr=lr),
            gamma=gamma,
        )

    if algorithm in ("dqn", "double_dqn", "dueling_dqn"):
        use_dueling = algorithm == "dueling_dqn"
        q_cls = GraphDuelingQNetwork if use_dueling else GraphQNetwork
        q_net = q_cls(node_in_dim, 0, hidden_dim, num_actions)
        target_net = q_cls(node_in_dim, 0, hidden_dim, num_actions)
        buf = ReplayBuffer(5000)
        agent_cls = DoubleDQNAgent if algorithm == "double_dqn" else DQNAgent
        return agent_cls(
            q_net, target_net,
            torch.optim.Adam(q_net.parameters(), lr=lr),
            gamma=gamma, replay_buffer=buf, batch_size=16,
        )

    if algorithm == "ppo":
        ac_net = GraphActorCriticNetwork(node_in_dim, 0, hidden_dim, num_actions)
        agent = PPOAgent(
            actor_critic=ac_net,
            optimizer=torch.optim.Adam(ac_net.parameters(), lr=lr),
            gamma=gamma,
        )
        agent.actor_critic = ac_net
        return agent

    raise ValueError(f"Unknown discrete algorithm '{algorithm}'")


def _run_episode_collect_based(
    agent: Any,
    algorithm: str,
    env_obj: Any,
    seed: int,
    ep: int,
    num_actions: int,
    gen: torch.Generator,
) -> Tuple[float, Dict[str, Any]]:
    """Run one episode for REINFORCE or PPO (collect-then-update pattern)."""
    info: Dict[str, Any] = {}
    if algorithm == "reinforce":
        env_obj.reset(seed=seed + ep)
        trajectory = agent.collect_episode(env_obj, generator=gen)
        ep_return = float(trajectory.get("total_return", 0.0))
        agent.update(trajectory)
    else:  # ppo
        n_steps = min(50, max(num_actions * 2, 10))
        rollout_data = agent.collect_rollout(env_obj, n_steps=n_steps, generator=gen)
        ep_return = float(sum(rollout_data["rewards"]))
        agent.update(rollout_data)
    return ep_return, info


def _run_episode_step_loop(
    agent: Any,
    algorithm: str,
    env_obj: Any,
    seed: int,
    ep: int,
    node_in_dim: int,
    gen: torch.Generator,
) -> Tuple[float, Dict[str, Any]]:
    """Run one episode via the step-by-step gym loop (baselines, A/C, DQN)."""
    obs = env_obj.reset(seed=seed + ep)
    done = truncated = False
    ep_return = 0.0
    info: Dict[str, Any] = {}
    batch_for_ac: List[Any] = []
    obs_list_a2c: List[Any] = []
    acts_a2c: List[int] = []
    rews_a2c: List[float] = []
    dones_a2c: List[bool] = []
    vals_a2c: List[float] = []

    while not (done or truncated):
        action = agent.select_action(obs, generator=gen)
        next_obs, reward, done, truncated, info = env_obj.step(action)

        if hasattr(agent, "buffer"):
            agent.buffer.push(obs, action, reward, next_obs, done or truncated)
            agent.update()

        if algorithm == "actor_critic":
            batch_for_ac.append((obs, action, reward, next_obs, done or truncated))
        elif algorithm == "a2c":
            nf_a = obs.get("node_features", torch.zeros(1, node_in_dim))
            ei_a = obs.get("edge_index", torch.zeros((2, 0), dtype=torch.long))
            with torch.no_grad():
                _, val_t = agent.net(nf_a, ei_a)
            obs_list_a2c.append(obs)
            acts_a2c.append(action)
            rews_a2c.append(float(reward))
            dones_a2c.append(bool(done or truncated))
            vals_a2c.append(float(val_t.squeeze().item()))

        obs = next_obs
        ep_return += reward

    if algorithm == "actor_critic" and batch_for_ac:
        agent.update(batch=batch_for_ac)
    elif algorithm == "a2c" and obs_list_a2c:
        agent.update(rollout={
            "obs_list": obs_list_a2c, "actions": acts_a2c, "rewards": rews_a2c,
            "dones": dones_a2c, "values": vals_a2c,
        })

    return float(ep_return), info


_COLLECT_BASED_ALGOS = frozenset({"reinforce", "ppo"})


def _run_discrete(
    env_obj: Any,
    algorithm: str,
    episodes: int,
    seed: int,
    device: str,
    hidden_dim: int,
    gamma: float,
    lr: float,
    dashboard_dir: Optional[str],
    verbose: bool,
    env_name: str,
    gen: torch.Generator,
    cb_list=None,
) -> RLResult:
    """Run discrete action algorithm."""
    from tgraphx.rl.callbacks import CallbackList

    if cb_list is None:
        cb_list = CallbackList()

    obs = env_obj.reset(seed=seed)
    nf = obs.get("node_features", torch.zeros(1, 4))
    node_in_dim = nf.shape[-1]
    num_actions = int(getattr(env_obj, "action_space", 10))

    agent = _build_discrete_agent(algorithm, node_in_dim, hidden_dim, num_actions, lr, gamma)

    episode_returns: List[float] = []
    successes: List[bool] = []

    cb_list.on_train_start(algorithm=algorithm, episodes=episodes)

    for ep in range(episodes):
        cb_list.on_episode_start(episode=ep)

        if algorithm in _COLLECT_BASED_ALGOS:
            ep_return, info = _run_episode_collect_based(
                agent, algorithm, env_obj, seed, ep, num_actions, gen
            )
        else:
            ep_return, info = _run_episode_step_loop(
                agent, algorithm, env_obj, seed, ep, node_in_dim, gen
            )

        episode_returns.append(ep_return)
        successes.append(bool(info.get("success", False)))
        cb_list.on_episode_end(
            episode=ep, reward=float(ep_return),
            steps=int(info.get("steps", 0)),
            success=bool(info.get("success", False)),
        )

        if verbose:
            print(f"Episode {ep+1}/{episodes} | Return: {ep_return:.2f}")

        if cb_list.should_stop():
            break

    success_rate = float(sum(successes)) / max(len(successes), 1)
    mean_return = float(sum(episode_returns)) / max(len(episode_returns), 1)

    metrics: Dict[str, Any] = {
        "episode_returns": episode_returns,
        "success_rate": success_rate,
        "mean_return": mean_return,
        "algorithm": algorithm,
        "environment": env_name,
    }
    if algorithm in ("dqn", "double_dqn", "dueling_dqn") and hasattr(agent, "epsilon_schedule"):
        metrics["final_epsilon"] = float(
            agent.epsilon_schedule.get_epsilon(getattr(agent, "_step_count", 0))
        )

    config = {
        "algorithm": algorithm,
        "episodes": episodes,
        "seed": seed,
        "device": device,
        "hidden_dim": hidden_dim,
        "gamma": gamma,
        "lr": lr,
    }

    cb_list.on_train_end(metrics=metrics)

    report_path = None
    if dashboard_dir:
        os.makedirs(dashboard_dir, exist_ok=True)
        report_path = os.path.join(dashboard_dir, f"rl_{algorithm}_{env_name}.json")
        with open(report_path, "w") as f:
            json.dump({"metrics": metrics, "config": config}, f, indent=2, default=str)

    result = RLResult(metrics=metrics, config=config, report_path=report_path)
    result.stopped_early = cb_list.should_stop()
    return result


def _run_continuous(
    env_obj: Any,
    algorithm: str,
    episodes: int,
    seed: int,
    device: str,
    hidden_dim: int,
    gamma: float,
    lr: float,
    dashboard_dir: Optional[str],
    verbose: bool,
    env_name: str,
    gen: torch.Generator,
) -> RLResult:
    """Run continuous action algorithm."""
    import copy
    from tgraphx.rl.algorithms.continuous import (
        ContinuousGraphActor, StochasticGraphActor,
        ContinuousGraphCritic, TwinContinuousGraphCritic,
        OUNoise, GraphDDPGAgent, GraphDelayedDDPGAgent, GraphTD3Agent, GraphSACAgent,
    )
    from tgraphx.rl.algorithms.replay_buffer import ReplayBuffer

    obs = env_obj.reset(seed=seed)
    nf = obs.get("node_features", torch.zeros(1, 8))
    node_in_dim = nf.shape[-1]
    action_dim = obs.get("action_space_bounds", {}).get("dim", 8)

    episode_returns: List[float] = []
    successes: List[bool] = []
    buf = ReplayBuffer(5000)

    if algorithm in ("ddpg", "delayed_ddpg"):
        actor = ContinuousGraphActor(node_in_dim, 0, hidden_dim, action_dim)
        target_actor = copy.deepcopy(actor)
        critic = ContinuousGraphCritic(node_in_dim, 0, action_dim, hidden_dim)
        target_critic = copy.deepcopy(critic)
        actor_opt = torch.optim.Adam(actor.parameters(), lr=lr)
        critic_opt = torch.optim.Adam(critic.parameters(), lr=lr)
        noise = OUNoise(action_dim, seed=seed)

        if algorithm == "ddpg":
            agent = GraphDDPGAgent(
                actor, critic, target_actor, target_critic,
                actor_opt, critic_opt, gamma=gamma, noise=noise, replay_buffer=buf, batch_size=16,
            )
        else:
            agent = GraphDelayedDDPGAgent(
                actor, critic, target_actor, target_critic,
                actor_opt, critic_opt, gamma=gamma, noise=noise, replay_buffer=buf, batch_size=16,
            )

    elif algorithm == "td3":
        actor = ContinuousGraphActor(node_in_dim, 0, hidden_dim, action_dim)
        target_actor = copy.deepcopy(actor)
        twin_critic = TwinContinuousGraphCritic(node_in_dim, 0, action_dim, hidden_dim)
        target_twin_critic = copy.deepcopy(twin_critic)
        actor_opt = torch.optim.Adam(actor.parameters(), lr=lr)
        critic_opt = torch.optim.Adam(twin_critic.parameters(), lr=lr)
        agent = GraphTD3Agent(
            actor, twin_critic, target_actor, target_twin_critic,
            actor_opt, critic_opt, gamma=gamma, replay_buffer=buf, batch_size=16,
        )

    elif algorithm == "sac":
        actor = StochasticGraphActor(node_in_dim, 0, hidden_dim, action_dim)
        twin_critic = TwinContinuousGraphCritic(node_in_dim, 0, action_dim, hidden_dim)
        target_twin_critic = copy.deepcopy(twin_critic)
        actor_opt = torch.optim.Adam(actor.parameters(), lr=lr)
        critic_opt = torch.optim.Adam(twin_critic.parameters(), lr=lr)
        agent = GraphSACAgent(
            actor, twin_critic, target_twin_critic,
            actor_opt, critic_opt, gamma=gamma, replay_buffer=buf, batch_size=16,
        )

    else:
        raise ValueError(f"Unknown continuous algorithm '{algorithm}'")

    global_step = 0

    for ep in range(episodes):
        obs = env_obj.reset(seed=seed + ep)
        ep_return = 0.0
        done = False
        truncated = False

        while not (done or truncated):
            action = agent.select_action(obs, generator=gen)

            if isinstance(action, torch.Tensor):
                action_np = action.detach().cpu()
            else:
                action_np = action

            next_obs, reward, done, truncated, info = env_obj.step(action_np)

            # Store in replay buffer (detach action for storage)
            action_stored = action_np.clone() if isinstance(action_np, torch.Tensor) else action_np
            buf.push(obs, action_stored, reward, next_obs, done or truncated)

            # Update
            if buf.is_ready(16):
                if algorithm == "td3":
                    agent.update(step=global_step)
                else:
                    agent.update()

            obs = next_obs
            ep_return += reward
            global_step += 1

        episode_returns.append(float(ep_return))
        successes.append(bool(info.get("success", False)))

        if verbose:
            print(f"Episode {ep+1}/{episodes} | Return: {ep_return:.2f}")

    success_rate = float(sum(successes)) / max(len(successes), 1)
    mean_return = float(sum(episode_returns)) / max(len(episode_returns), 1)

    metrics: Dict[str, Any] = {
        "episode_returns": episode_returns,
        "success_rate": success_rate,
        "mean_return": mean_return,
        "algorithm": algorithm,
        "environment": env_name,
    }

    config = {
        "algorithm": algorithm,
        "episodes": episodes,
        "seed": seed,
        "device": device,
        "hidden_dim": hidden_dim,
        "gamma": gamma,
        "lr": lr,
    }

    report_path = None
    if dashboard_dir:
        os.makedirs(dashboard_dir, exist_ok=True)
        report_path = os.path.join(dashboard_dir, f"rl_{algorithm}_{env_name}.json")
        with open(report_path, "w") as f:
            json.dump({"metrics": metrics, "config": config}, f, indent=2, default=str)

    return RLResult(metrics=metrics, config=config, report_path=report_path)
