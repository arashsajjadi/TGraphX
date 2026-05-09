"""TGraphX Graph Reinforcement Learning Subpackage.

Provides graph RL environments, policy/value/Q networks, and standard algorithms.

Limitation: These implementations are small-scale research tools.
For production RL, use dedicated libraries (stable-baselines3, RLlib, etc.).

Stability: Experimental (v0.7.0+).
"""
from .environments import (
    GraphEnvConfig,
    GraphEnv,
    GraphNavigationEnv,
    GraphColoringEnv,
    MaxCutEnv,
    VertexCoverEnv,
    GraphGenerationEnv,
    KGPathReasoningEnv,
    ContinuousGraphActionSpace,
    ContinuousNavigationEnv,
    ContinuousGraphEditEnv,
    ShortestPathEnv,
)
from .networks import (
    StateFeatureProjector,
    ActionFeatureProjector,
    GraphPolicyNetwork,
    MaskedCategoricalPolicy,
    NodeActionPolicy,
    EdgeActionPolicy,
    GraphEditPolicy,
    GraphValueNetwork,
    GraphQNetwork,
    GraphDuelingQNetwork,
    GraphActorCriticNetwork,
)
from .algorithms import (
    ReplayBuffer,
    RolloutBuffer,
    BaseAgent,
    REINFORCEAgent,
    ActorCriticAgent,
    A2CAgent,
    DQNAgent,
    DoubleDQNAgent,
    PPOAgent,
    GraphDDPGAgent,
    GraphDelayedDDPGAgent,
    GraphTD3Agent,
    GraphSACAgent,
    ContinuousGraphActor,
    StochasticGraphActor,
    ContinuousGraphCritic,
    TwinContinuousGraphCritic,
    soft_update,
    OUNoise,
    GaussianNoise,
    RandomPolicy,
    GreedyPolicy,
)
from .high_level_api import (
    run_graph_rl,
    make_graph_env,
    make_graph_policy,
    list_graph_rl_algorithms,
    RLResult,
)
from .exploration import (
    EpsilonGreedy,
    LinearEpsilonDecay,
    BoltzmannExploration,
    UCBExploration,
    EntropyRegularizer,
)
from .metrics import (
    episodic_return_mean,
    episodic_return_std,
    success_rate,
    episode_length_mean,
    policy_entropy,
    approximate_kl,
    explained_variance,
    gradient_norm,
    action_validity_rate,
    td_error_mean,
)
from .config import RLTrainingConfig, PolicyConfig, RewardConfig, ConstraintConfig
from .reports import write_graph_rl_env_report, write_graph_rl_training_report

__all__ = [
    # Environments
    "GraphEnvConfig",
    "GraphEnv",
    "GraphNavigationEnv",
    "GraphColoringEnv",
    "MaxCutEnv",
    "VertexCoverEnv",
    "GraphGenerationEnv",
    "KGPathReasoningEnv",
    "ContinuousGraphActionSpace",
    "ContinuousNavigationEnv",
    "ContinuousGraphEditEnv",
    "ShortestPathEnv",
    # Networks
    "StateFeatureProjector",
    "ActionFeatureProjector",
    "GraphPolicyNetwork",
    "MaskedCategoricalPolicy",
    "NodeActionPolicy",
    "EdgeActionPolicy",
    "GraphEditPolicy",
    "GraphValueNetwork",
    "GraphQNetwork",
    "GraphDuelingQNetwork",
    "GraphActorCriticNetwork",
    # Algorithms
    "ReplayBuffer",
    "RolloutBuffer",
    "BaseAgent",
    "REINFORCEAgent",
    "ActorCriticAgent",
    "A2CAgent",
    "DQNAgent",
    "DoubleDQNAgent",
    "PPOAgent",
    "GraphDDPGAgent",
    "GraphDelayedDDPGAgent",
    "GraphTD3Agent",
    "GraphSACAgent",
    "ContinuousGraphActor",
    "StochasticGraphActor",
    "ContinuousGraphCritic",
    "TwinContinuousGraphCritic",
    "soft_update",
    "OUNoise",
    "GaussianNoise",
    "RandomPolicy",
    "GreedyPolicy",
    # High-level API
    "run_graph_rl",
    "make_graph_env",
    "make_graph_policy",
    "list_graph_rl_algorithms",
    "RLResult",
    # Exploration
    "EpsilonGreedy",
    "LinearEpsilonDecay",
    "BoltzmannExploration",
    "UCBExploration",
    "EntropyRegularizer",
    # Metrics
    "episodic_return_mean",
    "episodic_return_std",
    "success_rate",
    "episode_length_mean",
    "policy_entropy",
    "approximate_kl",
    "explained_variance",
    "gradient_norm",
    "action_validity_rate",
    "td_error_mean",
    # Config
    "RLTrainingConfig",
    "PolicyConfig",
    "RewardConfig",
    "ConstraintConfig",
    # Reports
    "write_graph_rl_env_report",
    "write_graph_rl_training_report",
]
