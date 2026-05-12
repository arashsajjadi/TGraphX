"""TGraphX user-experience layer (v1.4.0+, enhanced v1.4.1).

High-impact, mathematically-safe utilities that reduce boilerplate without
weakening tensor-native semantics. Every function in this module is:

- Safe: it does not silently flatten tensor-valued node features.
- Honest: it raises actionable errors instead of misleading silent successes.
- Tested: every public alias has a regression test.
- Reproducible: deterministic when ``seed`` is provided.
- Device-aware: works on CPU; CUDA where applicable.

Stability: Beta (v1.4.0+).
"""
from __future__ import annotations

from .validation import (
    validate_graph,
    assert_tensor_native,
    check_graph_invariants,
)
from .describe import (
    describe,
    summary,
)
from .reproducible_ctx import (
    reproducible,
    seeded,
    reproducibility_state,
)
from .leakage import (
    check_leakage,
    leakage_report,
    validate_split_policy,
)
from .serialization import (
    save,
    load,
    save_tgraphx,
    load_tgraphx,
)
from .graph_construction import (
    knn_graph,
    build_class_prototypes,
    build_prototype_graph,
    image_to_patch_graph,
)
from .dashboard_audit import (
    audit_run_dir,
    dashboard_audit,
)
from .workflow import (
    workflow,
    run_workflow,
    list_workflow_tasks,
)
from .compare import (
    compare,
)
from .public_api import (
    public_api,
    api_status,
    list_aliases,
)
# v1.4.1 helpers
from .helpers import (
    classify_nodes,
    node_classification,
    fit_node_classifier,
    train_node_classifier,
    kg_completion,
    fit_kg,
    train_kg,
    make_graph,
    build_graph,
    explain_error,
    troubleshoot_error,
    debug_batch,
    batch_summary,
    assert_batch_consistent,
    dataset_card,
    model_card,
    benchmark_card,
    audit_package_readiness,
    WorkflowResult,
)
# v1.4.1 generation/RL/evolution wrappers
from .generation_wrappers import (
    generate_graph,
    graph_generator,
    generate,
    evaluate_generated_graphs,
    graph_generation_report,
    compare_generated_graphs,
    generation_metrics,
    optimize_graph,
    evolve_graph,
    graph_evolution,
    run_evolution,
    train_graph_rl,
    graph_rl,
    run_rl,
    audit_generation_run,
    audit_evolution_run,
    audit_rl_run,
)

__all__ = [
    # v1.4.0 Validation
    "validate_graph", "assert_tensor_native", "check_graph_invariants",
    # Describe / summary
    "describe", "summary",
    # Reproducibility
    "reproducible", "seeded", "reproducibility_state",
    # Leakage
    "check_leakage", "leakage_report", "validate_split_policy",
    # Serialization
    "save", "load", "save_tgraphx", "load_tgraphx",
    # Graph construction
    "knn_graph", "build_class_prototypes", "build_prototype_graph", "image_to_patch_graph",
    # Dashboard
    "audit_run_dir", "dashboard_audit",
    # Workflow
    "workflow", "run_workflow", "list_workflow_tasks",
    # Compare
    "compare",
    # Public API registry
    "public_api", "api_status", "list_aliases",
    # v1.4.1 one-call helpers
    "classify_nodes", "node_classification", "fit_node_classifier", "train_node_classifier",
    "kg_completion", "fit_kg", "train_kg",
    "make_graph", "build_graph",
    "explain_error", "troubleshoot_error",
    "debug_batch", "batch_summary", "assert_batch_consistent",
    "dataset_card", "model_card", "benchmark_card",
    "audit_package_readiness",
    "WorkflowResult",
    # v1.4.1 generation/RL/evolution
    "generate_graph", "graph_generator", "generate",
    "evaluate_generated_graphs", "graph_generation_report",
    "compare_generated_graphs", "generation_metrics",
    "optimize_graph", "evolve_graph", "graph_evolution", "run_evolution",
    "train_graph_rl", "graph_rl", "run_rl",
    "audit_generation_run", "audit_evolution_run", "audit_rl_run",
]
