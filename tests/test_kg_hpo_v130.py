"""KG HPO (hyperparameter search) tests (v1.3)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch


@pytest.fixture
def tiny_kg():
    from tgraphx.kg import KnowledgeGraph
    torch.manual_seed(0)
    heads = torch.randint(0, 15, (60,))
    rels = torch.randint(0, 2, (60,))
    tails = torch.randint(0, 15, (60,))
    return KnowledgeGraph.from_hrt(
        heads, rels, tails, num_entities=15, num_relations=2
    )


class TestKGHPOGrid:
    def test_grid_search_covers_all_combos(self, tiny_kg):
        from tgraphx.kg import run_kg_hpo
        result = run_kg_hpo(
            tiny_kg,
            model_names=["TransE", "DistMult"],
            search_space={"embedding_dim": [8, 16]},
            strategy="grid",
            epochs=1, seed=0,
        )
        # 2 models × 2 embedding dims = 4 trials
        assert len(result.trials) == 4

    def test_max_trials_cap(self, tiny_kg):
        from tgraphx.kg import run_kg_hpo
        result = run_kg_hpo(
            tiny_kg,
            model_names=["TransE", "DistMult", "SimplE"],
            search_space={"embedding_dim": [8, 16]},
            strategy="grid",
            max_trials=3,
            epochs=1, seed=0,
        )
        assert len(result.trials) <= 3


class TestKGHPORandom:
    def test_random_strategy_deterministic(self, tiny_kg):
        from tgraphx.kg import run_kg_hpo
        r1 = run_kg_hpo(
            tiny_kg,
            model_names=["TransE", "DistMult"],
            search_space={"embedding_dim": [8, 16]},
            strategy="random",
            max_trials=2,
            epochs=1, seed=42,
        )
        r2 = run_kg_hpo(
            tiny_kg,
            model_names=["TransE", "DistMult"],
            search_space={"embedding_dim": [8, 16]},
            strategy="random",
            max_trials=2,
            epochs=1, seed=42,
        )
        # Same seed → same trial order → same best model.
        assert r1.best_model_name == r2.best_model_name
        assert r1.best_config == r2.best_config


class TestKGHPOBestSelection:
    def test_best_trial_has_highest_mrr(self, tiny_kg):
        from tgraphx.kg import run_kg_hpo
        result = run_kg_hpo(
            tiny_kg,
            model_names=["TransE", "DistMult"],
            search_space={"embedding_dim": [8]},
            metric="mrr",
            epochs=2, seed=0,
        )
        ok_trials = [t for t in result.trials if t.status == "ok"]
        best_mrr = max(t.metrics["mrr"] for t in ok_trials)
        assert result.best_metrics["mrr"] == pytest.approx(best_mrr, abs=1e-6)

    def test_mr_uses_lower_is_better(self, tiny_kg):
        from tgraphx.kg import run_kg_hpo
        result = run_kg_hpo(
            tiny_kg,
            model_names=["TransE", "DistMult"],
            search_space={"embedding_dim": [8]},
            metric="mr",
            epochs=2, seed=0,
        )
        ok_trials = [t for t in result.trials if t.status == "ok"]
        best_mr = min(t.metrics["mr"] for t in ok_trials)
        assert result.best_metrics["mr"] == pytest.approx(best_mr, abs=1e-6)


class TestKGHPOResultObject:
    def test_result_to_dict_json_serialisable(self, tiny_kg):
        from tgraphx.kg import run_kg_hpo
        result = run_kg_hpo(
            tiny_kg,
            model_names=["SimplE"],
            search_space={"embedding_dim": [8]},
            epochs=1, seed=0,
        )
        d = result.to_dict()
        json.dumps(d)  # Must not raise.

    def test_summary_returns_string(self, tiny_kg):
        from tgraphx.kg import run_kg_hpo
        result = run_kg_hpo(
            tiny_kg,
            model_names=["TransE"],
            search_space={"embedding_dim": [8]},
            epochs=1, seed=0,
        )
        s = result.summary()
        assert isinstance(s, str)
        assert "best model" in s.lower() or "Best model" in s

    def test_write_dashboard_artifacts(self, tmp_path, tiny_kg):
        from tgraphx.kg import run_kg_hpo
        result = run_kg_hpo(
            tiny_kg,
            model_names=["TransE"],
            search_space={"embedding_dim": [8]},
            epochs=1, seed=0,
            dashboard_dir=str(tmp_path / "hpo_run"),
        )
        assert (tmp_path / "hpo_run" / "kg_hpo_results.json").exists()
        assert (tmp_path / "hpo_run" / "metrics_summary.json").exists()
        # JSON must be valid.
        data = json.loads((tmp_path / "hpo_run" / "kg_hpo_results.json").read_text())
        assert "best_model_name" in data


class TestKGHPOErrors:
    def test_unknown_metric(self, tiny_kg):
        from tgraphx.kg import run_kg_hpo
        with pytest.raises(ValueError, match="metric"):
            run_kg_hpo(tiny_kg, metric="bad_metric", epochs=1)

    def test_unknown_strategy(self, tiny_kg):
        from tgraphx.kg import run_kg_hpo
        with pytest.raises(ValueError, match="strategy"):
            run_kg_hpo(tiny_kg, strategy="bayesian", epochs=1)

    def test_unknown_model_name(self, tiny_kg):
        from tgraphx.kg import run_kg_hpo
        with pytest.raises(ValueError, match="model"):
            run_kg_hpo(tiny_kg, model_names=["FakeModel"], epochs=1)

    def test_no_mutation_of_input_kg(self, tiny_kg):
        from tgraphx.kg import run_kg_hpo
        original_triples = tiny_kg.triples.clone()
        run_kg_hpo(tiny_kg, model_names=["TransE"],
                   search_space={"embedding_dim": [8]}, epochs=1, seed=0)
        assert torch.equal(tiny_kg.triples, original_triples)
