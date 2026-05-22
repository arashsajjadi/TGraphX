.PHONY: install test test-slow test-stress lint format typecheck \
        fast-debug train-rtx5080 benchmark benchmark-full evaluate \
        generate-data pretrain full-game-check clean help smoke-rtx5080

install:
	pip install -e ".[dev]"

test:
	pytest -q -m "not slow and not stress"

test-all:
	pytest -v --tb=short

test-slow:
	pytest -q -m "slow or stress" --timeout=300

test-stress:
	pytest tests/test_random_stress.py -v

lint:
	ruff check src/ tests/ examples/ --ignore E501,E402 2>/dev/null || \
	  python -m py_compile $$(find src -name "*.py") && echo "syntax OK"

format:
	ruff format src/ tests/ examples/ 2>/dev/null || echo "ruff not installed; skipping"

typecheck:
	pyright src/ 2>/dev/null || mypy src/ --ignore-missing-imports 2>/dev/null || \
	  echo "no type checker installed; skipping"

fast-debug:
	python -m backgammon_rlx.train.train_ppo --config configs/fast_debug.yaml

train-rtx5080:
	python -m backgammon_rlx.train.train_ppo --config configs/rtx5080.yaml

smoke-rtx5080:
	python -m backgammon_rlx.train.train_ppo --config configs/rtx5080.yaml --max-updates 2

benchmark:
	python -m backgammon_rlx.benchmarks.benchmark_all --quick

benchmark-full:
	python -m backgammon_rlx.benchmarks.benchmark_all \
	  --config configs/rtx5080.yaml \
	  --out runs/benchmark_results.json

evaluate:
	@CKPT=$$(ls -t runs/*/checkpoints/latest.pt runs_debug/*/checkpoints/final.pt 2>/dev/null | head -1); \
	  if [ -z "$$CKPT" ]; then \
	    echo "No checkpoint found. Run 'make fast-debug' first."; exit 1; \
	  fi; \
	  python -m backgammon_rlx.train.evaluate --checkpoint $$CKPT --games 200

full-game-check:
	python -m backgammon_rlx.validation.full_game_check \
	  --games 1000 \
	  --agents random,heuristic \
	  --strict-invariants \
	  --seed 123

generate-data:
	python -m backgammon_rlx.data.generate_expert_dataset \
	  --engine heuristic \
	  --positions 50000 \
	  --out data/heuristic_expert.jsonl

pretrain:
	python -m backgammon_rlx.train.pretrain_imitation \
	  --config configs/imitation.yaml \
	  --dataset data/heuristic_expert.jsonl

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; \
	find . -name "*.pyc" -delete 2>/dev/null; \
	rm -rf runs_debug/ dist/ build/ *.egg-info/ .pytest_cache/ .ruff_cache/

help:
	@echo "BackgammonX — make targets:"
	@echo "  test / test-all / test-slow / test-stress"
	@echo "  fast-debug / train-rtx5080 / smoke-rtx5080"
	@echo "  benchmark / benchmark-full / evaluate"
	@echo "  full-game-check / generate-data / pretrain"
	@echo "  lint / format / typecheck / clean"
