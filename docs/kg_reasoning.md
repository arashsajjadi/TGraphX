# KG Reasoning

`tgraphx.kg.reasoning` provides lightweight symbolic reasoning utilities.

**Stability: Experimental**

## Path extraction

```python
from tgraphx.kg import PathExtractor, generate_synthetic_kg

kg = generate_synthetic_kg(20, 3, 50, seed=0)
extractor = PathExtractor(kg, max_path_length=2, max_paths_per_pair=20)

# All paths from entity 0 to entity 5 (up to length 2).
paths = extractor.paths(0, 5)
# Returns list of relation-ID tuples, e.g. [(0,), (1, 2), ...]

# Body paths for relation 0 (candidate rule bodies).
body_paths = extractor.extract_all_paths(target_relation=0)
# {(1, 2): 3, (0,): 1, ...}  (path → support count)
```

### Complexity note

Path extraction is BFS with `O(E^max_path_length)` worst-case.
Use `max_path_length=2` and `max_paths_per_pair` guards for practical performance.

## Horn rule candidate mining

```python
from tgraphx.kg.reasoning import mine_horn_rules

rules = mine_horn_rules(
    kg, max_body_length=2, min_support=2, min_confidence=0.1, max_rules=20,
)
for rule in rules:
    print(rule.body, "=>", rule.head_relation,
          f"conf={rule.confidence:.3f} lift={rule.lift:.2f}")
```

`mine_horn_rules` is O(N_r × E^max_body_length).  Suitable for small KGs (< 10k triples) or with small `max_body_length=1`.

## Logical constraint checking

```python
from tgraphx.kg import LogicalConstraintChecker

checker = LogicalConstraintChecker(
    kg,
    symmetric_relations={0, 2},       # relations 0 and 2 should be symmetric
    antisymmetric_relations={1},       # relation 1 should be antisymmetric
    inverse_pairs={3: 4},              # relation 3 and 4 are inverses
)
summary = checker.violation_summary()
# {"symmetric": {"count": 2, "examples": [...]}, ...}
```

## Limitations

- `mine_horn_rules` uses a nested PathExtractor call; for large KGs this can be slow.  Limit with `max_rules`, `min_support`.
- Probabilistic rule mining (AMIE+, AnyBURL) is not implemented — only exact support/confidence counts.
- Reasoning is over the observed KG only; no latent entity inference.
