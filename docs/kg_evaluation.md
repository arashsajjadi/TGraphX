# KG Filtered Ranking Evaluation

## Definitions

For each test triple (h, r, t):

**Tail prediction:**
``s_e = f(h, r, e)``  for all e ∈ E

**Raw tail rank:**
``rank_raw(t) = 1 + |{e : s_e > s_t}|``

**Filtered tail rank:**
Known positives (h, r, e) are removed from the ranking (except e = t):
``rank_filt(t) = 1 + |{e : s_e > s_t  AND  (h, r, e) not in T_pos \ {t}}|``

**Head prediction** follows the same logic.

**Tie policy (average):**
``rank = 1 + (strictly_higher) + 0.5 * (equal, not target)``

**Metrics:**
- MR = mean(rank)
- MRR = mean(1/rank)
- Hits@K = mean(rank ≤ K)

## Usage

```python
from tgraphx.kg import KGEvaluator

evaluator = KGEvaluator(
    train_triples, valid_triples, test_triples,
    num_entities=N_e, chunk_size=50_000,
)
result = evaluator.evaluate(model, test_triples)
print(result.filt_mrr, result.filt_hits[10])
```

## Performance

Candidate scoring is chunked (``chunk_size`` entities at a time).
Memory usage: O(chunk_size × embedding_dim × 4 bytes).
Set `chunk_size` based on your GPU/RAM budget.

No autograd is retained in evaluation — scores are detached.
