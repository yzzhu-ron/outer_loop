# What the fitted probe channels distinguish

These are post-hoc diagnostics of `natural/results/models.json`. They read source-fitted utilities and likelihoods, without reading target task outcomes. The six standard environment models contain only three distinct source fits: changing the held-out backend does not change the other two source families.

Every source fit has four distinct optimal four-bucket action vectors. Consequently the decision-entropy baseline is information gain exactly: its world-to-decision-label map is one-to-one, so label entropy equals world entropy for **every posterior**, including every possible observation branch. This explains their identical acquisition values and reported results under the same tie convention. It is a structural duplication of the baseline, not independent evidence favoring one acquisition objective.

Action indices are original=0, keywords=1, semantic=2, HyDE=3, decomposed=4. The source vectors are:

| Source world | Four-bucket argmax vector |
| --- | --- |
| FiQA BM25 | (0, 2, 0, 1) |
| FiQA dense | (0, 2, 0, 0) |
| NFCorpus BM25 | (1, 3, 1, 1) |
| NFCorpus dense | (3, 3, 3, 3) |
| SciFact BM25 | (1, 1, 0, 0) |
| SciFact dense | (3, 3, 1, 3) |

Each fit uses the four rows outside its held-out family. These labels use the frozen implementation's NumPy argmax convention and fitted mean utilities. They do not assert that the population-optimal action vectors differ.

The eight categorical channels are also pairwise Blackwell-incomparable in each fitted model. This resolves the earlier uncertainty about whether the new channels escaped the original Gaussian total order.

| Held-out family | Distinct decision labels | Incomparable channel pairs | Smallest certified directed distance lower bound |
| --- | ---: | ---: | ---: |
| FiQA | 4/4 | 28/28 | 0.0038364 |
| NFCorpus | 4/4 | 28/28 | 0.0126226 |
| SciFact | 4/4 | 28/28 | 0.0233250 |

The distance is max-entry probability error, not retrieval regret. For each ordered pair of channels A and B, a small LP minimizes `max(abs(A G - B))` over nonnegative matrices G whose rows sum to one. All 168 ordered comparisons returned successful solutions, and their stochastic-matrix constraints and residuals were recomputed independently. The numerical threshold was 1e-9.

For a solver-independent check of positive distances, the saved artifact includes an exact rational separating functional W for every direction. For every stochastic G,

```
<W, A G> <= sum_i max_j (A^T W)[i,j].
```

Hölder's inequality therefore gives

```
||B - A G||_infinity >=
  (<W, B> - sum_i max_j (A^T W)[i,j]) / sum(abs(W)).
```

Any nonzero W gives this bound. Its validity does not require an optimal or dual-feasible LP solution. The code rounds proposed weights to rational numbers and evaluates this expression exactly after converting saved decimal channel probabilities to fractions and normalizing each row exactly. The maximum resulting probability change is saved; it is at floating-rounding scale. Every comparison has a strictly positive rational lower bound, ruling out a garbling for these precisely defined fitted channels. This is an elementary Blackwell separation calculation, not a new theorem.

Pairwise incomparability only excludes a universal ordering over **all** downstream decision problems at equal cost. It does not imply different channel rankings for the particular fitted retrieval utilities, an acquisition advantage, accurate source-to-target transfer, or statistically established incomparability of the true observation distributions. Those are separate empirical questions.

Reproduce from the repository root:

```
papers/learning-to-explore-search-environments/pilot/.venv/bin/python papers/learning-to-explore-search-environments/full_paper/theory/channel_diagnostics.py
```

`test_channel_diagnostics.py` contains six focused tests: identity/erasure ordering, incomparable deterministic bit channels, validity with arbitrary separating weights, unique versus merged decision labels, rejection of LP failure as evidence, and invalid probability input. `channel_diagnostics.json` saves source/code hashes, solver versions, every garbling witness, every rational separator, and the six label maps.
