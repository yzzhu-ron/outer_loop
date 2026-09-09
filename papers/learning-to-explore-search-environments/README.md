# Search environment exploration

This track now has an executable **exploratory ceiling-and-signal pilot** on
the full SciFact and FiQA corpora, using BM25 and a pinned MiniLM encoder.
It follows the first feasibility stage of the
[proposal](search-environment-exploration-proposal.html), which remains the
original discussion document.

Start with the [result ledger](pilot/RESULTS.md), the three
[figures](pilot/results/figures/), and the frozen
[pilot protocol](pilot/PROTOCOL.md). Machine-readable scores, profiles,
conditional intervals, source models, and provenance are in
[`pilot/results/`](pilot/results/).

This reduced pilot uses deterministic lexical transformations and extractive
document probes. It does **not** implement semantic LLM rewrites, learned probe
selection, or an end-to-end search agent. Neither corpus is an untouched final
test family after this pilot.

## Reproduce

From the repository root, create a separate environment and install the exact
dependency versions used for the run (Python 3.14.0 on macOS arm64):

```bash
uv venv --python 3.14 papers/learning-to-explore-search-environments/pilot/.venv
uv pip install \
  --python papers/learning-to-explore-search-environments/pilot/.venv/bin/python \
  -r papers/learning-to-explore-search-environments/pilot/requirements.lock
```

Download approximately 20 MB of corpus archives and the pinned public MiniLM
model. The downloader verifies the corpus SHA-256 checksums. The model, indexes,
raw corpora, and intermediate retrieval caches stay in ignored local directories.

```bash
papers/learning-to-explore-search-environments/pilot/.venv/bin/python \
  papers/learning-to-explore-search-environments/pilot/prepare_data.py
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false \
  papers/learning-to-explore-search-environments/pilot/.venv/bin/python \
  papers/learning-to-explore-search-environments/pilot/run_pilot.py
papers/learning-to-explore-search-environments/pilot/.venv/bin/python \
  papers/learning-to-explore-search-environments/pilot/audit_probes.py
MPLCONFIGDIR=/tmp/search-exploration-matplotlib \
  papers/learning-to-explore-search-environments/pilot/.venv/bin/python \
  papers/learning-to-explore-search-environments/pilot/plot_results.py
```

The runner defaults to offline model loading after preparation. The first run
builds embeddings for every corpus document; later runs reuse valid caches.
Different hardware or numeric libraries can affect dense-score ties and timing.
`--analyze-only` rebuilds summaries from local outcome caches only when the
input-code/protocol contract (including the runner) and sample parameters still match. The original
generation hash is preserved separately from the analysis-code hashes.

Run the contract, ranking, and cache checks:

```bash
papers/learning-to-explore-search-environments/pilot/.venv/bin/python \
  -m unittest discover -s papers/learning-to-explore-search-environments/pilot/tests -v
```

## What the implementation enforces

- Document/normalized-duplicate groups are split before relevance eligibility
  is computed. Every original document remains indexed. Every positive support
  document of a selected evaluation query belongs to the held-out pool.
- Onboarding accepts sampled documents and a search interface, with no real
  queries or qrels. Its persisted profile contains only numeric aggregates.
- Both retrieval configurations of a corpus stay together in each source/target
  fold. Query routing is trained on the opposite corpus's training queries;
  disjoint source query IDs calibrate the passive profile correction.
- A probe samples one document and executes two searches. Invalid probes still
  pay for sampling. Caches accelerate experiments while virtual interaction
  costs remain charged. The 256-operation cap can be underspent.
- The evaluator keeps task labels and oracle choices separate from eligible
  methods. Oracle results indicate possible headroom, not deployable quality.

The code is local to this paper because these data, probe, and retrieval
assumptions are not generic `outer_loop` control-plane interfaces.
