# Search environment exploration

The continuation now has a [technical blog draft](full_paper/searchprobe-blog.html),
[new full-paper experiments](full_paper/README.md), and a
[portable CPU evidence replay](full_paper/replay/README.md).
The earlier report and frozen experiments below are preserved.

This track turns the [proposal](search-environment-exploration-proposal.html)
into a semantic retrieval experiment, executable decision-theory examples, and
an installable diagnostic tool.

Start with the [research report](searchprobe-research-report.html), which joins
the evidence and an interactive counterexample. The
[semantic experiment](semantic_pilot/README.md) uses SciFact, FiQA and NFCorpus,
BM25 and MiniLM, five query actions, and four probe selectors. Its source-trained
router and selector never use target-task labels during onboarding. All three
corpora remain exploratory, and retrieval scores do not measure answer correctness.

The [completed results](semantic_pilot/RESULTS.md) diagnose a router that could
not change actions within the allowed budget. A separately frozen source-world
intervention removes that bottleneck: SciFact/dense nDCG@10 rises from .6376 to
.6795 after 32 probes. Its conditional interval includes zero; results across
the other environments are mixed. Both experiments and every selector remain
in the evidence ledger.

[SearchProbe](../../packages/searchprobe/) runs locally with Python's standard
library. It audits paired-query logs for collisions, ties, saturation, coverage
and costs. A separate command computes the exact minimax decision radius of a
declared source utility model, including conflicts that pairwise checks miss.
The third command, `response-audit`, determines whether declared score bounds
allow any change in a router's decisions, without task relevance labels.
The [theory note](theory/theory.md) gives the assumptions, proofs and limitations;
the tool does not certify transfer to an unseen corpus.

The first [lexical pilot](pilot/RESULTS.md) is preserved with its
[figures](pilot/results/figures/), [protocol](pilot/PROTOCOL.md), and
[machine-readable evidence](pilot/results/). Its zero profile gain motivated
the stronger semantic experiment. Follow the
[semantic reproduction instructions](semantic_pilot/README.md#reproduce) for
the continuation; the commands below reproduce only that original lexical pilot.

## Reproduce the lexical pilot

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
