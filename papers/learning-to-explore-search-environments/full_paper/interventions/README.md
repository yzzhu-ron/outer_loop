# Within-corpus backend interventions: feasibility and adapter

The cached inputs support a controlled hybrid-backend study. The adapter and
endpoint validation are complete; **no real-cache interior-mixture task utilities, probe
outcomes, fitted models, or intervention-study results have been computed**.

**The target anchor is an unresolved design requirement.** Within-source-corpus
centering can isolate how probes and action utility change with the backend. A
policy facing one hidden target backend cannot subtract the target corpus's mean
over all seven backends, inspect another target backend, or use target reference
utility labels. A centered source predictor alone does not identify absolute
target action ordering. A deployable study must specify a source-only intercept
and/or an observable, charged target anchor. Otherwise the centered analysis must
remain an evaluator-only mechanism diagnostic. The next study needs its own
frozen protocol before any new outcomes are inspected.

## What the adapter reconstructs

`adapter.py` reads the semantic pilot's frozen prepared queries, generated rewrite
strings, sampled document IDs, and per-request BM25/dense top10 ranklists. It never
generates new queries or modifies old caches. `task_requests` preserves the
train/calibration/utility/test partitions. `probe_candidates` restores the known
document ID, comparison requests, candidate ID, features, family, and logical
cost; invalid or identical actions retain the existing exclusion rule.

For each raw query, the proposed operator is weighted RRF with constant 60:
`score(d) = (1−λ)/(60+rank_BM25(d)) + λ/(60+rank_dense(d))`.
An absent document contributes zero. Sort by descending score, break ties by
lexicographic document ID, then return at most ten documents. Lambda 0/1 exactly
returns BM25/dense, including short or empty results; zero-weight-only documents
are excluded. The proposed grid is `{0, .125, .25, .5, .75, .875, 1}`.

For decomposed actions, first combine BM25 and dense for **each subquery**, then
apply the existing unweighted RRF60 across those returned lists. Combining the
already-fused endpoint action lists is a different operator. The tests contain a
small example where the two orders produce different first-ranked documents.

The adapter exposes separate logical and underlying search costs. Its default
`execution="both"` charges two component searches at every lambda, including
endpoints, so policy-visible costs do not reveal endpoint identity. A two-subquery
action costs two wrapper calls and four component searches. A paired original
versus decomposed probe costs three wrapper calls and six component searches.
Repeated request strings still incur logical serving costs. `execution="active"`
is an explicit alternative that skips zero-weight components; it is unsuitable
for the hidden-backend contract if those differing costs reveal lambda. Cached
replay does not imply zero serving cost. Document sampling and generation setup
remain separately charged under the existing protocol; generated actions share
one rewrite bundle rather than three independent generations.

## Endpoint validation

The audit uses archived rankings and document-only observations, never task
relevance scores. All reconstructed request strings are present in both endpoint
caches. Candidate IDs, ordering, features, costs, task-action rankings, and probe
observations match exactly. Counts below are per endpoint; candidate occurrences
include all three document-pool seeds and are not independent samples.

| Corpus | Required raw queries | Task-action rankings | Probe observations | BM25 empty lists |
|---|---:|---:|---:|---:|
| FiQA | 2,625 | 1,710 | 384 | 0 |
| NFCorpus | 1,917 | 1,285 | 384 | 36 |
| SciFact | 2,624 | 1,710 | 384 | 0 |

`endpoint_audit.v1.json` records every checked count, the thirteen source-file
hashes/sizes, source cache identities, and the prepared code contract. The keyword
engine hash must match that contract before audit execution. `freeze.v1.json`
records this implementation's code, operator specification, tests, and audit
hashes. These files freeze preparation and provenance, not a finished inference
protocol or empirical finding about interior lambda values.

From this directory, using the existing pilot environment:

```sh
../../pilot/.venv/bin/python -m unittest discover -s . -p 'test_adapter.py'
../../pilot/.venv/bin/python adapter.py --output /tmp/intervention-endpoint-audit.json
```

## Limits that the eventual study must retain

This is fusion over a union of at most twenty cached documents per raw query,
not a full-corpus hybrid index or interpolation of retriever scores. No document
outside that truncated support can be recovered. Responses are piecewise
constant in lambda. With ten BM25 hits, a dense-only document cannot outrank even
the lowest BM25-only hit until `λ > 61/131 ≈ .46565`; at `.125/.25`, shared documents
may reorder the BM25 support, but dense-only documents cannot enter its top10.
The symmetric restriction applies at `.75/.875`. Report distinct observable
states rather than assuming the seven grid points create seven rich backends.

Hold out every lambda of a target family together. The seven variants share
queries, documents, generation, and endpoint rankings; they are paired engineered
states, not seven independent corpora. Resampling must preserve those shared
units. Source utility-to-lambda shuffles should occur within corpus to preserve
family means. The evaluator owns both endpoint caches and lambda; a policy may
access only selected, charged responses through the declared interface. Decide
whether full returned ranklists or only summary observations are allowed before
claiming observability. These three families were already explored, so a later
positive result here remains exploratory until checked on new families/backends.
