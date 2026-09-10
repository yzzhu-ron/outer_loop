# Replay the recorded semantic milestone on a CPU

This replay recalculates the original milestone's main quality comparisons and exact supplied-model certificates from committed per-query outcomes. It takes well under a minute locally after installation and uses about 6.9 MB of evidence. Python 3.10 or later and the dependency-free SearchProbe package are sufficient. No NumPy, scikit-learn, PyTorch, MLX, GPU, model files, or ignored caches are needed.

This is **analysis replay of saved outcomes**. It does not regenerate document retrieval, relevance scores from qrels, text generation, model fitting, probe choices, or bootstrap samples. It cannot independently establish the accuracy of the original retrieval measurements. The manifest pins 34 inputs from commit `e3306228a485037c662a313e591f17128e8f8ae0`, preserving the earlier milestone separately from later research.

## Start without a repository checkout

Run in a new directory. Install the development package and download these two small replay files:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install "searchprobe @ git+https://github.com/yzzhu-ron/outer_loop.git@codex/searchprobe-full-paper#subdirectory=packages/searchprobe"
replay_url=https://raw.githubusercontent.com/yzzhu-ron/outer_loop/codex/searchprobe-full-paper/papers/learning-to-explore-search-environments/full_paper/replay
curl -fsSLo replay.py "$replay_url/replay.py"
curl -fsSLo evidence_manifest.json "$replay_url/evidence_manifest.json"
python replay.py --download --evidence-dir evidence --output replay-report.json
```

Git and access to this repository are required for installation. The development script/package URLs follow a branch; the evidence itself is pinned to an immutable commit and checked against SHA-256 hashes. There is no PyPI release. `--download` retrieves only the manifest's files from GitHub. It reuses files with matching hashes and refuses to replace differing files. Subsequent runs are offline:

```sh
python replay.py --evidence-dir evidence --output replay-report.json
searchprobe-demo --output-dir editable-examples
```

The second command uses synthetic inputs bundled inside the installed wheel. It needs no download, and its output directory includes instructions for editing and rerunning the three diagnostics. On Windows, activate the virtual environment with `.venv\Scripts\activate` and download `replay.py` plus `evidence_manifest.json` from this directory using your browser or PowerShell; the Python commands are identical.

## Use the committed inputs from a checkout

From the repository root, after installing the package:

```sh
python -m pip install ./packages/searchprobe
python papers/learning-to-explore-search-environments/full_paper/replay/replay.py \
  --evidence-dir papers/learning-to-explore-search-environments/semantic_pilot/results \
  --output /tmp/searchprobe-replay.json
```

No evidence files are changed. `--output -` writes JSON to stdout. Exit `0` means all requested checks completed; malformed, missing, or inconsistent evidence exits `2`. The report output cannot overwrite an evidence file or its manifest. See [reference_report.json](reference_report.json) for a saved replay output.

## What is recalculated

| Evidence | Check |
| --- | --- |
| Six environments, 730 query-backend rows | Exact query ID/order, action order, seed/budget grids, and bounded recorded outcome arrays. The two backends share query populations. |
| Six headroom summaries | Original/source-router quality, target-best fixed action, per-query oracle, and oracle gaps. |
| 450 original and 360 follow-up runs | Select each recorded action's outcome from the original per-action matrix; recalculate nDCG and Recall means, paired deltas, and action-change rates. |
| 18 follow-up baselines | Recalculate each baseline's quality; verify the original-router and original-query references. |
| 600 paired comparison rows | Recalculate the contrast means. Stored interval endpoints are hash-checked and checked for ordering/range; their bootstrap samples are **not** regenerated. |
| Six response and six source decision models | Rerun the public SearchProbe APIs and compare their full reports with the saved reports. Response results remain conditional on supplied scores/bounds; source radii remain conditional on supplied utility matrices. |

The JSON report separates the environment summaries, certificates, replay counts, input manifest hash, calculation-code hashes, and work **not** replayed. Numeric summary comparisons use absolute tolerance `1e-12`; certificate arithmetic is exact for the supplied rational/decimal inputs. No new significance or generalization claim follows from matching the saved calculations.

The full cost ledger, original report builder, source fitting, and retrieval regeneration require additional caches or runtime context. Use the [semantic experiment instructions](../../semantic_pilot/README.md#reproduce) for that larger Apple Silicon/MLX run.

## Test the portable path

From the repository root:

```sh
PYTHONPATH=packages/searchprobe/src python -m unittest discover \
  -s papers/learning-to-explore-search-environments/full_paper/replay/tests -v
```

After building and installing a wheel into a **fresh** environment, run this smoke test with that environment's Python:

```sh
python papers/learning-to-explore-search-environments/full_paper/replay/smoke_installed.py \
  --python /path/to/fresh-venv/bin/python
```

It copies only the 34 committed files and two replay files to a temporary directory, removes `PYTHONPATH`/`PYTHONHOME`, checks the installed demo entry point, reruns all three exported examples, and executes the real replay. It performs no network access and does not load ignored caches.
