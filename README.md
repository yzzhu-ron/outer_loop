# outer_loop

`outer_loop` studies learning in artifact space: a frozen model proposes edits
to a prompt, program, harness, memory, or evaluation set; an application runs
the artifact; a noisy evaluator returns evidence; and a selection rule decides
what survives. The model weights need not change for the surrounding system to
improve.

This repository contains the reusable control-plane code, one web-agent
application, the theory manuscripts, prospective empirical paper work, and the
literature synthesis that connects them.

## Repository map

```text
packages/outer_loop/       application-neutral Python package
applications/shopgym/      ShopGym harness-search case study and all of its papers
experiments/               future cross-application studies
papers/
  learning-beyond-gradients/
                            general theory paper, reviews, simulations, companions
  benchmark-compression-for-retriever-evaluation/
                            frontier proposal, assessment, alternative IR directions
literature/                 systems review and theory reading notes
```

The organizing rule is ownership. General search machinery belongs in the
package. Domain assumptions, data, experiment drivers, results, and manuscripts
belong to the application that makes those assumptions. A paper about the
general method belongs under `papers/`. Literature is evidence, not runtime
code.

## Current research tracks

| Track | Question | Status | Start here |
|---|---|---|---|
| Core package | What minimal interfaces and selection primitives recur across artifact-search systems? | Working prototype | [`packages/outer_loop/`](packages/outer_loop/) |
| Learning Beyond Gradients | What can be guaranteed about proposal priors, feedback, noisy selection, archives, and description length? | ICLR 2027 manuscript generation | [`papers/learning-beyond-gradients/iclr-2027/`](papers/learning-beyond-gradients/iclr-2027/) |
| ShopGym | Can a frozen browser agent improve through statistically gated search over its harness? | Application prototype; live evidence pending | [`applications/shopgym/`](applications/shopgym/) |
| Benchmark Compression | Can a compact, nested query set preserve retriever conclusions across unseen system families? | Proposal under reassessment; novelty and validity gaps; experiments pending | [Proposal](papers/benchmark-compression-for-retriever-evaluation/) · [Assessment](papers/benchmark-compression-for-retriever-evaluation/frontier-assessment.html) |
| IR and Self-Improving Search | Which learning problem offers a stronger empirical research direction? | Ranked hypotheses with novelty and effort ratings; pilot selection pending | [Research directions](papers/benchmark-compression-for-retriever-evaluation/research-directions.html) |
| Literature | How do GEPA, autoresearch, Heuristic Learning, and Meta-Harness instantiate the same loop? | Research synthesis | [`literature/outer-loop-systems-review/`](literature/outer-loop-systems-review/) |

## Why `applications/shopgym`, not `adapters/shopgym`

The ShopGym work defines a mutable harness space, a trace model, evaluation
protocols, experiments, results, and manuscript claims. Those are application
semantics. An adapter would only translate interfaces. The confidence gate,
archive behavior, selection policy, and proposal/evaluation contracts form the
shared infrastructure.

ShopGym is therefore one application of the outer-loop program. The next
submission direction remains an explicit research decision.

## Install and run

The repository is a small `uv` workspace. To run the framework and ShopGym unit
tests without installing unrelated research dependencies:

```bash
PYTHONPATH=packages/outer_loop/src:applications/shopgym/src \
  python -m unittest discover packages/outer_loop/tests
PYTHONPATH=packages/outer_loop/src:applications/shopgym/src \
  python -m unittest discover applications/shopgym/tests
```

To reproduce the seeded ShopGym simulations:

```bash
uv sync --package outer-loop-shopgym
uv run --package outer-loop-shopgym shopgym-experiments
```

The committed ShopGym tables are simulation outputs, not live-shop results.
See the [application README](applications/shopgym/) for the exact evidence
boundary and the work required before making empirical claims.

## Reference repositories

The literature review studies third-party projects without vendoring their Git
histories. Clone them into the ignored paths only when source-level inspection
is needed:

```bash
git clone https://github.com/trinkle23897/learning-beyond-gradients literature/outer-loop-systems-review/1-heuristic-learning/learning-beyond-gradients
git clone https://github.com/gepa-ai/gepa literature/outer-loop-systems-review/2-gepa/gepa
git clone https://github.com/karpathy/autoresearch literature/outer-loop-systems-review/3-karpathy-auto-research/autoresearch
git clone https://github.com/stanford-iris-lab/meta-harness literature/outer-loop-systems-review/4-meta-harness/meta-harness
```
