# ShopGym application

This directory studies one application of outer-loop artifact search: improving
the harness around a frozen browser agent on ShopGym e-commerce tasks. The
mutable artifact is a five-axis harness configuration—observation modality,
action vocabulary, retained context, scaffolding, and retry policy. ShopGym is
the task environment and verifier.

Calling this an **application** is intentional. The work owns a domain model,
an evaluator, experimental claims, and paper drafts. An adapter would only
translate between interfaces.

## Research records

There are two ShopGym-specific paper records, both kept here:

- [`papers/shopgym-submission/`](papers/shopgym-submission/) contains the source
  ledger, review notes, and harness-track proposal associated with the ShopGym
  submission ([arXiv:2605.16116](https://arxiv.org/abs/2605.16116)).
- [`papers/certified-harness-search/`](papers/certified-harness-search/) contains
  the follow-up draft on gated, archive-based harness search. The earlier
  concept pages are retained in [`papers/research-notes/`](papers/research-notes/).

## Implementation

```text
configs/                 harness space and experiment parameters
src/shopgym/harness.py   mutable artifact definition
src/shopgym/environment.py
                         mock environment and live Playwright integration
src/shopgym/proposer.py  trace-conditioned and random proposers
src/shopgym/evaluation.py
                         application implementation of the core evaluation contract
src/shopgym/search.py    ShopGym experiment orchestration
src/shopgym/experiments/ four study drivers
results/                 committed tables and plots
papers/                  every ShopGym-specific manuscript and research note
```

The reusable archive, confidence gate, and selection policy are imported from
[`../../packages/outer_loop/`](../../packages/outer_loop/). The shared package
does not depend on ShopGym.

## Reproduce the simulation studies

From the repository root:

```bash
uv sync --package outer-loop-shopgym
uv run --package outer-loop-shopgym shopgym-experiments
```

Use `--full` for the larger phase 1–3 settings and `--multi-shop` to add phase
4. Outputs are written to this directory's `results/` tree regardless of the
shell's working directory.

The committed result tables are generated from the seeded mock/calibrated
environment. Their evidentiary scope is protocol and software validation. A
paper making empirical claims about web-agent improvement needs the live
environment, frozen task splits, model/API records, repeated trials, and
held-out-shop evaluation.

The live `PlaywrightShopGymEnv` requires a running ShopGym instance, Playwright
Chromium, and an `ANTHROPIC_API_KEY`. The public upstream implementation is
[agentic-foundation-modeling-research/shop-gym](https://github.com/agentic-foundation-modeling-research/shop-gym).
