# Where this work sits

This is a targeted comparison, not a completed novelty search. The public
abstracts below were checked on 10 September 2026. The original proposal
contains a broader starting bibliography. A theoretical observation being
useful for this project does not make it a new theorem.

| Prior work | What is already established | What this experiment tests |
| --- | --- | --- |
| [DREAM: Decoupling Exploration and Exploitation for Meta-Reinforcement Learning without Sacrifices](https://arxiv.org/abs/2008.02790) (ICML 2021) | Separate objectives can learn exploitation that identifies task-relevant information and exploration that recovers it. Generic task-relevant exploration is an existing idea. | Can source real-query labels teach acquisition from synthetic document-rediscovery observations, before any target task query is available? |
| [GPL: Generative Pseudo Labeling for Unsupervised Domain Adaptation of Dense Retrieval](https://arxiv.org/abs/2112.07577) | A query generator plus cross-encoder pseudo-labeling supports unsupervised adaptation of dense retrieval to a target domain. Synthetic-query adaptation is an existing idea and an eventual strong comparator. | Keep the retriever and index frozen; use a small numerical profile to choose among query actions. Does the changed contract preserve useful adaptation at its complete cost? |
| [HyDE: Precise Zero-Shot Dense Retrieval without Relevance Labels](https://arxiv.org/abs/2212.10496) | Query-conditioned hypothetical passages can provide dense retrieval representations without relevance labels. They may contain false details. | HyDE is one action in our shared menu. Its generation is attributed, and the output is never treated as evidence. Applying the passage to BM25 or our pinned MiniLM is an experimental variant, not a reproduction of the original Contriever system. |
| [Adaptive Submodularity: Theory and Applications in Active Learning and Stochastic Optimization](https://arxiv.org/abs/1003.3967) | Greedy adaptive policies have approximation guarantees under structural conditions such as adaptive submodularity. | Our XOR construction shows that unrestricted decision value need not satisfy a diminishing-return condition. Zero one-step gains alone cannot justify stopping. We do not claim a new general greedy guarantee. |

The minimax radius and its dual certificate use standard finite statistical
decision theory and linear-programming duality. The indistinguishability
argument uses standard two-point testing bounds; the noisy reconstruction
bound combines Hoeffding concentration with the argmax inequality. The linear
kernel condition is a familiar observability/identifiability condition, closely
related to ideas used in partial monitoring. The Gaussian variance reduction
is a standard rank-one posterior covariance update.

The concrete output here is a search-specific experimental contract, executable
counterexamples and an audit tool that keeps three questions separate:

1. Can two proposed query actions even produce different observable outcomes?
2. Do those outcomes constrain the useful action under an explicit source model?
3. Does that source relationship transfer to real held-out target tasks?

`searchprobe audit` addresses structural and observed endpoint problems in the
first question. `searchprobe decision-audit` calculates a conditional certificate
for a supplied utility model in the second. Neither command certifies the third;
that needs held-out experiments and justified transfer assumptions.

A stronger research claim would require a search-specific model or acquisition
rule that adds something beyond these ingredients, plus evidence against strong
synthetic-adaptation and inference-time baselines. The current evidence should
be evaluated on those terms, without claiming novelty for value of information,
query generation, or minimax duality themselves.
