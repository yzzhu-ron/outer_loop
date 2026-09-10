# Exact results and what they establish

The suite reproduces four distinct mechanisms: nuisance information can consume a budget; one-step decision value can miss complementary probes; adaptive designs can beat the best fixed design; and a correct planner can still fail when its task mixture or observation-to-utility model is wrong. These are constructed examples of established ideas, not new theorems or evidence of retrieval efficacy.

At the predefined task weight **w = 3/4** and budget **two unit-cost probes**, exact expected correctness is:

| Method | XOR panel | Gated panel |
| --- | ---: | ---: |
| No probe | 1/2 | 1/2 |
| Source-optimal fixed subset | 7/8 | 3/4 |
| Uniform random | 151/240 | 75/112 |
| Environment information gain | 9/16 | 9/16 |
| Decision-region entropy | 5/8 | 5/8 |
| Myopic decision value | 3/4 | 3/4 |
| Forced-budget myopic value | 3/4 | 3/4 |
| Two-step decision lookahead | 7/8 | 7/8 |
| Full-horizon decision DP | 7/8 | 7/8 |
| Privileged full-world oracle | 1 | 1 |

Every operational method except no-probe spends exactly two operations in this table. The full-world oracle has privileged information and is not a deployable cost comparator.

In XOR, the noisy proxy has immediate value while neither component alone predicts parity. At w = 1, myopic value and decision-region entropy select the proxy, achieve 3/4, and stop after one probe even with a budget of two. The component pair achieves one. The best fixed subset matches the adaptive planner in the XOR panel; this example establishes a planning trap, not an adaptive-selection advantage.

The gated panel supplies that separate comparison. A gate observation tells the planner whether to inspect `r` or `s`. Gate-first adaptation achieves 7/8 at w = 3/4 with two probes, versus 3/4 for the exhaustive best fixed subset. At four probes the adaptive planner achieves one, while the best fixed subset achieves 15/16. This is an exact mechanism comparison within the supplied world model.

Perfect decisions do not require complete environment identification. At w = 1 and two probes, the optimal XOR design retains four bits of environment entropy and the optimal gated design retains five; both obtain correctness one. All 15 XOR probe pairs and all 21 gated probe pairs have two-way equality/difference witnesses proving Blackwell incomparability. The high-entropy nuisance channel is therefore not being claimed Blackwell superior to the task probes.

The controls delimit these conclusions:

- **Task mixture:** retaining source w = 3/4 when the actual workload changes to w = 1/4 reduces two-probe DP quality to 5/8 in both panels. Supplying the new mixture before acquisition restores 7/8. Decision-region entropy achieves 7/8 under the hidden shift, exceeding the misspecified decision planner. Observations alone cannot reveal this independently changed workload.
- **Channel mismatch:** at extra bit-flip noise 1/4, the two-probe XOR source planner obtains 19/32, versus 5/8 for DP with calibrated channels. Gated DP obtains 41/64 with either model at this budget: calibration is not always beneficial. At noise 1/2 every non-world-oracle method has exact quality 1/2.
- **Shuffled source alignment:** with five fixed utility-row permutations, four-probe DP quality ranges from 27/64 to 141/256 in XOR (mean 627/1280), and from 119/256 to 145/256 in gated (mean 333/640). The control preserves utility marginals while disrupting their relationship to observations. Five permutations do not justify a population significance claim.
- **Candidate-order ties:** at w = 3/4 and two probes, environment-IG quality ranges from 1/2 to 9/16 in XOR and from 1/2 to 11/16 in gated across the three registered priority orders. Myopic value remains 3/4 and decision-region entropy remains 5/8 at that point. At four probes, tie order matters more: XOR environment IG ranges from 5/8 to 15/16 at w = 3/4, and forced myopic value ranges from 3/4 to one at w = 1. The two-probe planning examples do not imply a tie-order-independent greedy ranking at other budgets. All order-sensitive results remain in the ledger.

The useful research artifact is the falsifiable test suite and its strong comparators. A retrieval contribution still requires a transferable observation/utility model, realistic probe costs, meaningful action diversity, and natural-environment evidence. Abstract unit-cost probes, supplied finite worlds, exact likelihoods, and an exhaustive eight-router menu make this suite unsuitable for estimating real deployment gains.
