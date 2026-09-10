"""POSTHOC source-only support check for a possible contextual development test.

No target test utilities enter this exploratory calculation. The monolithic
query-evidence archive also contains prior endpoint scores; only its train/cal
query metadata is indexed here. This is not a registration or a transfer claim.
"""

from pathlib import Path
import argparse
import hashlib
import json

import numpy as np


ROOT = Path(__file__).resolve().parent
PAPER = ROOT.parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit():
    paths = [ROOT.parent / "intervention_transfer/inputs.v1.json",
             ROOT.parent / "query_transfer/evidence.json"]
    inputs, query = [json.loads(p.read_text()) for p in paths]
    data, support = {}, {}
    for family, corpus in inputs["corpora"].items():
        data[family], support[family] = {}, {}
        for part in ("train", "calibration"):
            q = query["families"][family]["partitions"][part]
            worlds = [corpus["worlds"][str(j)][part] for j in range(len(inputs["lambda_grid"]))]
            assert all(world["ids"] == q["ids"] for world in worlds)
            scores = np.array([world["scores"] for world in worlds])[:, :, :11]
            buckets = np.array(q["buckets"])
            counts = np.bincount(buckets, minlength=4)
            # Every existing source partition has all four buckets. Do not
            # silently extrapolate this small exploratory check to empty cells.
            assert (counts > 0).all()
            conditional = np.stack([scores[:, buckets == b].mean(axis=1) for b in range(4)])
            data[family][part] = scores, buckets, conditional
            support[family][part] = counts.tolist()
    folds = {}
    for target in sorted(data):
        sources = [family for family in sorted(data) if family != target]
        global_gap = np.mean([data[f]["train"][0].mean(axis=(0, 1)) for f in sources], axis=0)
        global_gap -= global_gap[0]
        bucket_gap = np.mean([(data[f]["train"][2] - data[f]["train"][2][:, :, :1]).mean(axis=1)
                              for f in sources], axis=0)
        grid = []
        for eta in (0., .5, 1.):
            policies = (global_gap[None, :] + eta * (bucket_gap - global_gap[None, :])).argmax(axis=1)
            values = []
            for source in sources:
                scores, buckets, _ = data[source]["calibration"]
                values.append(float(scores[:, np.arange(len(buckets)), policies[buckets]].mean()))
            grid.append({"eta": eta, "policies_by_bucket": policies.tolist(),
                         "source_calibration_ndcg": float(np.mean(values)),
                         "per_source_calibration_ndcg": dict(zip(sources, values))})
        best = max(item["source_calibration_ndcg"] for item in grid)
        selected = next(item for item in grid if item["source_calibration_ndcg"] >= best - 1e-12)
        folds[target] = {"sources": sources, "source_calibration_grid": grid,
                         "selected_eta": selected["eta"],
                         "calibration_gain_over_global": selected["source_calibration_ndcg"] - grid[0]["source_calibration_ndcg"]}
    return {"stage": "POSTHOC SOURCE-ONLY EXPLORATORY SUPPORT CHECK",
            "limitations": "No target test scores enter any calculation. The monolithic query archive physically also contains prior endpoint scores; only train/cal query metadata is indexed. No new-family performance claim. Sparse source cells and source-calibration selection remain limitations.",
            "rule": "Train source global and four-bucket relative-to-original utility means from128 queries; average lambdas and source families equally. Shrink bucket gap to global with eta in[0,.5,1], choose on32 source calibration queries per family with uniform lambdas; ties prefer smaller eta.",
            "policy_ties": "Lowest policy index; all11 policies within cap2. No probe-dependent term.",
            "source_sha256": {str(p.relative_to(PAPER)): digest(p) for p in paths},
            "script_sha256": digest(Path(__file__)), "bucket_counts": support, "folds": folds}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "posthoc_context_support.v1.json")
    args = parser.parse_args()
    result = audit()
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "folds": result["folds"]}, indent=2))


if __name__ == "__main__":
    main()
