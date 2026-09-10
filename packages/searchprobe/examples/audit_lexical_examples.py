"""Adapt the first pilot's recorded generator examples, without retrieval.

This is an audit of the small saved example sample, not the full experiment.
No query strings or outcomes are reconstructed from task labels. The original
pilot's profile logs do not retain query strings, so they cannot be faithfully
converted to the full paired-query contract without rerunning generation.
"""

import argparse
import json
from pathlib import Path

from searchprobe import audit_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probe_audit", type=Path, help="First pilot's results/probe_audit.json")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.resolve() == args.probe_audit.resolve() or (args.output.exists() and args.output.samefile(args.probe_audit)):
        parser.error("--output must differ from the saved pilot audit")
    source = json.loads(args.probe_audit.read_text(encoding="utf-8"))
    records = []
    for i, example in enumerate(source["examples"]):
        records.append({
            "probe_id": f"lexical-saved-example-{i}",
            "family": f"{example['dataset']}:{example['family']}",
            "actions": ["original", example["assigned_alternative"]],
            "queries": [example["probe"].strip(), example["transformed_probe"]],
            "cost": {"values": [0, 0], "total": 0, "unit": "additional_search_calls"},
        })
    report = audit_records(records, min_per_cell=1)
    report["adapter_provenance"] = {
        "source": str(args.probe_audit),
        "scope": "Actual saved generator examples only; an illustrative convenience sample, not an estimate of whole-pilot collision rates.",
        "cost_scope": "No retrieval performed by this adapter. Zero means additional search calls during this audit, not the cost of the original pilot.",
        "outcome_scope": "Saved examples contain no retrieval outcomes; none are fabricated or joined from future-task labels.",
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Audited {len(records)} recorded generator examples; wrote {args.output}")


if __name__ == "__main__":
    main()
