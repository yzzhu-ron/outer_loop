"""Command-line interface. JSON report on stdout or in --output; summary on stderr."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .audit import audit_jsonl


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="searchprobe", description="Audit search-probe logs, source decision models, or conditional router score bounds.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit", help="Check query collisions, outcome contrast, coverage, and costs")
    audit.add_argument("jsonl", type=Path, help="Input UTF-8 JSONL, one probe per nonblank line")
    audit.add_argument("--output", default="-", help="JSON report path; - (default) writes to stdout")
    audit.add_argument("--min-per-cell", type=int, default=3, help="Coverage warning threshold, not a power calculation (default: 3)")
    audit.add_argument("--expect-pair", nargs=2, action="append", metavar=("LEFT", "RIGHT"), help="Declare an expected unordered action pair; repeat for additional pairs")
    audit.add_argument("--expect-bucket", action="append", help="Declare an expected bucket; repeat for additional buckets")
    audit.add_argument("--fail-on-warnings", action="store_true", help="Return exit code 1 when structural/outcome warnings are present")
    decision = subparsers.add_parser("decision-audit", help="Compute conditional decision radius from labeled source utilities; never a target guarantee")
    decision.add_argument("model", type=Path, help="One source_utility_model JSON object, with provenance and assumptions")
    decision.add_argument("--output", default="-", help="JSON report path; - (default) writes to stdout")
    decision.add_argument("--max-vertex-systems", type=int, default=20_000, help="Exact solver work limit (default: 20000); large models need an LP solver")
    response = subparsers.add_parser("response-audit", help="Certify conditional argmax stability under explicitly supplied score-correction bounds")
    response.add_argument("model", type=Path, help="One score_box_model JSON object, with provenance and bound assumptions")
    response.add_argument("--output", default="-", help="JSON report path; - (default) writes to stdout")
    args = parser.parse_args(argv)
    try:
        source_path = args.jsonl if args.command == "audit" else args.model
        if args.output != "-":
            output_path = Path(args.output)
            if output_path.resolve() == source_path.resolve() or (output_path.exists() and source_path.exists() and output_path.samefile(source_path)):
                parser.error("--output must differ from the input log")
        if args.command == "audit":
            report = audit_jsonl(args.jsonl, expected_action_pairs=args.expect_pair,
                                 expected_buckets=args.expect_bucket, min_per_cell=args.min_per_cell)
        elif args.command == "decision-audit":
            # Keep this optional source-label pathway out of the default audit.
            from .decisions import audit_decision_json
            report = audit_decision_json(args.model, max_vertex_systems=args.max_vertex_systems)
        else:
            from .responsiveness import audit_response_json
            report = audit_response_json(args.model)
        rendered = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        if args.output == "-":
            sys.stdout.write(rendered)
        else:
            Path(args.output).write_text(rendered, encoding="utf-8")
    except (OSError, UnicodeError, ValueError, ArithmeticError) as exc:
        parser.error(str(exc))
    if args.command == "decision-audit":
        print(f"searchprobe decision-audit: model={report['model_id']}; source-model radius={report['decision_radius']['exact']}; "
              f"{report['certificate']['witness_support']} witness worlds", file=sys.stderr)
        print("  Conditional on the supplied source model; not a target-regret guarantee.", file=sys.stderr)
        return 0
    if args.command == "response-audit":
        print(f"searchprobe response-audit: model={report['model_id']}; "
              f"{report['counts']['certified_unchanged']}/{report['counts']['rows']} rows certified unchanged; "
              f"absolute mean utility-change bound={report['mean_utility_change_bound']['absolute_upper_bound']['exact']}", file=sys.stderr)
        print("  Conditional score-box result; supplied bounds are not verified. Not a target-regret or generalization guarantee.", file=sys.stderr)
        return 0
    counts = report["counts"]
    print(f"searchprobe: {report['status']}; {counts['valid_records']}/{counts['records']} valid probes; "
          f"{counts['exact_identical_query_pairs']} exact query collisions; "
          f"{counts['nonzero_contrast_pairs']}/{counts['observed_records']} observed endpoint contrasts", file=sys.stderr)
    for diagnostic in report["diagnostics"]:
        print(f"  {diagnostic['severity']}: {diagnostic['code']}: {diagnostic['evidence']}", file=sys.stderr)
    if report["status"] == "error":
        return 2
    if report["status"] == "warning" and args.fail_on_warnings:
        return 1
    return 0
