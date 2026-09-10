"""Export strict single-query probe audits without modifying frozen raw traces.

The experiment's raw traces render each action's query group as a newline-
joined string, and report nominal per-selected-pair search costs. Decomposed
actions may fuse multiple retrieval requests and therefore cannot be treated
as one submitted query. This exporter excludes that action from the strict
single-query audit, renames the cost unit, and records the conversion's scope.
It does not alter generation, retrieval, selection, or experiment measurements.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
PACKAGE_SOURCE = ROOT.parents[2] / "packages" / "searchprobe" / "src"
sys.path.insert(0, str(PACKAGE_SOURCE))
from searchprobe import validate_record


STRICT_COST_UNIT = "nominal_selected_probe_search_calls"
SINGLE_QUERY_ACTIONS = {"keywords", "semantic", "hyde"}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON object key {key!r}")
        result[key] = value
    return result


def convert_trace(path):
    """Validate a frozen raw trace, preserving strings and all recorded ranks."""
    raw_bytes = Path(path).read_bytes()
    records, excluded, nominal_raw_cost = [], [], 0
    raw_count = 0
    for line_number, line in enumerate(raw_bytes.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        raw_count += 1
        try:
            record = json.loads(line, object_pairs_hook=_unique_object)
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
        errors = validate_record(record)
        if errors:
            raise ValueError(f"{path}:{line_number}: raw record fails schema validation: {errors}")
        if record["cost"]["unit"] != "search_calls":
            raise ValueError(f"{path}:{line_number}: expected frozen raw cost unit 'search_calls'")
        if record["actions"][0] != "original":
            raise ValueError(f"{path}:{line_number}: expected original as the reference action")
        action = record["actions"][1]
        nominal_raw_cost += record["cost"]["total"]
        if action == "decomposed":
            excluded.append({"probe_id": record["probe_id"], "action": action,
                             "reason": "decomposed_action_may_fuse_multiple_requests",
                             "nominal_pair_cost": record["cost"]["total"]})
            continue
        if action not in SINGLE_QUERY_ACTIONS:
            raise ValueError(f"{path}:{line_number}: unsupported action {action!r}; review the export contract before adding it")
        if record["cost"]["values"] != [1, 1]:
            raise ValueError(f"{path}:{line_number}: a strict single-query pair must declare one nominal request per side")
        strict = deepcopy(record)
        strict["cost"]["unit"] = STRICT_COST_UNIT
        records.append(strict)
    return records, {
        "raw_records": raw_count, "retained_records": len(records), "excluded_records": len(excluded),
        "raw_sha256": sha256(raw_bytes).hexdigest(), "excluded": excluded,
        "raw_nominal_pair_cost_total": nominal_raw_cost,
        "retained_nominal_pair_cost_total": sum(record["cost"]["total"] for record in records),
        "excluded_nominal_pair_cost_total": sum(record["nominal_pair_cost"] for record in excluded),
    }


def export_traces(results_dir, *, python_executable=sys.executable):
    """Export all saved raw traces and run the actual searchprobe CLI on each."""
    results = Path(results_dir)
    raw_files = sorted((results / "probe_traces").glob("*.jsonl"))
    if not raw_files:
        raise ValueError(f"No raw probe traces in {results / 'probe_traces'}; run retrieval first")
    # Validate every source file before writing outputs, and never mutate raw
    # files or the original probe_diagnostics directory.
    converted = [(path, *convert_trace(path)) for path in raw_files]
    strict_dir, reports_dir = results / "strict_probe_traces", results / "strict_probe_diagnostics"
    protected = [(results / name).resolve() for name in ("probe_traces", "probe_diagnostics")]
    destinations = [results / "trace_provenance.json"]
    for source_path in raw_files:
        destinations.extend((strict_dir / source_path.name, reports_dir / (source_path.stem + ".json")))
    if any(destination.resolve().is_relative_to(source) for destination in destinations for source in protected):
        raise ValueError("A derived output resolves inside a protected raw directory; remove the output symlink before exporting")
    strict_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PACKAGE_SOURCE) + (os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else "")
    files = []
    for source_path, records, details in converted:
        strict_path = strict_dir / source_path.name
        report_path = reports_dir / (source_path.stem + ".json")
        strict_text = "".join(json.dumps(record, ensure_ascii=True, allow_nan=False) + "\n" for record in records)
        strict_path.write_text(strict_text, encoding="utf-8")
        command = [str(python_executable), "-m", "searchprobe", "audit", str(strict_path), "--output", str(report_path)]
        completed = subprocess.run(command, env=environment, text=True, capture_output=True, check=False)
        # An empty retained subset deliberately produces the CLI's no-valid-
        # probes error report. Record it as absence of evidence, never a pass.
        if completed.returncode != 0 and not (not records and completed.returncode == 2 and report_path.exists()):
            raise RuntimeError(f"CLI audit failed for {source_path.name}: {completed.stderr.strip()}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        details.update({"raw_trace": str(source_path.relative_to(results)),
                        "strict_trace": str(strict_path.relative_to(results)),
                        "strict_report": str(report_path.relative_to(results)),
                        "strict_sha256": sha256(strict_text.encode("utf-8")).hexdigest(),
                        "strict_audit_status": report["status"], "strict_audit_exit_code": completed.returncode})
        files.append(details)
        print(f"{source_path.name}: retained {len(records)}/{details['raw_records']}; "
              f"excluded {details['excluded_records']} decomposed pairs; strict audit {report['status']}")
    provenance = {
        "schema_version": 1,
        "scope": "Strict single-query audit of admitted experiment candidates, derived from immutable raw traces. This is not a log of probes selected by a deployment policy.",
        "directories": {"raw_traces": "probe_traces", "raw_reports": "probe_diagnostics",
                        "strict_traces": "strict_probe_traces", "strict_reports": "strict_probe_diagnostics"},
        "raw_query_fields": "Raw query strings render action inputs by joining each action's query group with a newline. Decomposed action outcomes may be ranks from fused multiple requests; these renderings are not always single submitted queries.",
        "strict_query_fields": "Only original-vs-keywords/semantic/hyde pairs with one nominal request per side are retained. Their query strings, action names, buckets, and observed target ranks are unchanged. Embedded newlines in a single query are preserved.",
        "exclusion_policy": "All decomposed action pairs are excluded from the strict single-query audit, including any one-request fallback cases, to avoid reinterpreting the frozen action pipeline. This does not exclude them from the experiment.",
        "upstream_filtering": "The raw traces already omit invalid-generation and exact-identical action pairs. Audit collision rates describe admitted candidates, not the unfiltered generator; consult the experiment's excluded-candidate records for upstream failures.",
        "cost_unit": STRICT_COST_UNIT,
        "cost_meaning": "Each pair records the nominal calls required if that pair were selected independently. The harness precomputes action panels, shares reference queries, and may cache results. Summed trace costs are not actual harness/backend requests or total experiment spend; sampling, generation, and setup require the experiment's separate ledger.",
        "conversion": "The only retained-record change is cost.unit: search_calls -> nominal_selected_probe_search_calls. Raw traces and raw reports are preserved; SHA-256 hashes identify the source and derived files.",
        "tool": "searchprobe audit CLI, executed separately for each strict JSONL file",
        "exporter_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "counts": {"files": len(files), "raw_records": sum(item["raw_records"] for item in files),
                   "retained_records": sum(item["retained_records"] for item in files),
                   "excluded_records": sum(item["excluded_records"] for item in files),
                   "files_without_retained_pairs": sum(item["retained_records"] == 0 for item in files)},
        "files": files,
    }
    destination = results / "trace_provenance.json"
    destination.write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote {destination}")
    return provenance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results", help="Experiment result directory containing probe_traces/")
    args = parser.parse_args(argv)
    try:
        export_traces(args.results)
    except (OSError, UnicodeError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
