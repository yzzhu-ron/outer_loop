"""Minimize existing exploratory caches; this command fits/evaluates no router."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parent
PAPER = ROOT.parents[1]
SEMANTIC = PAPER / "semantic_pilot"


def export():
    inputs = {}
    def read(path):
        inputs[str(path.relative_to(PAPER))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())

    manifest = read(SEMANTIC / "results/manifest.json")
    previous = read(SEMANTIC / "results/paired_outcomes.json")
    previous_models = read(SEMANTIC / "results/models.json")
    natural = read(PAPER / "full_paper/natural/results/paired.json")
    families, environments = {}, {}
    for path in sorted((SEMANTIC / "cache").glob("*_outcomes.json")):
        record = read(path)
        family, name = record["corpus_name"], record["name"]
        query_data = {split: {key: record["tasks"][split][key] for key in ("ids", "features", "buckets")}
                      for split in ("train", "calibration", "test")}
        if family in families and families[family]["partitions"] != query_data:
            raise ValueError(f"Backend query features/IDs differ for {family}")
        timing = manifest["environments"][name]
        all_ids = {qid for ids in timing["selected_task_ids"].values() for qid in ids}
        families.setdefault(family, {"partitions": query_data,
                                    "original_batched_query_encoding_seconds": timing["query_feature_encoding_seconds_shared"],
                                    "original_encoded_query_count": len(all_ids)})
        tasks = {split: {key: record["tasks"][split][key] for key in ("scores", "recalls")}
                 for split in ("train", "calibration", "test")}
        tasks["test"]["action_costs"] = record["tasks"]["test"]["action_costs"]
        ids = query_data["test"]["ids"]
        if ids != previous[name]["query_ids"] or ids != natural[name]["query_ids"]:
            raise ValueError(f"Comparator query IDs differ for {name}")
        environments[name] = {
            "family": family, "backend": record["backend"], "partitions": tasks,
            "archived_source_router_actions": previous[name]["source_router_actions"],
            "archived_source_router_model": previous_models[family + "::" + record["backend"]]["selected"],
            "rrf_all_actions_ndcg": natural[name]["baselines"]["rrf_all_actions"],
        }
    if len(families) != 3 or len(environments) != 6:
        raise ValueError("Expected the six existing development environments")
    payload = {"schema_version": 1, "stage": "Previously inspected development families; no new retrieval or confirmation data",
               "actions": manifest["actions"], "input_sha256": inputs,
               "feature_layout": {"lexical": [0, 8], "normalized_minilm": [8, 392]},
               "families": families, "environments": environments}
    destination = ROOT / "evidence.json"
    destination.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n")
    print(json.dumps({"path": str(destination), "bytes": destination.stat().st_size,
                      "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    export()
