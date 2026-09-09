"""Read BEIR archives and construct evaluator-owned document-disjoint splits.

The corpus remains intact. These helpers belong to the offline data/evaluation
layer; target adaptation must not receive qrels or query eligibility decisions.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import unicodedata
from zipfile import ZipFile


def _identifier(value, context: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError(f"{context}: expected a string or integer identifier")
    result = str(value)
    if not result:
        raise ValueError(f"{context}: empty identifier")
    return result


def _jsonl_rows(archive: ZipFile, member: str):
    with archive.open(member) as raw:
        with io.TextIOWrapper(raw, encoding="utf-8-sig") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{member}:{line_number}: invalid JSON") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"{member}:{line_number}: expected an object")
                yield line_number, row


def load_dataset(data_dir, name: str):
    """Return ``(docs, queries, qrels_by_split)`` from ``data_dir/name.zip``.

    Members are streamed directly from the ZIP; no paths are extracted. BEIR's
    usual dataset-named root and a flat archive root are supported. Malformed
    records, duplicate IDs, and conflicting duplicate judgments fail loudly.
    Missing document references are audited by :func:`eligible_queries` so the
    evaluator can record their effect on query retention.
    """
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError("name must be a simple dataset identifier")
    archive_path = Path(data_dir) / f"{name}.zip"
    with ZipFile(archive_path) as archive:
        names = archive.namelist()
        name_set = set(names)
        roots = [root for root in (f"{name}/", "")
                 if f"{root}corpus.jsonl" in name_set
                 and f"{root}queries.jsonl" in name_set]
        if len(roots) != 1:
            raise ValueError(f"{archive_path}: expected one corpus/queries root")
        root = roots[0]
        corpus_member = f"{root}corpus.jsonl"
        query_member = f"{root}queries.jsonl"
        qrels_prefix = f"{root}qrels/"
        qrels_members = sorted(member for member in name_set
                               if member.startswith(qrels_prefix)
                               and re.fullmatch(r"[A-Za-z0-9_-]+\.tsv",
                                                member[len(qrels_prefix):]))
        if not qrels_members:
            raise ValueError(f"{archive_path}: no qrels TSV files")
        for member in [corpus_member, query_member, *qrels_members]:
            if names.count(member) != 1:
                raise ValueError(f"{archive_path}: duplicate ZIP member {member}")

        docs = []
        doc_ids = set()
        for line_number, row in _jsonl_rows(archive, corpus_member):
            context = f"{corpus_member}:{line_number}"
            doc_id = _identifier(row.get("_id"), context)
            if doc_id in doc_ids:
                raise ValueError(f"{context}: duplicate document ID {doc_id}")
            title, body = row.get("title", ""), row.get("text", "")
            if not isinstance(title, str) or not isinstance(body, str):
                raise ValueError(f"{context}: title/text must be strings")
            docs.append({"_id": doc_id, "title": title, "text": body})
            doc_ids.add(doc_id)

        queries = {}
        for line_number, row in _jsonl_rows(archive, query_member):
            context = f"{query_member}:{line_number}"
            query_id = _identifier(row.get("_id"), context)
            if query_id in queries:
                raise ValueError(f"{context}: duplicate query ID {query_id}")
            if not isinstance(row.get("text"), str):
                raise ValueError(f"{context}: query text must be a string")
            queries[query_id] = row["text"]

        qrels_by_split = {}
        for member in qrels_members:
            split = member[len(qrels_prefix):-4]
            qrels = {}
            with archive.open(member) as raw:
                with io.TextIOWrapper(raw, encoding="utf-8-sig", newline="") as stream:
                    reader = csv.DictReader(stream, delimiter="\t")
                    if reader.fieldnames != ["query-id", "corpus-id", "score"]:
                        raise ValueError(f"{member}: unexpected qrels header")
                    for row in reader:
                        context = f"{member}:{reader.line_num}"
                        query_id = _identifier(row.get("query-id"), context)
                        doc_id = _identifier(row.get("corpus-id"), context)
                        if query_id not in queries:
                            raise ValueError(f"{context}: qrels references unknown query {query_id}")
                        try:
                            score = int(row["score"])
                        except (KeyError, ValueError, TypeError) as exc:
                            raise ValueError(f"{context}: relevance score must be an integer") from exc
                        judgments = qrels.setdefault(query_id, {})
                        if doc_id in judgments and judgments[doc_id] != score:
                            raise ValueError(f"{context}: conflicting duplicate judgment")
                        judgments[doc_id] = score
            qrels_by_split[split] = qrels
    return docs, queries, qrels_by_split


def _normalized_content(document: dict) -> tuple[str, bool, bool]:
    def normalize(value: str) -> str:
        return " ".join(unicodedata.normalize("NFKC", value).casefold().split())

    body = normalize(document.get("text", ""))
    title = normalize(document.get("title", ""))
    # Body equality groups title variants. Empty bodies use title equality.
    content = f"body\0{body}" if body else f"title\0{title}"
    return content, not bool(body), not bool(body or title)


def make_split(docs, seed: int = 20260909, exploration_fraction: float = 0.2):
    """Hash normalized duplicate groups into exploration and held-out pools.

    Assignment is independent of qrels, input order, and query content. The
    fraction is a group assignment probability, not an exact document quota.
    The function never removes documents from the retrieval corpus.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if not isinstance(exploration_fraction, (int, float)) or not math.isfinite(exploration_fraction):
        raise ValueError("exploration_fraction must be finite")
    if not 0 <= exploration_fraction <= 1:
        raise ValueError("exploration_fraction must be between zero and one")
    groups = {}
    all_ids = set()
    empty_body_count = empty_content_count = 0
    for document in docs:
        doc_id = _identifier(document.get("_id"), "split document")
        if doc_id in all_ids:
            raise ValueError(f"duplicate document ID {doc_id}")
        all_ids.add(doc_id)
        content, empty_body, empty_content = _normalized_content(document)
        empty_body_count += int(empty_body)
        empty_content_count += int(empty_content)
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        groups.setdefault(digest, []).append(doc_id)

    exploration_ids = set()
    exploration_group_count = 0
    for digest, group_ids in groups.items():
        assignment = hashlib.sha256(f"{seed}:{digest}".encode("ascii")).digest()
        draw = int.from_bytes(assignment[:8], "big") / 2**64
        if draw < exploration_fraction:
            exploration_ids.update(group_ids)
            exploration_group_count += 1
    heldout_ids = all_ids - exploration_ids
    duplicate_groups = [ids for ids in groups.values() if len(ids) > 1]
    audit = {
        "seed": seed,
        "requested_exploration_fraction": exploration_fraction,
        "assignment": "sha256(seed:sha256(normalized_content)), first 64 bits / 2**64",
        "normalization": "NFKC, casefold, whitespace collapse; body text, title fallback for empty body",
        "document_count": len(all_ids),
        "content_group_count": len(groups),
        "exploration_document_count": len(exploration_ids),
        "heldout_document_count": len(heldout_ids),
        "exploration_group_count": exploration_group_count,
        "heldout_group_count": len(groups) - exploration_group_count,
        "duplicate_group_count": len(duplicate_groups),
        "documents_in_duplicate_groups": sum(len(ids) for ids in duplicate_groups),
        "duplicate_excess_document_count": sum(len(ids) - 1 for ids in duplicate_groups),
        "empty_body_document_count": empty_body_count,
        "empty_content_document_count": empty_content_count,
        "document_id_overlap_count": len(exploration_ids & heldout_ids),
    }
    return exploration_ids, heldout_ids, audit


def eligible_queries(qrels, heldout_ids, all_doc_ids):
    """Retain queries with nonempty positive support entirely held out.

    Reject any query referencing a missing corpus document, even if that
    judgment is nonpositive. Returned qrels retain all original judgments for
    eligible queries; nonpositive judgments need not belong to the held-out
    pool. Exclusion reason counts are mutually exclusive in the order below.
    """
    heldout_ids, all_doc_ids = set(heldout_ids), set(all_doc_ids)
    if not heldout_ids <= all_doc_ids:
        raise ValueError("heldout_ids must be a subset of the full corpus IDs")
    kept = {}
    missing_document_ids = set()
    counts = {
        "missing_document_query_count": 0,
        "no_positive_support_query_count": 0,
        "partially_heldout_query_count": 0,
        "no_heldout_support_query_count": 0,
    }
    positive_qrel_count = 0
    eligible_positive_qrel_count = 0
    for query_id in sorted(qrels):
        judgments = qrels[query_id]
        if any(isinstance(score, bool) or not isinstance(score, int)
               for score in judgments.values()):
            raise ValueError(f"query {query_id}: relevance scores must be integers")
        positive_ids = {doc_id for doc_id, score in judgments.items() if score > 0}
        positive_qrel_count += len(positive_ids)
        missing = set(judgments) - all_doc_ids
        if missing:
            missing_document_ids.update(missing)
            counts["missing_document_query_count"] += 1
        elif not positive_ids:
            counts["no_positive_support_query_count"] += 1
        elif positive_ids <= heldout_ids:
            kept[query_id] = dict(judgments)
            eligible_positive_qrel_count += len(positive_ids)
        elif positive_ids & heldout_ids:
            counts["partially_heldout_query_count"] += 1
        else:
            counts["no_heldout_support_query_count"] += 1
    audit = {
        "rule": "nonempty positive support; every judged document exists; all positive support held out",
        "original_query_count": len(qrels),
        "eligible_query_count": len(kept),
        "excluded_query_count": len(qrels) - len(kept),
        "query_retention_fraction": len(kept) / len(qrels) if qrels else 0.0,
        "original_positive_judgment_count": positive_qrel_count,
        "eligible_positive_judgment_count": eligible_positive_qrel_count,
        "missing_document_count": len(missing_document_ids),
        "missing_document_ids": sorted(missing_document_ids),
        **counts,
    }
    return kept, audit
