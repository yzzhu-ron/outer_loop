"""Preserve realized generated text and validity/cost metadata for inspection.

The downloaded corpora and expensive local caches remain ignored. This smaller
public evidence export contains the actual query variants used by the run,
including invalid raw responses; it contains no model weights or qrels.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    prepared_path = ROOT / 'cache' / 'prepared.json'
    prepared = json.loads(prepared_path.read_text())
    output = ROOT / 'results' / 'generation_records.jsonl'
    counts = {}
    with output.open('w') as handle:
        for corpus, data in prepared['datasets'].items():
            rows = []
            for qid, record in data['task_rewrites'].items():
                rows.append({'corpus': corpus, 'kind': 'task_rewrite', 'query_id': qid,
                             'query': data['queries'][qid], 'generation': record})
            for document_id, record in data['probe_questions'].items():
                rows.append({'corpus': corpus, 'kind': 'probe_questions', 'source_document_id': document_id,
                             'generation': record})
            for key, record in data['probe_rewrites'].items():
                document_id, family = key.rsplit(':', 1)
                rows.append({'corpus': corpus, 'kind': 'probe_rewrite', 'source_document_id': document_id,
                             'family': family, 'query': data['probe_questions'][document_id][family], 'generation': record})
            counts[corpus] = len(rows)
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n')
    (ROOT / 'results' / 'generation_export.json').write_text(json.dumps({
        'records_per_corpus': counts, 'file': output.name,
        'sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
        'prepared_sha256': hashlib.sha256(prepared_path.read_bytes()).hexdigest(),
        'contract': prepared['contract'],
        'scope': 'Realized generated text, source document IDs, query IDs and generation metadata; no qrels. Corpus contents come from the checksum-pinned archives.',
        'source_truncation': 'See generation.py for fixed character bounds and per-record warnings.'
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
