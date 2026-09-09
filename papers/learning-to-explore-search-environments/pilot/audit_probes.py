"""Audit the frozen generator without retrieval outcomes or task labels."""
import json
from pathlib import Path
import random

from data import load_dataset, make_split
from engine import ACTIONS, feature_bucket, generate_probe, tokenize, transform

ROOT = Path(__file__).resolve().parent


def main():
    rows, examples = [], []
    for name in ('scifact', 'fiqa'):
        docs, _, _ = load_dataset(ROOT / 'data', name)
        exploration, _, _ = make_split(docs)
        by_id = {doc['_id']: doc for doc in docs}
        for seed in (11, 23, 47):
            ids = sorted(exploration)
            random.Random(seed).shuffle(ids)
            for family in ('exact_title', 'body_terms'):
                buckets = [0] * 4
                valid = identical = 0
                alternative_counts = {action: {'assigned': 0, 'identical': 0} for action in ACTIONS[1:]}
                lengths = []
                for i, doc_id in enumerate(ids[:64]):
                    query = generate_probe(by_id[doc_id], family)
                    if not query:
                        continue
                    valid += 1
                    buckets[feature_bucket(query)] += 1
                    lengths.append(len(tokenize(query)))
                    action = ACTIONS[1 + i % 3]
                    transformed = transform(query, action)
                    alternative_counts[action]['assigned'] += 1
                    is_identical = transformed == transform(query, 'original')
                    identical += is_identical
                    alternative_counts[action]['identical'] += is_identical
                    if seed == 11 and i < 3:
                        examples.append({'dataset': name, 'family': family,
                            'sampled_document_id': doc_id, 'title': by_id[doc_id]['title'],
                            'body_excerpt': by_id[doc_id]['text'][:250],
                            'probe': query, 'assigned_alternative': action,
                            'transformed_probe': transformed, 'identical_pair': is_identical})
                rows.append({'dataset': name, 'seed': seed, 'family': family,
                    'sampled_documents': min(64, len(ids)), 'valid': valid,
                    'invalid': min(64, len(ids)) - valid, 'identical_pairs': identical,
                    'nonidentical_pairs': valid - identical, 'buckets': buckets,
                    'minimum_query_tokens': min(lengths) if lengths else None,
                    'maximum_query_tokens': max(lengths) if lengths else None,
                    'action_pairs': alternative_counts})
    output = ROOT / 'results' / 'probe_audit.json'
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps({'scope': 'Offline audit; these text/ID examples never enter runtime profiles.',
                                  'histories': rows, 'examples': examples}, indent=2) + '\n')
    print(output)


if __name__ == '__main__':
    main()
