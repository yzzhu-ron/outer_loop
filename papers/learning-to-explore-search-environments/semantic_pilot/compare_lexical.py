"""Compare unchanged action outcomes with the preserved lexical pilot."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    old_path = ROOT.parent / 'pilot' / 'results' / 'paired_outcomes.json'
    new_path = ROOT / 'results' / 'paired_outcomes.json'
    old, new = (json.loads(p.read_text()) for p in (old_path, new_path))
    rows = []
    for environment, previous in sorted(old.items()):
        current = new[environment]
        if previous['query_ids'] != current['query_ids']:
            raise ValueError(f'{environment}: cannot compare differently ordered query cohorts')
        for index, action in enumerate(('original', 'keywords')):
            deltas = [b[index] - a[index] for a, b in zip(previous['action_ndcg'], current['action_scores'], strict=True)]
            rows.append({'environment': environment, 'action': action, 'n_queries': len(deltas),
                         'changed_query_count_at_1e_minus_12': sum(abs(x) > 1e-12 for x in deltas),
                         'maximum_absolute_ndcg_difference': max(map(abs, deltas)),
                         'mean_ndcg_difference': sum(deltas) / len(deltas)})
    (ROOT / 'results' / 'lexical_comparison.json').write_text(json.dumps({
        'unchanged_action_checks': rows,
        'all_shared_actions_match': all(r['changed_query_count_at_1e_minus_12'] == 0 for r in rows),
        'input_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (old_path, new_path)},
        'interpretation': 'Original and keyword queries, corpus, eligibility rule, backend and test cohort are unchanged. Generated action menus and source routers differ between pilots, so their changes cannot be attributed to a single factor.'
    }, indent=2) + '\n')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
