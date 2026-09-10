"""Export existing development evidence; never fit an acquisition policy here."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parent
SEMANTIC = ROOT.parents[1] / 'semantic_pilot'

def export():
    prepared_path = SEMANTIC / 'cache/prepared.json'
    prepared = json.loads(prepared_path.read_text())
    inputs = {str(prepared_path.relative_to(SEMANTIC)): hashlib.sha256(prepared_path.read_bytes()).hexdigest()}
    records = []
    for path in sorted((SEMANTIC / 'cache').glob('*_outcomes.json')):
        record = json.loads(path.read_text())
        inputs[str(path.relative_to(SEMANTIC))] = hashlib.sha256(path.read_bytes()).hexdigest()
        tasks = {}
        for split in ('train', 'test'):
            tasks[split] = {k:record['tasks'][split][k] for k in ('ids','buckets','scores','recalls','action_costs')}
        test = tasks['test']
        test['rankings'] = record['tasks']['test']['rankings']
        qrels = prepared['datasets'][record['corpus_name']]['qrels']['test']
        test['qrels'] = {qid:qrels[qid] for qid in test['ids']}
        pools = {}
        for seed,pool in record['pools'].items():
            candidates = []
            for candidate in pool['candidates']:
                c = {k:candidate[k] for k in ('id','action','bucket','family','cost')}
                # Same generated question has identical metadata across actions.
                # This conservative grouping can merge coincident feature vectors.
                group = [candidate['family'],candidate['bucket'],candidate['features']]
                c['question_group'] = hashlib.sha256(json.dumps(group,sort_keys=True).encode()).hexdigest()[:20]
                candidates.append(c)
            pools[seed] = {'candidates':candidates, 'observations':pool['observations'], 'setup_costs':pool['setup_costs']}
        records.append({k:record[k] for k in ('name','backend','corpus_name')} | {'tasks':tasks,'pools':pools})
    payload = {'status':'Previously inspected development families only; no new confirmation evidence.',
               'export_input_sha256':inputs,'records':records}
    output = ROOT / 'evidence.json'
    output.write_text(json.dumps(payload,separators=(',',':'),allow_nan=False)+'\n')
    print(output, output.stat().st_size, hashlib.sha256(output.read_bytes()).hexdigest())

if __name__ == '__main__':
    export()
