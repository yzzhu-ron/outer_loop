"""Explore a strong simple baseline: source-selected fixed RRF action subsets.

No onboarding and no novelty claim. All data are previously used development
families. The source-selected policy must be selected without target outcomes.
"""
from pathlib import Path
import hashlib
import itertools
import json
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parent
import importlib.util
_spec=importlib.util.spec_from_file_location('natural_fusion_metrics',ROOT.parent/'natural/experiment.py')
_metrics=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_metrics)
ndcg,rrf=_metrics.ndcg,_metrics.rrf
ACTIONS=('original','keywords','semantic','hyde','decomposed')
CALLS=(1,1,1,1,2)
SUBSETS=[list(s) for size in range(1,6) for s in itertools.combinations(range(5),size)]

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def export():
    semantic=ROOT.parents[1]/'semantic_pilot'
    prepared_path=semantic/'cache/prepared.json';prepared=json.loads(prepared_path.read_text())
    records=[];inputs={str(prepared_path.relative_to(semantic)):sha(prepared_path)}
    for path in sorted((semantic/'cache').glob('*_outcomes.json')):
        record=json.loads(path.read_text());inputs[str(path.relative_to(semantic))]=sha(path)
        tasks={}
        for split in ('train','test'):
            raw=record['tasks'][split]
            qrels=prepared['datasets'][record['corpus_name']]['qrels'][split]
            tasks[split]={k:raw[k] for k in ('ids','rankings','action_costs')}
            tasks[split]['qrels']={qid:qrels[qid] for qid in raw['ids']}
        records.append({k:record[k] for k in ('name','corpus_name','backend')}|{'tasks':tasks})
    (ROOT/'evidence.json').write_text(json.dumps({'status':'Previously seen development families','input_sha256':inputs,'records':records},separators=(',',':'))+'\n')

def measure(tasks):
    scores=[]
    for qid,ranks in zip(tasks['ids'],tasks['rankings']):
        scores.append([ndcg(ranks[subset[0]] if len(subset)==1 else rrf([ranks[a] for a in subset]),tasks['qrels'][qid]) for subset in SUBSETS])
    return np.array(scores)

def select(source_means,max_calls,allow_generation):
    eligible=[i for i,s in enumerate(SUBSETS) if sum(CALLS[a] for a in s)<=max_calls and (allow_generation or max(s)<2)]
    means=np.mean(source_means,axis=0)
    return min(eligible,key=lambda i:(-means[i],sum(CALLS[a] for a in SUBSETS[i]),len(SUBSETS[i]),SUBSETS[i]))

def main():
    if '--export' in sys.argv:return export()
    freeze=json.loads((ROOT/'freeze.json').read_text())
    for name,value in freeze['sha256'].items():
        if sha(ROOT/name)!=value:raise ValueError('Frozen input drift: '+name)
    if (ROOT/'results.json').exists():raise ValueError('Preserve existing results')
    started=time.perf_counter()
    records=json.loads((ROOT/'evidence.json').read_text())['records']
    # Compute source training utility tables. Target test utility stays unread.
    train={r['name']:measure(r['tasks']['train']) for r in records}
    choices={}
    for target in records:
        for backend_known in (True,False):
            sources=[s for s in records if s['corpus_name']!=target['corpus_name'] and (not backend_known or s['backend']==target['backend'])]
            source_means=[train[s['name']].mean(axis=0) for s in sources]
            for generation in (False,True):
                for calls in range(1,7):
                    choices[target['name'],backend_known,generation,calls]=select(source_means,calls,generation)
    # Evaluator boundary after every source-only subset has been chosen.
    rows=[];paired={};all_subsets=[]
    for target in records:
        test=target['tasks']['test'];scores=measure(test)
        paired[target['name']]={'query_ids':test['ids'],'subset_scores':scores.tolist()}
        for i,subset in enumerate(SUBSETS):
            all_subsets.append({'environment':target['name'],'subset':subset,'ndcg':float(scores[:,i].mean()),'search_calls':sum(CALLS[a] for a in subset),'uses_generation':max(subset)>=2})
        for (name,known,generation,calls),chosen in choices.items():
            if name!=target['name']:continue
            subset=SUBSETS[chosen]
            cost={key:float(np.mean([max(row[a].get(key,0) for a in subset) for row in test['action_costs']])) for key in ('llm_calls','input_tokens','output_tokens')}
            cost['search_calls']=sum(CALLS[a] for a in subset)
            rows.append({'environment':name,'backend_known':known,'generation_allowed':generation,'maximum_search_calls':calls,'subset_index':chosen,'subset':[ACTIONS[a] for a in subset], 'ndcg':float(scores[:,chosen].mean()),'per_task_cost':cost,
                         'delta_vs_all_action_rrf':float((scores[:,chosen]-scores[:,-1]).mean())})
    result={'status':'Exploratory simple-baseline development; no probe acquisition.', 'freeze':freeze,'subsets':[[ACTIONS[a] for a in s] for s in SUBSETS],'rows':rows,'all_subsets_posthoc_evaluator_only':all_subsets,'paired':paired,'seconds':time.perf_counter()-started}
    (ROOT/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for known in (True,False):
        print('backend_known',known)
        for calls in range(1,7):
            selected=[r for r in rows if r['backend_known']==known and r['generation_allowed'] and r['maximum_search_calls']==calls]
            print(calls,'ndcg',float(np.mean([r['ndcg'] for r in selected])),'actualcalls',float(np.mean([r['per_task_cost']['search_calls'] for r in selected])))

if __name__=='__main__':main()
