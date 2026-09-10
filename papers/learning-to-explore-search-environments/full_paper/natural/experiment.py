"""Finite categorical acquisition on committed, already-seen retrieval evidence.

This is development, not a confirmation test. Backend identity is hidden from
all selectors in this new experiment. Old experiments are untouched.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np

ROOT = Path(__file__).resolve().parent
METHODS = ('random','coverage','information_gain','decision_entropy','myopic_value','lookahead2')
COST_KEYS = ('search_calls','llm_calls','input_tokens','output_tokens','sample_calls')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def channel(candidate):
    if candidate['family'] not in ('direct_question','indirect_question') or candidate['action'] not in range(1,5):
        raise ValueError('Invalid probe metadata')
    return 4*int(candidate['family']=='indirect_question')+candidate['action']-1


def outcome(value):
    if not np.isfinite(value) or not -1 <= value <= 1:
        raise ValueError('RR difference outside [-1,1]')
    if abs(value) <= 1e-12:
        return 2
    return 0 if value < -.25 else 1 if value < 0 else 3 if value <= .25 else 4


def fit(sources, settings):
    """Only source train utilities and source probe observations enter this fit."""
    utilities, frequencies, likelihood = [], [], []
    for source in sources:
        train = source['tasks']['train']
        scores = np.asarray(train['scores'], dtype=float)
        buckets = np.asarray(train['buckets'], dtype=int)
        counts = np.bincount(buckets,minlength=4)
        global_mean = scores.mean(axis=0)
        alpha = settings['utility_shrinkage']
        utilities.append([(scores[buckets==b].sum(axis=0)+alpha*global_mean)/(counts[b]+alpha) for b in range(4)])
        frequencies.append(counts/len(buckets))
        hist = np.full((8,5),settings['dirichlet_per_outcome'],dtype=float)
        seen = {}
        for pool in source['pools'].values():
            for c in pool['candidates']:
                observed = pool['observations'][c['id']]['delta']
                value = (channel(c),outcome(observed))
                if c['id'] in seen:
                    if seen[c['id']] != value:
                        raise ValueError('Inconsistent repeated source observation')
                    continue
                seen[c['id']] = value
                hist[value] += 1
        likelihood.append(hist/hist.sum(axis=1,keepdims=True))
    utility = np.asarray(utilities)
    if settings.get('shuffle_utilities'):
        utility = utility[np.roll(np.arange(len(sources)),1)]
    return {'names':[s['name'] for s in sources], 'utilities':utility,
            'frequencies':np.mean(frequencies,axis=0), 'likelihood':np.asarray(likelihood)}


def route(model,p):
    return np.argmax(np.einsum('w,wba->ba',p,model['utilities']),axis=1)


def value(model,p):
    expected = np.einsum('w,wba->ba',p,model['utilities'])
    return float(model['frequencies']@np.max(expected,axis=1))


def entropy(p):
    p=np.asarray(p)
    p=p[p>0]
    return float(-np.sum(p*np.log2(p)))


def label_entropy(model,p):
    # Entropy of the entire four-bucket optimal-action vector in the source model.
    labels = np.argmax(model['utilities'],axis=2)
    masses={}
    for label,mass in zip(labels,p):
        key=tuple(label)
        masses[key]=masses.get(key,0)+float(mass)
    return entropy(list(masses.values()))


def branches(model,p,c):
    joint = p[:,None]*model['likelihood'][:,c,:]
    masses=joint.sum(axis=0)
    return [(float(m),joint[:,i]/m) for i,m in enumerate(masses) if m>0]


def gains(model,p,types,method,costs,remaining):
    base=value(model,p)
    result={}
    for c in types:
        kids=branches(model,p,c)
        if method=='information_gain':
            gain=entropy(p)-sum(m*entropy(post) for m,post in kids)
        elif method=='decision_entropy':
            gain=label_entropy(model,p)-sum(m*label_entropy(model,post) for m,post in kids)
        elif method=='myopic_value':
            gain=sum(m*value(model,post) for m,post in kids)-base
        elif method=='lookahead2':
            available=[other for other in types if costs[c]+costs[other]<=remaining]
            terminal=0.0
            for mass,post in kids:
                best=value(model,post)
                for other in available:
                    best=max(best,sum(m*value(model,nextp) for m,nextp in branches(model,post,other)))
                terminal+=mass*best
            gain=terminal-base
        else:
            raise ValueError('Unknown acquisition method')
        result[c]=max(0.0,gain)/costs[c]
    return result


def onboard(model,candidates,observe,method,budget,seed,settings):
    """Candidates have no outcomes; only the selected ID reaches the callback."""
    if method not in METHODS or budget<0:
        raise ValueError('Invalid method/budget')
    forbidden={'delta','scores','observations','utility','qrels','target_tasks','rankings','backend'}
    seen=set()
    for c in candidates:
        if forbidden.intersection(c) or c['id'] in seen or c['cost']<=0:
            raise ValueError('Invalid/outcome-bearing candidate')
        seen.add(c['id']); channel(c)
    rng=np.random.default_rng(seed)
    order=rng.permutation(len(candidates))
    priority={int(i):j for j,i in enumerate(order)}
    left=list(range(len(candidates)))
    p=np.full(len(model['names']),1/len(model['names']))
    trace=[];spent=0;counts=np.zeros(8,dtype=int)
    while left:
        allowed=[i for i in left if candidates[i]['cost']+spent<=budget]
        if not allowed: break
        if method=='random':
            selected=min(allowed,key=priority.get)
        elif method=='coverage':
            selected=min(allowed,key=lambda i:(counts[channel(candidates[i])],channel(candidates[i]),priority[i]))
        else:
            bytype={}
            for i in allowed:
                c=channel(candidates[i]); bytype[c]=min(i,bytype.get(c,i),key=priority.get)
            costs={c:candidates[i]['cost'] for c,i in bytype.items()}
            scores=gains(model,p,list(bytype),method,costs,budget-spent)
            # Equal zero gain falls back to coverage; do not hide useful
            # multi-step evidence by imposing a greedy stop in this natural test.
            selected=min(allowed,key=lambda i:(-round(scores[channel(candidates[i])],14),counts[channel(candidates[i])],priority[i]))
        c=candidates[selected];ch=channel(c)
        observed=observe(c['id'])
        y=outcome(float(observed['delta']))
        likelihood=model['likelihood'][:,ch,y]**settings['likelihood_power']
        p=p*likelihood;p=p/p.sum()
        spent+=c['cost'];counts[ch]+=1
        trace.append({'id':c['id'],'channel':ch,'outcome':y,'delta':observed['delta'],
                      'cost':c['cost'],'posterior':p.tolist(),'action_by_bucket':route(model,p).tolist()})
        # One contrast per original generated-question feature group.
        group=c['question_group']
        left=[i for i in left if candidates[i]['question_group']!=group]
    return p,trace,spent


def ndcg(ranking,qrels):
    ideal=sorted((v for v in qrels.values() if v>0),reverse=True)[:10]
    def dcg(values):return sum((2**v-1)/np.log2(i+2) for i,v in enumerate(values))
    denominator=dcg(ideal)
    return dcg([max(0,qrels.get(d,0)) for d in ranking[:10]])/denominator if denominator else 0.


def rrf(rankings):
    scores={}
    for ranking in rankings:
        for rank,doc in enumerate(ranking,1):scores[doc]=scores.get(doc,0)+1/(60+rank)
    return sorted(scores,key=lambda doc:(-scores[doc],doc))[:10]


def paired_interval(deltas,seed,draws=2000):
    x=np.asarray(deltas)
    rng=np.random.default_rng(seed)
    si=rng.integers(x.shape[0],size=(draws,x.shape[0]))
    qi=rng.integers(x.shape[1],size=(draws,x.shape[1]))
    boot=x[si[:,:,None],qi[:,None,:]].mean(axis=(1,2))
    return {'mean':float(x.mean()),'lo':float(np.quantile(boot,.025)),'hi':float(np.quantile(boot,.975))}


def evaluate(evidence,protocol):
    records=evidence['records'];settings=protocol['settings']
    run_rows=[];baseline_rows=[];outputs={};models={}
    configurations={'standard':{},'utility_rotation':{'shuffle_utilities':True},'tempered':{'likelihood_power':.25}}
    for target in records:
        sources=[s for s in records if s['corpus_name']!=target['corpus_name']]
        if len(sources)!=4:raise ValueError('Exactly four source worlds required')
        target_runs={}
        # First complete source fitting and selected-probe onboarding. No test
        # utilities, task IDs, query features, or backend identity enter either.
        for condition,overrides in configurations.items():
            config=settings|overrides
            model=fit(sources,config)
            models[target['name']+'/'+condition]={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in model.items()}
            for seed,pool in sorted(target['pools'].items()):
                for method in METHODS:
                    for budget in settings['search_call_budgets']:
                        p,trace,spent=onboard(model,pool['candidates'],lambda ident:pool['observations'][ident],method,budget,int(seed),config)
                        target_runs[(condition,seed,method,budget)]={'actions_by_bucket':route(model,p).tolist(),
                            'trace':trace,'search_calls':spent,'posterior':p.tolist()}
        # Evaluator boundary: now reveal existing test outcomes.
        test=target['tasks']['test'];scores=np.asarray(test['scores']);buckets=np.asarray(test['buckets']);row=np.arange(len(scores))
        model=fit(sources,settings);prior=np.full(4,.25)
        zero=route(model,prior)[buckets]
        source_fixed=int(np.argmax(np.mean([np.asarray(s['tasks']['train']['scores']).mean(axis=0) for s in sources],axis=0)))
        fixed_names=('original','keywords','semantic','hyde','decomposed')
        baselines={name:scores[:,i] for i,name in enumerate(fixed_names)}
        baselines['source_fixed']=scores[:,source_fixed]
        baselines['source_query_bucket']=scores[row,zero]
        baselines['target_best_fixed_oracle']=scores[:,int(np.argmax(scores.mean(axis=0)))]
        baselines['query_oracle']=scores.max(axis=1)
        baselines['rrf_all_actions']=np.asarray([ndcg(rrf(ranks),test['qrels'][qid]) for qid,ranks in zip(test['ids'],test['rankings'])])
        for name,values in baselines.items():
            baseline_rows.append({'environment':target['name'],'baseline':name,'ndcg':float(values.mean()),'n':len(scores)})
        saved={}
        for (condition,seed,method,budget),run in target_runs.items():
            actions=np.asarray(run['actions_by_bucket'])[buckets]
            values=scores[row,actions]
            pool=target['pools'][seed]
            setup=pool['setup_costs'] if budget else {}
            cost={key:setup.get(key,0) for key in COST_KEYS};cost['search_calls']+=run['search_calls']
            serving={key:float(np.mean([test['action_costs'][i][a].get(key,0) for i,a in enumerate(actions)])) for key in COST_KEYS}
            run_rows.append({'environment':target['name'],'condition':condition,'seed':seed,'method':method,'budget':budget,
                'ndcg':float(values.mean()),'changed_action_fraction':float(np.mean(actions!=zero)),
                'selected_pairs':len(run['trace']),'onboarding_cost':cost,'per_task_cost':serving})
            saved['/'.join(map(str,(condition,seed,method,budget)))]=run|{'actions':actions.tolist(),'ndcg':values.tolist()}
        outputs[target['name']]={'query_ids':test['ids'],'query_buckets':buckets.tolist(),'baselines':{k:v.tolist() for k,v in baselines.items()},'runs':saved}
        print('finished',target['name'],flush=True)
    intervals=[]
    for name,data in outputs.items():
        seeds=sorted({key.split('/')[1] for key in data['runs']})
        for condition in configurations:
            for method in METHODS:
                for budget in settings['search_call_budgets']:
                    values=np.asarray([data['runs'][f'{condition}/{seed}/{method}/{budget}']['ndcg'] for seed in seeds])
                    for comparator in ('source_query_bucket','original','hyde','rrf_all_actions','random','information_gain','myopic_value'):
                        reference=np.asarray([data['runs'][f'{condition}/{seed}/{comparator}/{budget}']['ndcg'] for seed in seeds]) if comparator in METHODS else np.asarray(data['baselines'][comparator])[None,:]
                        seed=int(hashlib.sha256(f'{name}/{condition}/{method}/{budget}/{comparator}'.encode()).hexdigest()[:8],16)
                        intervals.append({'environment':name,'condition':condition,'method':method,'budget':budget,'comparator':comparator,
                                          **paired_interval(values-reference,seed,settings['bootstrap_draws'])})
    return {'runs':run_rows,'baselines':baseline_rows,'paired':outputs,'models':models,'intervals':intervals}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'results')
    args=parser.parse_args(argv)
    manifest=json.loads((ROOT/'freeze.json').read_text())
    for filename,expected in manifest['sha256'].items():
        if digest(ROOT/filename)!=expected:raise ValueError('Frozen input drift: '+filename)
    if args.output_dir.exists():raise ValueError('Use a new output directory; existing results are immutable')
    args.output_dir.mkdir(parents=True)
    started=time.perf_counter()
    result=evaluate(json.loads((ROOT/'evidence.json').read_text()),json.loads((ROOT/'protocol.json').read_text()))
    for key,values in result.items():
        (args.output_dir/(key+'.json')).write_text(json.dumps(values,indent=2,allow_nan=False)+'\n')
    metadata={'status':'Exploratory development on previously inspected families', 'freeze':manifest,
              'seconds':time.perf_counter()-started,'numpy':np.__version__,'python':platform.python_version(),
              'output_sha256':{p.name:digest(p) for p in sorted(args.output_dir.glob('*.json'))}}
    (args.output_dir/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('seconds',metadata['seconds'])

if __name__=='__main__':main()
