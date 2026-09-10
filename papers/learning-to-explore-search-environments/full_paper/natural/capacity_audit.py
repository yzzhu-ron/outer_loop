"""Post-hoc evaluator: can any source-world posterior route well on this target?

Target labels enter only this audit, never the frozen acquisition experiment.
Weak inequalities allow arbitrary tie resolution, giving an optimistic upper
bound. A verified NumPy argmax witness provides a realizable lower bound. Report
the bracket rather than treating LP numerical feasibility as an exact theorem.
"""
from __future__ import annotations
import itertools
import json
from pathlib import Path
import hashlib
import numpy as np
from scipy.optimize import linprog

ROOT=Path(__file__).resolve().parent

def audit(utilities,buckets,scores):
    u=np.asarray(utilities,dtype=float)
    score=np.asarray(scores,dtype=float)
    buckets=np.asarray(buckets,dtype=int)
    w,b,a=u.shape
    sums=np.array([score[buckets==j].sum(axis=0) for j in range(b)])/len(score)
    policies=list(itertools.product(range(a),repeat=b))
    values={policy:float(sum(sums[j,policy[j]] for j in range(b))) for policy in policies}
    ordered=sorted(policies,key=lambda p:(-values[p],p))
    witness_best=None;upper=None;tested=0;status_counts={}
    for policy in ordered:
        # Stop once no remaining policy can improve the best verified witness.
        if witness_best is not None and values[policy]<witness_best['ndcg']-1e-12:break
        tested+=1
        constraints=np.array([u[:,j,other]-u[:,j,chosen] for j,chosen in enumerate(policy) for other in range(a) if other!=chosen])
        result=linprog(np.zeros(w),A_ub=constraints,b_ub=np.zeros(len(constraints)),A_eq=np.ones((1,w)),b_eq=[1.],bounds=[(0.,1.)]*w,method='highs')
        status_counts[str(result.status)]=status_counts.get(str(result.status),0)+1
        if not result.success:
            if result.status!=2:
                raise RuntimeError('LP failed without an infeasibility result: '+str(result.status))
            continue
        if not np.isfinite(result.x).all() or np.min(result.x)<-1e-7 or abs(float(np.sum(result.x))-1)>1e-7:
            raise ValueError('Invalid posterior witness')
        p=np.maximum(result.x,0);p=p/p.sum()
        residual=float(np.max(constraints@p))
        if residual>1e-7:raise ValueError('LP witness violates constraints')
        if upper is None:upper={'ndcg':values[policy],'relaxed_actions':list(policy),'posterior':p.tolist(),'max_constraint_residual':residual}
        actual=tuple(np.argmax(np.einsum('w,wba->ba',p,u),axis=1))
        realized=values[actual]
        if witness_best is None or realized>witness_best['ndcg']:
            witness_best={'ndcg':realized,'actions':[int(a) for a in actual],'posterior':p.tolist()}
    if upper is None or witness_best is None:raise ValueError('No feasible posterior policy')
    target_best_bucket=float(sum(np.max(s) for s in sums))
    return {'relaxed_upper_bound':upper,'verified_witness_lower_bound':witness_best,
            'bracket_width':upper['ndcg']-witness_best['ndcg'],'target_bucket_oracle':target_best_bucket,
            'target_query_oracle':float(score.max(axis=1).mean()),'policies_examined':tested,'solver_status_counts':status_counts,
            'method':'Enumeration of all 5^4 bucket policies, sorted by evaluator utility, with floating LP feasibility and verified deterministic argmax witnesses.'}

def main():
    evidence=json.loads((ROOT/'evidence.json').read_text())
    models=json.loads((ROOT/'results/models.json').read_text())
    baselines=json.loads((ROOT/'results/baselines.json').read_text())
    rows=[]
    for record in evidence['records']:
        test=record['tasks']['test'];u=models[record['name']+'/standard']['utilities']
        result=audit(u,test['buckets'],test['scores'])
        result['environment']=record['name']
        result['baselines']={r['baseline']:r['ndcg'] for r in baselines if r['environment']==record['name']}
        rows.append(result)
        print(record['name'],result['verified_witness_lower_bound']['ndcg'],result['relaxed_upper_bound']['ndcg'],result['target_bucket_oracle'])
    inputs=[ROOT/'evidence.json',ROOT/'results/models.json',ROOT/'results/baselines.json',Path(__file__)]
    output={'status':'Post-hoc diagnostic after natural-development-v1 results. Not a deployable target-label-free policy.',
            'numerical_limit':'Floating LP, tolerance1e-7 for feasibility; no claim of exact rational infeasibility certificate. Upper bounds allow arbitrary tie resolution, lower witnesses use deterministic argmax.',
            'input_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},'rows':rows}
    (ROOT/'capacity_audit.json').write_text(json.dumps(output,indent=2)+'\n')

if __name__=='__main__':main()
