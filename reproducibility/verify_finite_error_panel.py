"""Audit the finite-tolerance sensitivity calculation from target moments.

The historical script name is retained; there is no finite-tolerance panel
in the current main Figure 2.

No approximate implementation channel or noise process is constructed here.
Epsilon is a prescribed synthesis tolerance in the proved lower-bound theorem,
not a measured or simulated implementation error.
"""
from pathlib import Path
import hashlib
import json
import math
import csv
import numpy as np

from paths import ROOT
OUT=ROOT/'reproducibility/results'


def verify():
    from paths import prepare_output
    prepare_output()
    raw=np.load(OUT/'heisenberg4_moments.npz')
    saved=np.load(OUT/'heisenberg4_integer_bounds.npz')
    ct=math.log2(4/3);rows=[];reduction_rows=[];total=changed=0
    assert np.array_equal(raw['times'],saved['times'])
    for W in (.5,1.,2.,3.):
        p=raw[f'W{W:g}']
        assert p.shape==(200,50) and np.all(np.isfinite(p)) and np.all((p>0)&(p<=1))
        for eps in (0.,1e-6,1e-4):
            expected=np.maximum(0,np.ceil(-np.log2(np.minimum(1.,p+16*eps))/ct-1e-11)).astype(int)
            assert np.array_equal(expected,saved[f'W{W:g}_eps{eps:g}'])
        a=saved[f'W{W:g}_eps0'].astype(int)
        b=saved[f'W{W:g}_eps0.0001'].astype(int)
        diff=a-b
        assert set(np.unique(diff))<={0,1}
        mean=diff.mean(axis=0)
        paired_se=diff.std(axis=0,ddof=1)/math.sqrt(200)
        assert np.allclose(paired_se,np.sqrt(mean*(1-mean)/199),atol=1e-14,rtol=0)
        for j,t in enumerate(raw['times']):
            assert abs(mean[j]-(a[:,j].mean()-b[:,j].mean()))<1e-12
            reduction_rows.append({'W':W,'time':float(t),'epsilon':1e-4,
                                   'mean_reduction':float(mean[j]),'paired_standard_error':float(paired_se[j]),
                                   'zero_error_mean':float(a[:,j].mean()),'prescribed_error_mean':float(b[:,j].mean()),
                                   'number_reduced_by_one':int(diff[:,j].sum()),'sample_size':200})
        count=int(np.count_nonzero(diff));total+=diff.size;changed+=count
        rows.append({'W':W,'entries':diff.size,'entries_reduced_by_one':count,
                     'unchanged_fraction':float(np.mean(diff==0)),
                     'largest_mean_reduction':float(diff.mean(axis=0).max()),
                     'mean_zero_error_at_t500':float(a[:,-1].mean()),
                     'mean_prescribed_error_at_t500':float(b[:,-1].mean())})
    assert total==40000 and changed==3574
    with (OUT/'heisenberg4_bound_reduction.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(reduction_rows[0]));writer.writeheader();writer.writerows(reduction_rows)
    names=('heisenberg4_moments.npz','heisenberg4_integer_bounds.npz','heisenberg4.csv','heisenberg4_bound_reduction.csv')
    record={'date':'2026-10-21','same_target_moments_used_for_both_bounds':True,
            'current_manuscript_role':'Brief sensitivity statement and released arrays; no finite-error main-figure panel.',
            'separate_approximate_implementation_channel_simulated':False,
            'epsilon_is_prescribed_tolerance_not_observed_error':True,
            'epsilon':1e-4,'moment_correction':.0016,
            'all_120000_stored_integer_values_reproduced_from_formula':True,
            'paired_bound_entries':total,'entries_reduced_by_one':changed,
            'entries_unchanged':total-changed,'fraction_unchanged':(total-changed)/total,
            'by_W':rows,
            'interpretation':'The sensitivity statement compares theorem-based lower bounds at a prescribed tolerance; it is not a simulation of implementation errors or a cost reduction.',
            'derived_quantity':'The optional CSV contains means of paired differences b_0(p)-b_epsilon(p); no separate mean-reduction plot is used in the manuscript.',
            'paired_statistical_rows_checked':len(reduction_rows),
            'paired_standard_errors_verified':True,
            'derived_statistics_csv':'heisenberg4_bound_reduction.csv',
            'numerical_scope':'The stored target moments have a complete independent binary64 cross-check; no pointwise interval certification is asserted.',
            'sha256':{name:hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in names}}
    (OUT/'heisenberg4_finite_error_audit.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps(record,indent=2))


if __name__=='__main__':
    from paths import prepare_output
    prepare_output()
    verify()
