"""Recompute the author-confirmed four-site periodic model with saved inputs.

These are new disorder realizations, not a reconstruction of pixel values in
the historical plot. The exact nullity comparison uses parity, not a numerical
near-unit threshold. Floating-point integer witnesses use a downward guard.
"""
from pathlib import Path
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import csv
import hashlib
import json
import time
import numpy as np
from verify_revision import paulis, moments, choi, lower

from paths import ROOT
OUT=ROOT/'reproducibility/results'
W_VALUES=(.5,1.,2.,3.)
EPS_VALUES=(0.,1e-6,1e-4)


def hamiltonian(fields, basis):
    h=np.zeros((16,16),complex)
    weights=4**np.arange(3,-1,-1)
    for j in range(4):
        k=(j+1)%4
        for p,c in ((1,1.),(2,1.),(3,.2)):
            h+=c*basis[p*(weights[j]+weights[k])]
        h+=fields[j]*basis[3*weights[j]]
    return h


def generate():
    from paths import prepare_output
    prepare_output()
    start=time.perf_counter()
    basis=paulis(4)
    parity=basis[-1]
    rng=np.random.default_rng(20261021)
    times=np.geomspace(.1,500.,50)
    raw={}; inputs={}; bounds={'times':times}; rows=[]
    max_unitarity=0.; max_parity_commutator=0.
    for W in W_VALUES:
        fields=rng.uniform(-W,W,(200,4))
        inputs[f'W{W:g}']=fields
        vals=np.empty((200,len(times)))
        for i,hs in enumerate(fields):
            h=hamiltonian(hs,basis)
            assert np.array_equal(h@parity,parity@h)
            eig,vec=np.linalg.eigh(h)
            for j,t in enumerate(times):
                u=(vec*np.exp(-1j*t*eig))@vec.conj().T
                vals[i,j]=moments(choi(u),(2,),batch=256)[0]
                max_unitarity=max(max_unitarity,float(np.max(abs(u.conj().T@u-np.eye(16)))))
                max_parity_commutator=max(max_parity_commutator,float(np.max(abs(u@parity-parity@u))))
            if (i+1)%50==0:
                print(f'W={W:g}: {i+1}/200, elapsed {time.perf_counter()-start:.1f}s',flush=True)
        raw[f'W{W:g}']=vals
        for eps in EPS_VALUES:
            integers=np.array([[lower(p,eps) for p in row] for row in vals],dtype=np.int16)
            bounds[f'W{W:g}_eps{eps:g}']=integers
            for j,t in enumerate(times):
                rows.append({'W':W,'time':float(t),'epsilon':eps,
                             'mean_bound':float(integers[:,j].mean()),
                             'standard_error':float(integers[:,j].std(ddof=1)/np.sqrt(200)),
                             'fraction_above_nullity_ceiling':float(np.mean(integers[:,j]>7))})
    np.savez_compressed(OUT/'heisenberg4_fields.npz',**inputs)
    np.savez_compressed(OUT/'heisenberg4_moments.npz',times=times,**raw)
    np.savez_compressed(OUT/'heisenberg4_integer_bounds.npz',**bounds)
    with (OUT/'heisenberg4.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    model={'date':'2026-10-21','N':4,'boundary':'periodic; site 5 = site 1',
           'boundary_confirmed_by_author':True,'Delta':.2,'exchange_coefficient':1.,
           'fields':'independent uniform [-W,W] at each of four sites',
           'W_values':W_VALUES,'samples_per_W':200,'times':times.tolist(),
           'seed':20261021,'rng':'NumPy default_rng / PCG64; successive W draws',
           'epsilon_values':EPS_VALUES,'error':'half-diamond; adaptive correction 16 epsilon',
           'witness':'ceil(-log2(min(1,P2+16epsilon))/log2(4/3)) before averaging',
           'integer_guard':'subtract 1e-11 from the real-valued cost witness before ceiling',
           'nullity_benchmark':'nu(U)=8-log2 s(U)<=7 follows exactly from I and Z tensor 4; not an approximate-cost bound',
           'provenance':'Fresh calculations for the author-confirmed original model and disorder strengths. Historical raw disorder and exact time grid unavailable; no pixel or sample-by-sample reproduction claimed.',
           'maximum_unitarity_residual':max_unitarity,
           'maximum_parity_commutator_residual':max_parity_commutator,
           'numerical_scope':'Binary64 numerical simulation with independent cross-checks; not pointwise interval certification.'}
    (OUT/'heisenberg4_model.json').write_text(json.dumps(model,indent=2),encoding='utf-8')
    summary={}
    for W in W_VALUES:
        summary[f'W{W:g}']={}
        for eps in EPS_VALUES:
            b=bounds[f'W{W:g}_eps{eps:g}']; means=b.mean(axis=0)
            summary[f'W{W:g}'][f'eps{eps:g}']={
                'mean_at_first_time':float(means[0]),'mean_at_final_time':float(means[-1]),
                'minimum_mean_t_ge_10':float(means[times>=10].min()),
                'minimum_sample_bound_t_ge_10':int(b[:,times>=10].min()),
                'first_sampled_time_mean_above_7':float(times[np.flatnonzero(means>7)[0]]) if np.any(means>7) else None}
    (OUT/'heisenberg4_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    from paths import prepare_output
    prepare_output()
    generate()
