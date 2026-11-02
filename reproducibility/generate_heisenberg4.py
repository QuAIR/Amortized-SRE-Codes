"""Generate periodic four-site Heisenberg data and Choi-state witnesses."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import csv
import json
import time
import numpy as np
from compute_examples import paulis, moments, choi, lower

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
    model={'N':4,'boundary':'periodic','Delta':.2,'exchange_coefficient':1.,
           'fields':'independent uniform [-W,W] at each of four sites',
           'W_values':W_VALUES,'samples_per_W':200,
           'time_grid':{'spacing':'geometric','start':.1,'stop':500.,'count':50},
           'seed':20261021,'rng':'NumPy default_rng / PCG64; successive W draws',
           'epsilon_values':EPS_VALUES,'error':'half-diamond; adaptive correction 16 epsilon',
           'witness':'ceil(-log2(min(1,P2+16epsilon))/log2(4/3)) before averaging',
           'integer_guard':'subtract 1e-11 from the real-valued cost witness before ceiling',
           'nullity_benchmark':'[H,Z tensor 4]=0 implies s(U)>=2 and nu(U)<=7',
           'maximum_unitarity_residual':max_unitarity,
           'maximum_parity_commutator_residual':max_parity_commutator,
           'arithmetic':'binary64'}
    (OUT/'heisenberg4_model.json').write_text(json.dumps(model,indent=2),encoding='utf-8')
    print('Saved 40,000 target moments and 120,000 integer witnesses.',flush=True)


if __name__=='__main__':
    from paths import prepare_output
    prepare_output()
    generate()
