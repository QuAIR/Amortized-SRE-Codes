"""Check all four-site data using spin-basis evolution and direct PTM traces."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import csv
import hashlib
import itertools
import json
import math
import time
import numpy as np
from scipy.linalg import expm

from paths import ROOT
OUT=ROOT/'reproducibility/results'


def spin_hamiltonian(fields):
    h=np.zeros((16,16),complex)
    for b in range(16):
        bits=np.array([(b>>k)&1 for k in (3,2,1,0)])
        z=1-2*bits
        h[b,b]=np.dot(fields,z)+.2*np.dot(z,np.roll(z,-1))
        for j in range(4):
            k=(j+1)%4
            if bits[j]!=bits[k]:h[b^((1<<(3-j))|(1<<(3-k))),b]+=2
    return h


def verify():
    from paths import prepare_output
    prepare_output()
    start=time.perf_counter()
    names=('heisenberg4_fields.npz','heisenberg4_moments.npz','heisenberg4_integer_bounds.npz','heisenberg4.csv','heisenberg4_model.json')
    hashes={name:hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in names}
    fields=np.load(OUT/names[0]); raw=np.load(OUT/names[1]); stored=np.load(OUT/names[2])
    model=json.loads((OUT/names[4]).read_text(encoding='utf-8'))
    times=raw['times'];assert np.array_equal(times,np.geomspace(.1,500.,50))
    one=(np.eye(2),np.array([[0,1],[1,0]]),np.array([[0,-1j],[1j,0]]),np.diag([1,-1]))
    basis=np.array([np.kron(np.kron(np.kron(a,b),c),d) for a,b,c,d in itertools.product(one,repeat=4)])
    flat=basis.reshape(256,256).conj();parity=basis[-1]
    rng=np.random.default_rng(20261021)
    max_difference=0.; max_unitarity=0.;max_imaginary=0.;checked=0;integers=0
    min_distance_to_integer=math.inf
    # Identity boundary: exact t=0 has P2=1, nullity=0 and zero cost witness.
    identity=np.eye(16);assert np.array_equal(identity@parity,parity@identity)
    assert np.array_equal(basis.reshape(256,256).conj()@basis.reshape(256,256).T,16*np.eye(256))
    for W in (.5,1.,2.,3.):
        hs=fields[f'W{W:g}'];assert np.array_equal(hs,rng.uniform(-W,W,(200,4)))
        for i,hvec in enumerate(hs):
            h=spin_hamiltonian(hvec)
            assert np.array_equal(h@parity,parity@h)
            for j,t in enumerate(times):
                u=expm(-1j*t*h)
                transfer=flat@(u@basis@u.conj().T).reshape(256,256).T/16
                max_imaginary=max(max_imaginary,float(np.max(abs(transfer.imag))))
                p=float(np.sum(abs(transfer)**4)/256)
                max_difference=max(max_difference,abs(p-raw[f'W{W:g}'][i,j]))
                assert abs(p-raw[f'W{W:g}'][i,j])<3e-10
                max_unitarity=max(max_unitarity,float(np.max(abs(u.conj().T@u-np.eye(16)))))
                assert abs(transfer[0,0]-1)<3e-11 and abs(transfer[-1,-1]-1)<3e-11
                for eps in model['epsilon_values']:
                    value=-math.log2(min(1.,p+16*eps))/math.log2(4/3)
                    k=max(0,math.ceil(value-1e-11))
                    assert k==stored[f'W{W:g}_eps{eps:g}'][i,j]
                    min_distance_to_integer=min(min_distance_to_integer,abs(value-round(value)))
                    integers+=1
                checked+=1
            if (i+1)%50==0:print(f'Independent W={W:g}: {i+1}/200; {time.perf_counter()-start:.1f}s',flush=True)
    rows=list(csv.DictReader((OUT/'heisenberg4.csv').open(encoding='utf-8')))
    for row in rows:
        j=int(np.argmin(abs(times-float(row['time']))))
        b=stored[f"W{float(row['W']):g}_eps{float(row['epsilon']):g}"][:,j]
        assert abs(float(row['mean_bound'])-b.mean())<1e-12
        assert abs(float(row['standard_error'])-b.std(ddof=1)/math.sqrt(200))<1e-12
        assert abs(float(row['fraction_above_nullity_ceiling'])-np.mean(b>7))<1e-12
    assert checked==40000 and integers==120000 and len(rows)==600
    assert hashes=={name:hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in names}
    report={'evolutions_checked':checked,'integer_entries_checked':integers,'statistical_rows_checked':len(rows),
            'all_integer_entries_match':True,'raw_arrays_unchanged':True,
            'maximum_P2_difference':max_difference,'maximum_unitarity_residual':max_unitarity,
            'maximum_PTM_imaginary_residual':max_imaginary,
            'minimum_real_witness_distance_to_integer':min_distance_to_integer,
            'nullity_ceiling':7,'arithmetic':'binary64'}
    (OUT/'heisenberg4_checks.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(f'Four-site checks passed: {checked} evolutions, {integers} integer witnesses; max moment difference {max_difference:.3g}.')


if __name__=='__main__':
    from paths import prepare_output
    prepare_output()
    verify()
