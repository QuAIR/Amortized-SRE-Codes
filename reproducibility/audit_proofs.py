"""Fresh algebraic and adversarial diagnostics for the two proof audits.

Analytic, dimension-uniform arguments are in the manuscript/audit report.
Finite tests here are diagnostics, not proofs of unbounded ancillary claims.
This script does not import the previous verification implementation.
"""
from pathlib import Path
from fractions import Fraction
import itertools, json
import numpy as np
import sympy as sp

from paths import ROOT
OUT=ROOT/'reproducibility/results'
rng=np.random.default_rng(2026101802)
one=[np.eye(2),np.array([[0,1],[1,0]]),np.array([[0,-1j],[1j,0]]),np.diag([1,-1])]

def pauli_basis(n):
    result=[]
    for factors in itertools.product(one,repeat=n):
        p=np.ones((1,1),complex)
        for factor in factors:p=np.kron(p,factor)
        result.append(p)
    return np.array(result)

def projector(v):return np.outer(v,v.conj())
def vector(d):
    v=rng.normal(size=d)+1j*rng.normal(size=d)
    return v/np.linalg.norm(v)
def distance(a,b):return float(np.abs(np.linalg.eigvalsh(a-b)).sum()/2)
def poly(rho,ps):
    a=np.einsum('pij,ji->p',ps,rho).real
    return float(np.sum(a**4)/len(rho))
def gradient_state(rho,ps):
    a=np.einsum('pij,ji->p',ps,rho).real
    return np.einsum('p,pij->ij',a**3,ps)/len(rho)

def check_continuity():
    minimum_eig=1.;maximum_lip=0.;maximum_roof_ratio=0.;maximum_remainder=0.
    tested=0
    for n in (1,2,3):
        d=2**n;ps=pauli_basis(n)
        for _ in range(70):
            a=rng.normal(size=(d,d))+1j*rng.normal(size=(d,d))
            rho=a@a.conj().T;rho/=np.trace(rho)
            phi=projector(vector(d));psi=projector(vector(d))
            g=gradient_state(rho,ps)
            minimum_eig=min(minimum_eig,float(np.linalg.eigvalsh(g).min()))
            assert np.linalg.eigvalsh(g).min()>-1e-12
            assert abs(np.trace(g)-1)<1e-12
            ratio=abs(poly(rho,ps)-poly(phi,ps))/distance(rho,phi)
            maximum_lip=max(maximum_lip,ratio);assert ratio<=4+1e-10
            delta=psi-phi
            linear=4*np.trace(gradient_state(phi,ps)@delta).real
            remainder=poly(psi,ps)-poly(phi,ps)-linear
            infidelity=1-np.trace(psi@phi).real
            maximum_remainder=max(maximum_remainder,remainder/max(infidelity,1e-15))
            assert remainder<=12*infidelity+1e-10
            # Opposite transverse perturbations cancel to first order in the mean.
            v=vector(d);w=vector(d);w-=v*np.vdot(v,w);w/=np.linalg.norm(w)
            for h in (1e-1,1e-2,1e-3):
                ensemble=[projector(np.sqrt(1-h*h)*v+h*w),
                          projector(np.sqrt(1-h*h)*v-h*w)]
                mean=(ensemble[0]+ensemble[1])/2
                target=projector(v);epsilon=distance(mean,target)
                average=sum(poly(s,ps) for s in ensemble)/2
                excess=average-poly(target,ps)
                maximum_roof_ratio=max(maximum_roof_ratio,excess/epsilon)
                assert excess<=16*epsilon+1e-10
                tested+=1
    # Explicit counterexample to substituting a mixed Pauli polynomial for its roof.
    raw_mixed=poly(np.eye(2)/2,pauli_basis(1))
    assert raw_mixed==.5
    return dict(gradient_min_eigenvalue=minimum_eig,max_polynomial_lipschitz_ratio=maximum_lip,
                max_taylor_remainder_over_infidelity=maximum_remainder,
                max_tested_ensemble_roof_excess_over_distance=maximum_roof_ratio,
                near_pure_ensembles_checked=tested,
                maximally_mixed_qubit_polynomial=raw_mixed,
                maximally_mixed_qubit_roof=1,
                note='Ensemble checks do not optimize the roof; the Taylor proof covers every decomposition.')

def exact_ccz():
    def bitdot(x,z):return (x&z).bit_count()
    def p(x,z):
        matrix=sp.zeros(8)
        for y in range(8):matrix[y^x,y]=sp.I**bitdot(x,z)*(-1)**bitdot(z,y)
        return matrix
    u=sp.diag(1,1,1,1,1,1,1,-1)
    basis={(x,z):p(x,z) for x in range(8) for z in range(8)}
    blocks=[]
    for x in range(1,8):
        perpendicular=[k for k in range(8) if bitdot(x,k)%2==0]
        for z in range(8):
            result=sp.zeros(8)
            for k in range(8):
                coeff=sp.Integer(k==0)-sp.Rational(1,4)*(-1)**bitdot(k,7)*(1+(-1)**bitdot(k,x))
                assert (coeff==0)==(k not in perpendicular)
                phase=sp.I**(bitdot(x,z)-bitdot(x,z^k))
                if coeff:assert phase in (-1,1) and abs(coeff)==sp.Rational(1,2)
                result+=coeff*phase*basis[x,z^k]
            assert result==u*basis[x,z]*u
        remaining=set(range(8))
        while remaining:
            z=next(iter(remaining));coset=sorted(z^k for k in perpendicular)
            remaining-=set(coset)
            block=sp.Matrix([[(basis[x,a]*u*basis[x,b]*u).trace()/8 for b in coset] for a in coset])
            assert block.T*block==sp.eye(4)
            blocks.append(block)
    v=sp.ones(8,1)/sp.sqrt(8);out=u*v
    moments=[sp.simplify((out.H*matrix*out)[0]) for matrix in basis.values()]
    spectrum={str(k):moments.count(k) for k in set(moments)}
    assert sum(m**4 for m in moments)==sp.Rational(11,4)
    sx=sp.Matrix([[0,1],[1,0]]);sz=sp.diag(1,-1)
    a=sp.kronecker_product(sx,sx,sx,sx);b=sp.kronecker_product(sz,sz,sz,sz)
    qp=(sp.eye(16)+a)*(sp.eye(16)+b)
    qm=(sp.eye(16)-a)*(sp.eye(16)+b)
    assert all(value>=0 for value in qp.eigenvals())
    assert all(value>=0 for value in qm.eigenvals())
    assert 2*(sp.eye(16)+b)-qp==qm
    return dict(blocks=len(blocks),fourier_identity_checks=56,orthogonal=True,
                exact_output_P2='11/32',output_pauli_spectrum=spectrum,
                operator_difference_exact='2D-Q_plus=Q_minus >= 0')

def ancilla_diagnostics():
    rows=[]
    # Nonstabilizer references can be entangled with all three target qubits.
    for m in range(4):
        n=3+m;ps=pauli_basis(n)
        u=np.diag(np.repeat([1]*7+[-1],2**m))
        min_ratio=1.;min_diagonal=1.
        # Avoid a density-matrix quartic implementation of the previous code.
        diagonal_indices=[j for j,labels in enumerate(itertools.product(range(4),repeat=n))
                          if all(q in (0,3) for q in labels[:3])]
        for _ in range(12):
            v=vector(2**n);w=u@v
            a=np.einsum('i,pij,j->p',v.conj(),ps,v).real
            b=np.einsum('i,pij,j->p',w.conj(),ps,w).real
            ratio=np.sum(b**4)/np.sum(a**4)
            diagonal=np.sum(a[diagonal_indices]**4)/np.sum(a**4)
            min_ratio=min(min_ratio,ratio);min_diagonal=min(min_diagonal,diagonal)
            assert ratio>=11/32-1e-11 and diagonal>=1/8-1e-11
        rows.append(dict(reference_qubits=m,min_ratio=min_ratio,min_diagonal_fraction=min_diagonal))
    return rows

if __name__=='__main__':
    from paths import prepare_output
    prepare_output()
    result={'continuity':check_continuity(),'ccz':exact_ccz(),'arbitrary_reference_diagnostics':ancilla_diagnostics()}
    (OUT/'independent_proof_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)
