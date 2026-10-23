"""Independent verification and reproducible figures for the IEEE revision.

See reproducibility/README.md for environment and output-directory setup.
No optimizer output is used as a proof. Random tests are diagnostic only.
"""
from pathlib import Path
import csv, json, math, platform
import numpy as np
import scipy
from scipy.linalg import expm
import sympy as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from paths import ROOT
OUT = ROOT / 'reproducibility' / 'results'
C_T = math.log2(4/3)
I = np.eye(2, dtype=complex)
X = np.array([[0,1],[1,0]], complex)
Y = np.array([[0,-1j],[1j,0]], complex)
Z = np.diag([1.,-1.]).astype(complex)

def paulis(n):
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 0:
        raise ValueError('n must be a nonnegative integer number of qubits')
    a = np.array([[[1.]]], complex)
    for _ in range(n):
        a = np.array([np.kron(p,q) for p in a for q in (I,X,Y,Z)])
    return a

def moments(psi, orders=(2,3), batch=32):
    """All Pauli moments via XOR diagonals and Walsh-Hadamard transforms.

    Hermitian Pauli phases have modulus one; even absolute powers remove them.
    Memory O(batch*2**N), work O(N*4**N); no stochastic estimation.
    """
    psi = np.asarray(psi, complex)
    d = psi.size
    if psi.ndim != 1 or d == 0 or d & (d-1):
        raise ValueError('psi must be a nonempty vector of power-of-two length')
    if not np.all(np.isfinite(psi)) or abs(np.vdot(psi, psi)-1) > 1e-10:
        raise ValueError('psi must be finite and normalized (norm squared within 1e-10 of one)')
    orders = np.asarray(orders, float)
    if orders.ndim != 1 or not len(orders) or not np.all(np.isfinite(orders)) or np.any(orders <= 0):
        raise ValueError('orders must be a nonempty sequence of finite positive real numbers')
    if isinstance(batch, bool) or not isinstance(batch, (int, np.integer)) or batch <= 0:
        raise ValueError('batch must be a positive integer')
    idx = np.arange(d)
    totals = np.zeros(len(orders))
    for lo in range(0,d,batch):
        xs = np.arange(lo,min(lo+batch,d))
        a = psi[None,:] * psi[np.bitwise_xor(xs[:,None],idx)].conj()
        h = 1
        while h < d:
            b = a.reshape(len(xs),-1,2*h)
            left, right = b[:,:,:h].copy(), b[:,:,h:].copy()
            b[:,:,:h], b[:,:,h:] = left+right, left-right
            h *= 2
        sq = abs(a)**2
        for k,alpha in enumerate(orders):
            totals[k] += np.sum(sq**alpha)
    return totals/d

def choi(u):
    u = np.asarray(u, complex)
    if u.ndim != 2 or u.shape[0] == 0 or u.shape[0] != u.shape[1] or u.shape[0] & (u.shape[0]-1):
        raise ValueError('u must be a nonempty square matrix of power-of-two dimension')
    if not np.all(np.isfinite(u)) or not np.allclose(u.conj().T @ u, np.eye(len(u)), atol=1e-10, rtol=0):
        raise ValueError('u must be finite and unitary within absolute tolerance 1e-10')
    return u.reshape(-1)/math.sqrt(len(u))

def qft(n, shift=0):
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 0:
        raise ValueError('n must be a nonnegative integer number of qubits')
    d=2**n
    j=np.arange(d)+shift
    return np.exp(2j*np.pi*j[:,None]*j[None,:]/d)/math.sqrt(d)

def lower(a, eps=0., adaptive=True):
    """Binary64 witness estimate, NOT an interval-certified integer bound.

    The 1e-11 downward guard avoids upward rounding at known integer endpoints;
    it is not an error enclosure. Use certify_qft for certified QFT results.
    """
    a, eps = float(a), float(eps)
    if not math.isfinite(a) or a <= 0 or a > 1+1e-10:
        raise ValueError('a must be a finite positive Pauli moment, at most 1 (tolerance 1e-10)')
    if not math.isfinite(eps) or not 0 <= eps <= 1:
        raise ValueError('eps must be a half-diamond tolerance in [0, 1]')
    if not isinstance(adaptive, (bool, np.bool_)):
        raise ValueError('adaptive must be True or False')
    penalty=(16 if adaptive else 4)*eps
    x=-math.log2(min(1.,float(a)+penalty))/C_T
    return max(0,math.ceil(x-1e-11))

def ccrz(theta):
    # Rz(theta)=exp(-i theta Z/2), controls are the first two qubits.
    return np.diag([1]*6+[np.exp(-.5j*theta),np.exp(.5j*theta)])

def cswap(alpha):
    u=np.eye(8,dtype=complex)
    e=np.exp(1j*np.pi*alpha)
    u[5,5]=u[6,6]=(1+e)/2
    u[5,6]=u[6,5]=(1-e)/2
    return u

def heisenberg(n, fields, delta=.2):
    # Open boundary conditions, exchange coefficient set to one.
    def term(at):
        v=np.array([[1.]],complex)
        for j in range(n): v=np.kron(v,at.get(j,I))
        return v
    h=sum((term({j:X,j+1:X})+term({j:Y,j+1:Y})+
           delta*term({j:Z,j+1:Z}) for j in range(n-1)),np.zeros((2**n,2**n),complex))
    return h+sum(fields[j]*term({j:Z}) for j in range(n))

def verify():
    rng=np.random.default_rng(20261018)
    checks={}
    err=0.
    for n in range(1,5):
        ps=paulis(n)
        for _ in range(5):
            psi=rng.normal(size=2**n)+1j*rng.normal(size=2**n)
            psi/=np.linalg.norm(psi)
            direct=np.array([sum(abs(np.vdot(psi,p@psi))**(2*a) for p in ps)/2**n for a in (2,3)])
            err=max(err,float(max(abs(direct-moments(psi)))))
    assert err < 1e-12
    checks['walsh_vs_direct_max_error']=err
    t=np.diag([1,np.exp(1j*np.pi/4)])
    ccz=np.diag([1]*7+[-1])
    assert abs(moments(t@np.ones(2)/np.sqrt(2))[0]-.75)<1e-12
    assert abs(moments(ccz@np.ones(8)/np.sqrt(8))[0]-11/32)<1e-12
    checks['T_purity']=.75
    checks['CCZ_purity']=11/32
    minrat={}
    max_cont=0.
    for name,u,bound in [('T',t,.75),('CCZ',ccz,11/32)]:
        ratios=[]
        for m in range(4):
            full=np.kron(u,np.eye(2**m))
            for _ in range(80):
                psi=rng.normal(size=len(full))+1j*rng.normal(size=len(full))
                psi/=np.linalg.norm(psi)
                p=moments(psi)[0]; q=moments(full@psi)[0]
                ratios.append(q/p)
                phi=rng.normal(size=len(full))+1j*rng.normal(size=len(full))
                phi/=np.linalg.norm(phi)
                d=np.sqrt(max(0,1-abs(np.vdot(psi,phi))**2))
                max_cont=max(max_cont,abs(p-moments(phi)[0])/d)
        assert min(ratios)>=bound-1e-12
        minrat[name]=min(ratios)
    checks['random_ratio_minima_320_each']=minrat
    checks['random_purity_lipschitz_max']=float(max_cont)
    # Exact CCZ adjoint block structure, with rational complex arithmetic.
    sy=[sp.eye(2),sp.Matrix([[0,1],[1,0]]),sp.Matrix([[0,-sp.I],[sp.I,0]]),sp.diag(1,-1)]
    ps=[sp.kronecker_product(a,b,c) for a in sy for b in sy for c in sy]
    v=sp.diag(*([1]*7+[-1]))
    r=sp.Matrix(64,64,lambda i,j:sp.trace(ps[i]*v*ps[j]*v)/8)
    assert r.T*r==sp.eye(64)
    unseen=set(range(64)); sizes=[]
    while unseen:
        comp={min(unseen)}; todo=list(comp)
        while todo:
            j=todo.pop()
            new={i for i in range(64) if r[i,j]!=0}-comp
            comp|=new; todo.extend(new)
        unseen-=comp; sizes.append(len(comp))
    assert sorted(sizes)==[1]*8+[4]*14
    checks['CCZ_exact_adjoint_blocks']={'singletons':8,'four_dimensional':14,'orthogonal':True}
    # Four-copy PSD inequality for one qubit; tensoring proves arbitrary m.
    q=sum((sp.kronecker_product(p,p,p,p) for p in sy),sp.zeros(16))
    dz=sp.eye(16)+sp.kronecker_product(sy[3],sy[3],sy[3],sy[3])
    assert all(x>=0 for x in q.eigenvals())
    assert all(x>=0 for x in (2*dz-q).eigenvals())
    checks['four_copy_exact_PSD']=True
    return checks

def computations():
    from paths import prepare_output
    prepare_output()
    rows=[]
    for n in range(2,7):
        a2,a3=moments(choi(qft(n)))
        old=moments(choi(qft(n,1)))[0]
        row=dict(n=n,P2=float(a2),M2=float(-np.log2(a2)),P3=float(a3),
                 zero_error_witness=lower(a2),adaptive_1e_6=lower(a2,1e-6),
                 unitary_1e_3=lower(a2,1e-3,False),
                 old_one_based_P2=float(old),old_one_based_witness=lower(old))
        # Numerical margin below which the integer zero-error witness survives.
        k=lower(a2)
        margin=(.75**(k-1)-a2)/16
        row['adaptive_epsilon_threshold']=float(max(0,margin))
        rows.append(row)
        print('QFT',row,flush=True)
    with (OUT/'qft.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    # A fully specified fresh dynamics calculation, with saved disorder.
    rng=np.random.default_rng(20261018)
    times=np.linspace(0,3,61)
    eps_values=[0,1e-6,1e-4]
    allrows=[]; disorder={}; integer_arrays={}
    for w in (1.,3.):
        fields=rng.uniform(-w,w,(200,3));disorder[str(w)]=fields.tolist()
        vals=[]
        for hs in fields:
            eig,vec=np.linalg.eigh(heisenberg(3,hs))
            vals.append([moments(choi((vec*np.exp(-1j*tm*eig))@vec.conj().T),(2,))[0] for tm in times])
        vals=np.array(vals)
        for e in eps_values:
            b=np.array([[lower(a,e) for a in row] for row in vals])
            integer_arrays[f'W{w:g}_eps{e:g}']=b
            for j,tm in enumerate(times):
                allrows.append(dict(W=w,time=tm,epsilon=e,mean_bound=b[:,j].mean(),
                                    standard_error=b[:,j].std(ddof=1)/np.sqrt(len(b))))
        np.save(OUT/f'heisenberg_P2_W{w:g}.npy',vals)
        print('Heisenberg W =',w,':',len(fields),'realizations completed.',flush=True)
    np.savez_compressed(OUT/'heisenberg_integer_bounds.npz',times=times,**integer_arrays)
    (OUT/'heisenberg_disorder.json').write_text(json.dumps(disorder,indent=2),encoding='utf-8')
    with (OUT/'heisenberg.csv').open('w',newline='',encoding='utf-8') as f:
        wr=csv.DictWriter(f,fieldnames=allrows[0]);wr.writeheader();wr.writerows(allrows)
    # Controlled gates and error dependence: direct, independent calculations.
    gate_rows=[]
    for name,u in [('CCRz(pi/2)',ccrz(np.pi/2)),('CCRz(pi)',ccrz(np.pi)),
                   ('CSWAP(1/2)',cswap(.5)),('CSWAP(3/4)',cswap(.75)),('Fredkin',cswap(1.))]:
        a2,a3=moments(choi(u));k=lower(a2)
        margin=(.75**(k-1)-a2)/16
        gate_rows.append(dict(gate=name,P2=float(a2),P3=float(a3),M2=float(-np.log2(a2)),
                              zero_error_witness=k,adaptive_1e_6=lower(a2,1e-6),
                              adaptive_epsilon_threshold=float(max(0,margin))))
    (OUT/'controlled_gates.json').write_text(json.dumps(gate_rows,indent=2),encoding='utf-8')
    return {'qft':rows,'controlled_gates':gate_rows}

if __name__=='__main__':
    from paths import prepare_output
    prepare_output()
    checks=verify();print('Checks:',checks,flush=True)
    data=computations()
    summary={'versions':{'python':platform.python_version(),'numpy':np.__version__,
                         'scipy':scipy.__version__,'sympy':sp.__version__,'matplotlib':matplotlib.__version__},
             'seed':20261018,'checks':checks,**data}
    (OUT/'verification.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('All independent verification checks passed.',flush=True)
