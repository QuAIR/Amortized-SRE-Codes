import json
import numpy as np
import sympy as sp
from verify_revision import moments, choi
from paths import ROOT
OUT = ROOT / "reproducibility/results"

if __name__=='__main__':
    from paths import prepare_output
    prepare_output()
    z=sp.symbols('z',nonzero=True)
    sy=[sp.eye(2),sp.Matrix([[0,1],[1,0]]),sp.Matrix([[0,-sp.I],[sp.I,0]]),sp.diag(1,-1)]
    ps=[sp.kronecker_product(a,b,c) for a in sy for b in sy for c in sy]
    sw=sp.eye(8)
    sw[5,5]=sw[6,6]=(1+z)/2;sw[5,6]=sw[6,5]=(1-z)/2
    cr=sp.diag(*([1]*6+[1/z,z]))
    results={}
    for name,u in [('CSWAP',sw),('CCRz',cr)]:
        ud=u.T.subs(z,1/z)
        terms=[]; multiplicities={}
        for q in ps:
            v=u*q*ud
            for p in ps:
                tr=sp.expand(sum(p[i,j]*v[j,i] for i in range(8) for j in range(8))/8)
                if tr!=0:
                    terms.append(sp.expand(tr**4))
                    k=str(sp.expand(tr**2))
                    multiplicities[k]=multiplicities.get(k,0)+1
        p2=sp.expand(sum(terms)/64)
        results[name]={'P2':str(p2),'transfer_squared_multiplicities':multiplicities}
        print(name,p2,flush=True)
        # Independently evaluate at a non-special phase against Walsh moments.
        phi=.423
        numeric=complex(p2.subs(z,sp.exp(sp.I*phi)).evalf())
        un=np.array(u.subs(z,sp.exp(sp.I*phi)).evalf()).astype(complex)
        assert abs(numeric-moments(choi(un))[0])<1e-12
    (OUT/'exact_gate_polynomials.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
