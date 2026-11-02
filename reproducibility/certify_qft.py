"""Outward-rounded interval certificates of QFT Choi fourth moments.

Root-of-unity enclosures use mpmath interval arithmetic at 60 decimal digits.
All vector additions, subtractions, products, and divisions are rounded
outward with nextafter. Positive reductions include a conservative gamma_k
bound for any order of IEEE-754 binary64 additions. No random tests enter
these certificates.
"""
import json
import math
from fractions import Fraction
import numpy as np
from paths import ROOT, prepare_output
from mpmath import iv
OUT = ROOT / 'reproducibility/results'
iv.dps=60

def up(x):return np.nextafter(x,np.inf)
def down(x):return np.nextafter(x,-np.inf)

def positive_sum_up(a):
    a = np.asarray(a, float)
    if not np.all(np.isfinite(a)) or np.any(a < 0):
        raise ValueError('positive_sum_up requires finite nonnegative inputs')
    u=np.finfo(float).eps/2
    if a.size*u >= .5:
        raise ValueError('too many terms for this summation error bound')
    gamma=up((a.size*u)/(1-a.size*u))
    return float(up(np.sum(a)/down(1-gamma)))

def upper_qft(n,batch=16):
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 0:
        raise ValueError('n must be a nonnegative integer')
    if isinstance(batch, bool) or not isinstance(batch, (int, np.integer)) or batch <= 0:
        raise ValueError('batch must be a positive integer')
    d=2**n;D=d*d
    roots=[]
    for r in range(d):
        angle=2*iv.pi*r/d
        c,s=iv.cos(angle),iv.sin(angle)
        roots.append([down(float(c.a)),up(float(c.b)),down(float(s.a)),up(float(s.b))])
    roots=np.array(roots)
    j=np.arange(D);phase=((j//d)*(j%d))%d
    total=0.
    for lo in range(0,D,batch):
        xs=np.arange(lo,min(D,lo+batch))
        diff=(phase[None,:]-phase[np.bitwise_xor(xs[:,None],j)])%d
        rl=down(roots[diff,0]/D);rh=up(roots[diff,1]/D)
        il=down(roots[diff,2]/D);ih=up(roots[diff,3]/D)
        h=1
        while h<D:
            for lower_,upper_ in [(rl,rh),(il,ih)]:
                l=lower_.reshape(len(xs),-1,2*h);u=upper_.reshape(len(xs),-1,2*h)
                la,lb=l[:,:,:h].copy(),l[:,:,h:].copy()
                ua,ub=u[:,:,:h].copy(),u[:,:,h:].copy()
                l[:,:,:h]=down(la+lb);u[:,:,:h]=up(ua+ub)
                l[:,:,h:]=down(la-ub);u[:,:,h:]=up(ua-lb)
            h*=2
        ar=np.maximum(abs(rl),abs(rh));ai=np.maximum(abs(il),abs(ih))
        sq=up(up(ar*ar)+up(ai*ai))
        total=float(up(total+positive_sum_up(up(sq*sq))))
    return float(up(total/D))


def certified_lower(moment_upper, eps=0.):
    """Compare a certified upper enclosure with exact rational thresholds.

    This only certifies the integer if moment_upper really encloses the true
    target moment from above. eps is interpreted as its decimal string value.
    The correction is 16 eps for the adaptive, half-diamond-error theorem.
    """
    if not math.isfinite(moment_upper) or moment_upper <= 0:
        raise ValueError('moment_upper must be finite and strictly positive')
    if not math.isfinite(eps) or not 0 <= eps <= 1:
        raise ValueError('eps must be in [0, 1]')
    value = Fraction.from_float(float(moment_upper)) + 16*Fraction(str(eps))
    threshold, bound = Fraction(1), 0
    while value < threshold:
        threshold *= Fraction(3, 4)
        bound += 1
    return bound

if __name__=='__main__':
    prepare_output()
    rows=[]
    for n in range(2,7):
        upper=upper_qft(n)
        row={'n':n,'P2_upper':upper}
        for eps in [0.,1e-8,1e-6,1e-4]:
            row[f'bound_eps_{eps:g}']=certified_lower(upper,eps)
        rows.append(row);print(row,flush=True)
    (OUT/'qft_certified.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
