# Amortized stabilizer Rényi entropy

Code and numerical sources for *Amortized Stabilizer Rényi Entropy of Quantum
Dynamics*. The revised calculations include exact gate identities, certified
QFT Choi witnesses, and reproducible Heisenberg examples.

## Python: verify and reproduce

Install Python 3.12 and the pinned numerical packages in a virtual environment:

```sh
python -m pip install -r reproducibility/requirements.txt
python reproducibility/run_checks.py
python reproducibility/run_checks.py --full
```

The first check runs unit tests and verifies released file hashes. The full
check also reconstructs all 64,400 saved Heisenberg evolution points using an
independent Hamiltonian and evolution method, checks the exact gate algebra
and QFT intervals, and redraws the figures in `.tmp/audit/`.

See [reproducibility/README.md](reproducibility/README.md) for model parameters,
data provenance, commands for regeneration and the scope of each check.
Saved numerical sources are in [reproducibility/results/](reproducibility/results/)
and current scientific figures in [Figs/](Figs/). Generators require a separate
output directory; released data are protected by a SHA-256 manifest.

## MATLAB

Add this repository root to your MATLAB path. MATLAB R2024b is used in CI.
`SRE`, `SREChoi`, `PauliString` and `PauliMoments` use core MATLAB; exact
`UStabNul` additionally requires Symbolic Math Toolbox.

```matlab
T = diag([1 exp(1i*pi/4)]);
SRE(2,T*[1;1]/sqrt(2))       % log2(4/3)
SRE(1,T*[1;1]/sqrt(2))       % 1/2
SREChoi(2,T)                 % log2(4/3)
UStabNul(diag([sym(1) exp(sym(1i)*sym(pi)/4)])) % exactly 1
results = runtests('tests/matlab'); assertSuccess(results);
TcountQFT                    % numerical finite-tolerance QFT witnesses
```

`SRE` accepts a normalized pure vector or pure density matrix and positive
alpha, including the Shannon limit at alpha=1. Mixed matrices are rejected:
the Pauli polynomial is not the mixed-state roof in the proof. Computation is
exponential in the number of state qubits; the Walsh method avoids storing
the full dense Pauli basis.

`UStabNul(U)` now requires an **exactly constructed symbolic** matrix. The old
`abs(PTM entry)>0.99` rule incorrectly gives zero nullity for Rz(pi/100), whose
exact nullity is one. Numerical threshold counting remains explicitly
available as `UStabNul(U,'diagnostic',1e-12)` with a warning and an uncertified
status. Tightening a tolerance does not decide exact equality in general.
Converting a rounded matrix to `sym` does not restore its intended constants.

## Original data

The three MAT files in [Rz_magic_generation/](Rz_magic_generation/) are preserved
byte for byte from commit `012dc6e9205f8d3aa8f9d88d0a7a09f1899b8938`.
Their original optimizer and complete grid/configuration were not included in
that snapshot. They remain archived outputs rather than independently
reproduced global optima. The current Figure 1 uses the documented analytic
formulas and source CSV, separately from these MAT files. QETLAB and Manopt
were listed as original dependencies; the functions and new workflows above
are self-contained and do not use them.

The historical author-supplied image `Figs/t-count_lower_bounds_0919.png` is
retained for provenance. Current Figure 2 recalculates its periodic four-site
model with fully recorded new disorder samples; supplementary Figure 3 uses
the open three-site model.

## Interpretation

Choi-state SRE is a lower bound on amortized SRE. Finite-tolerance arrays apply
the proved correction to ideal-target moments at a prescribed half-diamond
tolerance. They do not model an implemented approximation. The N=4 nullity
line at seven is an exact ceiling on the zero-error nullity witness from
parity conservation. The Heisenberg results are independently cross-checked
binary64 calculations; QFT integer certification uses outward intervals.
Exact symbolic checks and finite random reference tests are distinguished in
the reproducibility guide.
