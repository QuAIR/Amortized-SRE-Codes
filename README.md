# Amortized stabilizer Rényi entropy

Code and data for *Amortized Stabilizer Rényi Entropy of Quantum Dynamics*.

## Python

Use Python 3.12 in a virtual environment:

```sh
python -m pip install -r reproducibility/requirements.txt
python reproducibility/run_checks.py
python reproducibility/run_checks.py --full
```

The full check verifies the gate identities, QFT certificates and all 64,400
Heisenberg evolution points, then redraws the figures in `.tmp/checks/`.
See [the reproducibility guide](reproducibility/README.md) for parameters and
regeneration commands.

## MATLAB

Add the repository root to the MATLAB path. The numerical functions use core
MATLAB; exact unitary nullity also requires Symbolic Math Toolbox.

```matlab
T = diag([1 exp(1i*pi/4)]);
SRE(2,T*[1;1]/sqrt(2))       % log2(4/3)
SRE(1,T*[1;1]/sqrt(2))       % 1/2
SREChoi(2,T)                 % log2(4/3)
UStabNul(diag([sym(1) exp(sym(1i)*sym(pi)/4)])) % 1
results = runtests('tests/matlab'); assertSuccess(results);
TcountQFT
```

`SRE` accepts normalized pure states and positive alpha, including alpha=1.
`UStabNul` evaluates exact nullity from exactly constructed symbolic matrices.
Floating-point threshold counts are available with
`UStabNul(U,'diagnostic',1e-12)` and are marked as numerical diagnostics.

## Data and figures

- [reproducibility/results/](reproducibility/results/): simulation inputs,
  moments, integer witnesses, analytic curves and calculation summaries.
- [Figs/](Figs/): Rz, periodic four-site Heisenberg and open three-site
  Heisenberg figures.
- [Rz_magic_generation/](Rz_magic_generation/): MATLAB data arrays for Rz gates.

A single [checksum manifest](reproducibility/data_manifest.json) identifies the
released data and figures. Generators write to a separate output directory.

Choi-state SRE gives a lower bound on amortized SRE. The Heisenberg calculations
use binary64 arithmetic and independent numerical checks; the QFT integers
use outward-rounded interval certificates. Finite-tolerance witnesses apply
the theorem to ideal-target moments at a prescribed half-diamond tolerance.
