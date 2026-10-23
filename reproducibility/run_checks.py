"""Run quick tests or the complete independent data audit in an isolated folder."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from integrity import verify_manifest
from paths import REPO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full', action='store_true', help='include symbolic/interval checks and every saved evolution')
    args = parser.parse_args()
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', PYTHONUTF8='1')
    output = REPO / '.tmp/audit'
    env['SRE_OUTPUT_ROOT'] = str(output)
    commands = [['-m','unittest','discover','-s','tests','-v']]
    if args.full:
        commands += [['reproducibility/paths.py'],
                     ['-c','import sys; sys.path.insert(0,"reproducibility"); from verify_revision import verify; print(verify())']]
        commands += [[f'reproducibility/{s}.py'] for s in (
            'certify_qft','symbolic_gates','audit_proofs','verify_figure_data',
            'verify_heisenberg4','verify_finite_error_panel','render_figures')]
    timings = []
    for command in commands:
        start = time.perf_counter()
        subprocess.run([sys.executable,'-X','utf8',*command],cwd=REPO,env=env,check=True)
        timings.append({'command':command,'seconds':time.perf_counter()-start})
    count = verify_manifest()
    if args.full:
        # Interval enclosures can be looser on a different machine; the integer
        # certificates, rather than JSON byte identity, are the reproducible claim.
        original = json.loads((REPO/'reproducibility/results/qft_certified.json').read_text(encoding='utf-8'))
        reproduced = json.loads((output/'reproducibility/results/qft_certified.json').read_text(encoding='utf-8'))
        for a,b in zip(original,reproduced,strict=True):
            for key in a:
                if key != 'P2_upper' and a[key] != b[key]:
                    raise AssertionError(f'QFT certificate changed: n={a["n"]}, {key}')
        (output/'run_summary.json').write_text(json.dumps({'passed':True,'released_files_unchanged':count,'commands':timings},indent=2),encoding='utf-8')
    print(f'Checks passed; {count} released files unchanged. Full audit outputs: {output}' if args.full else f'Quick checks passed; {count} released files verified.')


if __name__ == '__main__': main()
