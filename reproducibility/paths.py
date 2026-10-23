"""Separate immutable released data from regenerations and audit outputs."""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# Reading/rendering/checking uses the released data unless explicitly redirected.
ROOT = Path(os.environ.get('SRE_OUTPUT_ROOT', REPO)).expanduser().resolve()


def prepare_output():
    """Require a separate output root before running a data generator."""
    protected = [(REPO / name).resolve() for name in ('Figs', 'reproducibility', 'Rz_magic_generation', '.git')]
    if ROOT == REPO or any(ROOT.is_relative_to(p) or p.is_relative_to(ROOT) for p in protected):
        raise SystemExit('Set SRE_OUTPUT_ROOT to a separate directory (for example '
                         'REPO/.tmp/reproduced) before regenerating data.')
    for relative in ('reproducibility/results', 'Figs'):
        target = ROOT / relative
        if not target.resolve().is_relative_to(ROOT):
            raise SystemExit(f'Output path {relative} resolves outside SRE_OUTPUT_ROOT; remove the redirect first.')
        pending = [target] if target.exists() else []
        while pending:
            path = pending.pop()
            if path.is_symlink() or path.is_junction():
                raise SystemExit(f'Output trees must contain ordinary files/directories; found link: {path}')
            if path.is_dir():
                pending.extend(path.iterdir())
    (ROOT / 'reproducibility/results').mkdir(parents=True, exist_ok=True)
    (ROOT / 'Figs').mkdir(parents=True, exist_ok=True)


def prepare_audit():
    """Validate released hashes, then copy inputs to an isolated audit directory."""
    import shutil
    from integrity import verify_manifest
    verify_manifest(REPO)
    prepare_output()
    shutil.copytree(REPO / 'reproducibility/results',
                    ROOT / 'reproducibility/results', dirs_exist_ok=True)
    shutil.copytree(REPO / 'Figs', ROOT / 'Figs', dirs_exist_ok=True)


if __name__ == '__main__':
    prepare_audit()
    print(f'Validated released hashes and copied audit inputs to {ROOT}')
