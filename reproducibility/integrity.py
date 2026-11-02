"""Check released data/figure hashes without regenerating numerical outputs."""
import hashlib
import json
from pathlib import Path
from paths import REPO


def verify_manifest(root=REPO):
    root = Path(root).resolve()
    manifest = json.loads((root / 'reproducibility/data_manifest.json').read_text(encoding='utf-8'))
    for name, expected in manifest['sha256'].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f'Manifest path escapes repository: {name}')
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'Released file checksum mismatch: {name}')
    return len(manifest['sha256'])


if __name__ == '__main__':
    print(f'Verified {verify_manifest()} released data and figure files.')
