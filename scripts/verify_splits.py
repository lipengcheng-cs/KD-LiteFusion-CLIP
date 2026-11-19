import hashlib
import json
from pathlib import Path


def verify(root):
    summary = json.loads((root / 'splits/summary.json').read_text())
    checksums = {}
    for line in (root / 'splits/SHA256SUMS').read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        checksums[name] = digest
    seen = set()
    total = 0
    for split in ('train', 'val', 'test'):
        name = f'splits/{split}_ids.txt'
        path = root / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != checksums[name]:
            raise ValueError(f'Checksum mismatch: {name}')
        ids = path.read_text().splitlines()
        if not ids or any(not x.strip() for x in ids):
            raise ValueError(f'Empty ID in {split}')
        if len(ids) != len(set(ids)) or seen.intersection(ids):
            raise ValueError(f'Duplicate IDs or cross-split overlap: {split}')
        expected = summary['splits'][split]
        if len(ids) != expected['sample_count'] or len(set(ids)) != expected['unique_id_count']:
            raise ValueError(f'Count mismatch: {split}')
        seen.update(ids)
        total += len(ids)
        print(f'{split}: {len(ids)} IDs, checksum verified')
    if total != summary['total_samples'] or summary['cross_split_overlap'] != 0:
        raise ValueError('Summary mismatch')
    print(f'PASS: {total} unique IDs; no cross-split overlap')


if __name__ == '__main__':
    verify(Path(__file__).resolve().parents[1])
