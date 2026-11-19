import argparse
import csv
from pathlib import Path


def prepare_data(input_csv, splits_dir, output_dir):
    splits = {}
    all_ids = set()
    for split in ('train', 'val', 'test'):
        ids = (splits_dir / f'{split}_ids.txt').read_text().splitlines()
        if not ids or any(not value.strip() for value in ids):
            raise ValueError(f'Empty ID list or blank ID in {split}')
        if len(ids) != len(set(ids)) or all_ids.intersection(ids):
            raise ValueError(f'Duplicate IDs or overlapping splits: {split}')
        splits[split] = ids
        all_ids.update(ids)

    records = {}
    with input_csv.open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        required = {'sample_id', 'image_path', 'text', 'label'}
        if not fields or not required.issubset(fields):
            raise ValueError(f'Input CSV requires columns: {sorted(required)}')
        for row in reader:
            sample_id = row['sample_id']
            if sample_id not in all_ids:
                continue
            if sample_id in records:
                raise ValueError(f'Duplicate input sample: {sample_id}')
            records[sample_id] = row
    missing = all_ids.difference(records)
    if missing:
        raise ValueError(f'Missing {len(missing)} samples; example: {sorted(missing)[:5]}')

    output_fields = ['sample_id', 'image_path', 'text', 'label', 'split']
    destinations = [output_dir / f'{split}.csv' for split in splits]
    if any(path.resolve() == input_csv.resolve() for path in destinations):
        raise ValueError('Output must not overwrite the input manifest')
    if any(path.exists() for path in destinations):
        raise FileExistsError('Output split CSV already exists; use a new output directory')
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, ids in splits.items():
        with (output_dir / f'{split}.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=output_fields)
            writer.writeheader()
            for sample_id in ids:
                row = {key: records[sample_id][key] for key in output_fields if key != 'split'}
                row['split'] = split
                writer.writerow(row)
        print(f'{split}: {len(ids)} samples')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-csv', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--splits-dir', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'splits')
    args = parser.parse_args()
    prepare_data(args.input_csv, args.splits_dir, args.output_dir)
