# KD-LiteFusion-CLIP

Core implementation and fixed sample-ID splits. This is not a complete training reproduction package. No trained weights, raw images/text, or experimental results are included.

## Public files

- [Student model](student_model.py)
- [MKAN teacher model](teacher_model.py)
- [DKD loss](losses.py)
- [Basic configuration template](configs/basic.yaml): model fields are placeholders and must be supplied by the user.
- [Training IDs](splits/train_ids.txt): 5,171 samples
- [Validation IDs](splits/val_ids.txt): 989 samples
- [Test IDs](splits/test_ids.txt): 947 samples
- [Split summary](splits/summary.json)
- [SHA-256 checksums](splits/SHA256SUMS)
- [Standalone split verification](scripts/verify_splits.py)
- [Local data preparation](scripts/prepare_data.py)

## Verify fixed splits

Run from the repository root using Python 3; no third-party dependencies are required:

```bash
python scripts/verify_splits.py
```

The verifier checks checksums, counts, duplicate IDs and cross-split overlap. It does not evaluate model performance.

## Prepare local data

Provide a local CSV with sample_id,image_path,text,label columns. IDs must match the published lists. Raw data must be obtained separately.

```bash
python scripts/prepare_data.py --input-csv /path/to/manifest.csv --output-dir /path/to/new/output
```

The script selects and orders records using the fixed lists, checks missing and duplicate IDs, and writes train.csv, val.csv and test.csv. It does not randomly re-split data.
