"""Check the notebook's submission and four PNG outputs without extra packages."""
import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PLOTS = ['feature_distributions.png', 'enhanced_correlation.png',
         'model_performance.png', 'stacking_importance.png']


def verify(test_file, output_dir):
    with test_file.open(newline='') as handle:
        reader = csv.DictReader(handle)
        if 'PassengerId' not in (reader.fieldnames or []):
            raise ValueError('test.csv must contain PassengerId.')
        ids = [row['PassengerId'] for row in reader]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('Test IDs must be nonempty and unique.')
    with (output_dir / 'submission_advanced.csv').open(newline='') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ['PassengerId', 'Survived']:
            raise ValueError('Submission must have PassengerId,Survived columns.')
        rows = list(reader)
    if [row['PassengerId'] for row in rows] != ids:
        raise ValueError('Submission IDs must match every test ID in order.')
    if any(row['Survived'] not in {'0', '1'} or None in row for row in rows):
        raise ValueError('Predictions must be 0 or 1 with no extra columns.')
    for name in PLOTS:
        path = output_dir / name
        with path.open('rb') as handle:
            if handle.read(8) != b'\x89PNG\r\n\x1a\n' or path.stat().st_size <= 8:
                raise ValueError(f'Missing or invalid PNG: {name}')
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test', type=Path, default=ROOT / 'data/test.csv')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs')
    args = parser.parse_args()
    try:
        count = verify(args.test, args.output_dir)
    except (OSError, ValueError, KeyError, csv.Error) as error:
        parser.exit(2, f'ERROR: {error}\n')
    print(f'PASS: {count} ordered passenger predictions, binary labels, and {len(PLOTS)} PNG files.')


if __name__ == '__main__':
    main()
