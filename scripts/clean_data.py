"""Reproduce notebook 02's eligibility filter without overwriting existing data."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import zipfile

import pandas as pd


REQUIRED_COLUMNS = {'cancelled', 'dep_delay'}


def clean_data(source, output, *, zip_member='flight_data_2024.csv',
               chunksize=100_000, expected_input_rows=None, expected_output_rows=None):
    """Keep cancelled == 0 and known dep_delay; preserve order and other columns.

    Missing values outside these eligibility columns are reported, not repaired.
    Invalid nonnumeric eligibility values raise an error rather than being coerced.
    """
    source, output = Path(source), Path(output)
    if chunksize < 1:
        raise ValueError('chunksize must be positive')
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite existing output: {output}')
    stats = dict(input_rows=0, output_rows=0, excluded_rows=0,
                 cancelled_rows=0, missing_departure_delay_rows=0,
                 delayed_rows=0, not_delayed_rows=0,
                 input_missing_values={}, output_missing_values={})
    created = False
    try:
        with ExitStack() as stack:
            if source.suffix.lower() == '.zip':
                archive = stack.enter_context(zipfile.ZipFile(source))
                stream = stack.enter_context(archive.open(zip_member))
            else:
                stream = stack.enter_context(source.open('rb'))
            reader = stack.enter_context(pd.read_csv(
                stream, chunksize=chunksize, dtype={'cancellation_code': 'string'}))
            destination = None
            for chunk in reader:
                missing = REQUIRED_COLUMNS - set(chunk.columns)
                if missing:
                    raise ValueError(f'Missing required columns: {", ".join(sorted(missing))}')
                # Validate without converting missing values into valid observations.
                cancelled = pd.to_numeric(chunk.cancelled, errors='raise')
                delay = pd.to_numeric(chunk.dep_delay, errors='raise')
                keep = (cancelled == 0) & delay.notna()
                cleaned = chunk.loc[keep].copy()
                cleaned['delayed'] = (delay.loc[keep] >= 15).astype(int)
                stats['input_rows'] += len(chunk)
                stats['output_rows'] += len(cleaned)
                stats['cancelled_rows'] += int((cancelled == 1).sum())
                stats['missing_departure_delay_rows'] += int(delay.isna().sum())
                stats['delayed_rows'] += int(cleaned.delayed.sum())
                for key, frame in [('input_missing_values', chunk),
                                   ('output_missing_values', cleaned)]:
                    for column, count in frame.isna().sum().items():
                        stats[key][column] = stats[key].get(column, 0) + int(count)
                if destination is None:
                    # Exclusive creation protects existing files, including race cases.
                    destination = stack.enter_context(output.open('x', encoding='utf-8', newline=''))
                    created = True
                    cleaned.to_csv(destination, index=False)
                else:
                    cleaned.to_csv(destination, index=False, header=False)
            stats['excluded_rows'] = stats['input_rows'] - stats['output_rows']
            stats['not_delayed_rows'] = stats['output_rows'] - stats['delayed_rows']
            for key, expected in [('input_rows', expected_input_rows),
                                  ('output_rows', expected_output_rows)]:
                if expected is not None and stats[key] != expected:
                    raise ValueError(f'{key}: expected {expected}, found {stats[key]}')
            if stats['delayed_rows'] + stats['not_delayed_rows'] != stats['output_rows']:
                raise ValueError('Target counts do not reconcile')
    except BaseException:
        if created:
            output.unlink()
        raise
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Original CSV or ZIP')
    parser.add_argument('--output', type=Path, required=True, help='New CSV; must not exist')
    parser.add_argument('--zip-member', default='flight_data_2024.csv')
    parser.add_argument('--chunksize', type=int, default=100_000)
    parser.add_argument('--expected-input-rows', type=int)
    parser.add_argument('--expected-output-rows', type=int)
    args = parser.parse_args()
    try:
        report = clean_data(args.input, args.output, zip_member=args.zip_member,
                            chunksize=args.chunksize,
                            expected_input_rows=args.expected_input_rows,
                            expected_output_rows=args.expected_output_rows)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        parser.exit(1, f'Cleaning failed: {exc}\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
