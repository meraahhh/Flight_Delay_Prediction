"""Small synthetic checks; never read or modify the project's flight datasets."""
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

import pandas as pd

from scripts.clean_data import clean_data

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ('cancelled,dep_delay,diverted,cancellation_code,note\n'
          '0,-5,0,,early\n0,14,0,,\n0,15,1,,diverted\n'
          '1,45,0,A,cancelled\n0,,0,,unknown\n'
          '0,15,1,,diverted\n,30,0,,missing cancellation\n')


class CleaningTests(unittest.TestCase):
    def test_csv_and_zip_match_notebook_across_chunks(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'input.csv'
            source.write_text(SAMPLE)
            archive = root / 'input.zip'
            with zipfile.ZipFile(archive, 'w') as z:
                z.writestr('flights.csv', SAMPLE)
            expected_csv = io.StringIO()
            for index, raw in enumerate(pd.read_csv(
                    io.StringIO(SAMPLE), chunksize=2, dtype={'cancellation_code': 'string'})):
                expected = raw[(raw.cancelled == 0) & raw.dep_delay.notna()].copy()
                expected['delayed'] = (expected.dep_delay >= 15).astype(int)
                expected.to_csv(expected_csv, index=False, header=index == 0)
            for i, input_file in enumerate([source, archive]):
                output = root / f'output{i}.csv'
                report = clean_data(input_file, output, zip_member='flights.csv', chunksize=2,
                                    expected_input_rows=7, expected_output_rows=4)
                # Compare CSV serialization with the notebook's exact operations.
                self.assertEqual(output.read_text(), expected_csv.getvalue())
                self.assertEqual(report['delayed_rows'], 2)
                self.assertEqual(report['excluded_rows'], 3)
                self.assertEqual(report['output_missing_values']['note'], 1)
                self.assertEqual(report['missing_departure_delay_rows'], 1)
                self.assertEqual(report['cancelled_rows'], 1)
            self.assertEqual(source.read_text(), SAMPLE)

    def test_existing_output_and_same_input_are_protected(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'input.csv'
            source.write_text(SAMPLE)
            with self.assertRaises(FileExistsError):
                clean_data(source, source)
            output = Path(folder) / 'existing.csv'
            output.write_text('keep me')
            with self.assertRaises(FileExistsError):
                clean_data(source, output)
            self.assertEqual(output.read_text(), 'keep me')
            self.assertEqual(source.read_text(), SAMPLE)

    def test_validation_failures_leave_no_partial_output(self):
        cases = [('wrong\n1\n', {}),
                 ('cancelled,dep_delay\n0,15\n0,invalid\n', {'chunksize': 1}),
                 (SAMPLE, {'expected_input_rows': 8}),
                 (SAMPLE, {'expected_output_rows': 5}),
                 (SAMPLE, {'chunksize': 0})]
        with tempfile.TemporaryDirectory() as folder:
            source, output = Path(folder) / 'input.csv', Path(folder) / 'output.csv'
            for data, options in cases:
                with self.subTest(options=options, data=data):
                    source.write_text(data)
                    with self.assertRaises(ValueError):
                        clean_data(source, output, **options)
                    self.assertFalse(output.exists())

    def test_zero_eligible_rows_preserve_headers(self):
        with tempfile.TemporaryDirectory() as folder:
            source, output = Path(folder) / 'input.csv', Path(folder) / 'output.csv'
            source.write_text('cancelled,dep_delay\n1,20\n0,\n')
            report = clean_data(source, output, chunksize=1)
            self.assertEqual(report['output_rows'], 0)
            self.assertEqual(list(pd.read_csv(output).columns), ['cancelled', 'dep_delay', 'delayed'])

    def test_documented_cli_from_repository_root(self):
        with tempfile.TemporaryDirectory() as folder:
            source, output = Path(folder) / 'input.csv', Path(folder) / 'output.csv'
            source.write_text(SAMPLE)
            result = subprocess.run([sys.executable, '-B', 'scripts/clean_data.py',
                                     '--input', str(source), '--output', str(output),
                                     '--expected-input-rows', '7', '--expected-output-rows', '4'],
                                    cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('"output_rows": 4', result.stdout)


if __name__ == '__main__':
    unittest.main()
