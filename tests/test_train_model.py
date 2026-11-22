"""Synthetic training checks only: never train on the project's original data."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from scripts import train_model as training

ROOT = Path(__file__).resolve().parents[1]


def synthetic_data():
    rows = []
    for period, month in [('2024-03-04', 3), ('2024-11-04', 11)]:
        for i in range(1000):
            rows.append(dict(month=month, day_of_month=4, day_of_week=1,
                             op_unique_carrier='AA' if i % 2 else 'BB',
                             origin=f'A{i}', dest='XYZ', crs_dep_time=1200,
                             distance=(100 + i if month == 3 else 99999),
                             fl_date=period, dep_delay=15 if i % 2 else -5,
                             cancelled=0, delayed=i % 2))
    frame = pd.DataFrame(rows)
    frame.loc[::7, 'distance'] = np.nan
    return frame


class TrainingTests(unittest.TestCase):
    def test_pipeline_matches_deployed_settings_without_loading_it_in_training(self):
        deployed = joblib.load(training.DEPLOYED_MODEL)
        fresh = training.make_pipeline()
        self.assertEqual(fresh.named_steps['classifier'].get_params(),
                         deployed.named_steps['classifier'].get_params())
        self.assertEqual(training.FEATURES, deployed.feature_names_in_.tolist())
        self.assertEqual(repr(fresh.named_steps['preprocessing']),
                         repr(deployed.named_steps['preprocessing']))
        self.assertFalse(hasattr(fresh.named_steps['preprocessing'], 'transformers_'))

    def test_chunk_sampling_and_split_match_notebook(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'input.csv'
            synthetic_data().to_csv(source, index=False)
            before, after = [], []
            # Small chunks exercise partition rounding and boundary behavior.
            for chunk in pd.read_csv(source, chunksize=233):
                chunk.fl_date = pd.to_datetime(chunk.fl_date)
                for mask, accumulator in [(chunk.fl_date < training.CUTOFF, before),
                                           (chunk.fl_date >= training.CUTOFF, after)]:
                    part = chunk.loc[mask, training.FEATURES + ['delayed', 'fl_date']]
                    if len(part):
                        accumulator.append(part.sample(frac=0.06, random_state=42))
            expected_train, expected_val = train_test_split(
                pd.concat(before, ignore_index=True), test_size=0.2,
                stratify=pd.concat(before, ignore_index=True).delayed, random_state=42)
            with patch.object(training, 'CHUNKSIZE', 233):
                train, val, test, rows = training.load_splits(source)
            for actual, expected in [(train, expected_train), (val, expected_val),
                                      (test, pd.concat(after, ignore_index=True))]:
                pd.testing.assert_frame_equal(actual, expected)
            self.assertEqual(rows, 2000)

    def test_training_only_preprocessing_metrics_and_determinism(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'input.csv'
            synthetic_data().to_csv(source, index=False)
            original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            train, val, test, _ = training.load_splits(source)
            model, report = training.train_model(source, root / 'new.pkl')
            preprocessing = model.named_steps['preprocessing']
            encoder = preprocessing.named_transformers_['categorical'].named_steps['encoder']
            self.assertEqual(set(encoder.categories_[1]), set(train.origin))
            imputer = preprocessing.named_transformers_['numerical'].named_steps['imputer']
            np.testing.assert_allclose(imputer.statistics_, np.nanmedian(train[training.NUMERICAL], axis=0))
            self.assertTrue((test.fl_date >= training.CUTOFF).all())
            self.assertTrue((train.fl_date < training.CUTOFF).all())
            self.assertTrue(set(train.index).isdisjoint(val.index))
            self.assertEqual(report['split_sizes'], {'training': 48, 'validation': 12, 'test': 60})
            expected = confusion_matrix(test.delayed, model.predict(test[training.FEATURES]), labels=[0, 1])
            self.assertEqual(report['test_metrics']['confusion_matrix'], expected.tolist())
            tn, fp, fn, tp = expected.ravel()
            expected_precision = tp / (tp + fp) if tp + fp else 0
            expected_recall = tp / (tp + fn) if tp + fn else 0
            expected_f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0
            for name, value in [('accuracy', (tn + tp) / expected.sum()),
                                ('precision', expected_precision), ('recall', expected_recall),
                                ('f1', expected_f1)]:
                self.assertAlmostEqual(report['test_metrics'][name], value)
            for metric in ['accuracy', 'precision', 'recall', 'f1']:
                self.assertTrue(0 <= report['test_metrics'][metric] <= 1)
            restored = joblib.load(root / 'new.pkl')
            np.testing.assert_allclose(model.predict_proba(test[training.FEATURES]),
                                          restored.predict_proba(test[training.FEATURES]), rtol=1e-14, atol=1e-15)
            second, second_report = training.train_model(source, root / 'second.pkl')
            np.testing.assert_allclose(model.predict_proba(test[training.FEATURES]),
                                          second.predict_proba(test[training.FEATURES]), rtol=1e-14, atol=1e-15)
            self.assertEqual(report['test_metrics'], second_report['test_metrics'])
            self.assertEqual(json.loads((root / 'new.metrics.json').read_text()), report)
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), original_hash)

    def test_guards_and_invalid_input(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, output = root / 'input.csv', root / 'new.pkl'
            synthetic_data().to_csv(source, index=False)
            deployed_hash = hashlib.sha256(training.DEPLOYED_MODEL.read_bytes()).hexdigest()
            for path in [source, training.DEPLOYED_MODEL]:
                with self.assertRaises(ValueError):
                    training.train_model(source, path)
            output.write_bytes(b'keep')
            with self.assertRaises(FileExistsError):
                training.train_model(source, output)
            self.assertEqual(output.read_bytes(), b'keep')
            output.unlink()
            report = root / 'new.metrics.json'
            report.write_text('keep report')
            with self.assertRaises(FileExistsError):
                training.train_model(source, output)
            self.assertEqual(report.read_text(), 'keep report')
            report.unlink()
            for case in ['target', 'cancelled', 'date', 'missing column', 'one period']:
                frame = synthetic_data()
                if case == 'target': frame.loc[0, 'delayed'] = 1
                if case == 'cancelled': frame.loc[0, 'cancelled'] = 1
                if case == 'date': frame.loc[0, 'fl_date'] = None
                if case == 'missing column': frame = frame.drop(columns=['distance'])
                if case == 'one period': frame = frame.iloc[:1000]
                frame.to_csv(source, index=False)
                with self.subTest(case=case), self.assertRaises(ValueError):
                    training.train_model(source, output)
                self.assertFalse(output.exists())
                self.assertFalse(report.exists())
            self.assertEqual(hashlib.sha256(training.DEPLOYED_MODEL.read_bytes()).hexdigest(), deployed_hash)

    def test_cli_on_synthetic_data(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            synthetic_data().to_csv(root / 'input.csv', index=False)
            result = subprocess.run([sys.executable, '-B', 'scripts/train_model.py',
                                     '--input', str(root / 'input.csv'),
                                     '--output', str(root / 'new.pkl')],
                                    cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)['split_sizes']['test'], 60)


if __name__ == '__main__':
    unittest.main()
