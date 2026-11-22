"""Train a separate Random Forest using notebook 02's final sampled workflow."""
import argparse
import hashlib
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
DEPLOYED_MODEL = ROOT / 'models/flight_delay_rf_model_updated.pkl'
FEATURES = ['month', 'day_of_month', 'day_of_week', 'op_unique_carrier',
            'origin', 'dest', 'crs_dep_time', 'distance']
CATEGORICAL = ['op_unique_carrier', 'origin', 'dest']
NUMERICAL = ['month', 'day_of_month', 'day_of_week', 'crs_dep_time', 'distance']
SEED = 42
CHUNKSIZE = 100_000
SAMPLE_FRACTION = 0.06
CUTOFF = '2024-10-01'


def make_pipeline():
    """Construct fresh preprocessing; never read an intermediate or deployed model."""
    preprocessing = ColumnTransformer([
        ('categorical', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=True)),
        ]), CATEGORICAL),
        ('numerical', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
        ]), NUMERICAL),
    ])
    return Pipeline([
        ('preprocessing', preprocessing),
        ('classifier', RandomForestClassifier(
            n_estimators=150, max_depth=25, min_samples_leaf=10,
            class_weight='balanced', random_state=SEED, n_jobs=-1)),
    ])


def load_splits(source):
    """Preserve notebook chunk boundaries, per-partition sampling, and row order."""
    earlier, later = [], []
    rows = 0
    columns = FEATURES + ['delayed', 'fl_date', 'dep_delay', 'cancelled']
    with pd.read_csv(source, usecols=columns, chunksize=CHUNKSIZE) as reader:
        for chunk in reader:
            dates = pd.to_datetime(chunk.fl_date, errors='raise')
            if dates.isna().any() or not (dates.dt.year == 2024).all():
                raise ValueError('Expected non-missing 2024 flight dates')
            if (chunk.cancelled != 0).any() or chunk.dep_delay.isna().any():
                raise ValueError('Input must already satisfy the notebook cleaning rule')
            if chunk.delayed.isna().any() or not (
                    chunk.delayed == (chunk.dep_delay >= 15).astype(int)).all():
                raise ValueError('Target must equal int(dep_delay >= 15)')
            rows += len(chunk)
            # Keep exactly the same sampled frame columns as the final notebook.
            frame = chunk[FEATURES + ['delayed', 'fl_date']].copy()
            frame['fl_date'] = dates
            before = frame[dates < CUTOFF]
            after = frame[dates >= CUTOFF]
            if len(before):
                earlier.append(before.sample(frac=SAMPLE_FRACTION, random_state=SEED))
            if len(after):
                later.append(after.sample(frac=SAMPLE_FRACTION, random_state=SEED))
    if not earlier or not later:
        raise ValueError('Both January–September and October–December data are required')
    train_period = pd.concat(earlier, ignore_index=True)
    test = pd.concat(later, ignore_index=True)
    if test.empty or set(train_period.delayed.unique()) != {0, 1}:
        raise ValueError('Sampling needs a nonempty test set and both earlier-period classes')
    train, validation = train_test_split(
        train_period, test_size=0.2, random_state=SEED, stratify=train_period.delayed)
    if set(train.delayed.unique()) != {0, 1}:
        raise ValueError('Training must contain both target classes')
    return train, validation, test, rows


def evaluate(model, frame):
    # Match notebook cell 48: sklearn predict, with no threshold optimization.
    prediction = model.predict(frame[FEATURES])
    target = frame.delayed
    return {
        'accuracy': float(accuracy_score(target, prediction)),
        'precision': float(precision_score(target, prediction, zero_division=0)),
        'recall': float(recall_score(target, prediction, zero_division=0)),
        'f1': float(f1_score(target, prediction, zero_division=0)),
        'confusion_matrix': confusion_matrix(target, prediction, labels=[0, 1]).tolist(),
    }


def train_model(source, output, report_path=None):
    source, output = Path(source), Path(output)
    report_path = Path(report_path) if report_path else output.with_suffix('.metrics.json')
    paths = [source.resolve(), output.resolve(), report_path.resolve()]
    if len(set(paths)) != 3 or DEPLOYED_MODEL.resolve() in paths[1:]:
        raise ValueError('Source, new model, and report must be distinct; deployed model is protected')
    for path in [output, report_path]:
        if path.exists():
            raise FileExistsError(f'Refusing to overwrite existing output: {path}')
        if not path.parent.is_dir():
            raise ValueError(f'Output parent directory does not exist: {path.parent}')
    digest = hashlib.sha256()
    with source.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    train, validation, test, rows = load_splits(source)
    model = make_pipeline()
    model.fit(train[FEATURES], train.delayed)
    splits = {'training': train, 'validation': validation, 'test': test}
    report = {
        'source': {'file': source.name, 'sha256': digest.hexdigest(), 'rows': rows},
        'features': FEATURES, 'target': 'delayed = int(dep_delay >= 15)',
        'sampling': {'chunksize': CHUNKSIZE, 'fraction_per_partition_per_chunk': SAMPLE_FRACTION,
                     'cutoff': CUTOFF, 'random_state': SEED,
                     'validation_fraction': 0.2, 'stratified_validation': True},
        'split_sizes': {name: len(frame) for name, frame in splits.items()},
        'class_counts': {name: {str(label): int((frame.delayed == label).sum())
                               for label in [0, 1]} for name, frame in splits.items()},
        'classifier_parameters': model.named_steps['classifier'].get_params(),
        'evaluation_method': 'Pipeline.predict, matching notebook cell 48; no threshold tuning',
        'metric_scale': 'fractions from 0 to 1; confusion matrix rows actual, columns predicted [0, 1]',
        'validation_metrics': evaluate(model, validation),
        'test_metrics': evaluate(model, test),
        'versions': {'python': platform.python_version(), 'pandas': pd.__version__,
                     'numpy': np.__version__, 'scikit_learn': sklearn.__version__,
                     'joblib': joblib.__version__},
        'original_metrics_reproduced': 'Not established; compare full-data run with notebook results',
    }
    created = []
    try:
        with output.open('xb') as destination:
            created.append(output)
            joblib.dump(model, destination)
        with report_path.open('x', encoding='utf-8') as destination:
            created.append(report_path)
            json.dump(report, destination, indent=2)
            destination.write('\n')
    except BaseException:
        for path in created:
            path.unlink()
        raise
    return model, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Full cleaned 2024 CSV')
    parser.add_argument('--output', type=Path, required=True, help='New joblib model; must not exist')
    parser.add_argument('--report', type=Path, help='New JSON report; default: model path with .metrics.json suffix')
    args = parser.parse_args()
    try:
        _, report = train_model(args.input, args.output, args.report)
    except (OSError, ValueError) as exc:
        parser.exit(1, f'Training failed: {exc}\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
