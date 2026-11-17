"""Run with: python -B -m unittest discover -s tests -v"""
from datetime import date, time
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import joblib
import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'app'))
from dashboard_data import filter_data, summarize, totals


class DashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = pd.read_parquet(ROOT / 'data/analysis/flight_analysis_2024.parquet')
        cls.metadata = json.loads((ROOT / 'data/analysis/flight_analysis_2024.metadata.json').read_text())

    def test_artifact_integrity_and_monthly_reconciliation(self):
        path = ROOT / 'data/analysis/flight_analysis_2024.parquet'
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), self.metadata['artifact']['sha256'])
        self.assertEqual(len(self.data), self.metadata['artifact']['rows'])
        result = totals(self.data)
        for column in ['flights', 'delayed_flights', 'delay_rate']:
            self.assertEqual(result[column], self.metadata['totals'][column])
        for row in summarize(self.data, ['month']).itertuples():
            expected = self.metadata['monthly_totals'][str(row.month)]
            self.assertEqual(row.flights, expected['flights'])
            self.assertEqual(row.delayed_flights, expected['delayed_flights'])
        self.assertTrue((self.data.delayed_flights <= self.data.flights).all())
        self.assertTrue((self.data.flights > 0).all())

    def test_rates_are_weighted_and_empty_is_safe(self):
        sample = pd.DataFrame({'month': [1, 1], 'flights': [100, 1], 'delayed_flights': [10, 1],
                               'departure_delay_minutes_sum': [100., 30.],
                               'delayed_departure_minutes_sum': [200., 30.]})
        self.assertAlmostEqual(summarize(sample, ['month']).iloc[0].delay_rate_pct, 1100 / 101)
        empty = filter_data(self.data, {'month': []}, (0, 23))
        self.assertTrue(empty.empty)
        self.assertIsNone(totals(empty)['delay_rate'])

    def test_all_chart_specs_have_streamlit_safe_padding(self):
        app = AppTest.from_file(str(ROOT / 'app/app.py'), default_timeout=30).run()
        app.radio[0].set_value('Flight Analytics').run()
        for months in [['All'], list(range(1, 13)), [1]]:
            with self.subTest(months=months):
                app.multiselect(key='analysis_months').set_value(months).run()
                self.assertFalse(app.exception)
                charts = app.get('vega_lite_chart')
                self.assertEqual(len(charts), 7)
                for chart in charts:
                    spec = json.loads(chart.proto.spec)
                    # A scalar is valid Vega-Lite but crashes Streamlit's JS
                    # when its preprocessor writes padding.bottom.
                    self.assertIsInstance(spec['padding'], dict)
                    self.assertEqual(spec['padding'], {'left': 12, 'right': 12, 'top': 12, 'bottom': 20})
                    self.assertGreater(len(chart.proto.datasets), 0)
                expected = (self.metadata['monthly_totals']['1']['flights'] if months == [1]
                            else self.metadata['totals']['flights'])
                self.assertEqual(app.metric[0].value, f'{expected:,}')

    def test_model_unchanged_and_prediction_matches_pipeline(self):
        model_path = ROOT / 'models/flight_delay_rf_model_updated.pkl'
        self.assertEqual(hashlib.sha256(model_path.read_bytes()).hexdigest(),
                         '1d1de9db718e117e42cd1f70759e6523f17438a2722d91e9d66d6c5664d562d0')
        model = joblib.load(model_path)
        app = AppTest.from_file(str(ROOT / 'app/app.py'), default_timeout=30).run()
        self.assertFalse(app.exception)
        next(b for b in app.button if b.label == 'Predict Flight').click().run()
        for d in [date(2024, 1, 1), date(2024, 1, 7)]:
            app.date_input[0].set_value(d)
            app.time_input[0].set_value(time(18, 30))
            values = {box.label: box.value for box in app.selectbox}
            frame = pd.DataFrame([{
                'month': d.month, 'day_of_month': d.day, 'day_of_week': d.weekday() + 1,
                'op_unique_carrier': values['Airline'], 'origin': values['Departure Airport'],
                'dest': values['Destination Airport'], 'crs_dep_time': 1830,
                'distance': app.number_input[0].value}])
            expected = float(model.predict_proba(frame)[0][1])
            app.button[0].click().run()
            self.assertFalse(app.exception)
            cards = {m.label: m.value for m in app.metric}
            self.assertEqual(cards['Estimated delay probability'], f'{expected:.1%}')
            self.assertEqual(cards['Predicted status'], 'Delayed' if expected >= .5 else 'On Time')
            self.assertEqual(cards['Risk level'], 'High' if expected >= .7 else 'Moderate' if expected >= .4 else 'Low')
        origin = next(b.value for b in app.selectbox if b.label == 'Departure Airport')
        next(b for b in app.selectbox if b.label == 'Destination Airport').set_value(origin)
        app.button[0].click().run()
        self.assertIn('cannot be the same', app.error[0].value)

    def test_analysis_filters_empty_threshold_and_reset(self):
        app = AppTest.from_file(str(ROOT / 'app/app.py'), default_timeout=30).run()
        app.radio[0].set_value('Flight Analytics').run()
        self.assertFalse(app.exception)
        self.assertEqual(app.metric[0].value, f"{self.metadata['totals']['flights']:,}")
        self.assertEqual(len(app.get('vega_lite_chart')), 7)
        heat_spec = json.loads(app.get('vega_lite_chart')[-1].proto.spec)
        self.assertEqual(heat_spec['mark']['type'], 'rect')
        app.multiselect(key='analysis_months').set_value([6, 7, 8])
        app.multiselect(key='analysis_airlines').set_value(['AA']).run()
        selected = filter_data(self.data, {'month': [6, 7, 8], 'op_unique_carrier': ['AA']}, (0, 23))
        expected = totals(selected)
        self.assertFalse(app.exception)
        self.assertEqual(app.metric[0].value, f"{expected['flights']:,}")
        self.assertEqual(app.metric[2].value, f"{expected['delay_rate']:.2%}")
        app.number_input(key='analysis_minimum').set_value(10_000_000).run()
        self.assertFalse(app.exception)
        self.assertTrue(any('No groups meet' in info.value for info in app.info))
        self.assertTrue(any('No heatmap cells' in info.value for info in app.info))
        app.multiselect(key='analysis_months').set_value([]).run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.metric), 0)
        self.assertIn('No flights match', app.warning[0].value)
        next(b for b in app.button if b.label == 'Reset filters').click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.metric[0].value, f"{self.metadata['totals']['flights']:,}")

    def test_home_navigation_about_and_filter_persistence(self):
        app = AppTest.from_file(str(ROOT / 'app/app.py'), default_timeout=30).run()
        self.assertFalse(app.exception)
        self.assertEqual(app.radio[0].value, 'Home')
        self.assertEqual(len(app.sidebar), 0)
        next(b for b in app.button if b.label == 'View Analytics').click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.radio[0].value, 'Flight Analytics')
        app.multiselect(key='analysis_months').set_value([7]).run()
        expected = app.metric[0].value
        app.radio[0].set_value('About').run()
        self.assertFalse(app.exception)
        self.assertEqual({m.label: m.value for m in app.metric}, {
            'Accuracy': '72.56%', 'Precision': '27.04%', 'Recall': '37.89%', 'F1-Score': '31.56%'})
        app.radio[0].set_value('Flight Analytics').run()
        self.assertFalse(app.exception)
        self.assertEqual(app.multiselect(key='analysis_months').value, [7])
        self.assertEqual(app.metric[0].value, expected)
        self.assertEqual(len(app.get('vega_lite_chart')), 7)
        self.assertEqual(len(app.get('download_button')), 2)
        app.radio[0].set_value('Home').run()
        next(b for b in app.button if b.label == 'Predict Flight').click().run()
        self.assertEqual(app.radio[0].value, 'Flight Prediction')
        self.assertFalse(app.exception)

    def test_runs_from_deployment_files_without_source_csvs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copytree(ROOT / 'app', root / 'app', ignore=shutil.ignore_patterns('__pycache__'))
            (root / 'models').mkdir()
            shutil.copy2(ROOT / 'models/flight_delay_rf_model_updated.pkl', root / 'models')
            shutil.copytree(ROOT / 'data/analysis', root / 'data/analysis')
            shutil.copytree(ROOT / 'assets', root / 'assets')
            self.assertEqual(list(root.rglob('*.csv')), [])
            st.cache_resource.clear()
            app = AppTest.from_file(str(root / 'app/app.py'), default_timeout=30).run()
            self.assertFalse(app.exception)
            next(b for b in app.button if b.label == 'Predict Flight').click().run()
            app.button[0].click().run()
            self.assertFalse(app.exception)
            self.assertTrue(any(m.label == 'Estimated delay probability' for m in app.metric))
            app.radio[0].set_value('Flight Analytics').run()
            self.assertFalse(app.exception)
            self.assertEqual(app.metric[0].value, f"{self.metadata['totals']['flights']:,}")
            st.cache_resource.clear()


if __name__ == '__main__':
    unittest.main()
