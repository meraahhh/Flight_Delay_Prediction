"""Build dashboard artifacts locally; never train or load a prediction model."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KEYS = ["month", "op_unique_carrier", "origin", "dest", "day_of_week", "departure_hour"]
MEASURES = ["flights", "delayed_flights", "departure_delay_minutes_sum",
            "delayed_departure_minutes_sum"]


def build(source, output, original_zip=None):
    columns = KEYS[:-1] + ["year", "day_of_month", "fl_date", "crs_dep_time",
                            "cancelled", "dep_delay", "delayed"]
    parts = []
    rows = delayed = midnight = 0
    first_date = last_date = None
    monthly = []
    for chunk in pd.read_csv(source, usecols=columns, chunksize=200_000):
        if chunk[columns].isna().any().any():
            raise ValueError("Missing values in required analysis columns")
        dates = pd.to_datetime(chunk.fl_date, errors="raise")
        valid = ((chunk.cancelled == 0) & chunk.dep_delay.notna()
                 & (chunk.delayed == (chunk.dep_delay >= 15).astype(int))
                 & (chunk.year == 2024) & (dates.dt.year == 2024)
                 & (chunk.month == dates.dt.month)
                 & (chunk.day_of_month == dates.dt.day)
                 & (chunk.day_of_week == dates.dt.day_of_week + 1))
        scheduled = chunk.crs_dep_time
        valid_time = ((scheduled % 1 == 0) & scheduled.between(0, 2400)
                      & (scheduled % 100 < 60))
        if not valid.all() or not valid_time.all():
            raise ValueError("Source fails verified cleaning, target, calendar, or time rules")
        rows += len(chunk)
        delayed += int(chunk.delayed.sum())
        midnight += int((scheduled == 2400).sum())
        low, high = dates.min(), dates.max()
        first_date = low if first_date is None else min(first_date, low)
        last_date = high if last_date is None else max(last_date, high)
        chunk["departure_hour"] = (scheduled // 100).astype("uint8") % 24
        chunk["delayed_departure_minutes"] = chunk.dep_delay.where(chunk.delayed == 1, 0)
        monthly.append(chunk.groupby("month").agg(
            flights=("delayed", "size"), delayed_flights=("delayed", "sum")))
        parts.append(chunk.groupby(KEYS, as_index=False, observed=True).agg(
            flights=("delayed", "size"), delayed_flights=("delayed", "sum"),
            departure_delay_minutes_sum=("dep_delay", "sum"),
            delayed_departure_minutes_sum=("delayed_departure_minutes", "sum")))
    data = pd.concat(parts, ignore_index=True).groupby(KEYS, as_index=False).sum()
    for col in ["month", "day_of_week", "departure_hour"]:
        data[col] = data[col].astype("uint8")
    for col in ["op_unique_carrier", "origin", "dest"]:
        data[col] = data[col].astype("category")
    for col in ["flights", "delayed_flights"]:
        data[col] = data[col].astype("uint32")
    assert int(data.flights.sum()) == rows
    assert int(data.delayed_flights.sum()) == delayed
    output.mkdir(parents=True, exist_ok=True)
    artifact = output / "flight_analysis_2024.parquet"
    data.to_parquet(artifact, index=False, compression="snappy")
    restored = pd.read_parquet(artifact)
    pd.testing.assert_frame_equal(data, restored)
    summary = pd.concat(monthly).groupby(level=0).sum()
    metadata = {
        "schema_version": 1,
        "source": {"file": source.name, "description": "Full cleaned US flight dataset, 2024",
                   "bytes": source.stat().st_size, "sha256": file_hash(source)},
        "cleaning_rule": "cancelled == 0 and dep_delay is not missing; diverted flights retained",
        "target_rule": "delayed = int(dep_delay >= 15)",
        "weekday_mapping": {str(i + 1): day for i, day in enumerate(
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])},
        "scheduled_time_rule": "HHMM; 2400 grouped into hour 0 on the recorded flight date",
        "negative_delay_rule": "Early departures remain negative in the overall delay sum",
        "date_min": first_date.date().isoformat(), "date_max": last_date.date().isoformat(),
        "totals": {"flights": rows, "delayed_flights": delayed,
                   "not_delayed_flights": rows - delayed, "delay_rate": delayed / rows,
                   "departure_delay_minutes_sum": float(data.departure_delay_minutes_sum.sum()),
                   "delayed_departure_minutes_sum": float(data.delayed_departure_minutes_sum.sum()),
                   "scheduled_2400_records": midnight},
        "unique_counts": {k: int(data[k].nunique()) for k in
                          ["op_unique_carrier", "origin", "dest"]},
        "monthly_totals": {str(k): {c: int(v) for c, v in row.items()}
                           for k, row in summary.to_dict(orient="index").items()},
        "dimensions": KEYS, "measures": MEASURES,
        "measure_definitions": {
            "flights": "Count of eligible flights",
            "delayed_flights": "Count with dep_delay >= 15 minutes",
            "departure_delay_minutes_sum": "Sum of dep_delay, including negative early departures",
            "delayed_departure_minutes_sum": "Sum of dep_delay among delayed flights only"},
        "aggregation_rule": "Sum counts and minutes after filtering; divide summed measures, never average rates",
        "limitations": ["No individual flight records, exact-date filters, medians, or delay distributions",
                        "Observed associations do not establish causes; no live flight data"],
        "artifact": {"file": artifact.name, "rows": len(data), "bytes": artifact.stat().st_size,
                     "sha256": file_hash(artifact), "compression": "snappy"},
    }
    if original_zip is not None:
        raw_rows = cancelled = missing_delay = eligible = raw_delayed = 0
        raw_months = []
        with zipfile.ZipFile(original_zip) as archive:
            with archive.open('flight_data_2024.csv') as stream:
                for chunk in pd.read_csv(stream, usecols=['month', 'cancelled', 'dep_delay'], chunksize=200_000):
                    raw_rows += len(chunk)
                    cancelled += int((chunk.cancelled == 1).sum())
                    missing_delay += int(chunk.dep_delay.isna().sum())
                    valid = chunk[(chunk.cancelled == 0) & chunk.dep_delay.notna()].copy()
                    valid['delayed'] = (valid.dep_delay >= 15).astype(int)
                    eligible += len(valid)
                    raw_delayed += int(valid.delayed.sum())
                    raw_months.append(valid.groupby('month').agg(
                        flights=('delayed', 'size'), delayed_flights=('delayed', 'sum'),
                        departure_delay_minutes_sum=('dep_delay', 'sum')))
        raw_summary = pd.concat(raw_months).groupby(level=0).sum()
        grouped_summary = data.groupby('month', observed=True)[
            ['flights', 'delayed_flights', 'departure_delay_minutes_sum']].sum()
        pd.testing.assert_frame_equal(raw_summary, grouped_summary, check_dtype=False, check_index_type=False)
        if eligible != rows or raw_delayed != delayed:
            raise ValueError('Original ZIP does not reconcile with cleaned analysis data')
        metadata['original_source_verification'] = {
            'file': original_zip.name, 'sha256': file_hash(original_zip), 'rows': raw_rows,
            'cancelled_flights': cancelled, 'missing_departure_delay': missing_delay,
            'excluded_flights': raw_rows - eligible, 'eligible_flights': eligible,
            'eligible_delayed_flights': raw_delayed,
            'monthly_counts_and_departure_delay_sums_match': True,
            'note': 'Cancelled and missing-delay counts overlap; reconciliation checks aggregates, not row identity'}
    (output / "flight_analysis_2024.metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata["totals"], indent=2))
    print(json.dumps(metadata["artifact"], indent=2))


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/flight_data_cleaned.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "data/analysis")
    parser.add_argument("--original-zip", type=Path, help="Optionally reconcile against the original flight_data_2024.csv ZIP")
    args = parser.parse_args()
    build(args.source, args.output, args.original_zip)
