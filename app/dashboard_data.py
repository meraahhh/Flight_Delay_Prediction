"""Pure aggregation helpers shared by the dashboard and verification tests."""
import pandas as pd

DIMENSIONS = ["month", "op_unique_carrier", "origin", "dest", "day_of_week", "departure_hour"]
MEASURES = ["flights", "delayed_flights", "departure_delay_minutes_sum",
            "delayed_departure_minutes_sum"]


def filter_data(data, selections, hours):
    mask = data.departure_hour.between(*hours)
    for column, values in selections.items():
        # An empty multi-select deliberately means no matching records.
        mask &= data[column].isin(values)
    return data.loc[mask]


def summarize(data, dimensions):
    result = data.groupby(dimensions, observed=True, as_index=False)[MEASURES].sum()
    result["delay_rate_pct"] = 100 * result.delayed_flights / result.flights
    result["average_departure_delay"] = result.departure_delay_minutes_sum / result.flights
    return result


def totals(data):
    counts = data[MEASURES].sum()
    n, delayed = int(counts.flights), int(counts.delayed_flights)
    return {"flights": n, "delayed_flights": delayed,
            "delay_rate": delayed / n if n else None,
            "average_departure_delay": float(counts.departure_delay_minutes_sum) / n if n else None,
            "average_delayed_departure": float(counts.delayed_departure_minutes_sum) / delayed if delayed else None}
