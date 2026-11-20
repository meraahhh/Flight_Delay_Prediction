# Aviation Signature — Flight Delay Prediction System

A university Data Science project presented as an aviation website, combining a Random Forest flight-delay predictor with interactive analysis of the full cleaned 2024 US flight dataset.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

Use the horizontal navigation to visit **Home**, **Flight Prediction**, **Flight Analytics**, and **About**. Home includes working “Predict Flight” and “View Analytics” buttons. Analytics opens with four KPI cards. An accessible filter panel sits immediately below the KPIs. Four groups—Monthly & Time Analysis, Airline Performance, Airport Performance, and Detailed Data—start collapsed. They contain all seven charts, insights, the summary table, data notes, and existing downloads. Charts use a two-column layout on desktop and stack on mobile. Open a section to inspect it and click its title again to collapse it. Filter selections persist when navigating between sections. About explains the model and its unchanged evaluation metrics. The application works without the large original CSV or ZIP files.

## Prediction Dashboard

Enter the flight date, airline, departure airport, destination airport, local scheduled departure time, and distance in miles. Airline and airport options come from the saved model's encoder.

The existing `models/flight_delay_rf_model_updated.pkl` pipeline fills missing values, one-hot encodes categorical features, and uses a 150-tree Random Forest. It is not retrained by the app or the analysis preparation script.

- A departure delay means at least 15 minutes late.
- Prediction: delayed when the estimated probability is **≥50%**.
- Risk: Low **<40%**, Moderate **40–<70%**, High **≥70%**.
- Calendar inputs use **Monday=1 through Sunday=7**, matching the training data. This corrects the previous app's weekday shift; predictions can therefore differ from the earlier interface even though the saved model is unchanged.
- “On Time” includes departures less than 15 minutes late.

Random Forest was selected after comparisons with other models, including Balanced HistGradientBoosting. Existing final test metrics on **106,105 October–December 2024 flights** remain unchanged:

| Metric | Result |
|---|---|
| Accuracy | 72.56% |
| Precision | 27.04% |
| Recall | 37.89% |
| F1-Score | 31.56% |

Final training used 250,288 records. Predictions for future dates are estimates, not live flight tracking or official flight status. Select a valid direct route and its distance; airport combinations are not route-validated.

## Analysis Dashboard

Analysis uses the **full cleaned dataset**, not the model's training/testing sample:

- Original records: **7,079,081**.
- Eligible records: **6,982,766**.
- Delayed departures: **1,441,139**.
- Overall departure delay rate: **20.6385%**.

These verified reference figures are documented here; dashboard statistics are calculated from the packaged data at runtime.

### Filters and views

Filter by month, airline, departure airport, destination airport, weekday, and scheduled departure-hour range. Each multi-select defaults to **All**; remove All to select specific values. Clearing every value shows an empty-selection message. Reset filters restores defaults.

The dashboard includes:

- KPI cards for flight count, delayed count, delay rate, and average departure delay.
- Monthly and hourly trends, airline comparisons, weekday patterns, and top-15 departure/destination airport rankings.
- A weekday × departure-hour heatmap with flight counts in tooltips.
- Insights calculated from the current selection.
- Downloadable filtered summaries and source metadata.

The minimum-flight threshold defaults to **100** and applies to airline/airport rankings, heatmap cells, and insight comparisons. Trend charts retain all available groups. Missing heatmap cells mean unavailable or below-threshold data, not zero delay.

### Calculation rules

Keep flights with `cancelled == 0` and a non-missing `dep_delay`. Diverted flights remain included. Define `delayed = int(dep_delay >= 15)`.

The deployment table groups flights by:

```text
month, op_unique_carrier, origin, dest, day_of_week, departure_hour
```

Each group contains:

```text
flights, delayed_flights,
departure_delay_minutes_sum, delayed_departure_minutes_sum
```

After applying filters, rates divide **summed delayed counts by summed flight counts**; percentages are never averaged. Overall average departure delay includes negative values for early departures. The separate delayed-flight average includes only departures at least 15 minutes late.

Weekdays use Monday=1 through Sunday=7. Scheduled `2400` is grouped into hour `00` on the recorded flight date. Times are local scheduled departure hours, not a shared timezone.

### Deployment artifacts

Commit these files with the app and saved model:

| File | Purpose |
|---|---|
| `data/analysis/flight_analysis_2024.parquet` | 2,172,103 grouped rows; approximately 9.55 MB with Snappy compression |
| `data/analysis/flight_analysis_2024.metadata.json` | Source hashes, cleaning rules, weekday mapping, verified totals, monthly counts, artifact hash, and original-ZIP reconciliation |

The optimized loaded table is approximately 69.52 MB; total app memory also includes the model, charts, and temporary calculations. PyArrow provides Parquet support and Altair provides interactive chart tooltips.

The original `data/*.csv` and `data/*.zip` remain excluded by `.gitignore`. Files inside `data/analysis/` are included. Neither dashboard reads the large source files at runtime. For Streamlit Cloud, use `app/app.py` as the entrypoint and install `requirements.txt`. No hosting or GitHub changes are performed by the preparation script.

### Rebuild analysis data locally

Only rebuild when the full local cleaned dataset is available:

```bash
python scripts/prepare_analysis.py --original-zip data/flight_data_2024.csv.zip
```

The script reads the cleaned CSV in chunks, validates cleaning/target/calendar/time rules, groups the full data, verifies a Parquet round trip, and writes the two analysis artifacts. With `--original-zip`, it independently applies the cleaning rule to the original CSV inside the ZIP and checks monthly counts and departure-delay sums against the generated data. Reconciliation checks aggregates, not exact row identity.

Source and output locations can be overridden with `--source` and `--output`. The script never loads, retrains, or modifies a model, source dataset, or notebook.

## Validation

```bash
python -B -m unittest discover -s tests -v
```

Tests check artifact integrity, monthly reconciliation, weighted calculations, empty selections, combined filters, minimum thresholds, reset behavior, Home buttons, navigation, filter persistence, About metrics, Monday/Sunday predictions against the saved pipeline, unchanged model bytes, and startup/prediction/analysis from a temporary deployment copy containing no original CSVs.

## Project structure

```text
app/app.py                         Streamlit navigation and dashboards
app/dashboard_data.py              Shared filter and aggregation helpers
scripts/prepare_analysis.py         Reproducible local artifact builder
models/flight_delay_rf_model_updated.pkl
data/analysis/                      Packaged analysis data and metadata
notebooks/                         Original exploration and training notebooks
.streamlit/config.toml             White, soft blue, and deep navy theme
requirements.txt                   Pinned runtime dependencies
tests/test_dashboard.py            Data and Streamlit interaction checks
```

## Analysis limitations

The grouped table supports the documented combined filters, not individual flight records, exact-date filters, medians, or delay distributions. It excludes cancelled flights, so it cannot estimate cancellation rates. Comparisons describe historical associations, not causes or guaranteed future performance. Airlines operate different routes and schedules; small groups require caution. Analysis covers all eligible airports, independently of the narrower airport categories seen by the prediction model.

## Visual design and bundled photography

Aviation Signature uses an original premium aviation identity: deep navy `#14253D`, burgundy `#922F48`, warm ivory `#F5F0E9`, white `#FFFFFF`, and champagne gold `#C5A47E`. Burgundy actions, serif display typography, restrained card accents, a subtle navy photo overlay, and editorial About content are applied consistently across the four sections. No airline logos or proprietary brand assets are used. Home includes the headline “Fly Smarter. Travel with Confidence.”, a locally bundled aviation photograph, and functional navigation buttons. The prediction form uses a booking-style layout; analytics retains all seven charts, filters, KPI calculations, heatmap, and downloads.

`assets/aviation-hero.jpg` is a 1920 × 1440 photograph credited to Pixabay on Pexels, used under the [Pexels License](https://www.pexels.com/license/). Source, download details, and usage notes are recorded in [assets/ATTRIBUTION.md](assets/ATTRIBUTION.md). Visible credit appears beside the hero image. The image is read from local bytes and embedded, so Streamlit Cloud does not need a remote image request. Include the entire `assets/` directory when deploying. No AI-generated hero imagery is used.

Browser regression scripts cover header contrast and desktop/mobile navigation (`tests/browser_header.cjs`), all-month and January chart rendering (`tests/browser_analysis.cjs`), and complete website workflows, offline image loading, prediction results, About metrics, and downloads (`tests/browser_website.cjs`). They use Playwright as a separate development tool; it is not needed in Streamlit Cloud runtime dependencies.

### Airport display names

The Prediction page shows `IATA — Airport name` using the locally bundled `assets/airport_names.json` display reference, derived from [OurAirports](https://github.com/davidmegginson/ourairports-data) (public domain). Only unique active US/territory airport matches are included. PBI uses its verified historical name because it is the category used by the 2024 model; its code is never replaced with a newer identifier. The reference records its retrieval date, source hash, selection rules, license, and historical-code source. Unavailable names fall back to the original code. No external lookup is required at runtime. This display reference does not alter the model or the flight datasets.

The compact Prediction form retains all six inputs. Results use three cards—Predicted Status, Delay Probability, and Risk Level—with one visible historical-estimate note. Threshold details are available through card help icons and About.
