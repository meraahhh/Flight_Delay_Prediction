# Aviation Signature — Flight Delay Prediction System

A Python and Streamlit application for estimating departure-delay risk and exploring historical US flight patterns.

- **Website:** [Aviation Signature](https://meerah-flight-delay-prediction.streamlit.app/)
- **Repository:** [Flight_Delay_Prediction](https://github.com/meraahhh/Flight_Delay_Prediction)

## 1. Project Overview

Flight delays make travel planning uncertain. This system uses a saved Random Forest model to estimate whether a departure will be **at least 15 minutes late**, given scheduled flight details. A separate analytics dashboard explores the full cleaned 2024 dataset.

The binary target is `delayed = int(dep_delay >= 15)`, where `dep_delay` is departure delay in minutes. Class `1` means delayed; class `0` includes early departures and departures less than 15 minutes late. Consequently, the application's **On Time** label does not mean exactly on schedule.

Predictions are estimates based on historical data, **not live flight status**. The application does not train models or retrieve live flight information.

## 2. Project Architecture

```mermaid
flowchart TD
    RAW[Local 2024 flight CSV inside ZIP] --> CLEAN[Notebook 02: eligibility filtering and target creation]
    CLEAN --> CSV[Local full cleaned CSV]
    CSV --> SAMPLE[Notebook 02: sampled temporal partitions]
    SAMPLE --> PREP[Imputation and one-hot encoding]
    PREP --> TRAIN[Random Forest training and evaluation]
    TRAIN --> MODEL[Saved joblib pipeline]
    CSV --> BUILD[scripts/prepare_analysis.py]
    RAW -. optional reconciliation .-> BUILD
    BUILD --> ANALYSIS[Compact Parquet and metadata JSON]
    MODEL --> APP[Streamlit: app/app.py]
    ANALYSIS --> APP
    ASSETS[Bundled photograph and airport labels] --> APP
    APP --> GIT[GitHub main: code and runtime artifacts]
    GIT --> CLOUD[Streamlit Community Cloud]
    CLOUD --> USER[Browser: Home, Prediction, Analytics, About]
```

- **Offline notebooks** explore data, create the cleaned CSV, sample records, train models, and save the selected pipeline.
- **Analysis preparation** validates and aggregates the full cleaned data independently of model training.
- **The application** loads the saved pipeline and compact analytics artifacts. It calculates filtered dashboard statistics at runtime.
- **Deployment** needs the tracked runtime artifacts, not the large original datasets or notebook execution.

Training is notebook-based; it is not a fully automated build pipeline. See [Reproducing the Workflow](#11-reproducing-the-workflow) for prerequisites and gaps.

## 3. Repository Structure

The tracked project structure is:

```text
Flight_Delay_Prediction/
├── .gitignore
├── .streamlit/
│   └── config.toml
├── README.md
├── requirements.txt
├── app/
│   ├── app.py
│   ├── airport_labels.py
│   └── dashboard_data.py
├── assets/
│   ├── ATTRIBUTION.md
│   ├── airport_names.json
│   └── aviation-hero.jpg
├── data/
│   └── analysis/
│       ├── flight_analysis_2024.metadata.json
│       └── flight_analysis_2024.parquet
├── models/
│   └── flight_delay_rf_model_updated.pkl
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   └── 02_full_data_analysis.ipynb
├── scripts/
│   └── prepare_analysis.py
└── tests/
    ├── test_dashboard.py
    ├── browser_helpers.cjs
    ├── browser_header.cjs
    ├── browser_analysis.cjs
    ├── browser_website.cjs
    └── browser_quality.cjs
```

| Component | Responsibility |
|---|---|
| `app/app.py` | Four-page navigation, booking-style prediction form, charts, downloads, and interface styling |
| `app/dashboard_data.py` | Filtering, summed aggregation, weighted rates, and totals |
| `app/airport_labels.py` | Airport display-name lookup with code-only fallback |
| `scripts/prepare_analysis.py` | Chunked validation, full-data aggregation, Parquet generation, metadata, and optional raw-ZIP reconciliation |
| `models/flight_delay_rf_model_updated.pkl` | Deployed fitted preprocessing and Random Forest pipeline |
| `notebooks/01_data_understanding.ipynb` | Exploration of a 10,000-row sample |
| `notebooks/02_full_data_analysis.ipynb` | Full-data inspection and cleaning, sampled experiments, final temporal evaluation, and model saving |
| `data/analysis/` | Packaged full-data analytics and its provenance/validation metadata |
| `.streamlit/config.toml` | Light theme using navy, burgundy, and ivory; not Cloud account configuration |
| `tests/` | Python data/application checks and separate Playwright browser regressions |
| `assets/` | Locally bundled interface resources; no remote image request is required |

The photograph's source and license are recorded in [assets/ATTRIBUTION.md](assets/ATTRIBUTION.md). Airport labels have source and licensing metadata inside `airport_names.json`; they are display references, not model features. Neither asset license establishes a license for the flight dataset or project code.

## 4. Dataset

### Scope and provenance

The repository identifies the input as **US flight data for calendar year 2024**, covering January 1 through December 31. The local source ZIP is named `flight_data_2024.csv.zip` and contains `flight_data_2024.csv`.

**The notebooks and dataset metadata do not record an upstream provider, download URL, or flight-data license.** Do not assume a particular provider. Obtaining the original input and confirming its usage rights are manual prerequisites for rebuilding the workflow. The packaged application can run without obtaining it.

### Verified full-data totals

These reference values come from [the committed metadata](data/analysis/flight_analysis_2024.metadata.json), including raw-ZIP reconciliation. They are not the training-sample counts.

| Measure | Count |
|---|---:|
| Original flight records | 7,079,081 |
| Cancelled flights | 96,315 |
| Records with missing departure delay | 92,970 |
| Records excluded by the combined eligibility rule | 96,315 |
| Cleaned eligible records | 6,982,766 |
| Delayed: class `1` | 1,441,139 (20.6385%) |
| Not delayed: class `0` | 5,541,627 (79.3615%) |

Cancelled and missing-delay counts **overlap** and must not be added. Diverted flights are retained. The full cleaned data contains 15 carrier codes and 348 distinct airports in each of the origin and destination columns. Prediction choices instead come from categories learned by the saved model; they need not cover every full-data airport.

### Prediction features

| Model column | Meaning and application encoding |
|---|---|
| `month` | Scheduled date's month, 1–12 |
| `day_of_month` | Scheduled date's day |
| `day_of_week` | Monday = 1, Tuesday = 2, …, Sunday = 7 |
| `op_unique_carrier` | Operating carrier code |
| `origin` | Departure airport IATA code |
| `dest` | Destination airport IATA code |
| `crs_dep_time` | Local scheduled departure time as integer HHMM, e.g. 1430; not minutes since midnight |
| `distance` | Flight distance in miles, entered by the user |

The form collects six inputs; its date produces three calendar features. Airport names are shown alongside codes where available, while the model receives the original codes.

### Large files excluded from Git

`.gitignore` excludes direct `data/*.csv` and `data/*.zip` files. Local files include:

- `data/flight_data_2024.csv.zip`: original input.
- `data/flight_data_cleaned.csv`: full eligible dataset, 1,309,555,273 bytes.
- `data/flight_data_2024_sample.csv`: exploration sample.
- `data/flight_data_2024_data_dictionary.csv`: local column reference.

These are not supplied by a fresh clone. The previous `models/flight_delay_rf_model.pkl` is also ignored; the deployed **updated** artifact is tracked. Files nested in `data/analysis/` are tracked.

### Packaged analysis schema

The Snappy-compressed Parquet contains **2,172,103 grouped rows**, occupying **9,552,946 bytes** (about 9.55 MB). A row represents a combination of six dimensions, not one flight:

| Type | Columns |
|---|---|
| Dimensions | `month`, `op_unique_carrier`, `origin`, `dest`, `day_of_week`, `departure_hour` |
| Additive measures | `flights`, `delayed_flights`, `departure_delay_minutes_sum`, `delayed_departure_minutes_sum` |

Metadata records source/artifact SHA-256 hashes, sizes, cleaning rules, date range, weekday mapping, totals, monthly counts, and optional original-source reconciliation. This supports exact summed statistics for the available filter dimensions without deploying the original CSV.

## 5. Data Cleaning

`notebooks/02_full_data_analysis.ipynb` implements the following workflow (cell numbers below are **zero-based**, matching the notebook's JSON order):

1. **Inspect the original ZIP** in cells 0–3: count rows, inspect missing values, count cancellations/diversions, and calculate eligible records.
2. **Read in 100,000-row chunks** in cell 4, specifying a string dtype for `cancellation_code`.
3. **Keep only** `(cancelled == 0) & dep_delay.notna()`.
4. **Create** `delayed = (dep_delay >= 15).astype(int)`.
5. **Write the retained original columns plus the target** to `data/flight_data_cleaned.csv`, appending chunks with one header.
6. **Inspect remaining missing values and target counts** in cells 5–7. Missing values in unrelated columns are not globally dropped or filled by this cleaning step.

Cell 4 deletes an existing cleaned output before rebuilding it. Use a fresh output location or preserve an existing file before running that cell.

The implementation does **not** establish full-dataset deduplication, outlier clipping, removal of negative delays, or comprehensive invalid-value repair. The sample notebook checks duplicates only in its sample; that is not proof that the full dataset is duplicate-free. Negative delays represent early departures and remain in analytics sums.

### Additional analytics validation

`scripts/prepare_analysis.py` validates rather than silently repairs the cleaned input:

- Required analysis columns must be present and non-missing.
- Eligibility and target values must match the cleaning rules.
- Dates must be in 2024; year, month, day, and Monday-based weekday must agree.
- Scheduled HHMM values must be valid integers with valid minutes. `2400` is grouped into hour `0` on the recorded date; the date is not advanced.
- Summed aggregates must reconcile with source totals; the written Parquet must survive a read-back comparison.
- With `--original-zip`, counts and departure-delay sums are reconciled by month against independently filtered raw records. This verifies aggregates, not exact row identity.

The script does not deduplicate data, alter source files, or load/train a model.

## 6. Data Preprocessing

The deployed joblib artifact includes its fitted preprocessing, so callers supply the eight original feature columns rather than manually encoded arrays.

- **Categorical columns:** `op_unique_carrier`, `origin`, `dest`; most-frequent imputation, followed by `OneHotEncoder(handle_unknown="ignore")`.
- **Numerical columns:** calendar fields, scheduled HHMM, and distance; median imputation.
- **Scaling:** none in the deployed Random Forest. A separate logistic-regression experiment uses scaling; it is not the deployed pipeline.

### Splits used for the final saved model

The final workflow in notebook 02 samples **6% from each temporal partition within each 100,000-row chunk**, using `random_state=42`:

| Partition | Period / method | Records |
|---|---|---:|
| Earlier-period sample | January–September 2024 | 312,861 |
| Final training | 80% of that sample | 250,288 |
| Validation | 20% of that sample | 62,573 |
| Final test | October–December 2024 sample | 106,105 |

The training/validation split is stratified by target with `test_size=0.2`, `random_state=42`. The October–December sample is kept separate from final fitting. Earlier notebook experiments use a different 3% sample and random 80/20 split; those are not the final model's evaluation split.

### Leakage safeguards and limits

The final features use scheduled/calendar information, not observed departure delay, arrival outcomes, or the target. Preprocessing is fitted with the training pipeline; later-period flights are not part of final Random Forest fitting.

This is not evidence of a completely untouched evaluation process: the exploratory notebook compares thresholds/models using held-out results, and its HGB comparison reuses the Random Forest preprocessing object. A stricter experiment would isolate preprocessing objects and reserve an untouched evaluation set. The repository does not establish route/flight-group isolation across partitions. Do not claim a complete leakage audit.

## 7. Machine Learning Model

**Algorithm:** `sklearn.ensemble.RandomForestClassifier`, an ensemble of decision trees whose outputs are combined to estimate class probabilities.

**Artifact:** `models/flight_delay_rf_model_updated.pkl`, loaded using `joblib`. It is a scikit-learn `Pipeline` with `preprocessing` and `classifier` steps, approximately 15.42 MiB.

Verified saved classifier configuration:

| Parameter | Value |
|---|---|
| `n_estimators` | `150` |
| `max_depth` | `25` |
| `min_samples_leaf` | `10` |
| `class_weight` | `"balanced"` |
| `random_state` | `42` |
| `n_jobs` | `-1` |

Notebook 02 also contains logistic-regression, earlier Random Forest, and balanced HistGradientBoosting experiments. These are not additional models served by the application.

### Verified evaluation

Notebook 02 cell 48 reports the following final Random Forest results on **106,105 October–December 2024 sampled flights**. The About page displays the same rounded metrics.

| Metric | Result |
|---|---:|
| Accuracy | 72.56% |
| Precision, delayed class | 27.04% |
| Recall, delayed class | 37.89% |
| F1-score, delayed class | 31.56% |

Confusion matrix: rows are actual classes; columns are predicted classes.

| Actual / Predicted | Not delayed (`0`) | Delayed (`1`) |
|---|---:|---:|
| Not delayed (`0`) | 70,280 | 18,110 |
| Delayed (`1`) | 11,002 | 6,713 |

These are saved notebook evaluation results, not a fresh evaluation on every application startup. Tests verify that the displayed percentages agree with this matrix. The test records themselves are not committed as a separate dataset.

### Application decision rules

The application uses the class-1 value from `model.predict_proba(...)`:

- Predicted **Delayed** when probability is **≥ 0.50**; otherwise **On Time**.
- **Low** risk: probability < 0.40.
- **Moderate** risk: 0.40 ≤ probability < 0.70.
- **High** risk: probability ≥ 0.70.

Risk bands and the classification cutoff serve different purposes: a Moderate-risk flight can still be classified On Time. No probability-calibration stage is present; the displayed value is a model estimate, not a guaranteed frequency.

## 8. Application Features

### Home

Bundled airplane photograph, concise introduction, and functional **Predict Flight** / **View Analytics** buttons. Horizontal navigation connects all four pages. The Aviation Signature interface uses navy, burgundy, ivory, white, and champagne-gold accents with responsive layouts.

### Flight Prediction

Booking-style form for departure/destination airports, date, airline, local scheduled departure time, and distance. Airport/carrier choices come from the model encoder. Available airport names are display-only. The form rejects identical origin/destination airports and restricts distance to at least one mile; it does not validate route existence or calculate distance.

Results show **Predicted Status**, **Delay Probability**, and **Risk Level**, with a visible historical-data/non-live-status note.

### Flight Analytics

Four visible KPIs: flight count, delayed count, delay rate, and average departure delay. Filters cover month, airline, departure/destination airport, weekday, and inclusive local scheduled-hour range. Filter selections persist across navigation. Remove **All** to choose individual values; clearing a selection deliberately produces no matches. Reset restores defaults.

Four collapsible groups start closed:

| Group | Contents |
|---|---|
| Monthly & Time Analysis | Monthly delay chart, departure-hour chart, weekday chart, weekday × hour heatmap |
| Airline Performance | Airline delay comparison |
| Airport Performance | Departure and destination comparisons, up to 15 airports per ranking |
| Detailed Data | Calculated insights, summary table, data notes, CSV summary and JSON metadata downloads |

All **seven charts** retain tooltips and readable scales. Summary tables can group by any of the six dimensions. The minimum-flight control defaults to **100**, applying to rankings, heatmap cells, and insight comparisons; trend charts retain all available groups. Blank heatmap cells mean unavailable/below-threshold data, not zero delay.

Calculations after filtering use additive measures:

```text
delay rate = sum(delayed_flights) / sum(flights)
average departure delay = sum(departure_delay_minutes_sum) / sum(flights)
average delayed departure = sum(delayed_departure_minutes_sum) / sum(delayed_flights)
```

Percentages are never averaged across groups. The overall average includes negative early departures. Empty selections show a clear message instead of dividing by zero. Metadata downloads describe the source artifact, while CSV summaries reflect the current filters and selected grouping.

### About

Compact **How It Works**, **Model Performance**, and **Data & Limitations** sections, including verified metrics and the historical-data disclaimer.

## 9. Installation and Setup

### Prerequisites

- Git and Python with `pip` and `venv`.
- **Python 3.11.9 is verified in the local environment.** The repository does not pin the Python interpreter or provide a supported-version matrix; use Python 3.11 to match the verified setup.
- Internet access for the initial clone and dependency installation.

`requirements.txt` pins Streamlit 1.67.0, pandas 3.1.1, NumPy 2.4.6, scikit-learn 1.9.1, joblib 1.6.0, PyArrow 26.0.1, and Altair 6.3.0. PyArrow reads Parquet; Altair builds charts. Keep the pinned environment when loading the saved scikit-learn artifact.

### macOS / Unix shell

```bash
git clone https://github.com/meraahhh/Flight_Delay_Prediction.git
cd Flight_Delay_Prediction
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
```

### Windows PowerShell

```powershell
git clone https://github.com/meraahhh/Flight_Delay_Prediction.git
cd Flight_Delay_Prediction
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip check
```

If PowerShell blocks activation, use `.\.venv\Scripts\python.exe` instead of `python` in subsequent commands; activation is optional when calling the environment's interpreter directly. In Windows Command Prompt, activate with `.venv\Scripts\activate.bat`.

## 10. How to Run the Project

From the repository root, with the virtual environment active:

```bash
python -m streamlit run app/app.py
```

The expected default local address is **http://localhost:8501**. Use the address printed by Streamlit if its port or server configuration differs. Stop the server with `Ctrl+C`. The public review site remains [Aviation Signature](https://meerah-flight-delay-prediction.streamlit.app/).

### Runtime files

A normal clone already supplies:

- `app/app.py`, `app/dashboard_data.py`, and `app/airport_labels.py`.
- `models/flight_delay_rf_model_updated.pkl`.
- Both files in `data/analysis/`.
- `assets/aviation-hero.jpg`; retain `airport_names.json` for named airport labels and `ATTRIBUTION.md` for provenance.
- `requirements.txt` and `.streamlit/config.toml`.

**No data-preparation command is needed to run the committed application.** The original ZIP, cleaned CSV, notebooks, and old model are not runtime dependencies. Airport-name lookup has a code-only fallback; the photograph, model, and analytics artifacts should be bundled for the complete experience. No API credentials are used by the application.

### Optional: rebuild analytics

Only when the full cleaned CSV is available:

```bash
python scripts/prepare_analysis.py --help
python scripts/prepare_analysis.py
```

The default source is `data/flight_data_cleaned.csv`; the default output directory is `data/analysis`. To also reconcile against the original ZIP:

```bash
python scripts/prepare_analysis.py --original-zip data/flight_data_2024.csv.zip
```

For a separate build that leaves the committed artifacts untouched:

```bash
python scripts/prepare_analysis.py --source data/flight_data_cleaned.csv --output data/analysis_rebuild --original-zip data/flight_data_2024.csv.zip
```

The last command creates a new output directory; it is an optional example, not a tracked folder. Builds overwrite the two output artifacts in the selected directory. Omitting `--original-zip` omits raw-source reconciliation from the new metadata. The script never retrains or replaces a model.

## 11. Reproducing the Workflow

There are two distinct tasks: **running the supplied application** is directly reproducible from a clone; **recreating the trained artifact** requires the external dataset and manual notebook setup.

1. **Install the runtime** using section 9.
2. **Obtain the input data manually.** Place the original ZIP in `data/` with the filenames described above. No download script or upstream URL is supplied. The sample exploration notebook additionally requires its ignored sample CSV; no sample-generation script is provided.
3. **Install notebook tools separately**, if cleaning/training is required. These are development dependencies, not part of `requirements.txt`:

   ```bash
   python -m pip install jupyterlab matplotlib seaborn
   python -m jupyterlab
   ```

   Select the virtual environment's Python kernel. `01_data_understanding.ipynb` expects its sample at `../data/flight_data_2024_sample.csv` relative to the notebook working directory.

4. **Prepare notebook 02's working directory and input variable.** Its saved code uses root-relative `data/` and `models/` paths, and references `file_path` without defining it. Before running its cells, set the kernel working directory to the repository root. For example, when the kernel starts inside `notebooks/`, run this setup cell:

   ```python
   from pathlib import Path
   import os

   os.chdir(Path.cwd().parent)  # Only if the current directory is notebooks/
   file_path = Path("data/flight_data_2024.csv.zip")
   ```

   If the kernel already starts at the repository root, omit `os.chdir(...)`. This is required manual setup, not an existing repository script.

5. **Clean the data** using notebook 02 cells 0–7, preserving any existing cleaned output first. Inspect the resulting counts against metadata. There is no standalone cleaning CLI.
6. **Preprocess and train through the notebook.** Earlier cells define preprocessing, experiment pipelines, and the ignored `models/flight_delay_rf_model.pkl`. The later final workflow samples temporal partitions (cell 27), resets the final split (cell 40), clones the earlier model settings and fits the final Random Forest (cell 41), evaluates it (cell 48), and saves `models/flight_delay_rf_model_updated.pkl` (cell 52). These are notebook operations, not executable training commands supplied by a script.
7. **Prepare full-data analytics** with the `prepare_analysis.py` command in section 10. This uses the full cleaned CSV, independently of training samples.
8. **Run tests**, then launch Streamlit as shown below and in section 10.

### Reproduction caveats

Some earlier saved notebook outputs reflect smaller samples than the current 6% sampling code. The final split/evaluation cells report the consistent 250,288 / 62,573 / 106,105 counts documented here. The notebook is exploratory, relies on execution state, and shares a preprocessing object in a comparison experiment. It should not be represented as an unattended, guaranteed byte-identical retraining procedure. Keep the committed model when the goal is to reproduce the deployed application's behavior.

### Automated checks

Run the Python suite from the root:

```bash
python -B -m unittest discover -s tests -v
```

It checks data hashes/totals, weighted aggregation, filters and empty selections, thresholds, summary groupings, navigation, metrics against the recorded confusion matrix, Monday/Sunday predictions against the saved pipeline, and deployment without the original CSVs. These checks do not retrain the model or independently regenerate held-out evaluation predictions.

Browser scripts use **Node.js and Playwright separately**; neither is a Streamlit runtime requirement. A separate tools directory keeps their dependencies out of this repository. For example, on macOS:

```bash
npm install --prefix /tmp/flight-browser playwright
PLAYWRIGHT_BROWSERS_PATH=/tmp/flight-browser/browsers /tmp/flight-browser/node_modules/.bin/playwright install --only-shell chromium webkit
export NODE_PATH=/tmp/flight-browser/node_modules
export PLAYWRIGHT_BROWSERS_PATH=/tmp/flight-browser/browsers
```

On Windows PowerShell, an equivalent temporary setup is:

```powershell
$browserTools = Join-Path $env:TEMP "flight-browser"
npm install --prefix $browserTools playwright
$env:NODE_PATH = Join-Path $browserTools "node_modules"
$env:PLAYWRIGHT_BROWSERS_PATH = Join-Path $browserTools "browsers"
& "$browserTools\node_modules\.bin\playwright.cmd" install --only-shell chromium webkit
```

Start a test server in another terminal with the same Python environment:

```bash
python -m streamlit run app/app.py --server.port 8765 --server.headless true
```

Then run the browser scripts **sequentially**:

```bash
node tests/browser_header.cjs
node tests/browser_analysis.cjs
node tests/browser_website.cjs
node tests/browser_quality.cjs
```

Their default target is `http://127.0.0.1:8765`; `FLIGHT_APP_URL` can override it. Coverage includes Chromium/WebKit, navigation, all-month/January charts, predictions, downloads, and desktop/tablet/mobile layout checks. `browser_helpers.cjs` supports the other scripts and is not a separate test runner.

## 12. Deployment

The public application is hosted on **Streamlit Community Cloud**, backed by this GitHub repository:

[https://meerah-flight-delay-prediction.streamlit.app/](https://meerah-flight-delay-prediction.streamlit.app/)

To configure an equivalent deployment:

1. Connect the GitHub repository in Streamlit Community Cloud.
2. Select branch **`main`** and entrypoint **`app/app.py`**.
3. Choose a Python runtime compatible with the verified Python 3.11 environment and pinned dependencies.
4. Ensure the tracked model, `data/analysis/`, and `assets/` files are present.
5. Deploy and review Cloud logs if dependency installation or startup fails.

Cloud installs the root `requirements.txt`; `.streamlit/config.toml` supplies the theme. Runtime assets are resolved relative to the project files. No large original datasets need uploading.

Future pushes to the connected `main` branch are picked up by Streamlit Cloud; application/dependency changes trigger its update workflow. Review the public site after the update. Run tests before pushing. **No GitHub Actions workflow is included to enforce that policy**, and Cloud account/branch/runtime settings are managed outside this repository. A README-only change does not alter application behavior.

## 13. Limitations

- Training and analysis cover historical **2024 US flights**, not current worldwide operations. Future patterns may differ.
- No live flight tracking, weather feed, operational updates, or official flight-status integration is present.
- Final training uses a sample, while Analytics summarizes the full eligible data. Dashboard delay rates are not model accuracy estimates.
- Delayed-class precision (27.04%) and recall (37.89%) show substantial false positives and missed delays. The 72.56% accuracy should not be interpreted alone. For context, always predicting the majority class on the recorded test matrix would yield 83.30% accuracy; this is a calculated reference, not an additional trained experiment.
- Estimated probabilities are not explicitly calibrated. Risk bands are application rules, not airline operational guarantees.
- Route existence, carrier-route combinations, and entered distance are not checked against schedules. Unknown encoded categories are ignored by preprocessing, but UI choices remain limited to learned codes.
- Cancelled flights are excluded; neither the predictor nor these aggregates estimate cancellation risk.
- Analytics describes associations, not causes. Airlines and airports have different traffic mixes, and sparse comparisons require caution.
- The compact table cannot recover individual records, exact-date filters, medians, or full delay distributions. Local scheduled hours do not represent a shared timezone.
- Upstream dataset provenance/licensing and fully automated notebook reproduction remain documentation gaps in the source material.

## 14. Troubleshooting

| Problem | Checks and remedy |
|---|---|
| `ModuleNotFoundError` or missing Streamlit command | Use the intended virtual environment; run `python -m pip install -r requirements.txt`, then `python -m streamlit ...`. Check `python -m pip check`. |
| Model cannot be loaded | Restore `models/flight_delay_rf_model_updated.pkl` from Git. Do not substitute the ignored older model. Use the pinned scikit-learn/joblib environment; inspect the exception or Cloud logs. Only load trusted joblib artifacts. |
| Analytics cannot be loaded | Restore both committed `data/analysis/` files and install PyArrow. Rebuild only if the verified full cleaned CSV is available; the original CSV is not needed for normal startup. |
| `app/app.py`, requirements, or test paths not found | Change to the repository root before running the documented commands. Notebook working directories have separate requirements described in section 11. |
| Preparation script reports invalid data | Check required columns, eligibility, target, date/weekday agreement, and HHMM values. Do not bypass validation or substitute a sample for the full cleaned source. |
| Notebook raises `NameError: file_path` | Define the ZIP path and correct the kernel working directory before execution; see section 11. |
| Notebook cannot load the older model | The ignored `models/flight_delay_rf_model.pkl` is created by earlier experiment cells; a fresh clone does not contain it. Normal application startup uses the updated tracked model instead. |
| No flights match filters | Select at least one value in every dimension, broaden the route/hour filters, or use Reset filters. A higher minimum-flight threshold can also hide rankings/heatmap cells. |
| Prediction rejects the route | Origin and destination must differ. Enter distance in miles; the app does not determine route distance automatically. |
| Port is already in use | Stop another server or run `python -m streamlit run app/app.py --server.port 8502`; use the URL printed in the terminal. |
| Browser does not open automatically | Open the printed local address manually. In hosted environments, inspect the public deployment and its logs. |
| Chart rendering/browser exceptions | Confirm the pinned Streamlit/Altair versions, restart the server, and refresh the browser. Existing charts use explicit object-form padding for Streamlit chart compatibility; preserve it when editing specifications. Run the browser regressions after chart changes. |
| Playwright module/browser executable missing | Install the separate test tooling and browser binaries, and set `NODE_PATH` / `PLAYWRIGHT_BROWSERS_PATH` as shown in section 11. |
| Missing airport names or hero image | Restore the bundled `assets/` files. Airport names fall back to codes; the photograph is local and does not need a remote image service. |
