# Strawberry Creek Monitoring Group Anomaly Detection System

<p align="center">
  <img src="assets/SCMGlogo.jpg" width="400">
</p>

# StrawberryWatch

This is an unsupervised anomaly detection for the Strawberry Creek monitoring network. The system learns the creek's normal behavior from sensor data and flags deviations that look like spills or contamination events, without ever being trained on labeled anomalies. It treats the creek as a connected graph of sensor sites and combines a graph neural network with an Long Short Term Memory architecture to reason about both where a sensor sits in the flow and how its readings change over time.

This repository is the research and development counterpart to the production monitoring platform. It is where models are built, tested against historical events, and validated before anything is trusted for live alerting.

## What it does

The creek is monitored at eleven locations: UC botanical gardens, Women's Faculty Club (south fork 0), Stephens Hall (south fork 1), Downstream of Sather Gate (south fork 2), Weil Hall (south fork 3), Kingmann Hall Garden, University House, Giannini Hall (north fork 0), Wickson Footbridge (north fork 1, also sometimes labeled as scnf010), and Codornices Creek. Five of them are currently included in the flow model and form two paths that converge at Oxford (South Fork 1 to South Fork 2 to Oxford Street and North Fork 0 to North Fork 1, to Oxford Street). The eleventh site, Codornices, is a separate watershed monitored as a standalone point and deliberately left out of the flow graph. The system pulls sensor readings, merges in weather, learns what normal looks like across the whole network, and then scores new data by how badly the model fails to predict it. A large, sustained prediction error on conductivity that shows up across connected sites is the signature of a real event.

The model is unsupervised. It is trained only to predict the next reading from recent history. Anything it cannot predict well is, by definition, something it has not seen before, which is what an anomaly is.

## Repository layout

```
SCMG_AnDeSys/
  config/
    config.py              loads settings, defines locations and graph nodes
    settings.yaml          model architecture and training hyperparameters
  src/
    ingest/
      api_client.py        pulls sensor data from the public REST API
      sql_client.py        pulls richer data from the production database
      weather_client.py    near-term weather from NWS
      historical_weather_client.py   long-range weather from Open-Meteo
      data_loader.py       orchestrates loading, weather merge, rolling cache
    preprocessing/
      data_processor.py    missing-data handling, normalization, sequencing
    models/
      Dusk_Crayfish.py     the GCN plus LSTM model (the validated model)
      Flame_Skimmer.py     MC Dropout uncertainty model (work in progress)
      Water_Strider.py     Transformer temporal model (work in progress)
    training/
      trainer.py           training loop and threshold computation
    anomalies/
      anomaly_detector.py  scoring and rain-aware detection
    utils/
      graph_utils.py       builds the creek flow graph
      visualizations.py    dashboards and plots
      notifier.py          alert delivery
  tests/
    test_anomaly_detection.py   validation against labeled historical events
    conftest.py                 shared test fixtures
    test_models/                diagnostic scripts
    test_ingest/                API and client diagnostic scripts
  sandbox/
    simple_crayfish.py     three-feature pipeline, no weather, for experiments
    backfill_rain.py       one-time tool to add rain to labeled test CSVs
  scripts/
    clear_cache.py         utility to wipe the local data cache
  notebooks/
    Pulse.ipynb            exploratory analysis notebook
  assets/
    SCMGlogo.jpg
    SCMGBacklogo.png
  data/
    anomalies/             labeled anomaly windows
    normal/                labeled normal baselines
    raw_data/              per-site historical exports
    processed_data/        rolling cache (gitignored)
    rain_cache/            cached weather fetches
    train/                 train split staging (gitignored)
    test/                  test split staging (gitignored)
  models/                  saved model weights and metadata
  main.py                  entry point for training and detection
  run_live.py              continuous monitoring loop (runs inference every 15 minutes)
  .env.example             template for environment variables
  requirements.txt
```

## Setup

Clone the repository and create a virtual environment using Python 3.12, then install dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The system needs environment variables for data and weather access. Copy `.env.example` to `.env` in the repository root and fill in what applies to your setup.

```
# Public API (the token field is optional; the API currently requires no auth)
SCMG_API_TOKEN=
SCMG_API_BASE_URL=https://www.strawberrycreek.org/api/creek-data/

# NWS weather station -- no API key required, but NWS requires a contact email
# in the User-Agent string
NWS_STATION_ID=LBNL1
NWS_USER_AGENT=SCMG-AnDeSys/1.0 (your.email@example.com)
USE_NWS_WEATHER=true

# MySQL -- only needed with --data-source sql
MYSQL_HOST=
MYSQL_DATABASE_USER=
MYSQL_DATABASE_PASSWORD=
MYSQL_DATABASE_NAME=
MYSQL_PORT=3306

# Email alerts -- only needed if you want spill notifications
ALERT_EMAIL_SENDER=your_email@gmail.com
ALERT_EMAIL_PASSWORD=your_app_password
ALERT_EMAIL_RECEIVER=who_to_notify@gmail.com
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587

# How many days of data to keep in the rolling local cache
ROLLING_WINDOW_DAYS=90
```

The MySQL variables are only needed if you run with `--data-source sql`. Weather from NWS and Open-Meteo needs no key, but the NWS user-agent string must include a contact email per NWS requirements.

One network note. The Open-Meteo historical archive must be reachable for long-window weather fetches. If your machine restricts outbound traffic, the host `archive-api.open-meteo.com` needs to be allowed, or the pipeline will fall back to running without weather features.

## How to use it

The main entry point is `main.py`. It has three modes.

**Train** builds a fresh model on thirty days of data, computes a detection threshold from a validation split, and saves the weights and metadata.

```bash
python main.py --mode train --data-source api
```

**Inference** loads the saved model, pulls a short recent window (two days), and reports any anomalies. It falls back to training if no saved model exists.

```bash
python main.py --mode inference --data-source api
```

**Update** retrains on a fresh thirty-day window while reusing the existing setup, for keeping the model current as new data arrives. This is the default mode when `--mode` is not specified.

```bash
python main.py --mode update --data-source api
```

The `--data-source` flag chooses between the public API, which provides the three core sensor features, and the production SQL database, which provides more. The `--model` flag selects the architecture, defaulting to the validated one.

```bash
python main.py --mode train --data-source sql --model dusk_crayfish
```

Adding `--visualize` after any run produces a static dashboard and an interactive plot of the scores, thresholds, and flagged events.

To start the continuous monitoring loop, run `run_live.py` directly. It blocks indefinitely, calling `main.py --mode inference` via subprocess every 15 minutes, and exits cleanly on Ctrl-C.

```bash
python run_live.py
```

To validate the model against the labeled historical events, run the test suite. The `-s` flag shows the per-case diagnostic output, which is worth reading because each case prints its error curve and how many timesteps crossed the threshold.

```bash
pytest tests/test_anomaly_detection.py -v -s
```

The sandbox holds two standalone tools. `simple_crayfish.py` runs the whole pipeline on only the three always-present sensor features with no weather, useful for quick experiments or when weather is unavailable. `backfill_rain.py` is a one-time utility that fetches historical rain and writes it into the labeled test CSVs so the rain-aware threshold can be exercised in testing.

```bash
python sandbox/simple_crayfish.py --raw-dir data/raw_data --days 30 --mode train
python sandbox/backfill_rain.py --dirs data/anomalies data/normal
```

## How each part works

**Loading and the rolling cache.** `data_loader.py` pulls a window of sensor data from the chosen source, renames the raw sensor columns to internal names, merges in weather, and writes the result to a rolling cache on disk. The cache is not just a speed trick. New fetches are merged with what is already there, deduplicated on time and location, and trimmed to a rolling window of recent days, so the cache becomes a usable working set rather than getting overwritten each run. When data comes from the API tier it carries three sensor features: conductivity, depth, and temperature. The SQL tier can carry more, and any extra columns flow through automatically as model features without code changes, because feature selection picks up every numeric column that is not on an exclude list.

**Weather merge.** Weather comes from one of two sources depending on the window. For windows up to about five days, near-term observations come from the NWS station. For longer windows, the historical archive from Open-Meteo is used. Both return the same column names so nothing downstream needs to know which ran. Rainfall gets special handling. The archive reports rain as an hourly total, but the creek samples every fifteen minutes, so each hourly total is divided across the four sub-hourly rows it covers. This keeps cumulative rainfall correct when it is summed over a window, which matters because the rain-aware detection logic reads exactly that windowed sum.

**Missing data.** `data_processor.py` handles the reality that sensors drop out. It sorts every reading into one of three classes. Permanently absent means a sensor at a site has no data at all in the window. Transiently absent means it has data sometimes but not at this timestep. Present means a real reading. Short gaps under about a day are filled by interpolating over time. Longer gaps are left missing. Everything still missing is filled with zero, but only after normalization, so zero means the neutral average rather than a literal zero that would skew the model.

**Normalization and sequencing.** The data is z-score normalized using statistics fit only on complete rows, then reshaped into a three-dimensional array of time by node by feature. A sliding window cuts this into sequences of twenty-four timesteps, six hours of history, each paired with the single next timestep as the prediction target. Only windows where every timestep is valid become training examples.

**The graph.** `graph_utils.py` builds the creek as a directed graph from the locations and edges defined in config. Config currently defines five nodes: `north_fork_0`, `footbridge`, `south_fork_1`, `south_fork_2`, and `oxford`. The `footbridge` node is the Wickson Footbridge, the north fork 1 site, kept under its common name in the config. The edges follow water flow downhill. On the north path, `north_fork_0` flows into `footbridge`, which flows into `oxford`. On the south path, `south_fork_1` flows into `south_fork_2`, which flows into `oxford`. Oxford is the confluence of both paths. Codornices is registered as a physical monitoring site in the field but has no node in this graph, because it is a separate watershed and connecting it would create spatial relationships that do not physically exist.

**The model.** `Dusk_Crayfish.py` defines `DuskCrayfish`, which reads its whole architecture from config. For each timestep in the window, a graph convolution lets each site's representation absorb information from its upstream and downstream neighbors, then the sites are averaged into a single creek-state summary for that moment. The twenty-four summaries form a sequence that an LSTM reads to capture the recent temporal trend. That trend is expanded back to every node and a linear layer predicts the next timestep for every feature at every site. The graph handles space, the LSTM handles time, and the prediction uses both.

**Training.** `trainer.py` runs the training loop, minimizing mean squared error between predicted and actual next timesteps. On CPU it uses plain precision because mixed precision breaks the LSTM there. After training it computes the detection threshold as a high percentile of the validation conductivity errors, so the definition of anomalous is fixed at training time rather than recomputed on whatever is seen later. The feature list the model trained on is saved with the weights, which lets later runs reshape incoming data to match the model rather than rebuilding the model to match the data.

**Detection.** `anomaly_detector.py` scores each timestep and applies the threshold. Scoring follows a model-all, alert-one approach. The model predicts every feature, but only conductivity error counts toward the anomaly score, because conductivity is what actually responds to spills while other features add noise. The conductivity error is averaged across sites, so a deviation seen at several connected sensors scores higher than a single-site blip. On top of the base threshold, a rain-aware adjustment raises the bar during wet periods. For each timestep it sums rain over the preceding twelve hours, and if that exceeds a small amount it doubles the threshold there, so ordinary rain-driven conductivity changes do not trigger false alarms.

**The other models.** `Flame_Skimmer.py` and `Water_Strider.py` are works in progress and not yet wired into the model registry, so they cannot yet be selected with the `--model` flag. `Flame_Skimmer` uses the same spatial backbone as `DuskCrayfish` but adds Monte Carlo Dropout for uncertainty estimation: at inference time, dropout stays active and predictions are sampled thirty times, producing a mean and a standard deviation. The anomaly detector can then ask how far an observation falls from the predicted distribution rather than just from a point prediction. `Water_Strider` replaces the LSTM with a Transformer encoder and sinusoidal positional encoding. It is designed for scenarios with months to years of training data, where Transformers can exploit longer-range temporal dependencies that an LSTM would miss. Both are intended to slot into the registry in `main.py` once finished, selectable with the `--model` flag exactly like the current model, with nothing else in the pipeline changing.

## Testing and validation

The model is validated against a catalog of real documented events in the tests directory, including south fork spills, overnight south fork events, foam events, a fire hydrant spill, a botanical garden actuator malfunction, sprinkler events, and several rainfall events that should not be flagged. Each test runs a labeled window through the model and checks whether the conductivity error crosses the trained threshold, using the same rain-aware logic the live pipeline uses. Cases too short to build valid sequences are skipped rather than failed. The suite is the main guard against a change quietly breaking detection.

## Known limitations

The validated model covers five nodes on the main flow path and is being extended across the full network. It has no seasonal awareness yet, so data from an out-of-season period reads as mildly anomalous everywhere. By design it only scores on conductivity, so anomalies confined to other features are not flagged. And the rain adjustment only raises the threshold while rain is recent, so a delayed first-flush conductivity pulse arriving after the rain stops can still be flagged, which may be correct behavior depending on what you want the system to catch.

---

<p align="center">
  <img src="assets/SCMGBacklogo.png" width="400">
</p>
