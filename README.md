# From Prediction to Profit

## Designing, Deploying, and Benchmarking ML/DL Trading Bots for Live XAUUSD Paper-Trading

This repository contains the official code, notebooks, validation outputs, and deployment work for a Master of Science capstone project on machine-learning and deep-learning trading systems for XAUUSD.

The project evaluates whether predictive performance from LightGBM, LSTM, and CNN-LSTM models translates into economically meaningful trading performance after chronological validation, transaction costs, trading overlays, and risk controls.

> **Research and educational use only.** The system is intended for MetaTrader 5 demo or paper trading. It is not financial advice and is not intended for real-money deployment.

---

## Project objectives

The capstone aims to:

1. Build and validate a reproducible long-horizon XAUUSD dataset.
2. Compare suitable machine-learning and deep-learning model families under a consistent time-series methodology.
3. Evaluate price-only and volume-assisted feature sets.
4. Select a candidate using predictive, economic, and risk-based evidence.
5. test the selected candidate once on a final untouched historical holdout.
6. Deploy the selected system to MetaTrader 5 demo trading with complete decision, risk, order, and performance logging.
7. Prospectively compare the frozen research system with a clearly separated post-holdout deployment revision.

The project does **not** assume that a profitable model must be found. Negative and non-generalising results are retained and reported as research findings.

---

## Data

### Source

- Instrument: XAUUSD
- Provider: Dukascopy
- Price side: BID
- Base frequency: M1
- Timezone: UTC
- Historical period: approximately January 2016 to March 2026
- Raw M1 master size: approximately 3.63 million rows

The M1 data was validated and gap-aware resampled into M5 and M15 datasets.

### Final modelling timeframe

M15 was selected after exploratory feasibility analysis and a lightweight walk-forward screening benchmark.

### Model-ready datasets

The final M15 datasets contain approximately 237,001 rows:

- Price-only: 50 input features
- Volume-assisted: 51 input features
- Additional volume feature: `volume_z20`

Raw absolute OHLCV values are retained for reconstruction and auditing but are not supplied directly as model predictors.

Large raw and processed datasets are intentionally excluded from Git.

---

## Official research methodology

The project follows CRISP-DM and uses a nested chronological walk-forward design.

For each development fold:

1. Historical inner training period
2. One-year inner validation period for model or epoch selection
3. Refit using the complete outer training period
4. Separate one-year outer validation period for trading-overlay selection
5. One-year unseen development test period

Development test years:

- 2022
- 2023
- 2024

Final untouched holdout:

- 1 January 2025 to 31 March 2026

The evaluated model families were:

- LightGBM
- LSTM
- CNN-LSTM

Each family included:

- Vanilla price-only
- Vanilla volume-assisted
- Tuned price-only
- Tuned volume-assisted

---

## Trading evaluation

The prediction model outputs an upward-direction probability. A separate trading overlay maps the probability into long, short, or flat positions.

The development overlay grid evaluated:

- Probability thresholds: 0.51/0.49, 0.52/0.48, 0.53/0.47, and 0.55/0.45
- Minimum holding periods: 1, 2, or 3 M15 bars
- Maximum daily position-change events: 3, 5, or 8

Risk controls included:

- 2% daily loss stop
- 15% total drawdown stop
- Position closure at non-contiguous market gaps
- Stay-flat selection when no viable validation overlay existed

Transaction-cost sensitivity was evaluated at:

- 0.0 bps
- 0.5 bps
- 1.0 bps
- 2.0 bps
- 3.0 bps
- 5.0 bps

The primary development selection cost was 1.0 bps one-way.

---

## Main findings

### Development comparison

The cross-model comparison selected the M15 vanilla volume-assisted CNN-LSTM as the strongest development candidate.

Its mean development returns were approximately:

| One-way cost | Mean return |
|---:|---:|
| 0.0 bps | +10.37% |
| 0.5 bps | +6.57% |
| 1.0 bps | +2.26% |

Development performance was not stable:

- 2022: approximately +19.75% at 1.0 bps
- 2023: approximately -12.98% at 1.0 bps
- 2024: stay flat

### Final untouched holdout

The selected candidate was frozen and evaluated once on the January 2025 to March 2026 holdout.

Final holdout results were approximately:

| Metric | Result |
|---|---:|
| ROC-AUC | 0.5158 |
| Return at 0.0 bps | -8.80% |
| Return at 0.5 bps | -11.35% |
| Return at 1.0 bps | -13.08% |
| Sharpe at 1.0 bps | -1.56 |
| Maximum drawdown | -15.14% |

The final result did not support a robust or deployable historical trading edge. The holdout remains the official final research result and is not replaced by post-hoc model changes.

Key diagnostic findings included:

- Weak directional discrimination and probability calibration
- Target-horizon mismatch between next-bar prediction and multi-bar holding
- Poor short-side contribution
- Excessive turnover and transaction-cost erosion
- Fragile overlay selection
- Regime instability

---

## Deployment plan

Deployment is being developed for MetaTrader 5 demo trading using three controlled stages:

1. Offline replay and feature/prediction parity
2. Live MT5 shadow mode with no submitted orders
3. Parallel demo execution after replay and shadow acceptance criteria pass

Two systems are planned:

### Model A: frozen research benchmark

The exact Notebook 7 CNN-LSTM model, scaler, features, overlay, and risk controls.

### Model B: post-holdout exploratory deployment revision

The same CNN-LSTM and preprocessing artefacts, but with a separately versioned trading overlay intended to reduce weak short exposure, turnover, and cost erosion.

Model B will be evaluated only on new forward paper-trading data and will not replace the official historical holdout result.

Deployment will include:

- Completed M15 bar retrieval
- Exact online feature generation
- Model inference
- Independent strategy and risk state
- MT5 order reconciliation
- SQLite decision and execution audit logs
- Spread and slippage recording
- Restart-safe state
- Lightweight monitoring
- Model A versus Model B forward comparison

---

## Official notebook sequence

| Notebook | Purpose |
|---:|---|
| 01 | M5 versus M15 timeframe feasibility EDA |
| 02 | Lightweight LightGBM timeframe screening |
| 03 | Full M15 LightGBM evaluation |
| 04 | Full M15 LSTM evaluation |
| 05 | Full M15 CNN-LSTM evaluation |
| 06 | Cross-model comparison and candidate selection |
| 07 | Final untouched CNN-LSTM holdout evaluation |

The repository should contain only the final official capstone notebooks and their final outputs.

---

## Repository structure

```text
.
├── data/
│   └── capstone_methodology/
│       ├── docs/                  # Dataset metadata
│       └── reports/               # Data-quality and integrity evidence
├── notebook_outputs/              # Final tables, figures, and manifests
├── notebooks/                     # Official notebooks 01-07
├── scripts/                       # Data acquisition and preparation scripts
├── src/                           # Reusable Python source code
├── .env.example                   # Placeholder environment variables only
├── .gitignore
├── README.md
├── requirements.txt
├── requirements-research.txt
└── requirements-deployment.txt
```

Raw datasets, processed datasets, model binaries, Optuna databases, runtime logs, and credentials are intentionally excluded.

---

## Environment setup

Python 3.11 is recommended.

### Research environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-research.txt
```

The root `requirements.txt` installs the research environment for backward compatibility:

```powershell
pip install -r requirements.txt
```

### Deployment environment

The deployment environment is Windows-specific because the official MetaTrader 5 Python connector communicates with the locally installed MT5 terminal.

```powershell
python -m venv .venv-deployment
.\.venv-deployment\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-deployment.txt
```

The deployment dependency file is intentionally minimal and should be updated only after the final deployment architecture is frozen.

---

## Data pipeline

The official data pipeline is implemented by the following scripts:

```text
scripts/01_download_validate_dukascopy_xauusd_m1_daily.py
scripts/02_aggregate_dukascopy_daily_to_m1_master.py
scripts/06_prepare_m5_m15_relative_datasets_v2.py
scripts/07_verify_m5_m15_relative_pipeline.py
```

Run each script with `--help` where supported and review its configured input and output paths before execution.

Generated data belongs under `data/capstone_methodology/` and is excluded from Git except for metadata, integrity summaries, and documentation.

---

## Environment variables and credentials

Copy `.env.example` to `.env`:

```powershell
Copy-Item .env.example .env
```

Only `.env` should contain real credentials. Never place real values in `.env.example`.

The repository must never contain:

- MT5 login credentials
- Broker passwords
- Personal access tokens
- Service-account keys
- Google Drive credentials
- Private key files
- Runtime databases containing account information

Any credential ever committed to a public repository must be rotated immediately, even if it is later deleted.

---

## Reproducibility notes

Critical serialized-artifact versions are pinned:

- TensorFlow 2.20.0
- Keras 3.13.2
- scikit-learn 1.6.1
- LightGBM 4.6.0
- Optuna 4.9.0

Notebook output folders contain the tables, figures, configurations, and evidence required to reproduce the reported conclusions.

Large model and dataset artefacts are maintained outside Git and should be verified using their recorded hashes.

---

## Limitations

- XAUUSD is non-stationary and regime-dependent.
- Dukascopy historical BID data may differ from a deployment broker feed.
- Tick volume is not equivalent to centralised exchange volume.
- Weak classification improvements may not translate into economic value.
- Transaction costs and turnover materially affect results.
- A short MT5 paper-trading period cannot prove long-term profitability.
- The final holdout result was negative.

---

## Academic and trading disclaimer

This repository is an academic capstone artefact.

It is not financial advice, an investment recommendation, or evidence that the strategy will be profitable. Automated trading carries substantial risk. Any deployment must remain on a demo account unless separately reviewed and authorised outside the scope of this capstone.
