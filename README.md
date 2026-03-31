# NBA Game Predictor

NBA game prediction system that combines a machine learning model with a multi-factor weighted scoring engine. Consumes real-time data from the NBA API, ESPN, and optionally Vegas odds.

---

## Features

- **ML Model**: HistGradientBoostingClassifier trained on 4 seasons (2021-22 to 2024-25)
- **Multi-factor scoring**: 13 weighted factors as fallback or standalone mode
- **Real-time data**: recent form, advanced stats, injuries (ESPN), fatigue and travel factor
- **CLI interface** (`main.py`) and **graphical interface** (`GUI.py` with Tkinter)
- **PDF export** with per-game statistics (requires `reportlab`)
- **Vegas odds integration** via The Odds API (optional)

---

## Structure

```
NBA-Predictor/
├── main.py                # Interactive CLI
├── GUI.py                 # Graphical interface with Tkinter
├── data/
│   ├── dataset.csv        # Historical dataset (generated)
│   └── model.pkl          # Trained model (generated)
└── Servicios/
    ├── api_utils.py       # Retry decorator with exponential backoff
    ├── GetSchedule.py     # Next game between two teams (ScoreboardV2)
    ├── GetTier1.py        # Recent form, H2H, home/away splits
    ├── GetTier2.py        # Advanced stats, shooting, defense, clutch
    ├── GetTier3.py        # Injuries (ESPN), fatigue, minutes per player
    ├── Predict.py         # Inference with model.pkl + injury adjustment
    ├── TrainModel.py      # Training and evaluation pipeline
    ├── BuildDataset.py    # Historical dataset construction
    ├── team_locations.py  # Coordinates and time zones for all 30 teams
    └── vegas_odds.py      # Integration with The Odds API
```

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Main dependencies

| Package | Usage |
|---|---|
| `nba_api` | Official NBA statistics |
| `pandas` | Data manipulation |
| `scikit-learn` | ML model and evaluation |
| `joblib` | Model serialization |
| `requests` | ESPN API and Vegas odds |
| `reportlab` | PDF export (optional) |

---

## Quick start

### CLI

```bash
python main.py
```

It will prompt for the two team names and display the prediction with a factor breakdown.

### GUI

```bash
python GUI.py
```

Loads the day's games and runs predictions in parallel using threading.

---

## Training the model

The model is trained from scratch using historical data from the NBA API.

```bash
# 1. Build the dataset (may take several minutes due to rate limits)
python Servicios/BuildDataset.py

# 2. Train and save the model
python Servicios/TrainModel.py
```

The dataset includes 4 seasons with 10- and 5-game rolling features, season stats in expanding mode (no data leakage), schedule strength, travel factor, and head-to-head ratio.

---

## Environment variables

Create a `.env` file in the project root:

```
ODDS_API_KEY=your_api_key_here
```

The key is optional. Without it the system works normally, without displaying Vegas lines. It can be obtained at [the-odds-api.com](https://the-odds-api.com) (free tier: 500 requests/month).

---

## Pipeline architecture

```
nba_api / ESPN / Odds API
 │
 ├── GetTier1 → recent form, H2H, splits, travel
 ├── GetTier2 → advanced stats, shooting, defense
 └── GetTier3 → injuries, fatigue, minutes per player
 │
 ├── Predict.py → ML inference + injury adjustment
 └── calculate_prediction() → weighted scoring (fallback)
```

### Model features (11)

| Feature | Type |
|---|---|
| `diff_win_rate_10j` | Win rate differential (10 games) |
| `diff_pt_diff_10j` | Point differential |
| `diff_pts_allowed_10j` | Points allowed differential |
| `diff_season_def_rtg` | Season Defensive Rating differential |
| `diff_schedule_strength` | Schedule strength differential |
| `away_rest_days` | Away team rest days |
| `away_fatigue_7d` | Away team games in last 7 days |
| `home_b2b` | Is the home team playing back-to-back? |
| `home_rest_days` | Home team rest days |
| `travel_factor` | Travel fatigue factor (distance + time zone) |
| `h2h_ratio` | Head-to-head win ratio |

---

## Notes

- NBA API (stats.nba.com) rate limits require ~0.6s delays between calls. A full prediction takes approximately 60 seconds.
- The model is trained with a temporal split: seasons prior to 2024-25 as train and 2024-25 as test.
- The injury adjustment is applied in log-odds space over the ML model probability to maintain the [0, 1] range.
