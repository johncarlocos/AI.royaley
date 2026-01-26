# ROYALEY ML Training Pipeline Update

## Overview

This update implements the complete ML training pipeline that connects your data collectors to model training and predictions.

## What's Included

### 1. Training Service (`app/services/ml/training_service.py`)
The main orchestrator that:
- Loads and prepares training data from database
- Runs walk-forward validation
- Trains models using H2O/Sklearn/AutoGluon
- Calibrates probabilities
- Saves models to database
- Tracks training runs and performance

### 2. Feature Generator (`scripts/generate_features.py`)
Transforms collected data into ML-ready features:
- ELO ratings (historical and current)
- Team statistics (win%, PPG, point differential)
- Head-to-head history
- Momentum/form (last 5/10 games, streak, rest days)
- Weather impact (outdoor sports)
- Injury impact
- Odds features (opening line, movement)

### 3. Training CLI (`scripts/train_models.py`)
Command-line interface for model training:
```bash
# Train single model
python scripts/train_models.py --sport NFL --bet-type spread --framework h2o

# Train all bet types for a sport
python scripts/train_models.py --sport NBA --all-bet-types

# Train all models
python scripts/train_models.py --all --framework sklearn

# Use mock trainers (testing)
python scripts/train_models.py --sport NFL --bet-type spread --mock
```

### 4. Prediction CLI (`scripts/predict.py`)
Command-line interface for generating predictions:
```bash
# Today's predictions
python scripts/predict.py --sport NFL

# Specific date
python scripts/predict.py --sport NBA --date 2025-01-15

# All sports
python scripts/predict.py --all

# Filter by edge
python scripts/predict.py --sport NFL --min-edge 0.05 --tier A
```

### 5. Updated API Routes (`app/api/routes/models.py`)
- `/models/train` endpoint now actually triggers training
- Uses TrainingService in background task
- Updates training run status automatically

### 6. Updated Config (`app/core/config.py`)
- Added `FEATURES_PATH` and `DATASETS_PATH` settings

### 7. Updated ML Init (`app/services/ml/__init__.py`)
- Exports TrainingService, TrainingResult, get_training_service, train_model_task

## Installation

1. Extract the ZIP to your project root:
```bash
unzip -o ml_training_update.zip
```

2. Rebuild Docker:
```bash
docker compose up -d --build api
```

## Usage Flow

### Step 1: Collect Data (Already Done)
You've already implemented 7 collectors:
- ESPN, OddsAPI, Pinnacle, Tennis, Weather, SportsDB, nflfastR

### Step 2: Generate Features
```bash
# Generate features for NFL
docker exec royaley_api python scripts/generate_features.py --sport NFL

# Generate for all sports
docker exec royaley_api python scripts/generate_features.py --all
```

### Step 3: Train Models
```bash
# Train NFL spread model with H2O
docker exec royaley_api python scripts/train_models.py --sport NFL --bet-type spread

# Train all NFL bet types
docker exec royaley_api python scripts/train_models.py --sport NFL --all-bet-types

# Train with sklearn (faster for testing)
docker exec royaley_api python scripts/train_models.py --sport NFL --bet-type spread --framework sklearn
```

### Step 4: Generate Predictions
```bash
# Get today's NFL predictions
docker exec royaley_api python scripts/predict.py --sport NFL

# Get predictions for all sports
docker exec royaley_api python scripts/predict.py --all
```

## API Endpoints

### Train Model
```bash
POST /api/v1/models/train
{
    "sport_code": "NFL",
    "bet_type": "spread",
    "framework": "h2o",
    "max_runtime_seconds": 3600
}
```

### Get Training Status
```bash
GET /api/v1/models/training/{run_id}
```

### List Models
```bash
GET /api/v1/models?sport_code=NFL&bet_type=spread
```

### Promote Model to Production
```bash
POST /api/v1/models/{model_id}/promote
```

## Data Flow

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Collectors │ -> │   Database  │ -> │  Features   │ -> │   Models    │
│  (7 total)  │    │  (45 tables)│    │  Generator  │    │  Training   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
       │                  │                  │                  │
       v                  v                  v                  v
   ESPN, Odds,        Games, Teams,     ELO, Stats,       H2O, Sklearn,
   Pinnacle...        Odds, Weather     H2H, Weather      AutoGluon
                                                               │
                                                               v
                                                    ┌─────────────────┐
                                                    │   Predictions   │
                                                    │   Signal Tiers  │
                                                    │   Kelly Sizing  │
                                                    └─────────────────┘
```

## Features Computed

| Feature Category | Features |
|------------------|----------|
| **ELO** | home_elo, away_elo, elo_diff, home_elo_expected |
| **Team Stats** | win_pct, games_played, ppg, ppg_allowed, point_diff |
| **Momentum** | form_last5, form_last10, streak, rest_days |
| **H2H** | h2h_games, h2h_home_wins, avg_total, avg_margin |
| **Odds** | opening_spread, current_spread, spread_movement, ML |
| **Weather** | temperature, wind_speed, precipitation_prob, weather_impact |
| **Injuries** | home_injury_count, away_injury_count, injury_impact |

## Signal Tiers

| Tier | Edge | Description |
|------|------|-------------|
| A | ≥10% | Strong value, max bet |
| B | ≥5% | Good value, standard bet |
| C | ≥2% | Marginal value, small bet |
| D | <2% | No bet |

## Next Steps

1. **Import Historical Data**: Run collectors for historical data
2. **Generate Features**: Run feature generator for all games
3. **Train Models**: Train models for each sport/bet type
4. **Set Production Models**: Promote best models to production
5. **Automate**: Set up cron jobs for daily updates

## Cron Jobs (Suggested)

```bash
# Daily data collection (4 AM)
0 4 * * * docker exec royaley_api python scripts/master_import.py --source espn,odds_api,pinnacle

# Weekly model retraining (Sunday 2 AM)
0 2 * * 0 docker exec royaley_api python scripts/train_models.py --all --framework h2o

# Daily predictions (6 AM)
0 6 * * * docker exec royaley_api python scripts/predict.py --all --output /data/predictions/$(date +%Y-%m-%d).json
```

## Troubleshooting

### No games found for training
```bash
# Check game count
docker exec royaley_api python -c "
import asyncio
from app.core.database import db_manager
from app.models import Game, Sport
from sqlalchemy import select, func

async def check():
    await db_manager.initialize()
    async with db_manager.session() as session:
        result = await session.execute(
            select(Sport.code, func.count(Game.id))
            .join(Game, Sport.id == Game.sport_id)
            .group_by(Sport.code)
        )
        for code, count in result:
            print(f'{code}: {count} games')

asyncio.run(check())
"
```

### H2O not starting
```bash
# Check H2O installation
docker exec royaley_api pip list | grep h2o

# Use sklearn instead (faster, no Java required)
python scripts/train_models.py --sport NFL --bet-type spread --framework sklearn
```

### Training taking too long
```bash
# Reduce max runtime
python scripts/train_models.py --sport NFL --bet-type spread --max-runtime 600

# Use mock trainers for testing
python scripts/train_models.py --sport NFL --bet-type spread --mock
```
