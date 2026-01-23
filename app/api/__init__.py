"""
ROYALEY - API Module
FastAPI routes and schemas for the enterprise sports prediction platform.
"""

from app.api.routes import (
    auth,
    predictions,
    games,
    odds,
    betting,
    player_props,
    models,
    backtest,
    health,
    admin,
)

__all__ = [
    "auth",
    "predictions", 
    "games",
    "odds",
    "betting",
    "player_props",
    "models",
    "backtest",
    "health",
    "admin",
]
