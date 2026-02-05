"""
ROYALEY - API Routes Package
Enterprise-grade API endpoint modules

Complete routing infrastructure with 12 route modules:
- Authentication (auth)
- Predictions (predictions)
- Games (games)
- Odds (odds)
- Betting & Bankroll (betting)
- Player Props (player_props)
- ML Models (models)
- Backtesting (backtest)
- Administration (admin)
- Health Checks (health)
- Analytics (analytics)
- Monitoring (monitoring)
"""

from fastapi import APIRouter

# Import all route modules
from app.api.routes import auth
from app.api.routes import predictions
from app.api.routes import games
from app.api.routes import odds
from app.api.routes import betting
from app.api.routes import player_props
from app.api.routes import models
from app.api.routes import backtest
from app.api.routes import admin
from app.api.routes import health
from app.api.routes import analytics
from app.api.routes import monitoring
from app.api.routes import archive

# Create main API router
api_router = APIRouter()

# Include all route modules with prefixes and tags
api_router.include_router(
    auth.router,
    prefix="/auth",
    tags=["Authentication"]
)

api_router.include_router(
    predictions.router,
    prefix="/predictions",
    tags=["Predictions"]
)

api_router.include_router(
    games.router,
    prefix="/games",
    tags=["Games"]
)

api_router.include_router(
    odds.router,
    prefix="/odds",
    tags=["Odds"]
)

api_router.include_router(
    betting.router,
    prefix="/betting",
    tags=["Betting & Bankroll"]
)

api_router.include_router(
    player_props.router,
    prefix="/player-props",
    tags=["Player Props"]
)

api_router.include_router(
    models.router,
    prefix="/models",
    tags=["ML Models"]
)

api_router.include_router(
    backtest.router,
    prefix="/backtest",
    tags=["Backtesting"]
)

api_router.include_router(
    admin.router,
    prefix="/admin",
    tags=["Administration"]
)

api_router.include_router(
    health.router,
    prefix="/health",
    tags=["Health & Monitoring"]
)

api_router.include_router(
    analytics.router,
    prefix="/analytics",
    tags=["Analytics"]
)

api_router.include_router(
    monitoring.router,
    prefix="/monitoring",
    tags=["System Monitoring"]
)

api_router.include_router(
    archive.router,
    prefix="/archive",
    tags=["Raw Data Archive"]
)

# Export individual routers for direct imports
auth_router = auth.router
predictions_router = predictions.router
games_router = games.router
odds_router = odds.router
betting_router = betting.router
player_props_router = player_props.router
models_router = models.router
backtest_router = backtest.router
admin_router = admin.router
health_router = health.router
analytics_router = analytics.router
monitoring_router = monitoring.router
archive_router = archive.router

__all__ = [
    # Main router
    "api_router",
    
    # Individual routers
    "auth_router",
    "predictions_router",
    "games_router",
    "odds_router",
    "betting_router",
    "player_props_router",
    "models_router",
    "backtest_router",
    "admin_router",
    "health_router",
    "analytics_router",
    "monitoring_router",
    "archive_router",
    
    # Modules
    "auth",
    "predictions",
    "games",
    "odds",
    "betting",
    "player_props",
    "models",
    "backtest",
    "admin",
    "health",
    "analytics",
    "monitoring",
    "archive",
]