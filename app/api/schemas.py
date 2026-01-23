"""
LOYALEY - API Schemas
Phase 1: Pydantic Request/Response Models
"""

from datetime import datetime, date
from decimal import Decimal
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field, ConfigDict


# =============================================================================
# BASE SCHEMAS
# =============================================================================

class BaseSchema(BaseModel):
    """Base schema with common configuration."""
    model_config = ConfigDict(from_attributes=True)


class PaginatedResponse(BaseModel):
    """Paginated response wrapper."""
    items: List[Any]
    total: int
    page: int
    page_size: int
    pages: int


class MessageResponse(BaseModel):
    """Simple message response."""
    message: str
    success: bool = True


# =============================================================================
# AUTH SCHEMAS
# =============================================================================

class UserCreate(BaseModel):
    """User registration request."""
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=100)
    first_name: Optional[str] = Field(None, max_length=100)
    last_name: Optional[str] = Field(None, max_length=100)


class UserLogin(BaseModel):
    """User login request."""
    email: EmailStr
    password: str


class UserResponse(BaseSchema):
    """User response."""
    id: UUID
    email: str
    role: str
    is_active: bool
    is_verified: bool
    first_name: Optional[str]
    last_name: Optional[str]
    created_at: datetime


class TokenResponse(BaseModel):
    """JWT token response."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class TokenRefresh(BaseModel):
    """Token refresh request."""
    refresh_token: str


class TwoFactorSetup(BaseModel):
    """2FA setup response."""
    secret: str
    qr_uri: str


class TwoFactorVerify(BaseModel):
    """2FA verification request."""
    code: str


# =============================================================================
# SPORT SCHEMAS
# =============================================================================

class SportResponse(BaseSchema):
    """Sport response."""
    id: UUID
    code: str
    name: str
    feature_count: int
    is_active: bool
    config: Dict[str, Any]


class TeamResponse(BaseSchema):
    """Team response."""
    id: UUID
    sport_id: UUID
    external_id: str
    name: str
    abbreviation: str
    city: Optional[str]
    conference: Optional[str]
    division: Optional[str]
    elo_rating: float
    is_active: bool


class PlayerResponse(BaseSchema):
    """Player response."""
    id: UUID
    team_id: Optional[UUID]
    external_id: str
    name: str
    position: Optional[str]
    jersey_number: Optional[int]
    is_active: bool


# =============================================================================
# GAME SCHEMAS
# =============================================================================

class GameResponse(BaseSchema):
    """Game response."""
    id: UUID
    sport_id: UUID
    external_id: str
    home_team_id: UUID
    away_team_id: UUID
    game_date: datetime
    status: str
    home_score: Optional[int]
    away_score: Optional[int]
    is_overtime: bool
    broadcast: Optional[str]


class GameDetailResponse(GameResponse):
    """Game detail response with related data."""
    home_team: Optional[TeamResponse]
    away_team: Optional[TeamResponse]
    venue_name: Optional[str]


class GameFilter(BaseModel):
    """Game filter parameters."""
    sport_code: Optional[str] = None
    status: Optional[str] = None
    date_from: Optional[date] = None
    date_to: Optional[date] = None
    team_id: Optional[UUID] = None


# =============================================================================
# ODDS SCHEMAS
# =============================================================================

class OddsResponse(BaseSchema):
    """Odds response."""
    id: UUID
    game_id: UUID
    sportsbook_id: UUID
    market_type: str
    selection: str
    price: int
    line: Optional[float]
    is_current: bool
    recorded_at: datetime


class OddsWithSportsbook(OddsResponse):
    """Odds with sportsbook info."""
    sportsbook_name: str


class BestOddsResponse(BaseModel):
    """Best odds response."""
    game_id: UUID
    market_type: str
    selection: str
    best_price: int
    best_line: Optional[float]
    sportsbook: str
    all_odds: List[OddsWithSportsbook]


class OddsMovementResponse(BaseSchema):
    """Odds movement response."""
    id: UUID
    game_id: UUID
    market_type: str
    old_line: Optional[float]
    new_line: Optional[float]
    old_price: Optional[int]
    new_price: Optional[int]
    movement_size: Optional[float]
    detected_at: datetime


# =============================================================================
# PREDICTION SCHEMAS
# =============================================================================

class PredictionResponse(BaseSchema):
    """Prediction response."""
    id: UUID
    game_id: UUID
    bet_type: str
    predicted_side: str
    probability: float
    calibrated_probability: Optional[float]
    line_at_prediction: Optional[float]
    odds_at_prediction: Optional[int]
    edge: Optional[float]
    signal_tier: str
    kelly_fraction: Optional[float]
    recommended_bet_size: Optional[float]
    created_at: datetime


class PredictionDetailResponse(PredictionResponse):
    """Prediction with game and explanation."""
    game: Optional[GameDetailResponse]
    shap_explanations: List[Dict[str, Any]] = []


class PredictionFilter(BaseModel):
    """Prediction filter parameters."""
    sport_code: Optional[str] = None
    bet_type: Optional[str] = None
    signal_tier: Optional[str] = None
    min_probability: Optional[float] = None
    date_from: Optional[date] = None
    date_to: Optional[date] = None


# =============================================================================
# BETTING SCHEMAS
# =============================================================================

class BankrollResponse(BaseSchema):
    """Bankroll response."""
    id: UUID
    user_id: UUID
    initial_amount: Decimal
    current_amount: Decimal
    peak_amount: Decimal
    low_amount: Decimal
    currency: str
    roi: Optional[float] = None
    total_bets: int = 0


class BankrollCreate(BaseModel):
    """Create bankroll request."""
    initial_amount: Decimal = Field(..., gt=0)
    currency: str = "USD"


class BankrollUpdate(BaseModel):
    """Update bankroll request."""
    amount: Decimal
    transaction_type: str  # deposit, withdrawal


class BetCreate(BaseModel):
    """Create bet request."""
    prediction_id: Optional[UUID] = None
    game_id: UUID
    bet_type: str
    selection: str
    stake: Decimal = Field(..., gt=0)
    odds: int
    line: Optional[float] = None
    sportsbook: Optional[str] = None


class BetResponse(BaseSchema):
    """Bet response."""
    id: UUID
    user_id: UUID
    prediction_id: Optional[UUID]
    game_id: UUID
    bet_type: str
    selection: str
    stake: Decimal
    odds: int
    line: Optional[float]
    sportsbook: Optional[str]
    result: str
    profit_loss: Optional[Decimal]
    placed_at: datetime
    settled_at: Optional[datetime]


class BetSizingRequest(BaseModel):
    """Bet sizing calculation request."""
    probability: float = Field(..., gt=0, lt=1)
    odds: int
    bankroll: Optional[Decimal] = None


class BetSizingResponse(BaseModel):
    """Bet sizing calculation response."""
    full_kelly: float
    fractional_kelly: float
    recommended_stake: Decimal
    edge: float
    implied_probability: float


# =============================================================================
# HEALTH SCHEMAS
# =============================================================================

class HealthCheck(BaseModel):
    """Basic health check response."""
    status: str
    timestamp: datetime
    version: str


class ComponentHealth(BaseModel):
    """Component health status."""
    name: str
    status: str
    response_time_ms: Optional[int] = None
    message: Optional[str] = None


class DetailedHealthCheck(HealthCheck):
    """Detailed health check response."""
    components: List[ComponentHealth]
    database: str
    redis: str
    api_latency_ms: int


# =============================================================================
# DATA COLLECTION SCHEMAS
# =============================================================================

class CollectionStatus(BaseModel):
    """Data collection status."""
    collector: str
    last_run: Optional[datetime]
    records_collected: int
    success: bool
    error: Optional[str] = None


class CollectionTrigger(BaseModel):
    """Trigger data collection."""
    sport_code: Optional[str] = None
    collector: Optional[str] = None  # odds, espn, all


class CollectionResult(BaseModel):
    """Data collection result."""
    success: bool
    collectors_run: List[str]
    total_records: int
    errors: List[str] = []
    duration_seconds: float


# =============================================================================
# VALIDATION SCHEMAS
# =============================================================================

class ValidationCheckResult(BaseModel):
    """Single validation check result."""
    check_name: str
    passed: bool
    level: str
    message: str


class ValidationReport(BaseModel):
    """Validation report."""
    source: str
    total_checks: int
    passed_checks: int
    failed_checks: int
    warnings: int
    quality_score: float
    is_valid: bool
    results: List[ValidationCheckResult]
    timestamp: datetime


# =============================================================================
# ADMIN SCHEMAS
# =============================================================================

class SystemStatus(BaseModel):
    """System status overview."""
    database_connected: bool
    redis_connected: bool
    collectors_active: int
    pending_tasks: int
    last_collection: Optional[datetime]
    predictions_today: int
    system_health: str


class TaskStatus(BaseModel):
    """Scheduled task status."""
    task_name: str
    is_enabled: bool
    status: str
    last_run: Optional[datetime]
    next_run: Optional[datetime]
    last_error: Optional[str]
