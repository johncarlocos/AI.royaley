"""
ROYALEY - Database Models
Phase 1: Core Data Platform

Complete SQLAlchemy 2.0 models for the sports prediction platform.
"""

from datetime import datetime, date
from decimal import Decimal
from enum import Enum as PyEnum
from typing import List, Optional
from uuid import uuid4

from sqlalchemy import (
    Boolean, CheckConstraint, Column, Date, DateTime, Enum, Float,
    ForeignKey, Index, Integer, Numeric, String, Text, UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import ENUM as PG_ENUM, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.core.database import Base


#
# Base class imported from app.core.database


# =============================================================================
# ENUMS
# =============================================================================

class UserRole(str, PyEnum):
    USER = "user"
    PRO_USER = "pro_user"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"


class GameStatus(str, PyEnum):
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    FINAL = "final"
    POSTPONED = "postponed"
    CANCELLED = "cancelled"


class BetResult(str, PyEnum):
    PENDING = "pending"
    WIN = "win"
    LOSS = "loss"
    PUSH = "push"
    VOID = "void"


class SignalTier(str, PyEnum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"


class AlertSeverity(str, PyEnum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class HealthStatus(str, PyEnum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    DOWN = "down"


class MLFramework(str, PyEnum):
    H2O = "h2o"
    AUTOGLUON = "autogluon"
    SKLEARN = "sklearn"
    META = "meta"


class TaskStatus(str, PyEnum):
    IDLE = "idle"
    RUNNING = "running"
    FAILED = "failed"
    SUCCESS = "success"


# =============================================================================
# USERS & AUTHENTICATION
# =============================================================================

class User(Base):
    """User accounts with authentication and role management."""
    __tablename__ = "users"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[UserRole] = mapped_column(Enum(UserRole, values_callable=lambda obj: [e.value for e in obj]), default=UserRole.USER)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # 2FA fields
    two_factor_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    two_factor_secret: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    
    # Profile
    first_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    last_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now())
    last_login_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    # Relationships
    sessions: Mapped[List["Session"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    api_keys: Mapped[List["APIKey"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    preferences: Mapped[Optional["UserPreference"]] = relationship(back_populates="user", uselist=False)
    bankroll: Mapped[Optional["Bankroll"]] = relationship(back_populates="user", uselist=False)
    bets: Mapped[List["Bet"]] = relationship(back_populates="user")
    audit_logs: Mapped[List["AuditLog"]] = relationship(back_populates="user")


class Session(Base):
    """Active user sessions for token management."""
    __tablename__ = "sessions"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"))
    token_hash: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    ip_address: Mapped[Optional[str]] = mapped_column(String(45), nullable=True)
    user_agent: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    is_revoked: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    
    user: Mapped["User"] = relationship(back_populates="sessions")
    
    __table_args__ = (Index("ix_sessions_token_hash", "token_hash"),)


class APIKey(Base):
    """API keys for programmatic access."""
    __tablename__ = "api_keys"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"))
    key_hash: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    permissions: Mapped[dict] = mapped_column(JSONB, default=dict)
    rate_limit: Mapped[int] = mapped_column(Integer, default=100)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    last_used_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    
    user: Mapped["User"] = relationship(back_populates="api_keys")


class UserPreference(Base):
    """User preferences and settings."""
    __tablename__ = "user_preferences"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), unique=True)
    timezone: Mapped[str] = mapped_column(String(50), default="UTC")
    notification_settings: Mapped[dict] = mapped_column(JSONB, default=dict)
    display_preferences: Mapped[dict] = mapped_column(JSONB, default=dict)
    default_sport: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    odds_format: Mapped[str] = mapped_column(String(20), default="american")
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now())
    
    user: Mapped["User"] = relationship(back_populates="preferences")


class AuditLog(Base):
    """Audit trail for user actions."""
    __tablename__ = "audit_logs"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id: Mapped[Optional[UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    action: Mapped[str] = mapped_column(String(100), nullable=False)
    resource_type: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    resource_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    ip_address: Mapped[Optional[str]] = mapped_column(String(45), nullable=True)
    details: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    
    user: Mapped[Optional["User"]] = relationship(back_populates="audit_logs")
    
    __table_args__ = (Index("ix_audit_logs_created_at", "created_at"),)


# =============================================================================
# SPORTS DATA
# =============================================================================

class Sport(Base):
    """Sports configuration and metadata."""
    __tablename__ = "sports"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    code: Mapped[str] = mapped_column(String(10), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    api_key: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    feature_count: Mapped[int] = mapped_column(Integer, default=70)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    config: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    
    teams: Mapped[List["Team"]] = relationship(back_populates="sport")
    seasons: Mapped[List["Season"]] = relationship(back_populates="sport")
    ml_models: Mapped[List["MLModel"]] = relationship(back_populates="sport")


class Team(Base):
    """Team information with ELO ratings."""
    __tablename__ = "teams"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    sport_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("sports.id"))
    external_id: Mapped[str] = mapped_column(String(100), nullable=False)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    abbreviation: Mapped[str] = mapped_column(String(10), nullable=False)
    city: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    conference: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    division: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    elo_rating: Mapped[float] = mapped_column(Float, default=1500.0)
    logo_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now())
    
    sport: Mapped["Sport"] = relationship(back_populates="teams")
    players: Mapped[List["Player"]] = relationship(back_populates="team")
    home_games: Mapped[List["Game"]] = relationship(back_populates="home_team", foreign_keys="Game.home_team_id")
    away_games: Mapped[List["Game"]] = relationship(back_populates="away_team", foreign_keys="Game.away_team_id")
    stats: Mapped[List["TeamStats"]] = relationship(back_populates="team")
    injuries: Mapped[List["Injury"]] = relationship("Injury", back_populates="team")
    
    __table_args__ = (
        UniqueConstraint("sport_id", "external_id", name="uq_teams_sport_external"),
        Index("ix_teams_abbreviation", "abbreviation"),
    )


class Player(Base):
    """Player information for props betting."""
    __tablename__ = "players"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    team_id: Mapped[Optional[UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("teams.id"), nullable=True)
    external_id: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    position: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    jersey_number: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    birth_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    height: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    weight: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now())
    
    team: Mapped[Optional["Team"]] = relationship(back_populates="players")
    stats: Mapped[List["PlayerStats"]] = relationship(back_populates="player")
    props: Mapped[List["PlayerProp"]] = relationship(back_populates="player")
    injuries: Mapped[List["Injury"]] = relationship("Injury", back_populates="player")


class Venue(Base):
    """Venue/stadium information."""
    __tablename__ = "venues"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    city: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    state: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    country: Mapped[str] = mapped_column(String(100), default="USA")
    timezone: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    is_dome: Mapped[bool] = mapped_column(Boolean, default=False)
    surface: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    capacity: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    latitude: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    longitude: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    
    games: Mapped[List["Game"]] = relationship(back_populates="venue")


class Season(Base):
    """Season configuration by sport."""
    __tablename__ = "seasons"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    sport_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("sports.id"))
    year: Mapped[int] = mapped_column(Integer, nullable=False)
    name: Mapped[str] = mapped_column(String(50), nullable=False)
    start_date: Mapped[date] = mapped_column(Date, nullable=False)
    end_date: Mapped[date] = mapped_column(Date, nullable=False)
    is_current: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    
    sport: Mapped["Sport"] = relationship(back_populates="seasons")
    
    __table_args__ = (UniqueConstraint("sport_id", "year", name="uq_seasons_sport_year"),)


class Game(Base):
    """Game/event records with scores."""
    __tablename__ = "games"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    sport_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("sports.id", ondelete="CASCADE"), nullable=False)
    season_id: Mapped[Optional[UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("seasons.id", ondelete="SET NULL"), nullable=True)
    external_id: Mapped[Optional[str]] = mapped_column(String(100), unique=True, nullable=True)
    home_team_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("teams.id", ondelete="CASCADE"), nullable=False)
    away_team_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("teams.id", ondelete="CASCADE"), nullable=False)
    venue_id: Mapped[Optional[UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("venues.id", ondelete="SET NULL"), nullable=True)
    
    scheduled_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    status: Mapped[GameStatus] = mapped_column(
        Enum(GameStatus, values_callable=lambda obj: [e.value for e in obj]),
        default=GameStatus.SCHEDULED
    )
    
    # Scores
    home_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    away_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    
    # Rotation numbers
    home_rotation: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    away_rotation: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    
    # Live game info
    period: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    clock: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    
    # Weather data (JSONB)
    weather: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    home_team: Mapped["Team"] = relationship(back_populates="home_games", foreign_keys=[home_team_id])
    away_team: Mapped["Team"] = relationship(back_populates="away_games", foreign_keys=[away_team_id])
    venue: Mapped[Optional["Venue"]] = relationship(back_populates="games")
    odds: Mapped[List["Odds"]] = relationship(back_populates="game", cascade="all, delete-orphan")
    predictions: Mapped[List["Prediction"]] = relationship(back_populates="game")
    features: Mapped[Optional["GameFeature"]] = relationship(back_populates="game", uselist=False)
    game_injuries: Mapped[List["GameInjury"]] = relationship("GameInjury", back_populates="game")


class GameFeature(Base):
    """Pre-computed features for ML models."""
    __tablename__ = "game_features"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    game_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("games.id", ondelete="CASCADE"), unique=True)
    features: Mapped[dict] = mapped_column(JSONB, nullable=False)
    feature_version: Mapped[str] = mapped_column(String(20), default="1.0")
    computed_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    
    game: Mapped["Game"] = relationship(back_populates="features")


class TeamStats(Base):
    """Team statistics by season."""
    __tablename__ = "team_stats"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    team_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("teams.id"))
    season_id: Mapped[Optional[UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("seasons.id"), nullable=True)
    stat_type: Mapped[str] = mapped_column(String(50), nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    games_played: Mapped[int] = mapped_column(Integer, default=0)
    computed_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    
    team: Mapped["Team"] = relationship(back_populates="stats")
    
    __table_args__ = (Index("ix_team_stats_team_stat", "team_id", "stat_type"),)


class PlayerStats(Base):
    """Player statistics."""
    __tablename__ = "player_stats"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    player_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("players.id"))
    game_id: Mapped[Optional[UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("games.id"), nullable=True)
    season_id: Mapped[Optional[UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("seasons.id"), nullable=True)
    stat_type: Mapped[str] = mapped_column(String(50), nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    
    player: Mapped["Player"] = relationship(back_populates="stats")


# =============================================================================
# ODDS & MARKETS
# =============================================================================

class Sportsbook(Base):
    """Sportsbook information."""
    __tablename__ = "sportsbooks"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    key: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    is_sharp: Mapped[bool] = mapped_column(Boolean, default=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    priority: Mapped[int] = mapped_column(Integer, default=100)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    
    odds: Mapped[List["Odds"]] = relationship(back_populates="sportsbook")


class Odds(Base):
    """Historical and current odds from sportsbooks."""
    __tablename__ = "odds"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    game_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("games.id", ondelete="CASCADE"), nullable=False)
    sportsbook_id: Mapped[Optional[UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("sportsbooks.id", ondelete="CASCADE"), nullable=True)
    sportsbook_key: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    bet_type: Mapped[str] = mapped_column(String(50), nullable=False)  # spread, moneyline, total
    home_line: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    away_line: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    home_odds: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    away_odds: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    total: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    over_odds: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    under_odds: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    is_opening: Mapped[bool] = mapped_column(Boolean, default=False)
    recorded_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    
    game: Mapped["Game"] = relationship(back_populates="odds")
    sportsbook: Mapped[Optional["Sportsbook"]] = relationship(back_populates="odds")


class OddsMovement(Base):
    """Line movement tracking."""
    __tablename__ = "odds_movements"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    game_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("games.id", ondelete="CASCADE"), nullable=False)
    bet_type: Mapped[str] = mapped_column(String(50), nullable=False)
    previous_line: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    current_line: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    movement: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    is_steam: Mapped[bool] = mapped_column(Boolean, default=False)
    is_reverse: Mapped[bool] = mapped_column(Boolean, default=False)
    detected_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())


class ClosingLine(Base):
    """Closing lines for CLV calculation."""
    __tablename__ = "closing_lines"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    game_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("games.id", ondelete="CASCADE"), nullable=False, unique=True)
    spread_home: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    spread_away: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    total: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    moneyline_home: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    moneyline_away: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    source: Mapped[str] = mapped_column(String(50), default='pinnacle')
    recorded_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())


class ConsensusLine(Base):
    """Market consensus across sportsbooks."""
    __tablename__ = "consensus_lines"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    game_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("games.id", ondelete="CASCADE"), nullable=False)
    bet_type: Mapped[str] = mapped_column(String(50), nullable=False)
    consensus_line: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    public_bet_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    public_money_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sharp_action: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())


# =============================================================================
# PREDICTIONS
# =============================================================================

class Prediction(Base):
    """Model predictions with probabilities."""
    __tablename__ = "predictions"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    game_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("games.id"))
    model_id: Mapped[Optional[UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("ml_models.id"), nullable=True)
    
    bet_type: Mapped[str] = mapped_column(String(20), nullable=False)  # spread, moneyline, total
    predicted_side: Mapped[str] = mapped_column(String(20), nullable=False)  # home, away, over, under
    probability: Mapped[float] = mapped_column(Float, nullable=False)
    calibrated_probability: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    line_at_prediction: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    odds_at_prediction: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    
    edge: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    signal_tier: Mapped[SignalTier] = mapped_column(Enum(SignalTier, values_callable=lambda obj: [e.value for e in obj]), default=SignalTier.D)
    kelly_fraction: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    recommended_bet_size: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Integrity
    prediction_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    
    # Relationships
    game: Mapped["Game"] = relationship(back_populates="predictions")
    model: Mapped[Optional["MLModel"]] = relationship(back_populates="predictions")
    result: Mapped[Optional["PredictionResult"]] = relationship(back_populates="prediction", uselist=False)
    shap_explanations: Mapped[List["ShapExplanation"]] = relationship(back_populates="prediction")
    bets: Mapped[List["Bet"]] = relationship(back_populates="prediction")
    
    __table_args__ = (
        Index("ix_predictions_game_bet", "game_id", "bet_type"),
        Index("ix_predictions_tier", "signal_tier"),
        Index("ix_predictions_created", "created_at"),
    )


class PredictionResult(Base):
    """Graded prediction results."""
    __tablename__ = "prediction_results"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    prediction_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("predictions.id", ondelete="CASCADE"), unique=True)
    actual_result: Mapped[BetResult] = mapped_column(Enum(BetResult, values_callable=lambda obj: [e.value for e in obj]), nullable=False)
    closing_line: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    closing_odds: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    clv: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # Closing Line Value
    profit_loss: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    graded_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    
    prediction: Mapped["Prediction"] = relationship(back_populates="result")


class PlayerProp(Base):
    """Player prop predictions."""
    __tablename__ = "player_props"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    game_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("games.id"))
    player_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("players.id"))
    model_id: Mapped[Optional[UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("ml_models.id"), nullable=True)
    
    prop_type: Mapped[str] = mapped_column(String(50), nullable=False)  # points, rebounds, assists, etc.
    line: Mapped[float] = mapped_column(Float, nullable=False)
    predicted_value: Mapped[float] = mapped_column(Float, nullable=False)
    over_probability: Mapped[float] = mapped_column(Float, nullable=False)
    under_probability: Mapped[float] = mapped_column(Float, nullable=False)
    
    actual_value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    result: Mapped[Optional[BetResult]] = mapped_column(Enum(BetResult, values_callable=lambda obj: [e.value for e in obj]), nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    graded_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    player: Mapped["Player"] = relationship(back_populates="props")
    
    __table_args__ = (Index("ix_player_props_game_player", "game_id", "player_id"),)


class ShapExplanation(Base):
    """SHAP explanations for predictions."""
    __tablename__ = "shap_explanations"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    prediction_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("predictions.id", ondelete="CASCADE"))
    feature_name: Mapped[str] = mapped_column(String(100), nullable=False)
    feature_value: Mapped[float] = mapped_column(Float, nullable=False)
    shap_value: Mapped[float] = mapped_column(Float, nullable=False)
    impact_direction: Mapped[str] = mapped_column(String(10), nullable=False)  # positive, negative
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    
    prediction: Mapped["Prediction"] = relationship(back_populates="shap_explanations")


# =============================================================================
# ML MODELS
# =============================================================================

class MLModel(Base):
    """Trained ML model metadata."""
    __tablename__ = "ml_models"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    sport_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("sports.id"))
    bet_type: Mapped[str] = mapped_column(String(20), nullable=False)
    framework: Mapped[MLFramework] = mapped_column(Enum(MLFramework, values_callable=lambda obj: [e.value for e in obj]), nullable=False)
    version: Mapped[str] = mapped_column(String(20), nullable=False)
    file_path: Mapped[str] = mapped_column(String(500), nullable=False)
    is_production: Mapped[bool] = mapped_column(Boolean, default=False)
    performance_metrics: Mapped[dict] = mapped_column(JSONB, default=dict)
    feature_list: Mapped[Optional[list]] = mapped_column(JSONB, nullable=True)
    hyperparameters: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    training_samples: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    
    sport: Mapped["Sport"] = relationship(back_populates="ml_models")
    predictions: Mapped[List["Prediction"]] = relationship(back_populates="model")
    training_runs: Mapped[List["TrainingRun"]] = relationship(back_populates="model")
    performance_history: Mapped[List["ModelPerformance"]] = relationship(back_populates="model")
    feature_importances: Mapped[List["FeatureImportance"]] = relationship(back_populates="model")
    calibration: Mapped[Optional["CalibrationModel"]] = relationship(back_populates="model", uselist=False)
    
    __table_args__ = (Index("ix_ml_models_sport_bet", "sport_id", "bet_type"),)


class TrainingRun(Base):
    """Model training history."""
    __tablename__ = "training_runs"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    model_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("ml_models.id", ondelete="CASCADE"))
    started_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    status: Mapped[TaskStatus] = mapped_column(Enum(TaskStatus, values_callable=lambda obj: [e.value for e in obj]), default=TaskStatus.RUNNING)
    hyperparameters: Mapped[dict] = mapped_column(JSONB, default=dict)
    validation_metrics: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    training_duration_seconds: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    
    model: Mapped["MLModel"] = relationship(back_populates="training_runs")


class ModelPerformance(Base):
    """Daily model performance tracking."""
    __tablename__ = "model_performance"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    model_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("ml_models.id", ondelete="CASCADE"))
    date: Mapped[date] = mapped_column(Date, nullable=False)
    predictions_count: Mapped[int] = mapped_column(Integer, default=0)
    accuracy: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    auc: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    log_loss: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    brier_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    calibration_error: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    roi: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    clv_avg: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    model: Mapped["MLModel"] = relationship(back_populates="performance_history")
    
    __table_args__ = (UniqueConstraint("model_id", "date", name="uq_model_performance_date"),)


class FeatureImportance(Base):
    """Feature importance from trained models."""
    __tablename__ = "feature_importances"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    model_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("ml_models.id", ondelete="CASCADE"))
    feature_name: Mapped[str] = mapped_column(String(100), nullable=False)
    importance_score: Mapped[float] = mapped_column(Float, nullable=False)
    importance_rank: Mapped[int] = mapped_column(Integer, nullable=False)
    
    model: Mapped["MLModel"] = relationship(back_populates="feature_importances")


class CalibrationModel(Base):
    """Probability calibration models."""
    __tablename__ = "calibration_models"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    model_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("ml_models.id", ondelete="CASCADE"), unique=True)
    calibrator_type: Mapped[str] = mapped_column(String(20), nullable=False)  # isotonic, platt
    calibrator_path: Mapped[str] = mapped_column(String(500), nullable=False)
    ece_before: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ece_after: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    
    model: Mapped["MLModel"] = relationship(back_populates="calibration")


# =============================================================================
# BETTING
# =============================================================================

class Bankroll(Base):
    """User bankroll management."""
    __tablename__ = "bankrolls"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), unique=True)
    initial_amount: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    current_amount: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    peak_amount: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    low_amount: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    currency: Mapped[str] = mapped_column(String(3), default="USD")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now())
    
    user: Mapped["User"] = relationship(back_populates="bankroll")
    transactions: Mapped[List["BankrollTransaction"]] = relationship(back_populates="bankroll")


class Bet(Base):
    """Tracked bets."""
    __tablename__ = "bets"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"))
    prediction_id: Mapped[Optional[UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("predictions.id"), nullable=True)
    game_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("games.id"))
    
    bet_type: Mapped[str] = mapped_column(String(20), nullable=False)
    selection: Mapped[str] = mapped_column(String(20), nullable=False)
    stake: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    odds: Mapped[int] = mapped_column(Integer, nullable=False)
    line: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sportsbook: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    
    result: Mapped[BetResult] = mapped_column(Enum(BetResult, values_callable=lambda obj: [e.value for e in obj]), default=BetResult.PENDING)
    profit_loss: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 2), nullable=True)
    
    placed_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    settled_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    user: Mapped["User"] = relationship(back_populates="bets")
    prediction: Mapped[Optional["Prediction"]] = relationship(back_populates="bets")
    
    __table_args__ = (Index("ix_bets_user_placed", "user_id", "placed_at"),)


class BankrollTransaction(Base):
    """Bankroll transaction history."""
    __tablename__ = "bankroll_transactions"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    bankroll_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("bankrolls.id", ondelete="CASCADE"))
    transaction_type: Mapped[str] = mapped_column(String(20), nullable=False)  # deposit, withdrawal, bet, win
    amount: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    balance_after: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    reference_id: Mapped[Optional[UUID]] = mapped_column(UUID(as_uuid=True), nullable=True)  # bet_id for bet/win
    notes: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    
    bankroll: Mapped["Bankroll"] = relationship(back_populates="transactions")


# =============================================================================
# SYSTEM
# =============================================================================

class SystemSetting(Base):
    """System configuration settings."""
    __tablename__ = "system_settings"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    key: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    value: Mapped[str] = mapped_column(Text, nullable=False)
    value_type: Mapped[str] = mapped_column(String(20), default="string")
    description: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now())


class ScheduledTask(Base):
    """Background task scheduling."""
    __tablename__ = "scheduled_tasks"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    task_name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    cron_expression: Mapped[str] = mapped_column(String(100), nullable=False)
    is_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    last_run_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    next_run_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    status: Mapped[TaskStatus] = mapped_column(Enum(TaskStatus, values_callable=lambda obj: [e.value for e in obj]), default=TaskStatus.IDLE)
    last_error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    config: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())


class Alert(Base):
    """System alerts and notifications."""
    __tablename__ = "alerts"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    alert_type: Mapped[str] = mapped_column(String(50), nullable=False)
    severity: Mapped[AlertSeverity] = mapped_column(Enum(AlertSeverity, values_callable=lambda obj: [e.value for e in obj]), default=AlertSeverity.INFO)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    details: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    is_acknowledged: Mapped[bool] = mapped_column(Boolean, default=False)
    acknowledged_by: Mapped[Optional[UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    acknowledged_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    
    __table_args__ = (Index("ix_alerts_severity_created", "severity", "created_at"),)


class DataQualityCheck(Base):
    """Data quality audit logs."""
    __tablename__ = "data_quality_checks"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    check_type: Mapped[str] = mapped_column(String(50), nullable=False)
    source: Mapped[str] = mapped_column(String(50), nullable=False)
    passed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    records_checked: Mapped[int] = mapped_column(Integer, default=0)
    failed_count: Mapped[int] = mapped_column(Integer, default=0)
    quality_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    details: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    checked_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())


class SystemHealthSnapshot(Base):
    """System health metrics."""
    __tablename__ = "system_health_snapshots"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    component: Mapped[str] = mapped_column(String(50), nullable=False)
    status: Mapped[HealthStatus] = mapped_column(Enum(HealthStatus, values_callable=lambda obj: [e.value for e in obj]), default=HealthStatus.HEALTHY)
    response_time_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    metrics: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    recorded_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    
    __table_args__ = (Index("ix_health_component_time", "component", "recorded_at"),)


class BacktestRun(Base):
    """Backtesting history."""
    __tablename__ = "backtest_runs"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id: Mapped[Optional[UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    sport_code: Mapped[str] = mapped_column(String(10), nullable=False)
    bet_type: Mapped[str] = mapped_column(String(20), nullable=False)
    start_date: Mapped[date] = mapped_column(Date, nullable=False)
    end_date: Mapped[date] = mapped_column(Date, nullable=False)
    initial_bankroll: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    final_bankroll: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    total_bets: Mapped[int] = mapped_column(Integer, default=0)
    wins: Mapped[int] = mapped_column(Integer, default=0)
    losses: Mapped[int] = mapped_column(Integer, default=0)
    pushes: Mapped[int] = mapped_column(Integer, default=0)
    roi: Mapped[float] = mapped_column(Float, nullable=False)
    max_drawdown: Mapped[float] = mapped_column(Float, nullable=False)
    sharpe_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    config: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())


# =============================================================================
# ADDITIONAL TRACKING TABLES
# =============================================================================

class ELOHistory(Base):
    """Historical ELO rating tracking for teams."""
    __tablename__ = "elo_history"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    team_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("teams.id"), nullable=False)
    sport_code: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    game_id: Mapped[Optional[UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("games.id"), nullable=True)
    rating_before: Mapped[float] = mapped_column(Float, nullable=False)
    rating_after: Mapped[float] = mapped_column(Float, nullable=False)
    rating_change: Mapped[float] = mapped_column(Float, nullable=False)
    opponent_rating: Mapped[float] = mapped_column(Float, nullable=False)
    result: Mapped[str] = mapped_column(String(10), nullable=False)  # win/loss/draw
    margin: Mapped[int] = mapped_column(Integer, default=0)
    recorded_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    
    __table_args__ = (
        Index("ix_elo_history_team_date", "team_id", "recorded_at"),
        Index("ix_elo_history_sport_date", "sport_code", "recorded_at"),
    )


class CLVRecord(Base):
    """Closing Line Value tracking records."""
    __tablename__ = "clv_records"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    bet_id: Mapped[Optional[UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("bets.id"), nullable=True)
    prediction_id: Mapped[Optional[UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("predictions.id"), nullable=True)
    sport_code: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    bet_type: Mapped[str] = mapped_column(String(20), nullable=False)
    bet_side: Mapped[str] = mapped_column(String(20), nullable=False)
    bet_line: Mapped[float] = mapped_column(Float, nullable=False)
    closing_line: Mapped[float] = mapped_column(Float, nullable=False)
    clv_cents: Mapped[float] = mapped_column(Float, nullable=False)
    clv_percent: Mapped[float] = mapped_column(Float, nullable=False)
    is_positive: Mapped[bool] = mapped_column(Boolean, nullable=False)
    benchmark_book: Mapped[str] = mapped_column(String(50), default="pinnacle")
    recorded_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    
    __table_args__ = (
        Index("ix_clv_sport_date", "sport_code", "recorded_at"),
        Index("ix_clv_positive", "is_positive", "recorded_at"),
    )


class LineMovementAlert(Base):
    """Significant line movement alerts."""
    __tablename__ = "line_movement_alerts"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    game_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("games.id"), nullable=False)
    sport_code: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    bet_type: Mapped[str] = mapped_column(String(20), nullable=False)
    opening_line: Mapped[float] = mapped_column(Float, nullable=False)
    current_line: Mapped[float] = mapped_column(Float, nullable=False)
    movement: Mapped[float] = mapped_column(Float, nullable=False)
    movement_percent: Mapped[float] = mapped_column(Float, nullable=False)
    is_steam_move: Mapped[bool] = mapped_column(Boolean, default=False)
    is_reverse_movement: Mapped[bool] = mapped_column(Boolean, default=False)
    public_percentage: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sharp_action_detected: Mapped[bool] = mapped_column(Boolean, default=False)
    alert_level: Mapped[str] = mapped_column(String(20), default="normal")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    
    __table_args__ = (
        Index("ix_line_alert_game", "game_id"),
        Index("ix_line_alert_steam", "is_steam_move", "created_at"),
    )


# ============================================================================
# Additional Tables (41-43)
# ============================================================================

class Notification(Base):
    """User notifications for alerts and updates"""
    __tablename__ = "notifications"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    title = Column(String(255), nullable=False)
    message = Column(Text, nullable=False)
    notification_type = Column(String(50), nullable=False)  # prediction, alert, system, promo
    is_read = Column(Boolean, default=False)
    read_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="notifications")


class RateLimitLog(Base):
    """Rate limiting logs for API requests"""
    __tablename__ = "rate_limit_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    ip_address = Column(String(45), nullable=False)
    endpoint = Column(String(255), nullable=False)
    request_count = Column(Integer, default=1)
    window_start = Column(DateTime(timezone=True), server_default=func.now())
    blocked = Column(Boolean, default=False)
    
    # Relationships
    user = relationship("User")


class WeatherData(Base):
    """Weather data for outdoor sports predictions"""
    __tablename__ = "weather_data"
    
    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    game_id: Mapped[Optional[UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("games.id", ondelete="CASCADE"), nullable=True)
    venue_id: Mapped[Optional[UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("venues.id", ondelete="CASCADE"), nullable=True)
    temperature_f: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    feels_like_f: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    humidity_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    wind_speed_mph: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    wind_direction: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    precipitation_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    conditions: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    is_dome: Mapped[bool] = mapped_column(Boolean, default=False)
    recorded_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    
    # Relationships
    game = relationship("Game", back_populates="weather_data")


# Add relationships to User model
User.notifications = relationship("Notification", back_populates="user", cascade="all, delete-orphan")

# Add relationship to Game model for WeatherData table
Game.weather_data = relationship("WeatherData", back_populates="game", uselist=False)