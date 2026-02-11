"""
ROYALEY - Upcoming Games & Odds Models
Live prediction pipeline tables - completely separate from training data.

Tables:
  58. upcoming_games  - Live games from Odds API (NOT training data)
  59. upcoming_odds   - Current sportsbook lines for upcoming games
"""

from datetime import datetime
from typing import Optional, List
from uuid import uuid4, UUID

from sqlalchemy import (
    Boolean, DateTime, Enum, Float, ForeignKey, Index, Integer,
    String, Text, func, UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.models import Base, GameStatus


class UpcomingGame(Base):
    """
    Upcoming games fetched from Odds API.
    Completely separate from the 'games' table used for ML training.
    """
    __tablename__ = "upcoming_games"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    sport_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("sports.id", ondelete="CASCADE"), nullable=False)
    
    # Odds API external ID (unique per game)
    external_id: Mapped[str] = mapped_column(String(200), unique=True, nullable=False)
    
    # Teams - reference shared teams table
    home_team_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("teams.id", ondelete="CASCADE"), nullable=False)
    away_team_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("teams.id", ondelete="CASCADE"), nullable=False)
    
    # Team names stored directly (avoid extra joins, also covers cases where team isn't in teams table)
    home_team_name: Mapped[str] = mapped_column(String(200), nullable=False)
    away_team_name: Mapped[str] = mapped_column(String(200), nullable=False)
    
    # Schedule
    scheduled_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    status: Mapped[GameStatus] = mapped_column(
        Enum(GameStatus, values_callable=lambda obj: [e.value for e in obj]),
        default=GameStatus.SCHEDULED
    )
    
    # Scores (filled after game completes)
    home_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    away_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    
    # Source tracking
    source: Mapped[str] = mapped_column(String(50), default="odds_api")  # odds_api, pinnacle
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    sport: Mapped["Sport"] = relationship(foreign_keys=[sport_id])
    home_team: Mapped["Team"] = relationship(foreign_keys=[home_team_id])
    away_team: Mapped["Team"] = relationship(foreign_keys=[away_team_id])
    upcoming_odds: Mapped[List["UpcomingOdds"]] = relationship(back_populates="upcoming_game", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index("ix_upcoming_games_sport", "sport_id"),
        Index("ix_upcoming_games_scheduled", "scheduled_at"),
        Index("ix_upcoming_games_status", "status"),
        Index("ix_upcoming_games_sport_scheduled", "sport_id", "scheduled_at"),
    )
    
    def __repr__(self):
        return f"<UpcomingGame {self.away_team_name} @ {self.home_team_name} ({self.scheduled_at})>"


class UpcomingOdds(Base):
    """
    Current odds/lines for upcoming games from various sportsbooks.
    Separate from the 'odds' table used for ML training.
    """
    __tablename__ = "upcoming_odds"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    upcoming_game_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("upcoming_games.id", ondelete="CASCADE"), nullable=False)
    
    # Sportsbook info
    sportsbook_key: Mapped[str] = mapped_column(String(50), nullable=False)  # e.g., "pinnacle", "fanduel", "draftkings"
    sportsbook_name: Mapped[str] = mapped_column(String(100), nullable=False)
    is_sharp: Mapped[bool] = mapped_column(Boolean, default=False)  # True for Pinnacle
    
    # Market type: spread, moneyline, total
    bet_type: Mapped[str] = mapped_column(String(50), nullable=False)
    
    # Spread lines
    home_line: Mapped[Optional[float]] = mapped_column(Float, nullable=True)   # e.g., -3.5
    away_line: Mapped[Optional[float]] = mapped_column(Float, nullable=True)   # e.g., +3.5
    home_odds: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)   # e.g., -110
    away_odds: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)   # e.g., -110
    
    # Total lines
    total: Mapped[Optional[float]] = mapped_column(Float, nullable=True)       # e.g., 215.5
    over_odds: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)   # e.g., -110
    under_odds: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # e.g., -110
    
    # Moneyline odds
    home_ml: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)     # e.g., -150
    away_ml: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)     # e.g., +130
    
    # Source tracking
    source: Mapped[str] = mapped_column(String(50), default="odds_api")  # odds_api, pinnacle
    
    recorded_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    upcoming_game: Mapped["UpcomingGame"] = relationship(back_populates="upcoming_odds")
    
    __table_args__ = (
        Index("ix_upcoming_odds_game", "upcoming_game_id"),
        Index("ix_upcoming_odds_game_book_bet", "upcoming_game_id", "sportsbook_key", "bet_type"),
        UniqueConstraint("upcoming_game_id", "sportsbook_key", "bet_type", name="uq_upcoming_odds_game_book_bet"),
    )
    
    def __repr__(self):
        return f"<UpcomingOdds {self.sportsbook_key} {self.bet_type} game={self.upcoming_game_id}>"