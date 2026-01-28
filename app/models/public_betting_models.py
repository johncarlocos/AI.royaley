"""
ROYALEY - Public Betting Data Models
Tables for storing public betting data from Action Network and other sources.

Tables:
- public_betting: Individual game public betting percentages
- public_betting_history: Historical snapshots of public betting changes
- sharp_money_indicator: Sharp vs public money indicators
"""

from datetime import datetime, date
from decimal import Decimal
from typing import Optional
from uuid import uuid4

from sqlalchemy import (
    Boolean, Column, Date, DateTime, Float, ForeignKey, Index,
    Integer, String, Text, UniqueConstraint, func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


# =============================================================================
# PUBLIC BETTING DATA
# =============================================================================

class PublicBetting(Base):
    """
    Public betting percentages and money splits from Action Network.
    
    Contains:
    - % of bets on each side (spread, moneyline, total)
    - % of money on each side
    - Total bet count
    - Sharp vs public indicators
    """
    __tablename__ = "public_betting"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    game_id: Mapped[Optional[UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("games.id", ondelete="CASCADE"), nullable=True)
    sport_code: Mapped[str] = mapped_column(String(10), nullable=False)  # NFL, NBA, etc.
    
    # Game identification (for matching)
    external_game_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    home_team: Mapped[str] = mapped_column(String(100), nullable=False)
    away_team: Mapped[str] = mapped_column(String(100), nullable=False)
    game_date: Mapped[date] = mapped_column(Date, nullable=False)
    game_time: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    
    # Spread betting percentages
    spread_home_bet_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # % of bets on home spread
    spread_away_bet_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # % of bets on away spread
    spread_home_money_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # % of money on home spread
    spread_away_money_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # % of money on away spread
    spread_bet_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # Total spread bets
    spread_line: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # Current spread line
    spread_opening_line: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # Opening spread line
    
    # Moneyline betting percentages
    ml_home_bet_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # % of bets on home ML
    ml_away_bet_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # % of bets on away ML
    ml_home_money_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # % of money on home ML
    ml_away_money_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # % of money on away ML
    ml_bet_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # Total ML bets
    ml_home_odds: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # Current home ML odds
    ml_away_odds: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # Current away ML odds
    ml_home_opening: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # Opening home ML
    ml_away_opening: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # Opening away ML
    
    # Total (Over/Under) betting percentages
    total_over_bet_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # % of bets on over
    total_under_bet_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # % of bets on under
    total_over_money_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # % of money on over
    total_under_money_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # % of money on under
    total_bet_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # Total O/U bets
    total_line: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # Current total line
    total_opening_line: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # Opening total line
    
    # Sharp indicators
    is_sharp_spread: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)  # Sharp action on spread
    is_sharp_ml: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)  # Sharp action on ML
    is_sharp_total: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)  # Sharp action on total
    sharp_side_spread: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)  # home/away
    sharp_side_ml: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)  # home/away
    sharp_side_total: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)  # over/under
    
    # Reverse line movement indicators
    is_rlm_spread: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)  # Reverse line movement on spread
    is_rlm_total: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)  # Reverse line movement on total
    
    # Steam move detection
    is_steam_spread: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    is_steam_total: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    
    # Game result (filled after game completes)
    home_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    away_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    game_status: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)  # scheduled, final, etc.
    
    # Metadata
    source: Mapped[str] = mapped_column(String(50), default="action_network")
    scraped_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now())
    
    # Raw data storage
    raw_data: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    
    __table_args__ = (
        UniqueConstraint("sport_code", "home_team", "away_team", "game_date", "source", 
                        name="uq_public_betting_game"),
        Index("ix_public_betting_sport_date", "sport_code", "game_date"),
        Index("ix_public_betting_game_id", "game_id"),
        Index("ix_public_betting_sharp", "is_sharp_spread", "is_sharp_ml", "is_sharp_total"),
        Index("ix_public_betting_rlm", "is_rlm_spread", "is_rlm_total"),
    )


class PublicBettingHistory(Base):
    """
    Historical snapshots of public betting percentages.
    Tracks how public betting changes over time leading up to game.
    """
    __tablename__ = "public_betting_history"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    public_betting_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("public_betting.id", ondelete="CASCADE"))
    
    # Snapshot timing
    snapshot_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    hours_before_game: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Spread snapshot
    spread_home_bet_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    spread_home_money_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    spread_line: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    spread_bet_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    
    # Moneyline snapshot
    ml_home_bet_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ml_home_money_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ml_home_odds: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    ml_bet_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    
    # Total snapshot
    total_over_bet_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    total_over_money_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    total_line: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    total_bet_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    
    __table_args__ = (
        Index("ix_public_betting_history_pb_snapshot", "public_betting_id", "snapshot_at"),
    )


class SharpMoneyIndicator(Base):
    """
    Detailed sharp money tracking and indicators.
    Identifies professional/sharp bettor activity.
    """
    __tablename__ = "sharp_money_indicators"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    game_id: Mapped[Optional[UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("games.id", ondelete="CASCADE"), nullable=True)
    public_betting_id: Mapped[Optional[UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("public_betting.id", ondelete="SET NULL"), nullable=True)
    sport_code: Mapped[str] = mapped_column(String(10), nullable=False)
    
    # Game identification
    home_team: Mapped[str] = mapped_column(String(100), nullable=False)
    away_team: Mapped[str] = mapped_column(String(100), nullable=False)
    game_date: Mapped[date] = mapped_column(Date, nullable=False)
    
    # Bet type (spread, moneyline, total)
    bet_type: Mapped[str] = mapped_column(String(20), nullable=False)
    
    # Sharp indicator details
    indicator_type: Mapped[str] = mapped_column(String(50), nullable=False)  # steam_move, reverse_line, money_pct_divergence, etc.
    sharp_side: Mapped[str] = mapped_column(String(20), nullable=False)  # home, away, over, under
    confidence_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # 0-100
    
    # Line movement details
    line_before: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    line_after: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    line_movement: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Public vs money divergence
    public_bet_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    money_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    divergence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # money_pct - bet_pct
    
    # Outcome tracking
    result: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)  # win, loss, push
    profit_loss: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Metadata
    detected_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    source: Mapped[str] = mapped_column(String(50), default="action_network")
    
    __table_args__ = (
        Index("ix_sharp_indicators_sport_date", "sport_code", "game_date"),
        Index("ix_sharp_indicators_type", "indicator_type", "bet_type"),
    )


class FadePublicRecord(Base):
    """
    Records for tracking fade-the-public strategy performance.
    Tracks when to bet against heavy public action.
    """
    __tablename__ = "fade_public_records"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    public_betting_id: Mapped[Optional[UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("public_betting.id", ondelete="SET NULL"), nullable=True)
    sport_code: Mapped[str] = mapped_column(String(10), nullable=False)
    
    # Game identification
    home_team: Mapped[str] = mapped_column(String(100), nullable=False)
    away_team: Mapped[str] = mapped_column(String(100), nullable=False)
    game_date: Mapped[date] = mapped_column(Date, nullable=False)
    
    # Bet type
    bet_type: Mapped[str] = mapped_column(String(20), nullable=False)  # spread, moneyline, total
    
    # Public side (the side to fade)
    public_side: Mapped[str] = mapped_column(String(20), nullable=False)  # home, away, over, under
    public_bet_pct: Mapped[float] = mapped_column(Float, nullable=False)  # % of bets on public side
    public_money_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # % of money on public side
    
    # Fade threshold used
    fade_threshold: Mapped[float] = mapped_column(Float, nullable=False)  # e.g., 70% = fade when public > 70%
    
    # Bet details
    fade_side: Mapped[str] = mapped_column(String(20), nullable=False)  # The side to bet (opposite of public)
    line_at_bet: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    odds_at_bet: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    
    # Result
    result: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)  # win, loss, push
    profit_loss: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # Assuming $100 bet
    closing_line: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    clv: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # Closing line value
    
    # Final scores
    home_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    away_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    
    # Metadata
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    graded_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    __table_args__ = (
        Index("ix_fade_public_sport_date", "sport_code", "game_date"),
        Index("ix_fade_public_result", "result", "sport_code"),
        Index("ix_fade_public_threshold", "fade_threshold", "public_bet_pct"),
    )