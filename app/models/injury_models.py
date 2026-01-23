"""
ROYALEY - Additional Database Models
Injury tracking models

Add this to your existing models.py or import from here.
"""

from datetime import datetime, date
from typing import Optional, List
from uuid import uuid4

from sqlalchemy import (
    Boolean, Column, Date, DateTime, Float, ForeignKey, 
    Index, Integer, String, UniqueConstraint, func
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class Injury(Base):
    """Player injury tracking."""
    __tablename__ = "injuries"
    
    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    player_id: Mapped[Optional[UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("players.id", ondelete="CASCADE"), nullable=True
    )
    team_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("teams.id"), nullable=False)
    sport_code: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    
    # Player info
    player_name: Mapped[str] = mapped_column(String(200), nullable=False)
    position: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    
    # Injury details
    injury_type: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    body_part: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    status: Mapped[str] = mapped_column(String(50), nullable=False)  # Out, Doubtful, Questionable, Probable, IR
    status_detail: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    
    # Impact
    games_missed: Mapped[int] = mapped_column(Integer, default=0)
    is_starter: Mapped[bool] = mapped_column(Boolean, default=False)
    impact_score: Mapped[float] = mapped_column(Float, default=0.0)
    
    # Dates
    injury_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    expected_return: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    first_reported: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    last_updated: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now())
    
    # Source
    source: Mapped[str] = mapped_column(String(50), default="espn")
    external_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    
    # Relationships
    player: Mapped[Optional["Player"]] = relationship("Player", back_populates="injuries")
    team: Mapped["Team"] = relationship("Team", back_populates="injuries")
    game_injuries: Mapped[List["GameInjury"]] = relationship("GameInjury", back_populates="injury")
    
    __table_args__ = (
        Index("ix_injuries_team_status", "team_id", "status"),
        Index("ix_injuries_sport_date", "sport_code", "last_updated"),
    )
    
    @property
    def is_out(self) -> bool:
        return self.status in ["Out", "IR", "Suspended"]
    
    @property
    def is_questionable(self) -> bool:
        return self.status in ["Questionable", "Doubtful", "Day-to-Day"]


class GameInjury(Base):
    """Junction table linking injuries to specific games."""
    __tablename__ = "game_injuries"
    
    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    game_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("games.id", ondelete="CASCADE"), nullable=False
    )
    injury_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("injuries.id", ondelete="CASCADE"), nullable=False
    )
    team_side: Mapped[str] = mapped_column(String(10), nullable=False)  # home, away
    impact_on_game: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    
    # Relationships
    game: Mapped["Game"] = relationship("Game", back_populates="game_injuries")
    injury: Mapped["Injury"] = relationship("Injury", back_populates="game_injuries")
    
    __table_args__ = (
        UniqueConstraint("game_id", "injury_id", name="uq_game_injuries"),
    )


# Add these relationships to existing models in models.py:
# 
# In Player class:
#     injuries: Mapped[List["Injury"]] = relationship("Injury", back_populates="player")
#
# In Team class:
#     injuries: Mapped[List["Injury"]] = relationship("Injury", back_populates="team")
#
# In Game class:
#     game_injuries: Mapped[List["GameInjury"]] = relationship("GameInjury", back_populates="game")
