"""
ROYALEY - Master Data Architecture Models
Data Unification Layer: 9 tables that create a single source of truth
across all 27 data collectors.

Tables:
  TIER 1 - Core:   master_teams, master_players, master_games
  TIER 2 - Maps:   team_mappings, player_mappings, game_mappings, venue_mappings
  TIER 3 - Infra:  source_registry, mapping_audit_log
"""

from datetime import datetime, date
from typing import List, Optional
from uuid import uuid4

from sqlalchemy import (
    Boolean, Date, DateTime, Float, ForeignKey, Index, Integer,
    String, Text, UniqueConstraint, func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


# =============================================================================
# TIER 3 — INFRASTRUCTURE (defined first, referenced by others)
# =============================================================================

class SourceRegistry(Base):
    """
    Registry of all data sources with priority and trust levels.
    Used to decide which source's data wins on conflict.
    """
    __tablename__ = "source_registry"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    key: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    source_type: Mapped[str] = mapped_column(String(30), nullable=False)  # api, scraper, file
    priority: Mapped[int] = mapped_column(Integer, default=50)  # 1=highest trust
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    # Capabilities
    provides_teams: Mapped[bool] = mapped_column(Boolean, default=False)
    provides_players: Mapped[bool] = mapped_column(Boolean, default=False)
    provides_games: Mapped[bool] = mapped_column(Boolean, default=False)
    provides_odds: Mapped[bool] = mapped_column(Boolean, default=False)
    provides_stats: Mapped[bool] = mapped_column(Boolean, default=False)
    sports_covered: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)  # ["NFL","NBA",…]

    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now())


class MappingAuditLog(Base):
    """Audit trail for every mapping change — debug and verify."""
    __tablename__ = "mapping_audit_log"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    entity_type: Mapped[str] = mapped_column(String(30), nullable=False)  # team, player, game, venue
    entity_id: Mapped[str] = mapped_column(String(100), nullable=False)
    action: Mapped[str] = mapped_column(String(30), nullable=False)  # created, merged, split, deleted
    old_value: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    new_value: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    performed_by: Mapped[str] = mapped_column(String(50), default="system")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())

    __table_args__ = (
        Index("ix_audit_entity", "entity_type", "entity_id"),
        Index("ix_audit_created", "created_at"),
    )


# =============================================================================
# TIER 1 — CORE MASTER TABLES
# =============================================================================

class MasterTeam(Base):
    """
    Canonical team record. One row per real team across the world.
    ~800 rows: 500 pro teams + 300 college (FBS/D1).
    """
    __tablename__ = "master_teams"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    sport_code: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    canonical_name: Mapped[str] = mapped_column(String(150), nullable=False)
    short_name: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    abbreviation: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    city: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    state: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    conference: Mapped[Optional[str]] = mapped_column(String(80), nullable=True)
    division: Mapped[Optional[str]] = mapped_column(String(80), nullable=True)
    venue_id: Mapped[Optional[UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("venues.id"), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    founded_year: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    metadata_json: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now())

    # Relationships
    team_mappings: Mapped[List["TeamMapping"]] = relationship(back_populates="master_team", cascade="all, delete-orphan")
    home_master_games: Mapped[List["MasterGame"]] = relationship(
        back_populates="home_master_team", foreign_keys="MasterGame.home_master_team_id"
    )
    away_master_games: Mapped[List["MasterGame"]] = relationship(
        back_populates="away_master_team", foreign_keys="MasterGame.away_master_team_id"
    )

    __table_args__ = (
        UniqueConstraint("sport_code", "canonical_name", name="uq_master_teams_sport_name"),
        Index("ix_master_teams_abbr", "abbreviation"),
    )


class MasterPlayer(Base):
    """
    Canonical player record. One row per real person.
    For tennis: these ARE the competitors (no team).
    For team sports: linked to master_teams.
    """
    __tablename__ = "master_players"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    sport_code: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    master_team_id: Mapped[Optional[UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("master_teams.id"), nullable=True
    )
    canonical_name: Mapped[str] = mapped_column(String(200), nullable=False)
    first_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    last_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    position: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    birth_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    height_inches: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    weight_lbs: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    nationality: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    hand: Mapped[Optional[str]] = mapped_column(String(5), nullable=True)  # L/R/A — tennis
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    metadata_json: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now())

    # Relationships
    player_mappings: Mapped[List["PlayerMapping"]] = relationship(back_populates="master_player", cascade="all, delete-orphan")
    home_master_games: Mapped[List["MasterGame"]] = relationship(
        back_populates="home_master_player", foreign_keys="MasterGame.home_master_player_id"
    )
    away_master_games: Mapped[List["MasterGame"]] = relationship(
        back_populates="away_master_player", foreign_keys="MasterGame.away_master_player_id"
    )

    __table_args__ = (
        UniqueConstraint("sport_code", "canonical_name", name="uq_master_players_sport_name"),
        Index("ix_master_players_team", "master_team_id"),
        Index("ix_master_players_last", "last_name"),
    )


class MasterGame(Base):
    """
    THE most critical table. One row per real game/match.
    All odds, stats, weather, predictions link here.
    ~200K rows (deduplicated from 340K source records).
    """
    __tablename__ = "master_games"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    sport_code: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    season: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    season_type: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)  # regular, playoff, preseason

    scheduled_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    # Team sports
    home_master_team_id: Mapped[Optional[UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("master_teams.id"), nullable=True
    )
    away_master_team_id: Mapped[Optional[UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("master_teams.id"), nullable=True
    )

    # Individual sports (tennis)
    home_master_player_id: Mapped[Optional[UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("master_players.id"), nullable=True
    )
    away_master_player_id: Mapped[Optional[UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("master_players.id"), nullable=True
    )

    venue_id: Mapped[Optional[UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("venues.id"), nullable=True
    )

    # Result
    status: Mapped[str] = mapped_column(String(20), default="scheduled")
    home_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    away_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Quarter / period scores
    score_detail: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    is_neutral_site: Mapped[bool] = mapped_column(Boolean, default=False)
    is_playoff: Mapped[bool] = mapped_column(Boolean, default=False)

    # Which source we trusted for core data
    primary_source: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now())

    # Relationships
    home_master_team: Mapped[Optional["MasterTeam"]] = relationship(
        back_populates="home_master_games", foreign_keys=[home_master_team_id]
    )
    away_master_team: Mapped[Optional["MasterTeam"]] = relationship(
        back_populates="away_master_games", foreign_keys=[away_master_team_id]
    )
    home_master_player: Mapped[Optional["MasterPlayer"]] = relationship(
        back_populates="home_master_games", foreign_keys=[home_master_player_id]
    )
    away_master_player: Mapped[Optional["MasterPlayer"]] = relationship(
        back_populates="away_master_games", foreign_keys=[away_master_player_id]
    )
    game_mappings: Mapped[List["GameMapping"]] = relationship(back_populates="master_game", cascade="all, delete-orphan")

    __table_args__ = (
        # Prevent duplicate games: same sport, same day, same matchup
        Index("ix_master_games_dedup", "sport_code", "scheduled_at", "home_master_team_id", "away_master_team_id"),
        Index("ix_master_games_date", "sport_code", "scheduled_at"),
        Index("ix_master_games_status", "status"),
    )


# =============================================================================
# TIER 2 — MAPPING TABLES  (source → master)
# =============================================================================

class TeamMapping(Base):
    """
    Maps every source-specific team name/id to the canonical master_team.
    Example: "LA Lakers" (espn), "Los Angeles Lakers" (bdl), "L.A. Lakers" (pinnacle) → 1 master row.
    """
    __tablename__ = "team_mappings"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    master_team_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("master_teams.id", ondelete="CASCADE"), nullable=False
    )
    source_key: Mapped[str] = mapped_column(String(50), nullable=False)  # "espn", "bdl", "pinnacle"
    source_team_name: Mapped[str] = mapped_column(String(200), nullable=False)
    source_external_id: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    source_team_db_id: Mapped[Optional[UUID]] = mapped_column(UUID(as_uuid=True), nullable=True)  # FK to teams.id
    confidence: Mapped[float] = mapped_column(Float, default=1.0)  # 0.0-1.0
    verified: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())

    master_team: Mapped["MasterTeam"] = relationship(back_populates="team_mappings")

    __table_args__ = (
        UniqueConstraint("source_key", "source_team_name", name="uq_team_map_source_name"),
        Index("ix_team_map_master", "master_team_id"),
        Index("ix_team_map_source", "source_key"),
    )


class PlayerMapping(Base):
    """Maps source player records to canonical master_player."""
    __tablename__ = "player_mappings"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    master_player_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("master_players.id", ondelete="CASCADE"), nullable=False
    )
    source_key: Mapped[str] = mapped_column(String(50), nullable=False)
    source_player_name: Mapped[str] = mapped_column(String(200), nullable=False)
    source_external_id: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    source_player_db_id: Mapped[Optional[UUID]] = mapped_column(UUID(as_uuid=True), nullable=True)
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    verified: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())

    master_player: Mapped["MasterPlayer"] = relationship(back_populates="player_mappings")

    __table_args__ = (
        UniqueConstraint("source_key", "source_external_id", name="uq_player_map_source_ext"),
        Index("ix_player_map_master", "master_player_id"),
        Index("ix_player_map_source", "source_key"),
        Index("ix_player_map_name", "source_player_name"),
    )


class GameMapping(Base):
    """Maps source game records to canonical master_game."""
    __tablename__ = "game_mappings"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    master_game_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("master_games.id", ondelete="CASCADE"), nullable=False
    )
    source_key: Mapped[str] = mapped_column(String(50), nullable=False)
    source_external_id: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    source_game_db_id: Mapped[Optional[UUID]] = mapped_column(UUID(as_uuid=True), nullable=True)  # FK to games.id
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())

    master_game: Mapped["MasterGame"] = relationship(back_populates="game_mappings")

    __table_args__ = (
        UniqueConstraint("source_key", "source_external_id", name="uq_game_map_source_ext"),
        Index("ix_game_map_master", "master_game_id"),
        Index("ix_game_map_source", "source_key"),
    )


class VenueMapping(Base):
    """Maps source venue name variants to canonical venues.id."""
    __tablename__ = "venue_mappings"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    venue_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("venues.id", ondelete="CASCADE"), nullable=False
    )
    source_key: Mapped[str] = mapped_column(String(50), nullable=False)
    source_venue_name: Mapped[str] = mapped_column(String(200), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())

    __table_args__ = (
        UniqueConstraint("source_key", "source_venue_name", name="uq_venue_map_source_name"),
        Index("ix_venue_map_venue", "venue_id"),
    )
