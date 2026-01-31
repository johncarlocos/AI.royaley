"""
ROYALEY - Master Data Service
Core resolver: every collector calls this to get canonical IDs.

Usage in any collector:
    from app.services.master_data import MasterDataService
    mds = MasterDataService(session)
    master_team = await mds.resolve_team("espn", "LA Lakers", sport_code="NBA")
    master_game = await mds.resolve_game("espn", "401772771", sport_code="NBA",
                                          scheduled_at=dt, home_team=mt_home, away_team=mt_away)
"""

import logging
from datetime import datetime, timedelta
from typing import Optional
from uuid import UUID

from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.master_data_models import (
    MasterTeam, MasterPlayer, MasterGame,
    TeamMapping, PlayerMapping, GameMapping,
    MappingAuditLog,
)

logger = logging.getLogger(__name__)


class MasterDataService:
    """
    Resolves source-specific names/IDs to canonical master records.
    Designed for high throughput: caches lookups per session.
    """

    def __init__(self, session: AsyncSession):
        self.session = session
        # In-session cache to avoid repeated DB hits during a single collection run
        self._team_cache: dict[tuple[str, str], MasterTeam] = {}
        self._player_cache: dict[tuple[str, str], MasterPlayer] = {}

    # =========================================================================
    # TEAM RESOLUTION
    # =========================================================================

    async def resolve_team(
        self,
        source_key: str,
        source_name: str,
        sport_code: str,
        source_external_id: Optional[str] = None,
        source_db_id: Optional[UUID] = None,
        auto_create: bool = False,
    ) -> Optional[MasterTeam]:
        """
        Resolve a source-specific team name to the canonical MasterTeam.

        Steps:
          1. Check in-session cache
          2. Look up team_mappings by (source_key, source_name)
          3. Look up by (source_key, source_external_id) if provided
          4. If auto_create: fuzzy-search master_teams, create mapping if match found
          5. Return None if unresolved (logged for manual review)
        """
        cache_key = (source_key, source_name)
        if cache_key in self._team_cache:
            return self._team_cache[cache_key]

        # Step 2: exact mapping lookup by name
        result = await self.session.execute(
            select(TeamMapping).where(
                TeamMapping.source_key == source_key,
                TeamMapping.source_team_name == source_name,
            )
        )
        mapping = result.scalars().first()

        if mapping:
            master = await self._get_master_team(mapping.master_team_id)
            if master:
                self._team_cache[cache_key] = master
                return master

        # Step 3: lookup by external_id
        if source_external_id:
            result = await self.session.execute(
                select(TeamMapping).where(
                    TeamMapping.source_key == source_key,
                    TeamMapping.source_external_id == source_external_id,
                )
            )
            mapping = result.scalars().first()
            if mapping:
                master = await self._get_master_team(mapping.master_team_id)
                if master:
                    self._team_cache[cache_key] = master
                    # Also create a name-based mapping for future lookups
                    await self._create_team_mapping(
                        master.id, source_key, source_name,
                        source_external_id, source_db_id, confidence=0.95
                    )
                    return master

        # Step 4: try case-insensitive match against master_teams
        result = await self.session.execute(
            select(MasterTeam).where(
                MasterTeam.sport_code == sport_code,
                func.lower(MasterTeam.canonical_name) == source_name.lower(),
            )
        )
        master = result.scalars().first()
        if master:
            await self._create_team_mapping(
                master.id, source_key, source_name,
                source_external_id, source_db_id, confidence=1.0
            )
            self._team_cache[cache_key] = master
            return master

        # Try abbreviation match
        if source_name and len(source_name) <= 10:
            result = await self.session.execute(
                select(MasterTeam).where(
                    MasterTeam.sport_code == sport_code,
                    func.upper(MasterTeam.abbreviation) == source_name.upper(),
                )
            )
            master = result.scalars().first()
            if master:
                await self._create_team_mapping(
                    master.id, source_key, source_name,
                    source_external_id, source_db_id, confidence=0.9
                )
                self._team_cache[cache_key] = master
                return master

        if not auto_create:
            logger.warning(f"Unresolved team: source={source_key} name={source_name} sport={sport_code}")
            return None

        # Auto-create: create a new MasterTeam (only when explicitly allowed)
        master = MasterTeam(
            sport_code=sport_code,
            canonical_name=source_name,
            abbreviation=source_name[:10] if source_name else None,
            is_active=True,
        )
        self.session.add(master)
        await self.session.flush()

        await self._create_team_mapping(
            master.id, source_key, source_name,
            source_external_id, source_db_id, confidence=0.7
        )
        self._team_cache[cache_key] = master
        logger.info(f"Auto-created master team: {source_name} ({sport_code})")
        return master

    async def _get_master_team(self, team_id: UUID) -> Optional[MasterTeam]:
        result = await self.session.execute(
            select(MasterTeam).where(MasterTeam.id == team_id)
        )
        return result.scalars().first()

    async def _create_team_mapping(
        self, master_id: UUID, source_key: str, source_name: str,
        source_ext_id: Optional[str], source_db_id: Optional[UUID],
        confidence: float = 1.0,
    ):
        try:
            mapping = TeamMapping(
                master_team_id=master_id,
                source_key=source_key,
                source_team_name=source_name,
                source_external_id=source_ext_id,
                source_team_db_id=source_db_id,
                confidence=confidence,
                verified=confidence >= 1.0,
            )
            self.session.add(mapping)
            await self.session.flush()
        except Exception:
            await self.session.rollback()
            logger.debug(f"Mapping already exists: {source_key}/{source_name}")

    # =========================================================================
    # PLAYER RESOLUTION
    # =========================================================================

    async def resolve_player(
        self,
        source_key: str,
        source_name: str,
        sport_code: str,
        source_external_id: Optional[str] = None,
        source_db_id: Optional[UUID] = None,
        master_team_id: Optional[UUID] = None,
        auto_create: bool = False,
    ) -> Optional[MasterPlayer]:
        """Resolve a source player to canonical MasterPlayer."""
        cache_key = (source_key, source_external_id or source_name)
        if cache_key in self._player_cache:
            return self._player_cache[cache_key]

        # Lookup by external_id first (more reliable than name)
        if source_external_id:
            result = await self.session.execute(
                select(PlayerMapping).where(
                    PlayerMapping.source_key == source_key,
                    PlayerMapping.source_external_id == source_external_id,
                )
            )
            mapping = result.scalars().first()
            if mapping:
                master = await self._get_master_player(mapping.master_player_id)
                if master:
                    self._player_cache[cache_key] = master
                    return master

        # Lookup by name
        result = await self.session.execute(
            select(PlayerMapping).where(
                PlayerMapping.source_key == source_key,
                PlayerMapping.source_player_name == source_name,
            )
        )
        mapping = result.scalars().first()
        if mapping:
            master = await self._get_master_player(mapping.master_player_id)
            if master:
                self._player_cache[cache_key] = master
                return master

        # Direct match on master_players
        result = await self.session.execute(
            select(MasterPlayer).where(
                MasterPlayer.sport_code == sport_code,
                func.lower(MasterPlayer.canonical_name) == source_name.strip().lower(),
            )
        )
        master = result.scalars().first()
        if master:
            await self._create_player_mapping(
                master.id, source_key, source_name,
                source_external_id, source_db_id, confidence=1.0
            )
            self._player_cache[cache_key] = master
            return master

        if not auto_create:
            return None

        # Parse first/last name
        parts = source_name.strip().split(None, 1)
        first = parts[0] if parts else ""
        last = parts[1] if len(parts) > 1 else parts[0] if parts else ""

        master = MasterPlayer(
            sport_code=sport_code,
            master_team_id=master_team_id,
            canonical_name=source_name.strip(),
            first_name=first,
            last_name=last,
            is_active=True,
        )
        self.session.add(master)
        await self.session.flush()

        await self._create_player_mapping(
            master.id, source_key, source_name,
            source_external_id, source_db_id, confidence=0.7
        )
        self._player_cache[cache_key] = master
        return master

    async def _get_master_player(self, player_id: UUID) -> Optional[MasterPlayer]:
        result = await self.session.execute(
            select(MasterPlayer).where(MasterPlayer.id == player_id)
        )
        return result.scalars().first()

    async def _create_player_mapping(
        self, master_id: UUID, source_key: str, source_name: str,
        source_ext_id: Optional[str], source_db_id: Optional[UUID],
        confidence: float = 1.0,
    ):
        try:
            mapping = PlayerMapping(
                master_player_id=master_id,
                source_key=source_key,
                source_player_name=source_name,
                source_external_id=source_ext_id,
                source_player_db_id=source_db_id,
                confidence=confidence,
                verified=confidence >= 1.0,
            )
            self.session.add(mapping)
            await self.session.flush()
        except Exception:
            await self.session.rollback()
            logger.debug(f"Player mapping exists: {source_key}/{source_ext_id}")

    # =========================================================================
    # GAME RESOLUTION
    # =========================================================================

    async def resolve_game(
        self,
        source_key: str,
        source_external_id: str,
        sport_code: str,
        scheduled_at: datetime,
        home_master_team_id: Optional[UUID] = None,
        away_master_team_id: Optional[UUID] = None,
        home_master_player_id: Optional[UUID] = None,
        away_master_player_id: Optional[UUID] = None,
        source_db_id: Optional[UUID] = None,
        home_score: Optional[int] = None,
        away_score: Optional[int] = None,
        status: str = "scheduled",
        auto_create: bool = True,
    ) -> Optional[MasterGame]:
        """
        Resolve a source game to canonical MasterGame.
        Matching logic: same sport + same teams + within 24h window.
        """
        # Step 1: existing mapping
        if source_external_id:
            result = await self.session.execute(
                select(GameMapping).where(
                    GameMapping.source_key == source_key,
                    GameMapping.source_external_id == source_external_id,
                )
            )
            mapping = result.scalars().first()
            if mapping:
                master = await self._get_master_game(mapping.master_game_id)
                if master:
                    # Update score if we have it
                    if home_score is not None and master.home_score is None:
                        master.home_score = home_score
                        master.away_score = away_score
                        master.status = status
                    return master

        # Step 2: find by sport + teams + date window (±24h)
        window_start = scheduled_at - timedelta(hours=24)
        window_end = scheduled_at + timedelta(hours=24)

        conditions = [
            MasterGame.sport_code == sport_code,
            MasterGame.scheduled_at.between(window_start, window_end),
        ]

        if home_master_team_id and away_master_team_id:
            conditions.extend([
                MasterGame.home_master_team_id == home_master_team_id,
                MasterGame.away_master_team_id == away_master_team_id,
            ])
        elif home_master_player_id and away_master_player_id:
            conditions.extend([
                MasterGame.home_master_player_id == home_master_player_id,
                MasterGame.away_master_player_id == away_master_player_id,
            ])
        else:
            # Can't match without participants
            if not auto_create:
                return None

        result = await self.session.execute(
            select(MasterGame).where(and_(*conditions))
        )
        master = result.scalars().first()

        if master:
            # Create mapping for this source
            await self._create_game_mapping(master.id, source_key, source_external_id, source_db_id)
            # Update score if available
            if home_score is not None and master.home_score is None:
                master.home_score = home_score
                master.away_score = away_score
                master.status = status
            return master

        if not auto_create:
            return None

        # Step 3: create new master game
        master = MasterGame(
            sport_code=sport_code,
            scheduled_at=scheduled_at,
            home_master_team_id=home_master_team_id,
            away_master_team_id=away_master_team_id,
            home_master_player_id=home_master_player_id,
            away_master_player_id=away_master_player_id,
            home_score=home_score,
            away_score=away_score,
            status=status,
            primary_source=source_key,
        )
        self.session.add(master)
        await self.session.flush()

        await self._create_game_mapping(master.id, source_key, source_external_id, source_db_id)
        return master

    async def _get_master_game(self, game_id: UUID) -> Optional[MasterGame]:
        result = await self.session.execute(
            select(MasterGame).where(MasterGame.id == game_id)
        )
        return result.scalars().first()

    async def _create_game_mapping(
        self, master_id: UUID, source_key: str,
        source_ext_id: Optional[str], source_db_id: Optional[UUID],
    ):
        try:
            mapping = GameMapping(
                master_game_id=master_id,
                source_key=source_key,
                source_external_id=source_ext_id,
                source_game_db_id=source_db_id,
            )
            self.session.add(mapping)
            await self.session.flush()
        except Exception:
            await self.session.rollback()
            logger.debug(f"Game mapping exists: {source_key}/{source_ext_id}")

    # =========================================================================
    # AUDIT LOGGING
    # =========================================================================

    async def log_audit(
        self, entity_type: str, entity_id: str,
        action: str, old_value: str = None, new_value: str = None,
    ):
        entry = MappingAuditLog(
            entity_type=entity_type,
            entity_id=str(entity_id),
            action=action,
            old_value=old_value,
            new_value=new_value,
        )
        self.session.add(entry)
