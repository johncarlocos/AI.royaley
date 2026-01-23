"""
LOYALEY - Games API Routes
Enterprise-grade game management endpoints
"""

from datetime import datetime, date, timedelta
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.api.dependencies import get_current_user
from app.core.cache import cache_manager


router = APIRouter(tags=["games"])


# ============================================================================
# SCHEMAS
# ============================================================================

class TeamInfo(BaseModel):
    id: int
    name: str
    abbreviation: str
    elo_rating: Optional[float] = None


class GameBase(BaseModel):
    id: int
    external_id: str
    sport_code: str
    home_team: TeamInfo
    away_team: TeamInfo
    game_date: datetime
    venue: Optional[str] = None
    status: str  # scheduled, in_progress, final, postponed, cancelled
    home_score: Optional[int] = None
    away_score: Optional[int] = None
    period: Optional[str] = None
    is_neutral_site: bool = False


class GameDetail(GameBase):
    home_team_record: Optional[str] = None
    away_team_record: Optional[str] = None
    weather: Optional[dict] = None
    broadcast: Optional[str] = None
    attendance: Optional[int] = None
    features: Optional[dict] = None
    predictions_count: int = 0


class GameListResponse(BaseModel):
    games: List[GameBase]
    total: int
    page: int
    per_page: int


class GameFeatures(BaseModel):
    game_id: int
    sport_code: str
    home_elo: float
    away_elo: float
    home_offensive_rating: Optional[float] = None
    home_defensive_rating: Optional[float] = None
    away_offensive_rating: Optional[float] = None
    away_defensive_rating: Optional[float] = None
    home_rest_days: int
    away_rest_days: int
    home_b2b: bool
    away_b2b: bool
    home_win_streak: int
    away_win_streak: int
    h2h_home_wins: int
    h2h_away_wins: int
    home_last5_wins: int
    away_last5_wins: int
    spread_movement: Optional[float] = None
    total_movement: Optional[float] = None
    computed_at: datetime


class GameScheduleResponse(BaseModel):
    date: date
    sport_code: Optional[str]
    games: List[GameBase]
    total_games: int


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.get("", response_model=GameListResponse)
async def get_games(
    sport: Optional[str] = Query(None, description="Filter by sport code"),
    status: Optional[str] = Query(None, description="Filter by status"),
    date_from: Optional[date] = Query(None, description="Filter from date"),
    date_to: Optional[date] = Query(None, description="Filter to date"),
    team_id: Optional[int] = Query(None, description="Filter by team ID"),
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Get games with filtering and pagination.
    """
    conditions = []
    params = {}
    
    if sport:
        conditions.append("g.sport_code = :sport")
        params["sport"] = sport.upper()
    if status:
        conditions.append("g.status = :status")
        params["status"] = status
    if date_from:
        conditions.append("DATE(g.game_date) >= :date_from")
        params["date_from"] = date_from
    if date_to:
        conditions.append("DATE(g.game_date) <= :date_to")
        params["date_to"] = date_to
    if team_id:
        conditions.append("(g.home_team_id = :team_id OR g.away_team_id = :team_id)")
        params["team_id"] = team_id
    
    where_clause = " AND ".join(conditions) if conditions else "1=1"
    
    # Count total
    count_query = f"SELECT COUNT(*) FROM games g WHERE {where_clause}"
    count_result = await db.execute(count_query, params)
    total = count_result.scalar() or 0
    
    # Get paginated results
    offset = (page - 1) * per_page
    params["limit"] = per_page
    params["offset"] = offset
    
    query = f"""
        SELECT 
            g.*,
            ht.id as home_id, ht.name as home_name, ht.abbreviation as home_abbr, ht.elo_rating as home_elo,
            at.id as away_id, at.name as away_name, at.abbreviation as away_abbr, at.elo_rating as away_elo
        FROM games g
        JOIN teams ht ON g.home_team_id = ht.id
        JOIN teams at ON g.away_team_id = at.id
        WHERE {where_clause}
        ORDER BY g.game_date DESC
        LIMIT :limit OFFSET :offset
    """
    
    result = await db.execute(query, params)
    rows = result.fetchall()
    
    games = [
        GameBase(
            id=row.id,
            external_id=row.external_id,
            sport_code=row.sport_code,
            home_team=TeamInfo(
                id=row.home_id,
                name=row.home_name,
                abbreviation=row.home_abbr,
                elo_rating=row.home_elo
            ),
            away_team=TeamInfo(
                id=row.away_id,
                name=row.away_name,
                abbreviation=row.away_abbr,
                elo_rating=row.away_elo
            ),
            game_date=row.game_date,
            venue=row.venue,
            status=row.status,
            home_score=row.home_score,
            away_score=row.away_score,
            period=row.period,
            is_neutral_site=row.is_neutral_site or False
        )
        for row in rows
    ]
    
    return GameListResponse(
        games=games,
        total=total,
        page=page,
        per_page=per_page
    )


@router.get("/today", response_model=GameScheduleResponse)
async def get_todays_games(
    sport: Optional[str] = Query(None, description="Filter by sport code"),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Get today's game schedule.
    """
    today = date.today()
    cache_key = f"games:today:{sport or 'all'}"
    
    cached = await cache_manager.get(cache_key)
    if cached:
        return cached
    
    conditions = ["DATE(g.game_date) = :today"]
    params = {"today": today}
    
    if sport:
        conditions.append("g.sport_code = :sport")
        params["sport"] = sport.upper()
    
    where_clause = " AND ".join(conditions)
    
    query = f"""
        SELECT 
            g.*,
            ht.id as home_id, ht.name as home_name, ht.abbreviation as home_abbr, ht.elo_rating as home_elo,
            at.id as away_id, at.name as away_name, at.abbreviation as away_abbr, at.elo_rating as away_elo
        FROM games g
        JOIN teams ht ON g.home_team_id = ht.id
        JOIN teams at ON g.away_team_id = at.id
        WHERE {where_clause}
        ORDER BY g.game_date ASC
    """
    
    result = await db.execute(query, params)
    rows = result.fetchall()
    
    games = [
        GameBase(
            id=row.id,
            external_id=row.external_id,
            sport_code=row.sport_code,
            home_team=TeamInfo(
                id=row.home_id,
                name=row.home_name,
                abbreviation=row.home_abbr,
                elo_rating=row.home_elo
            ),
            away_team=TeamInfo(
                id=row.away_id,
                name=row.away_name,
                abbreviation=row.away_abbr,
                elo_rating=row.away_elo
            ),
            game_date=row.game_date,
            venue=row.venue,
            status=row.status,
            home_score=row.home_score,
            away_score=row.away_score,
            period=row.period,
            is_neutral_site=row.is_neutral_site or False
        )
        for row in rows
    ]
    
    response = GameScheduleResponse(
        date=today,
        sport_code=sport,
        games=games,
        total_games=len(games)
    )
    
    # Cache for 5 minutes
    await cache_manager.set(cache_key, response.dict(), ttl=300)
    
    return response


@router.get("/upcoming", response_model=List[GameBase])
async def get_upcoming_games(
    sport: Optional[str] = Query(None, description="Filter by sport code"),
    days: int = Query(7, ge=1, le=30, description="Days ahead to look"),
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Get upcoming scheduled games.
    """
    now = datetime.utcnow()
    end_date = now + timedelta(days=days)
    
    conditions = [
        "g.game_date >= :now",
        "g.game_date <= :end_date",
        "g.status = 'scheduled'"
    ]
    params = {"now": now, "end_date": end_date}
    
    if sport:
        conditions.append("g.sport_code = :sport")
        params["sport"] = sport.upper()
    
    where_clause = " AND ".join(conditions)
    params["limit"] = limit
    
    query = f"""
        SELECT 
            g.*,
            ht.id as home_id, ht.name as home_name, ht.abbreviation as home_abbr, ht.elo_rating as home_elo,
            at.id as away_id, at.name as away_name, at.abbreviation as away_abbr, at.elo_rating as away_elo
        FROM games g
        JOIN teams ht ON g.home_team_id = ht.id
        JOIN teams at ON g.away_team_id = at.id
        WHERE {where_clause}
        ORDER BY g.game_date ASC
        LIMIT :limit
    """
    
    result = await db.execute(query, params)
    rows = result.fetchall()
    
    return [
        GameBase(
            id=row.id,
            external_id=row.external_id,
            sport_code=row.sport_code,
            home_team=TeamInfo(
                id=row.home_id,
                name=row.home_name,
                abbreviation=row.home_abbr,
                elo_rating=row.home_elo
            ),
            away_team=TeamInfo(
                id=row.away_id,
                name=row.away_name,
                abbreviation=row.away_abbr,
                elo_rating=row.away_elo
            ),
            game_date=row.game_date,
            venue=row.venue,
            status=row.status,
            home_score=row.home_score,
            away_score=row.away_score,
            period=row.period,
            is_neutral_site=row.is_neutral_site or False
        )
        for row in rows
    ]


@router.get("/{game_id}", response_model=GameDetail)
async def get_game_detail(
    game_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Get detailed game information including features and prediction count.
    """
    query = """
        SELECT 
            g.*,
            ht.id as home_id, ht.name as home_name, ht.abbreviation as home_abbr, ht.elo_rating as home_elo,
            at.id as away_id, at.name as away_name, at.abbreviation as away_abbr, at.elo_rating as away_elo,
            gf.features_json,
            (SELECT COUNT(*) FROM predictions WHERE game_id = g.id) as predictions_count
        FROM games g
        JOIN teams ht ON g.home_team_id = ht.id
        JOIN teams at ON g.away_team_id = at.id
        LEFT JOIN game_features gf ON g.id = gf.game_id
        WHERE g.id = :game_id
    """
    
    result = await db.execute(query, {"game_id": game_id})
    row = result.fetchone()
    
    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Game {game_id} not found"
        )
    
    # Parse features
    features = None
    if row.features_json:
        import json
        try:
            features = json.loads(row.features_json) if isinstance(row.features_json, str) else row.features_json
        except json.JSONDecodeError:
            pass
    
    return GameDetail(
        id=row.id,
        external_id=row.external_id,
        sport_code=row.sport_code,
        home_team=TeamInfo(
            id=row.home_id,
            name=row.home_name,
            abbreviation=row.home_abbr,
            elo_rating=row.home_elo
        ),
        away_team=TeamInfo(
            id=row.away_id,
            name=row.away_name,
            abbreviation=row.away_abbr,
            elo_rating=row.away_elo
        ),
        game_date=row.game_date,
        venue=row.venue,
        status=row.status,
        home_score=row.home_score,
        away_score=row.away_score,
        period=row.period,
        is_neutral_site=row.is_neutral_site or False,
        home_team_record=row.home_team_record,
        away_team_record=row.away_team_record,
        weather=row.weather_json if hasattr(row, 'weather_json') else None,
        broadcast=row.broadcast if hasattr(row, 'broadcast') else None,
        attendance=row.attendance if hasattr(row, 'attendance') else None,
        features=features,
        predictions_count=row.predictions_count or 0
    )


@router.get("/{game_id}/features", response_model=GameFeatures)
async def get_game_features(
    game_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Get computed features for a game used in ML predictions.
    """
    query = """
        SELECT 
            gf.*,
            g.sport_code
        FROM game_features gf
        JOIN games g ON gf.game_id = g.id
        WHERE gf.game_id = :game_id
    """
    
    result = await db.execute(query, {"game_id": game_id})
    row = result.fetchone()
    
    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Features for game {game_id} not found"
        )
    
    return GameFeatures(
        game_id=row.game_id,
        sport_code=row.sport_code,
        home_elo=row.home_elo,
        away_elo=row.away_elo,
        home_offensive_rating=row.home_offensive_rating,
        home_defensive_rating=row.home_defensive_rating,
        away_offensive_rating=row.away_offensive_rating,
        away_defensive_rating=row.away_defensive_rating,
        home_rest_days=row.home_rest_days,
        away_rest_days=row.away_rest_days,
        home_b2b=row.home_b2b,
        away_b2b=row.away_b2b,
        home_win_streak=row.home_win_streak,
        away_win_streak=row.away_win_streak,
        h2h_home_wins=row.h2h_home_wins,
        h2h_away_wins=row.h2h_away_wins,
        home_last5_wins=row.home_last5_wins,
        away_last5_wins=row.away_last5_wins,
        spread_movement=row.spread_movement,
        total_movement=row.total_movement,
        computed_at=row.computed_at
    )


@router.get("/sport/{sport_code}", response_model=List[GameBase])
async def get_games_by_sport(
    sport_code: str,
    days: int = Query(7, ge=1, le=90, description="Days to look back"),
    status: Optional[str] = Query(None, description="Filter by status"),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Get games for a specific sport.
    """
    valid_sports = ["NFL", "NCAAF", "CFL", "NBA", "NCAAB", "WNBA", "NHL", "MLB", "ATP", "WTA"]
    sport_upper = sport_code.upper()
    
    if sport_upper not in valid_sports:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid sport code. Valid codes: {', '.join(valid_sports)}"
        )
    
    from_date = date.today() - timedelta(days=days)
    
    conditions = [
        "g.sport_code = :sport_code",
        "DATE(g.game_date) >= :from_date"
    ]
    params = {"sport_code": sport_upper, "from_date": from_date}
    
    if status:
        conditions.append("g.status = :status")
        params["status"] = status
    
    where_clause = " AND ".join(conditions)
    
    query = f"""
        SELECT 
            g.*,
            ht.id as home_id, ht.name as home_name, ht.abbreviation as home_abbr, ht.elo_rating as home_elo,
            at.id as away_id, at.name as away_name, at.abbreviation as away_abbr, at.elo_rating as away_elo
        FROM games g
        JOIN teams ht ON g.home_team_id = ht.id
        JOIN teams at ON g.away_team_id = at.id
        WHERE {where_clause}
        ORDER BY g.game_date DESC
        LIMIT 500
    """
    
    result = await db.execute(query, params)
    rows = result.fetchall()
    
    return [
        GameBase(
            id=row.id,
            external_id=row.external_id,
            sport_code=row.sport_code,
            home_team=TeamInfo(
                id=row.home_id,
                name=row.home_name,
                abbreviation=row.home_abbr,
                elo_rating=row.home_elo
            ),
            away_team=TeamInfo(
                id=row.away_id,
                name=row.away_name,
                abbreviation=row.away_abbr,
                elo_rating=row.away_elo
            ),
            game_date=row.game_date,
            venue=row.venue,
            status=row.status,
            home_score=row.home_score,
            away_score=row.away_score,
            period=row.period,
            is_neutral_site=row.is_neutral_site or False
        )
        for row in rows
    ]


@router.post("/refresh")
async def refresh_games(
    sport: Optional[str] = Query(None, description="Sport code to refresh"),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Trigger refresh of game data from external sources.
    Requires admin role.
    """
    if current_user.get("role") not in ["admin", "system"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    from app.services.collectors.espn_collector import espn_collector
    
    try:
        if sport:
            result = await espn_collector.collect_games(sport_code=sport.upper())
        else:
            result = await espn_collector.collect_all_games()
        
        # Clear cache
        await cache_manager.delete_pattern("games:*")
        
        return {
            "status": "success",
            "games_updated": result.get("updated", 0),
            "games_created": result.get("created", 0),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to refresh games: {str(e)}"
        )


@router.post("/{game_id}/compute-features")
async def compute_game_features(
    game_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Compute or recompute features for a specific game.
    Requires admin role.
    """
    if current_user.get("role") not in ["admin", "system"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    # Verify game exists
    game_query = "SELECT id, sport_code FROM games WHERE id = :game_id"
    game_result = await db.execute(game_query, {"game_id": game_id})
    game = game_result.fetchone()
    
    if not game:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Game {game_id} not found"
        )
    
    from app.services.ml.feature_engineering import feature_engineer
    
    try:
        features = await feature_engineer.compute_features(
            game_id=game_id,
            sport_code=game.sport_code
        )
        
        return {
            "status": "success",
            "game_id": game_id,
            "features_computed": len(features),
            "computed_at": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to compute features: {str(e)}"
        )
