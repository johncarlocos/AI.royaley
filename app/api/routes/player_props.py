"""
LOYALEY - Player Props API Routes
Enterprise-grade player props predictions endpoints
"""

from datetime import datetime, date, timedelta
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.api.dependencies import get_current_user
from app.core.cache import cache_manager


router = APIRouter(tags=["player-props"])


# ============================================================================
# SCHEMAS
# ============================================================================

class PlayerInfo(BaseModel):
    id: int
    external_id: str
    name: str
    team_id: int
    team_name: str
    team_abbreviation: str
    position: Optional[str] = None
    sport_code: str


class PropPrediction(BaseModel):
    id: int
    player_id: int
    player_name: str
    game_id: int
    prop_type: str  # points, rebounds, assists, etc.
    line: float
    predicted_value: float
    over_probability: float
    under_probability: float
    edge: float
    signal_tier: str
    recommended_side: str  # over or under
    confidence: float
    model_id: int
    locked_at: datetime
    is_graded: bool = False
    result: Optional[str] = None
    actual_value: Optional[float] = None


class PropDetail(PropPrediction):
    sport_code: str
    home_team: str
    away_team: str
    game_date: datetime
    player_season_avg: Optional[float] = None
    player_last5_avg: Optional[float] = None
    opponent_rank: Optional[int] = None
    factors: List[dict] = []


class PropListResponse(BaseModel):
    props: List[PropPrediction]
    total: int
    page: int
    per_page: int


class PropStats(BaseModel):
    total_props: int
    graded_props: int
    win_rate: float
    roi: float
    by_prop_type: dict
    by_sport: dict
    by_tier: dict


class PlayerSeasonStats(BaseModel):
    player_id: int
    player_name: str
    sport_code: str
    season: str
    games_played: int
    stats: dict


class PropTypeInfo(BaseModel):
    prop_type: str
    display_name: str
    sport_codes: List[str]
    description: str


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.get("", response_model=PropListResponse)
async def get_player_props(
    sport: Optional[str] = Query(None, description="Filter by sport code"),
    prop_type: Optional[str] = Query(None, description="Filter by prop type"),
    signal_tier: Optional[str] = Query(None, description="Filter by signal tier"),
    player_id: Optional[int] = Query(None, description="Filter by player ID"),
    date_from: Optional[date] = Query(None),
    date_to: Optional[date] = Query(None),
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Get player prop predictions with filtering and pagination.
    """
    conditions = []
    params = {}
    
    if sport:
        conditions.append("g.sport_code = :sport")
        params["sport"] = sport.upper()
    if prop_type:
        conditions.append("pp.prop_type = :prop_type")
        params["prop_type"] = prop_type
    if signal_tier:
        conditions.append("pp.signal_tier = :signal_tier")
        params["signal_tier"] = signal_tier
    if player_id:
        conditions.append("pp.player_id = :player_id")
        params["player_id"] = player_id
    if date_from:
        conditions.append("DATE(pp.locked_at) >= :date_from")
        params["date_from"] = date_from
    if date_to:
        conditions.append("DATE(pp.locked_at) <= :date_to")
        params["date_to"] = date_to
    
    where_clause = " AND ".join(conditions) if conditions else "1=1"
    
    # Count total
    count_query = f"""
        SELECT COUNT(*) FROM player_props pp
        JOIN games g ON pp.game_id = g.id
        WHERE {where_clause}
    """
    count_result = await db.execute(count_query, params)
    total = count_result.scalar() or 0
    
    # Get paginated results
    offset = (page - 1) * per_page
    params["limit"] = per_page
    params["offset"] = offset
    
    query = f"""
        SELECT 
            pp.*,
            p.name as player_name
        FROM player_props pp
        JOIN players p ON pp.player_id = p.id
        JOIN games g ON pp.game_id = g.id
        WHERE {where_clause}
        ORDER BY pp.locked_at DESC
        LIMIT :limit OFFSET :offset
    """
    
    result = await db.execute(query, params)
    rows = result.fetchall()
    
    props = [
        PropPrediction(
            id=row.id,
            player_id=row.player_id,
            player_name=row.player_name,
            game_id=row.game_id,
            prop_type=row.prop_type,
            line=row.line,
            predicted_value=row.predicted_value,
            over_probability=row.over_probability,
            under_probability=row.under_probability,
            edge=row.edge,
            signal_tier=row.signal_tier,
            recommended_side=row.recommended_side,
            confidence=row.confidence,
            model_id=row.model_id,
            locked_at=row.locked_at,
            is_graded=row.is_graded or False,
            result=row.result,
            actual_value=row.actual_value
        )
        for row in rows
    ]
    
    return PropListResponse(
        props=props,
        total=total,
        page=page,
        per_page=per_page
    )


@router.get("/game/{game_id}", response_model=List[PropPrediction])
async def get_props_by_game(
    game_id: int,
    prop_type: Optional[str] = Query(None),
    signal_tier: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Get all player prop predictions for a specific game.
    """
    conditions = ["pp.game_id = :game_id"]
    params = {"game_id": game_id}
    
    if prop_type:
        conditions.append("pp.prop_type = :prop_type")
        params["prop_type"] = prop_type
    if signal_tier:
        conditions.append("pp.signal_tier = :signal_tier")
        params["signal_tier"] = signal_tier
    
    where_clause = " AND ".join(conditions)
    
    query = f"""
        SELECT 
            pp.*,
            p.name as player_name
        FROM player_props pp
        JOIN players p ON pp.player_id = p.id
        WHERE {where_clause}
        ORDER BY pp.edge DESC
    """
    
    result = await db.execute(query, params)
    rows = result.fetchall()
    
    return [
        PropPrediction(
            id=row.id,
            player_id=row.player_id,
            player_name=row.player_name,
            game_id=row.game_id,
            prop_type=row.prop_type,
            line=row.line,
            predicted_value=row.predicted_value,
            over_probability=row.over_probability,
            under_probability=row.under_probability,
            edge=row.edge,
            signal_tier=row.signal_tier,
            recommended_side=row.recommended_side,
            confidence=row.confidence,
            model_id=row.model_id,
            locked_at=row.locked_at,
            is_graded=row.is_graded or False,
            result=row.result,
            actual_value=row.actual_value
        )
        for row in rows
    ]


@router.get("/player/{player_id}", response_model=List[PropPrediction])
async def get_props_by_player(
    player_id: int,
    days: int = Query(30, ge=1, le=90),
    prop_type: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Get player prop prediction history for a specific player.
    """
    from_date = date.today() - timedelta(days=days)
    
    conditions = [
        "pp.player_id = :player_id",
        "DATE(pp.locked_at) >= :from_date"
    ]
    params = {"player_id": player_id, "from_date": from_date}
    
    if prop_type:
        conditions.append("pp.prop_type = :prop_type")
        params["prop_type"] = prop_type
    
    where_clause = " AND ".join(conditions)
    
    query = f"""
        SELECT 
            pp.*,
            p.name as player_name
        FROM player_props pp
        JOIN players p ON pp.player_id = p.id
        WHERE {where_clause}
        ORDER BY pp.locked_at DESC
        LIMIT 100
    """
    
    result = await db.execute(query, params)
    rows = result.fetchall()
    
    return [
        PropPrediction(
            id=row.id,
            player_id=row.player_id,
            player_name=row.player_name,
            game_id=row.game_id,
            prop_type=row.prop_type,
            line=row.line,
            predicted_value=row.predicted_value,
            over_probability=row.over_probability,
            under_probability=row.under_probability,
            edge=row.edge,
            signal_tier=row.signal_tier,
            recommended_side=row.recommended_side,
            confidence=row.confidence,
            model_id=row.model_id,
            locked_at=row.locked_at,
            is_graded=row.is_graded or False,
            result=row.result,
            actual_value=row.actual_value
        )
        for row in rows
    ]


@router.get("/today", response_model=List[PropPrediction])
async def get_todays_props(
    sport: Optional[str] = Query(None),
    signal_tier: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Get today's player prop predictions.
    """
    cache_key = f"props:today:{sport or 'all'}:{signal_tier or 'all'}"
    
    cached = await cache_manager.get(cache_key)
    if cached:
        return cached
    
    today = date.today()
    
    conditions = ["DATE(g.game_date) = :today"]
    params = {"today": today}
    
    if sport:
        conditions.append("g.sport_code = :sport")
        params["sport"] = sport.upper()
    if signal_tier:
        conditions.append("pp.signal_tier = :signal_tier")
        params["signal_tier"] = signal_tier
    
    where_clause = " AND ".join(conditions)
    
    query = f"""
        SELECT 
            pp.*,
            p.name as player_name
        FROM player_props pp
        JOIN players p ON pp.player_id = p.id
        JOIN games g ON pp.game_id = g.id
        WHERE {where_clause}
        ORDER BY pp.edge DESC
    """
    
    result = await db.execute(query, params)
    rows = result.fetchall()
    
    props = [
        PropPrediction(
            id=row.id,
            player_id=row.player_id,
            player_name=row.player_name,
            game_id=row.game_id,
            prop_type=row.prop_type,
            line=row.line,
            predicted_value=row.predicted_value,
            over_probability=row.over_probability,
            under_probability=row.under_probability,
            edge=row.edge,
            signal_tier=row.signal_tier,
            recommended_side=row.recommended_side,
            confidence=row.confidence,
            model_id=row.model_id,
            locked_at=row.locked_at,
            is_graded=row.is_graded or False,
            result=row.result,
            actual_value=row.actual_value
        )
        for row in rows
    ]
    
    # Cache for 5 minutes
    await cache_manager.set(cache_key, [p.dict() for p in props], ttl=300)
    
    return props


@router.get("/{prop_id}", response_model=PropDetail)
async def get_prop_detail(
    prop_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Get detailed information about a specific player prop prediction.
    """
    query = """
        SELECT 
            pp.*,
            p.name as player_name,
            g.sport_code,
            g.game_date,
            ht.name as home_team,
            at.name as away_team
        FROM player_props pp
        JOIN players p ON pp.player_id = p.id
        JOIN games g ON pp.game_id = g.id
        JOIN teams ht ON g.home_team_id = ht.id
        JOIN teams at ON g.away_team_id = at.id
        WHERE pp.id = :prop_id
    """
    
    result = await db.execute(query, {"prop_id": prop_id})
    row = result.fetchone()
    
    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Player prop {prop_id} not found"
        )
    
    # Get player averages
    avg_query = """
        SELECT 
            AVG(value) as season_avg
        FROM player_game_stats
        WHERE player_id = :player_id AND stat_type = :prop_type
    """
    
    avg_result = await db.execute(avg_query, {
        "player_id": row.player_id,
        "prop_type": row.prop_type
    })
    avg_row = avg_result.fetchone()
    
    # Get last 5 games average
    last5_query = """
        SELECT AVG(value) as last5_avg
        FROM (
            SELECT value FROM player_game_stats
            WHERE player_id = :player_id AND stat_type = :prop_type
            ORDER BY game_id DESC
            LIMIT 5
        ) sub
    """
    
    last5_result = await db.execute(last5_query, {
        "player_id": row.player_id,
        "prop_type": row.prop_type
    })
    last5_row = last5_result.fetchone()
    
    # Parse factors
    factors = []
    if row.factors_json:
        import json
        try:
            factors = json.loads(row.factors_json) if isinstance(row.factors_json, str) else row.factors_json
        except json.JSONDecodeError:
            pass
    
    return PropDetail(
        id=row.id,
        player_id=row.player_id,
        player_name=row.player_name,
        game_id=row.game_id,
        prop_type=row.prop_type,
        line=row.line,
        predicted_value=row.predicted_value,
        over_probability=row.over_probability,
        under_probability=row.under_probability,
        edge=row.edge,
        signal_tier=row.signal_tier,
        recommended_side=row.recommended_side,
        confidence=row.confidence,
        model_id=row.model_id,
        locked_at=row.locked_at,
        is_graded=row.is_graded or False,
        result=row.result,
        actual_value=row.actual_value,
        sport_code=row.sport_code,
        home_team=row.home_team,
        away_team=row.away_team,
        game_date=row.game_date,
        player_season_avg=avg_row.season_avg if avg_row else None,
        player_last5_avg=last5_row.last5_avg if last5_row else None,
        opponent_rank=row.opponent_rank if hasattr(row, 'opponent_rank') else None,
        factors=factors
    )


@router.get("/stats", response_model=PropStats)
async def get_prop_stats(
    days: int = Query(30, ge=1, le=365),
    sport: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Get player props statistics and performance metrics.
    """
    from_date = date.today() - timedelta(days=days)
    
    conditions = ["DATE(pp.locked_at) >= :from_date"]
    params = {"from_date": from_date}
    
    if sport:
        conditions.append("g.sport_code = :sport")
        params["sport"] = sport.upper()
    
    where_clause = " AND ".join(conditions)
    
    # Overall stats
    overall_query = f"""
        SELECT 
            COUNT(*) as total_props,
            SUM(CASE WHEN pp.is_graded THEN 1 ELSE 0 END) as graded_props,
            AVG(CASE WHEN pp.is_graded AND pp.result = 'win' THEN 1.0 
                     WHEN pp.is_graded THEN 0.0 END) as win_rate
        FROM player_props pp
        JOIN games g ON pp.game_id = g.id
        WHERE {where_clause}
    """
    
    overall_result = await db.execute(overall_query, params)
    overall = overall_result.fetchone()
    
    # By prop type
    type_query = f"""
        SELECT 
            pp.prop_type,
            COUNT(*) as total,
            AVG(CASE WHEN pp.is_graded AND pp.result = 'win' THEN 1.0 
                     WHEN pp.is_graded THEN 0.0 END) as win_rate
        FROM player_props pp
        JOIN games g ON pp.game_id = g.id
        WHERE {where_clause}
        GROUP BY pp.prop_type
    """
    
    type_result = await db.execute(type_query, params)
    by_prop_type = {
        row.prop_type: {
            "total": row.total,
            "win_rate": round((row.win_rate or 0) * 100, 2)
        }
        for row in type_result.fetchall()
    }
    
    # By sport
    sport_query = f"""
        SELECT 
            g.sport_code,
            COUNT(*) as total,
            AVG(CASE WHEN pp.is_graded AND pp.result = 'win' THEN 1.0 
                     WHEN pp.is_graded THEN 0.0 END) as win_rate
        FROM player_props pp
        JOIN games g ON pp.game_id = g.id
        WHERE {where_clause}
        GROUP BY g.sport_code
    """
    
    sport_result = await db.execute(sport_query, params)
    by_sport = {
        row.sport_code: {
            "total": row.total,
            "win_rate": round((row.win_rate or 0) * 100, 2)
        }
        for row in sport_result.fetchall()
    }
    
    # By tier
    tier_query = f"""
        SELECT 
            pp.signal_tier,
            COUNT(*) as total,
            AVG(CASE WHEN pp.is_graded AND pp.result = 'win' THEN 1.0 
                     WHEN pp.is_graded THEN 0.0 END) as win_rate
        FROM player_props pp
        JOIN games g ON pp.game_id = g.id
        WHERE {where_clause}
        GROUP BY pp.signal_tier
    """
    
    tier_result = await db.execute(tier_query, params)
    by_tier = {
        row.signal_tier: {
            "total": row.total,
            "win_rate": round((row.win_rate or 0) * 100, 2)
        }
        for row in tier_result.fetchall()
    }
    
    return PropStats(
        total_props=overall.total_props or 0,
        graded_props=overall.graded_props or 0,
        win_rate=round((overall.win_rate or 0) * 100, 2),
        roi=0,  # Would need bet tracking for props
        by_prop_type=by_prop_type,
        by_sport=by_sport,
        by_tier=by_tier
    )


@router.get("/players/search", response_model=List[PlayerInfo])
async def search_players(
    query: str = Query(..., min_length=2, description="Search query"),
    sport: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=50),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Search for players by name.
    """
    conditions = ["p.name ILIKE :query"]
    params = {"query": f"%{query}%", "limit": limit}
    
    if sport:
        conditions.append("p.sport_code = :sport")
        params["sport"] = sport.upper()
    
    where_clause = " AND ".join(conditions)
    
    search_query = f"""
        SELECT 
            p.*,
            t.name as team_name,
            t.abbreviation as team_abbreviation
        FROM players p
        JOIN teams t ON p.team_id = t.id
        WHERE {where_clause}
        ORDER BY p.name
        LIMIT :limit
    """
    
    result = await db.execute(search_query, params)
    rows = result.fetchall()
    
    return [
        PlayerInfo(
            id=row.id,
            external_id=row.external_id,
            name=row.name,
            team_id=row.team_id,
            team_name=row.team_name,
            team_abbreviation=row.team_abbreviation,
            position=row.position,
            sport_code=row.sport_code
        )
        for row in rows
    ]


@router.get("/players/{player_id}/stats", response_model=PlayerSeasonStats)
async def get_player_season_stats(
    player_id: int,
    season: Optional[str] = Query(None, description="Season (e.g., 2024-25)"),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Get player's season statistics.
    """
    # Get player info
    player_query = """
        SELECT id, name, sport_code FROM players WHERE id = :player_id
    """
    player_result = await db.execute(player_query, {"player_id": player_id})
    player = player_result.fetchone()
    
    if not player:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Player {player_id} not found"
        )
    
    # Determine season
    if not season:
        today = date.today()
        if today.month >= 9:
            season = f"{today.year}-{str(today.year + 1)[2:]}"
        else:
            season = f"{today.year - 1}-{str(today.year)[2:]}"
    
    # Get stats
    stats_query = """
        SELECT 
            stat_type,
            COUNT(*) as games,
            AVG(value) as average,
            MAX(value) as max,
            MIN(value) as min
        FROM player_game_stats pgs
        JOIN games g ON pgs.game_id = g.id
        WHERE pgs.player_id = :player_id AND g.season = :season
        GROUP BY stat_type
    """
    
    stats_result = await db.execute(stats_query, {
        "player_id": player_id,
        "season": season
    })
    stats_rows = stats_result.fetchall()
    
    stats = {}
    games_played = 0
    
    for row in stats_rows:
        stats[row.stat_type] = {
            "average": round(row.average, 1) if row.average else 0,
            "max": row.max,
            "min": row.min
        }
        games_played = max(games_played, row.games)
    
    return PlayerSeasonStats(
        player_id=player.id,
        player_name=player.name,
        sport_code=player.sport_code,
        season=season,
        games_played=games_played,
        stats=stats
    )


@router.get("/prop-types", response_model=List[PropTypeInfo])
async def get_prop_types(
    sport: Optional[str] = Query(None),
    current_user: dict = Depends(get_current_user)
):
    """
    Get available prop types by sport.
    """
    prop_types = [
        PropTypeInfo(
            prop_type="points",
            display_name="Points",
            sport_codes=["NBA", "NCAAB", "WNBA"],
            description="Total points scored by the player"
        ),
        PropTypeInfo(
            prop_type="rebounds",
            display_name="Rebounds",
            sport_codes=["NBA", "NCAAB", "WNBA"],
            description="Total rebounds (offensive + defensive)"
        ),
        PropTypeInfo(
            prop_type="assists",
            display_name="Assists",
            sport_codes=["NBA", "NCAAB", "WNBA"],
            description="Total assists"
        ),
        PropTypeInfo(
            prop_type="pra",
            display_name="Points + Rebounds + Assists",
            sport_codes=["NBA", "NCAAB", "WNBA"],
            description="Combined points, rebounds, and assists"
        ),
        PropTypeInfo(
            prop_type="threes",
            display_name="Three-Pointers Made",
            sport_codes=["NBA", "NCAAB", "WNBA"],
            description="Number of three-point shots made"
        ),
        PropTypeInfo(
            prop_type="steals",
            display_name="Steals",
            sport_codes=["NBA", "NCAAB", "WNBA"],
            description="Total steals"
        ),
        PropTypeInfo(
            prop_type="blocks",
            display_name="Blocks",
            sport_codes=["NBA", "NCAAB", "WNBA"],
            description="Total blocks"
        ),
        PropTypeInfo(
            prop_type="passing_yards",
            display_name="Passing Yards",
            sport_codes=["NFL", "NCAAF", "CFL"],
            description="Total passing yards"
        ),
        PropTypeInfo(
            prop_type="passing_tds",
            display_name="Passing Touchdowns",
            sport_codes=["NFL", "NCAAF", "CFL"],
            description="Number of passing touchdowns"
        ),
        PropTypeInfo(
            prop_type="rushing_yards",
            display_name="Rushing Yards",
            sport_codes=["NFL", "NCAAF", "CFL"],
            description="Total rushing yards"
        ),
        PropTypeInfo(
            prop_type="receiving_yards",
            display_name="Receiving Yards",
            sport_codes=["NFL", "NCAAF", "CFL"],
            description="Total receiving yards"
        ),
        PropTypeInfo(
            prop_type="receptions",
            display_name="Receptions",
            sport_codes=["NFL", "NCAAF", "CFL"],
            description="Number of receptions"
        ),
        PropTypeInfo(
            prop_type="strikeouts",
            display_name="Strikeouts",
            sport_codes=["MLB"],
            description="Pitcher strikeouts"
        ),
        PropTypeInfo(
            prop_type="hits",
            display_name="Hits",
            sport_codes=["MLB"],
            description="Batter hits"
        ),
        PropTypeInfo(
            prop_type="total_bases",
            display_name="Total Bases",
            sport_codes=["MLB"],
            description="Batter total bases"
        ),
        PropTypeInfo(
            prop_type="rbis",
            display_name="RBIs",
            sport_codes=["MLB"],
            description="Runs batted in"
        ),
        PropTypeInfo(
            prop_type="goals",
            display_name="Goals",
            sport_codes=["NHL"],
            description="Goals scored"
        ),
        PropTypeInfo(
            prop_type="hockey_assists",
            display_name="Assists",
            sport_codes=["NHL"],
            description="Hockey assists"
        ),
        PropTypeInfo(
            prop_type="hockey_points",
            display_name="Points",
            sport_codes=["NHL"],
            description="Goals + Assists"
        ),
        PropTypeInfo(
            prop_type="shots_on_goal",
            display_name="Shots on Goal",
            sport_codes=["NHL"],
            description="Total shots on goal"
        )
    ]
    
    if sport:
        sport_upper = sport.upper()
        prop_types = [pt for pt in prop_types if sport_upper in pt.sport_codes]
    
    return prop_types


@router.post("/generate")
async def generate_prop_predictions(
    game_id: Optional[int] = Query(None),
    sport: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Generate player prop predictions for a game or sport.
    Requires admin role.
    """
    if current_user.get("role") not in ["admin", "system"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    from app.services.player_props.predictor import props_predictor
    
    try:
        if game_id:
            result = await props_predictor.generate_for_game(game_id)
        elif sport:
            result = await props_predictor.generate_for_sport(sport.upper())
        else:
            result = await props_predictor.generate_all()
        
        return {
            "status": "success",
            "props_generated": result.get("generated", 0),
            "games_processed": result.get("games", 0),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate props: {str(e)}"
        )
