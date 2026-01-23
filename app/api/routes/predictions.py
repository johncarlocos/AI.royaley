"""
ROYALEY - Predictions API Routes
Enterprise-grade predictions endpoints with filtering, pagination, and SHAP explanations
"""

from datetime import datetime, date, timedelta
from typing import Optional, List
from uuid import UUID, uuid4
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import select, func, and_, or_, text, cast, String
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload, load_only, joinedload, aliased

from app.core.database import get_db
from app.api.dependencies import get_current_user
from app.core.cache import cache_manager, CachePrefix
from app.models import (
    Prediction as DBPrediction, 
    PredictionResult,
    Game, 
    Sport,
    Team,
    SignalTier, 
    User, 
    UserRole,
    GameStatus
)


router = APIRouter(tags=["predictions"])


# ============================================================================
# SCHEMAS
# ============================================================================

class SHAPExplanation(BaseModel):
    feature: str
    value: float
    impact: str  # positive or negative


class PredictionBase(BaseModel):
    id: str  # UUID
    game_id: str  # UUID
    sport_code: Optional[str] = None
    # Frontend-compatible fields
    sport: Optional[str] = None  # Alias for sport_code
    home_team: Optional[str] = None
    away_team: Optional[str] = None
    game_time: Optional[datetime] = None  # Game start time (from game_date)
    line: Optional[float] = None  # Alias for line_at_prediction
    odds: Optional[int] = None  # Alias for odds_at_prediction
    status: Optional[str] = None  # Computed: "pending", "win", "loss", "push"
    # Original fields (kept for backwards compatibility)
    bet_type: str  # spread, moneyline, total
    predicted_side: str
    probability: float
    edge: Optional[float] = None
    signal_tier: Optional[str] = None  # A, B, C, D
    line_at_prediction: Optional[float] = None
    odds_at_prediction: Optional[int] = None
    kelly_fraction: Optional[float] = None
    recommended_bet: Optional[float] = None
    prediction_hash: str
    locked_at: datetime
    model_id: Optional[str] = None  # UUID
    model_version: str = "1.0.0"
    is_graded: bool = False
    result: Optional[str] = None  # win, loss, push
    actual_outcome: Optional[str] = None
    profit_loss: Optional[float] = None
    clv: Optional[float] = None


class PredictionDetail(PredictionBase):
    home_team: str
    away_team: str
    game_date: datetime
    shap_explanations: List[SHAPExplanation] = []
    closing_line: Optional[float] = None
    closing_odds: Optional[int] = None


class PredictionListResponse(BaseModel):
    predictions: List[PredictionBase]
    total: int
    page: int
    per_page: int
    total_pages: int


class PredictionStats(BaseModel):
    total_predictions: int
    graded_predictions: int
    pending_predictions: int
    win_rate: float
    roi: float
    clv_average: float
    tier_a_count: int
    tier_a_win_rate: float
    tier_b_count: int
    tier_b_win_rate: float
    by_sport: dict
    by_bet_type: dict


class GeneratePredictionsRequest(BaseModel):
    sport_code: Optional[str] = None
    game_ids: Optional[List[int]] = None
    bet_types: List[str] = ["spread", "moneyline", "total"]


class GeneratePredictionsResponse(BaseModel):
    generated_count: int
    predictions: List[PredictionBase]
    errors: List[str] = []


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.get("", response_model=PredictionListResponse)
async def get_predictions(
    sport: Optional[str] = Query(None, description="Filter by sport code"),
    bet_type: Optional[str] = Query(None, description="Filter by bet type"),
    signal_tier: Optional[str] = Query(None, description="Filter by signal tier (A, B, C, D)"),
    is_graded: Optional[bool] = Query(None, description="Filter by graded status"),
    result: Optional[str] = Query(None, description="Filter by result (win, loss, push)"),
    date_from: Optional[date] = Query(None, description="Filter from date"),
    date_to: Optional[date] = Query(None, description="Filter to date"),
    min_probability: Optional[float] = Query(None, ge=0, le=1, description="Minimum probability"),
    min_edge: Optional[float] = Query(None, description="Minimum edge"),
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(50, ge=1, le=100, description="Items per page"),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Get predictions with comprehensive filtering and pagination.
    """
    # Build cache key
    cache_key = f"predictions:{sport}:{bet_type}:{signal_tier}:{is_graded}:{result}:{date_from}:{date_to}:{min_probability}:{min_edge}:{page}:{per_page}"
    
    # Check cache
    cached = await cache_manager.get(cache_key)
    if cached:
        return cached
    
    # Build base query with joins for filtering
    base_query = select(DBPrediction).join(Game).join(Sport)
    
    # Apply filters
    conditions = []
    
    if sport:
        conditions.append(Sport.code == sport.upper())
    if bet_type:
        conditions.append(DBPrediction.bet_type == bet_type)
    if signal_tier:
        # Compare using string value since database column is VARCHAR(1), not PostgreSQL enum
        # Cast the enum column to string for comparison to avoid enum type casting
        try:
            tier_value = signal_tier.upper()
            # Cast the enum column to string and compare with string value
            conditions.append(cast(DBPrediction.signal_tier, String) == tier_value)
        except (ValueError, AttributeError):
            # Invalid tier value, skip this filter
            pass
    if date_from:
        conditions.append(func.date(DBPrediction.created_at) >= date_from)
    if date_to:
        conditions.append(func.date(DBPrediction.created_at) <= date_to)
    if min_probability is not None:
        conditions.append(DBPrediction.probability >= min_probability)
    if min_edge is not None:
        conditions.append(DBPrediction.edge >= min_edge)
    if is_graded is not None:
        if is_graded:
            conditions.append(DBPrediction.result.has())
        else:
            conditions.append(~DBPrediction.result.has())
    if result:
        conditions.append(DBPrediction.result.has(PredictionResult.actual_result == result))
    
    if conditions:
        base_query = base_query.where(and_(*conditions))
    
    # Get total count - count IDs only (avoids selecting columns that don't exist in DB)
    count_query = select(func.count(DBPrediction.id)).select_from(DBPrediction).join(Game).join(Sport)
    if conditions:
        count_query = count_query.where(and_(*conditions))
    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0
    
    # Apply pagination and ordering to data query
    offset = (page - 1) * per_page
    data_query = base_query.options(
        load_only(
            DBPrediction.id,
            DBPrediction.game_id,
            DBPrediction.model_id,
            DBPrediction.bet_type,
            DBPrediction.predicted_side,
            DBPrediction.probability,
            DBPrediction.line_at_prediction,
            DBPrediction.odds_at_prediction,
            DBPrediction.edge,
            DBPrediction.signal_tier,
            DBPrediction.kelly_fraction,
            DBPrediction.prediction_hash,
            DBPrediction.created_at
        ),
        selectinload(DBPrediction.game),
        selectinload(DBPrediction.result).load_only(
            PredictionResult.id,
            PredictionResult.prediction_id,
            PredictionResult.actual_result,
            PredictionResult.closing_line,
            PredictionResult.clv,
            PredictionResult.profit_loss,
            PredictionResult.graded_at
        )
    ).order_by(DBPrediction.created_at.desc()).limit(per_page).offset(offset)
    
    # Execute query
    result = await db.execute(data_query)
    predictions_list = result.scalars().all()
    
    # Get sport codes and team names for all games
    game_ids = [pred.game_id for pred in predictions_list if pred.game_id]
    sport_codes_map = {}
    team_names_map = {}  # game_id -> {"home_team": name, "away_team": name, "game_date": datetime}
    if game_ids:
        # Get sport codes
        sport_query = select(Game.id, Sport.code).join(Sport, Game.sport_id == Sport.id).where(Game.id.in_(game_ids))
        sport_result = await db.execute(sport_query)
        sport_codes_map = {row.id: row.code for row in sport_result.all()}
        
        # Get team names and game dates using raw SQL to avoid relationship loading issues
        team_query = text("""
            SELECT 
                g.id,
                g.game_date,
                ht.name as home_team_name,
                at.name as away_team_name
            FROM games g
            JOIN teams ht ON g.home_team_id = ht.id
            JOIN teams at ON g.away_team_id = at.id
            WHERE g.id = ANY(:game_ids)
        """)
        team_result = await db.execute(team_query, {"game_ids": [str(gid) for gid in game_ids]})
        for row in team_result:
            team_names_map[UUID(str(row.id))] = {
                "home_team": row.home_team_name,
                "away_team": row.away_team_name,
                "game_date": row.game_date
            }
    
    # Convert to response models
    predictions = []
    for pred in predictions_list:
        sport_code = sport_codes_map.get(pred.game_id) if pred.game_id else None
        pred_result = pred.result
        game_info = team_names_map.get(pred.game_id, {}) if pred.game_id else {}
        
        # Compute status from is_graded and result
        status = "pending"
        if pred_result is not None and pred_result.actual_result:
            status = pred_result.actual_result.value  # "win", "loss", or "push"
        
        predictions.append(
            PredictionBase(
                id=str(pred.id),
                game_id=str(pred.game_id),
                sport_code=sport_code,
                sport=sport_code,  # Frontend-compatible field
                home_team=game_info.get("home_team"),
                away_team=game_info.get("away_team"),
                game_time=game_info.get("game_date"),
                line=pred.line_at_prediction,  # Frontend-compatible field
                odds=pred.odds_at_prediction,  # Frontend-compatible field
                status=status,  # Computed status
                bet_type=pred.bet_type,
                predicted_side=pred.predicted_side,
                probability=pred.probability,
                edge=pred.edge,
                signal_tier=pred.signal_tier.value if pred.signal_tier else None,
                line_at_prediction=pred.line_at_prediction,
                odds_at_prediction=pred.odds_at_prediction,
                kelly_fraction=pred.kelly_fraction,
                recommended_bet=None,  # recommended_bet_size column doesn't exist in database
                prediction_hash=pred.prediction_hash,
                locked_at=pred.created_at,  # Using created_at as locked_at
                model_id=str(pred.model_id) if pred.model_id else None,
                model_version="1.0.0",  # TODO: Get from model relationship
                is_graded=pred_result is not None,
                result=pred_result.actual_result.value if pred_result and pred_result.actual_result else None,
                actual_outcome=pred_result.actual_result.value if pred_result and pred_result.actual_result else None,
                profit_loss=pred_result.profit_loss if pred_result else None,
                clv=pred_result.clv if pred_result else None
            )
        )
    
    total_pages = (total + per_page - 1) // per_page
    
    response = PredictionListResponse(
        predictions=predictions,
        total=total,
        page=page,
        per_page=per_page,
        total_pages=total_pages
    )
    
    # Cache for 60 seconds
    await cache_manager.set(cache_key, response.dict(), ttl=60)
    
    return response


@router.get("/today", response_model=List[PredictionBase])
async def get_todays_predictions(
    sport: Optional[str] = Query(None, description="Filter by sport code"),
    signal_tier: Optional[str] = Query(None, description="Filter by signal tier"),
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Get today's predictions with optional sport and tier filtering.
    """
    # Check if demo user - return empty list for demo mode
    if hasattr(current_user, 'id') and str(current_user.id) == "00000000-0000-0000-0000-000000000000":
        return []
    
    cache_key = f"predictions:today:{sport}:{signal_tier}"
    
    cached = await cache_manager.get(cache_key)
    if cached:
        return cached
    
    try:
        today = date.today()
        
        # Build base query with joins for filtering
        base_query = select(DBPrediction).join(Game).join(Sport)
        
        # Apply filters
        conditions = []
        conditions.append(func.date(DBPrediction.created_at) == today)
        
        if sport:
            conditions.append(Sport.code == sport.upper())
        if signal_tier:
            # Compare using string value since database column is VARCHAR(1), not PostgreSQL enum
            try:
                tier_value = signal_tier.upper()
                conditions.append(cast(DBPrediction.signal_tier, String) == tier_value)
            except (ValueError, AttributeError):
                # Invalid tier value, skip this filter
                pass
        
        if conditions:
            base_query = base_query.where(and_(*conditions))
        
        # Execute query with eager loading
        data_query = base_query.options(
            load_only(
                DBPrediction.id,
                DBPrediction.game_id,
                DBPrediction.model_id,
                DBPrediction.bet_type,
                DBPrediction.predicted_side,
                DBPrediction.probability,
                DBPrediction.line_at_prediction,
                DBPrediction.odds_at_prediction,
                DBPrediction.edge,
                DBPrediction.signal_tier,
                DBPrediction.kelly_fraction,
                DBPrediction.prediction_hash,
                DBPrediction.created_at
            ),
            selectinload(DBPrediction.game),
            selectinload(DBPrediction.result).load_only(
                PredictionResult.id,
                PredictionResult.prediction_id,
                PredictionResult.actual_result,
                PredictionResult.closing_line,
                PredictionResult.clv,
                PredictionResult.profit_loss,
                PredictionResult.graded_at
            )
        ).order_by(DBPrediction.probability.desc())
        
        result = await db.execute(data_query)
        predictions_list = result.scalars().all()
        
        # Get sport codes and team names for all games
        game_ids = [pred.game_id for pred in predictions_list if pred.game_id]
        sport_codes_map = {}
        team_names_map = {}  # game_id -> {"home_team": name, "away_team": name, "game_date": datetime}
        if game_ids:
            # Get sport codes
            sport_query = select(Game.id, Sport.code).join(Sport, Game.sport_id == Sport.id).where(Game.id.in_(game_ids))
            sport_result = await db.execute(sport_query)
            sport_codes_map = {row.id: row.code for row in sport_result.all()}
            
            # Get team names and game dates using raw SQL to avoid relationship loading issues
            team_query = text("""
                SELECT 
                    g.id,
                    g.game_date,
                    ht.name as home_team_name,
                    at.name as away_team_name
                FROM games g
                JOIN teams ht ON g.home_team_id = ht.id
                JOIN teams at ON g.away_team_id = at.id
                WHERE g.id = ANY(:game_ids)
            """)
            team_result = await db.execute(team_query, {"game_ids": [str(gid) for gid in game_ids]})
            for row in team_result:
                team_names_map[UUID(str(row.id))] = {
                    "home_team": row.home_team_name,
                    "away_team": row.away_team_name,
                    "game_date": row.game_date
                }
        
        # Convert to response models
        predictions = []
        for pred in predictions_list:
            sport_code = sport_codes_map.get(pred.game_id) if pred.game_id else None
            pred_result = pred.result
            game_info = team_names_map.get(pred.game_id, {}) if pred.game_id else {}
            
            # Compute status from is_graded and result
            status = "pending"
            if pred_result is not None and pred_result.actual_result:
                status = pred_result.actual_result.value  # "win", "loss", or "push"
            
            predictions.append(
                PredictionBase(
                    id=str(pred.id),
                    game_id=str(pred.game_id),
                    sport_code=sport_code,
                    sport=sport_code,  # Frontend-compatible field
                    home_team=game_info.get("home_team"),
                    away_team=game_info.get("away_team"),
                    game_time=game_info.get("game_date"),
                    line=pred.line_at_prediction,  # Frontend-compatible field
                    odds=pred.odds_at_prediction,  # Frontend-compatible field
                    status=status,  # Computed status
                    bet_type=pred.bet_type,
                    predicted_side=pred.predicted_side,
                    probability=pred.probability,
                    edge=pred.edge,
                    signal_tier=pred.signal_tier.value if pred.signal_tier else None,
                    line_at_prediction=pred.line_at_prediction,
                    odds_at_prediction=pred.odds_at_prediction,
                    kelly_fraction=pred.kelly_fraction,
                    recommended_bet=None,  # recommended_bet_size column doesn't exist in database
                    prediction_hash=pred.prediction_hash,
                    locked_at=pred.created_at,  # Using created_at as locked_at
                    model_id=str(pred.model_id) if pred.model_id else None,
                    model_version="1.0.0",  # TODO: Get from model relationship
                    is_graded=pred_result is not None,
                    result=pred_result.actual_result.value if pred_result and pred_result.actual_result else None,
                    actual_outcome=pred_result.actual_result.value if pred_result and pred_result.actual_result else None,
                    profit_loss=pred_result.profit_loss if pred_result else None,
                    clv=pred_result.clv if pred_result else None
                )
            )
        
        # Cache for 5 minutes
        await cache_manager.set(cache_key, [p.dict() for p in predictions], ttl=300)
        
        return predictions
    except Exception:
        # If database error, return empty list for demo mode
        if hasattr(current_user, 'id') and str(current_user.id) == "00000000-0000-0000-0000-000000000000":
            return []
        raise


@router.get("/sport/{sport_code}", response_model=List[PredictionBase])
async def get_predictions_by_sport(
    sport_code: str,
    days: int = Query(7, ge=1, le=90, description="Number of days to look back"),
    signal_tier: Optional[str] = Query(None, description="Filter by signal tier"),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Get predictions for a specific sport.
    """
    valid_sports = ["NFL", "NCAAF", "CFL", "NBA", "NCAAB", "WNBA", "NHL", "MLB", "ATP", "WTA"]
    if sport_code.upper() not in valid_sports:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid sport code. Valid codes: {', '.join(valid_sports)}"
        )
    
    from_date = date.today() - timedelta(days=days)
    
    conditions = [
        "sport_code = :sport_code",
        "DATE(created_at) >= :from_date"
    ]
    params = {"sport_code": sport_code.upper(), "from_date": from_date}
    
    if signal_tier:
        conditions.append("signal_tier = :signal_tier")
        params["signal_tier"] = signal_tier
    
    where_clause = " AND ".join(conditions)
    
    query = text(f"""
        SELECT 
            p.*,
            m.version as model_version
        FROM predictions p
        LEFT JOIN ml_models m ON p.model_id = m.id
        WHERE {where_clause}
        ORDER BY p.created_at DESC
        LIMIT 500
    """)
    
    result = await db.execute(query, params)
    rows = result.fetchall()
    
    return [
        PredictionBase(
            id=row.id,
            game_id=row.game_id,
            sport_code=row.sport_code,
            bet_type=row.bet_type,
            predicted_side=row.predicted_side,
            probability=row.probability,
            edge=row.edge,
            signal_tier=row.signal_tier,
            line_at_prediction=row.line_at_prediction,
            odds_at_prediction=row.odds_at_prediction,
            kelly_fraction=row.kelly_fraction,
            recommended_bet=row.recommended_bet,
            prediction_hash=row.prediction_hash,
            locked_at=row.created_at,  # Using created_at as locked_at since locked_at column doesn't exist
            model_id=row.model_id,
            model_version=row.model_version or "1.0.0",
            is_graded=row.is_graded,
            result=row.result,
            actual_outcome=row.actual_outcome,
            profit_loss=row.profit_loss,
            clv=row.clv
        )
        for row in rows
    ]


@router.get("/{prediction_id}", response_model=PredictionDetail)
async def get_prediction_detail(
    prediction_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Get detailed information about a specific prediction including SHAP explanations.
    """
    query = text("""
        SELECT 
            p.*,
            m.version as model_version,
            g.home_team_id,
            g.away_team_id,
            g.game_date,
            ht.name as home_team,
            at.name as away_team,
            cl.closing_spread,
            cl.closing_total,
            cl.closing_home_ml,
            cl.closing_away_ml
        FROM predictions p
        LEFT JOIN ml_models m ON p.model_id = m.id
        LEFT JOIN games g ON p.game_id = g.id
        LEFT JOIN teams ht ON g.home_team_id = ht.id
        LEFT JOIN teams at ON g.away_team_id = at.id
        LEFT JOIN closing_lines cl ON p.game_id = cl.game_id
        WHERE p.id = :prediction_id
    """)
    
    result = await db.execute(query, {"prediction_id": prediction_id})
    row = result.fetchone()
    
    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prediction {prediction_id} not found"
        )
    
    # Parse SHAP explanations from JSON
    shap_explanations = []
    if row.shap_values:
        import json
        try:
            shap_data = json.loads(row.shap_values) if isinstance(row.shap_values, str) else row.shap_values
            shap_explanations = [
                SHAPExplanation(
                    feature=item["feature"],
                    value=item["value"],
                    impact="positive" if item["value"] > 0 else "negative"
                )
                for item in shap_data[:10]  # Top 10 features
            ]
        except (json.JSONDecodeError, KeyError):
            pass
    
    # Determine closing line based on bet type
    closing_line = None
    closing_odds = None
    if row.bet_type == "spread":
        closing_line = row.closing_spread
    elif row.bet_type == "total":
        closing_line = row.closing_total
    elif row.bet_type == "moneyline":
        closing_odds = row.closing_home_ml if row.predicted_side == "home" else row.closing_away_ml
    
    return PredictionDetail(
        id=row.id,
        game_id=row.game_id,
        sport_code=row.sport_code,
        bet_type=row.bet_type,
        predicted_side=row.predicted_side,
        probability=row.probability,
        edge=row.edge,
        signal_tier=row.signal_tier,
        line_at_prediction=row.line_at_prediction,
        odds_at_prediction=row.odds_at_prediction,
        kelly_fraction=row.kelly_fraction,
        recommended_bet=row.recommended_bet,
        prediction_hash=row.prediction_hash,
        locked_at=row.created_at,  # Using created_at as locked_at since locked_at column doesn't exist
        model_id=row.model_id,
        model_version=row.model_version or "1.0.0",
        is_graded=row.is_graded,
        result=row.result,
        actual_outcome=row.actual_outcome,
        profit_loss=row.profit_loss,
        clv=row.clv,
        home_team=row.home_team or "Unknown",
        away_team=row.away_team or "Unknown",
        game_date=row.game_date,
        shap_explanations=shap_explanations,
        closing_line=closing_line,
        closing_odds=closing_odds
    )


@router.get("/stats", response_model=PredictionStats)
async def get_prediction_stats(
    sport: Optional[str] = Query(None, description="Filter by sport code"),
    days: int = Query(30, ge=1, le=365, description="Number of days to analyze"),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Get prediction statistics and performance metrics.
    """
    cache_key = f"prediction_stats:{sport}:{days}"
    
    cached = await cache_manager.get(cache_key)
    if cached:
        return cached
    
    from_date = date.today() - timedelta(days=days)
    
    sport_condition = "AND sport_code = :sport" if sport else ""
    params = {"from_date": from_date}
    if sport:
        params["sport"] = sport
    
    # Overall stats
    stats_query = text(f"""
        SELECT 
            COUNT(*) as total_predictions,
            SUM(CASE WHEN is_graded THEN 1 ELSE 0 END) as graded_predictions,
            SUM(CASE WHEN NOT is_graded THEN 1 ELSE 0 END) as pending_predictions,
            AVG(CASE WHEN is_graded AND result = 'win' THEN 1.0 WHEN is_graded THEN 0.0 END) as win_rate,
            SUM(COALESCE(profit_loss, 0)) / NULLIF(COUNT(CASE WHEN is_graded THEN 1 END), 0) as roi,
            AVG(clv) as clv_average,
            SUM(CASE WHEN signal_tier = 'A' THEN 1 ELSE 0 END) as tier_a_count,
            AVG(CASE WHEN signal_tier = 'A' AND is_graded AND result = 'win' THEN 1.0 
                     WHEN signal_tier = 'A' AND is_graded THEN 0.0 END) as tier_a_win_rate,
            SUM(CASE WHEN signal_tier = 'B' THEN 1 ELSE 0 END) as tier_b_count,
            AVG(CASE WHEN signal_tier = 'B' AND is_graded AND result = 'win' THEN 1.0 
                     WHEN signal_tier = 'B' AND is_graded THEN 0.0 END) as tier_b_win_rate
        FROM predictions
        WHERE DATE(created_at) >= :from_date {sport_condition}
    """)
    
    result = await db.execute(stats_query, params)
    row = result.fetchone()
    
    # By sport breakdown
    sport_query = text(f"""
        SELECT 
            sport_code,
            COUNT(*) as total,
            AVG(CASE WHEN is_graded AND result = 'win' THEN 1.0 WHEN is_graded THEN 0.0 END) as win_rate,
            AVG(clv) as avg_clv
        FROM predictions
        WHERE DATE(created_at) >= :from_date {sport_condition}
        GROUP BY sport_code
    """)
    
    sport_result = await db.execute(sport_query, params)
    sport_rows = sport_result.fetchall()
    
    by_sport = {
        r.sport_code: {
            "total": r.total,
            "win_rate": round(r.win_rate * 100, 2) if r.win_rate else 0,
            "avg_clv": round(r.avg_clv, 4) if r.avg_clv else 0
        }
        for r in sport_rows
    }
    
    # By bet type breakdown
    bet_type_query = text(f"""
        SELECT 
            bet_type,
            COUNT(*) as total,
            AVG(CASE WHEN is_graded AND result = 'win' THEN 1.0 WHEN is_graded THEN 0.0 END) as win_rate,
            AVG(clv) as avg_clv
        FROM predictions
        WHERE DATE(created_at) >= :from_date {sport_condition}
        GROUP BY bet_type
    """)
    
    bet_result = await db.execute(bet_type_query, params)
    bet_rows = bet_result.fetchall()
    
    by_bet_type = {
        r.bet_type: {
            "total": r.total,
            "win_rate": round(r.win_rate * 100, 2) if r.win_rate else 0,
            "avg_clv": round(r.avg_clv, 4) if r.avg_clv else 0
        }
        for r in bet_rows
    }
    
    stats = PredictionStats(
        total_predictions=row.total_predictions or 0,
        graded_predictions=row.graded_predictions or 0,
        pending_predictions=row.pending_predictions or 0,
        win_rate=round((row.win_rate or 0) * 100, 2),
        roi=round((row.roi or 0) * 100, 2),
        clv_average=round(row.clv_average or 0, 4),
        tier_a_count=row.tier_a_count or 0,
        tier_a_win_rate=round((row.tier_a_win_rate or 0) * 100, 2),
        tier_b_count=row.tier_b_count or 0,
        tier_b_win_rate=round((row.tier_b_win_rate or 0) * 100, 2),
        by_sport=by_sport,
        by_bet_type=by_bet_type
    )
    
    # Cache for 5 minutes
    await cache_manager.set(cache_key, stats.dict(), ttl=300)
    
    return stats


@router.post("/generate", response_model=GeneratePredictionsResponse)
async def generate_predictions(
    request: GeneratePredictionsRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Generate predictions for upcoming games and save them to the database.
    Requires admin role.
    """
    # Check if user has admin or system role
    user_role = current_user.role.value if hasattr(current_user.role, 'value') else str(current_user.role)
    if user_role not in ["admin", "super_admin", "system"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required to generate predictions"
        )
    
    # Import prediction engine types
    from app.services.ml.prediction_engine import (
        create_advanced_prediction_engine,
        BetType,
        OddsInfo,
        FrameworkPrediction,
        ModelFramework,
        PredictedSide,
        SituationalModifiers
    )
    from app.models import Odds as DBOdds
    
    prediction_engine = create_advanced_prediction_engine()
    
    try:
        saved_predictions = []
        saved_count = 0
        errors = []
        
        # Build query to fetch games
        base_query = select(Game).join(Sport, Game.sport_id == Sport.id)
        conditions = []
        
        # Filter by sport_code if provided
        if request.sport_code:
            conditions.append(Sport.code == request.sport_code.upper())
        
        # Filter by game_ids if provided
        if request.game_ids:
            conditions.append(Game.id.in_([UUID(gid) if isinstance(gid, str) else gid for gid in request.game_ids]))
        
        # Only fetch scheduled games that haven't started yet
        conditions.append(Game.status == GameStatus.SCHEDULED)
        # Use naive UTC datetime for comparison with TIMESTAMP WITHOUT TIME ZONE
        now_utc = datetime.utcnow()
        conditions.append(Game.game_date >= now_utc)
        
        if conditions:
            base_query = base_query.where(and_(*conditions))
        
        # Limit to reasonable number of games
        base_query = base_query.limit(100).order_by(Game.game_date.asc())
        
        # Execute query to get games with sport codes
        # Use join to get sport code in one query
        result = await db.execute(
            base_query.options(
                selectinload(Game.home_team), 
                selectinload(Game.away_team)
            )
        )
        games = result.scalars().all()
        
        if not games:
            return GeneratePredictionsResponse(
                generated_count=0,
                predictions=[],
                errors=[f"No eligible games found for prediction generation (sport_code: {request.sport_code}, game_count: 0)"]
            )
        
        # Get sport codes for all games in a single query
        game_sport_ids = {game.id: game.sport_id for game in games}
        sport_ids_list = list(set(game_sport_ids.values()))
        sport_codes_query = select(Sport.id, Sport.code).where(Sport.id.in_(sport_ids_list))
        sport_codes_result = await db.execute(sport_codes_query)
        sport_codes_map = {row.id: row.code for row in sport_codes_result.all()}
        
        # Create game_id to sport_code mapping
        game_sport_codes_map = {game.id: sport_codes_map.get(game.sport_id, "NBA") for game in games}
        
        # Collect all predictions before saving (store with game_id for later mapping)
        predictions_list = []  # List of tuples: (prediction_obj, game_id)
        
        # Process each game
        for game in games:
            try:
                # Get sport code from the map
                sport_code = sport_codes_map.get(game.sport_id, "NBA")  # Default fallback
                
                # Fetch all current odds for this game
                odds_query = select(DBOdds).where(
                    DBOdds.game_id == game.id,
                    DBOdds.is_current == True
                ).order_by(DBOdds.recorded_at.desc())
                odds_result = await db.execute(odds_query)
                odds_records = odds_result.scalars().all()
                
                if not odds_records:
                    errors.append(f"No current odds found for game {game.id}")
                    continue
                
                # Aggregate odds from multiple rows into OddsInfo
                # Odds are stored as separate rows per market_type/selection
                spread_home_line = None
                spread_home_odds = None
                spread_away_line = None
                spread_away_odds = None
                moneyline_home = None
                moneyline_away = None
                total_line = None
                total_over_odds = None
                total_under_odds = None
                
                for odds_row in odds_records:
                    if odds_row.market_type == 'spread':
                        if odds_row.selection == 'home':
                            spread_home_line = odds_row.line
                            spread_home_odds = odds_row.price
                        elif odds_row.selection == 'away':
                            spread_away_line = odds_row.line
                            spread_away_odds = odds_row.price
                    elif odds_row.market_type == 'moneyline':
                        if odds_row.selection == 'home':
                            moneyline_home = odds_row.price
                        elif odds_row.selection == 'away':
                            moneyline_away = odds_row.price
                    elif odds_row.market_type == 'total':
                        if odds_row.line is not None:
                            total_line = odds_row.line
                        if odds_row.selection == 'over':
                            total_over_odds = odds_row.price
                        elif odds_row.selection == 'under':
                            total_under_odds = odds_row.price
                
                # Build OddsInfo object
                odds_info = OddsInfo(
                    game_id=str(game.id),
                    spread_home=spread_home_line,
                    spread_away=spread_away_line,
                    spread_home_odds=spread_home_odds,
                    spread_away_odds=spread_away_odds,
                    moneyline_home=moneyline_home,
                    moneyline_away=moneyline_away,
                    total_line=total_line,
                    total_over_odds=total_over_odds,
                    total_under_odds=total_under_odds,
                )
                
                # Build features dictionary using basic ELO from teams
                # Note: game_features table doesn't exist in database, so using team ELO directly
                features = {
                    'home_elo': game.home_team.elo_rating if game.home_team and hasattr(game.home_team, 'elo_rating') else 1500,
                    'away_elo': game.away_team.elo_rating if game.away_team and hasattr(game.away_team, 'elo_rating') else 1500,
                }
                
                # Create placeholder framework predictions
                # In production, these would come from actual ML models
                elo_diff = features.get('home_elo', 1500) - features.get('away_elo', 1500)
                base_prob = 0.5 + (elo_diff / 400.0) * 0.1  # Simple ELO-based probability
                base_prob = max(0.1, min(0.9, base_prob))  # Clamp between 0.1 and 0.9
                
                framework_predictions = {
                    'h2o': FrameworkPrediction(
                        framework=ModelFramework.H2O,
                        probability=base_prob,
                        recent_accuracy=0.55,
                        recent_clv=0.01,
                    ),
                    'sklearn': FrameworkPrediction(
                        framework=ModelFramework.SKLEARN,
                        probability=base_prob + 0.02,
                        recent_accuracy=0.53,
                        recent_clv=0.008,
                    ),
                    'autogluon': FrameworkPrediction(
                        framework=ModelFramework.AUTOGLUON,
                        probability=base_prob - 0.02,
                        recent_accuracy=0.57,
                        recent_clv=0.012,
                    ),
                }
                
                # Generate predictions for each requested bet type
                for bet_type_str in request.bet_types:
                    try:
                        # Convert bet type string to enum
                        bet_type_map = {
                            'spread': BetType.SPREAD,
                            'moneyline': BetType.MONEYLINE,
                            'total': BetType.TOTAL,
                        }
                        bet_type = bet_type_map.get(bet_type_str.lower())
                        if not bet_type:
                            continue  # Skip unsupported bet types
                        
                        # Generate prediction using the engine
                        prediction = prediction_engine.generate_prediction(
                            game_id=str(game.id),
                            sport=sport_code,
                            home_team=game.home_team.name if game.home_team else "Home",
                            away_team=game.away_team.name if game.away_team else "Away",
                            game_date=game.game_date,
                            bet_type=bet_type,
                            features=features,
                            odds_info=odds_info,
                            framework_predictions=framework_predictions,
                        )
                        
                        # Add prediction to list for processing
                        predictions_list.append(prediction)
                        
                    except Exception as e:
                        import logging
                        logger = logging.getLogger(__name__)
                        logger.error(f"Error generating prediction for game {game.id}, bet_type {bet_type_str}: {e}", exc_info=True)
                        errors.append(f"Error generating prediction for game {game.id}, bet_type {bet_type_str}: {str(e)}")
                        continue
                        
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Error processing game {game.id}: {e}", exc_info=True)
                errors.append(f"Error processing game {game.id}: {str(e)}")
                continue
        
        # Now process all predictions and save to database
        for pred_obj in predictions_list:
            try:
                # Handle both dataclass and dict objects
                if hasattr(pred_obj, 'to_dict'):
                    # It's a dataclass Prediction object
                    pred_data = pred_obj.to_dict()
                elif isinstance(pred_obj, dict):
                    # It's already a dictionary
                    pred_data = pred_obj
                else:
                    # Try to convert using dict() or get attributes
                    pred_data = dict(pred_obj) if hasattr(pred_obj, '__dict__') else {}
                
                # Extract values with fallbacks
                game_id_str = pred_data.get("game_id") or (getattr(pred_obj, 'game_id', None) if hasattr(pred_obj, 'game_id') else None)
                if not game_id_str:
                    errors.append(f"Missing game_id in prediction")
                    continue
                
                # Convert game_id to UUID
                try:
                    game_id = game_id_str if isinstance(game_id_str, UUID) else UUID(str(game_id_str))
                except (ValueError, AttributeError):
                    errors.append(f"Invalid game_id format: {game_id_str}")
                    continue
                
                # Get prediction hash
                pred_hash = pred_data.get("prediction_hash") or (getattr(pred_obj, 'prediction_hash', None) if hasattr(pred_obj, 'prediction_hash') else None)
                if not pred_hash:
                    errors.append(f"Missing prediction_hash for game {game_id_str}")
                    continue
                
                # Check if prediction already exists (by hash) - only select id to avoid non-existent columns
                existing = await db.execute(
                    select(DBPrediction.id).where(DBPrediction.prediction_hash == pred_hash)
                )
                if existing.scalar_one_or_none():
                    continue  # Skip if already exists
                
                # Extract bet_type and predicted_side (handle both .value for enums and direct strings)
                bet_type = pred_data.get("bet_type", "")
                if hasattr(bet_type, 'value'):
                    bet_type = bet_type.value
                
                predicted_side = pred_data.get("predicted_side", "")
                if hasattr(predicted_side, 'value'):
                    predicted_side = predicted_side.value
                
                # Get signal_tier
                signal_tier_val = pred_data.get("signal_tier") or (getattr(pred_obj, 'signal_tier', None) if hasattr(pred_obj, 'signal_tier') else "D")
                if hasattr(signal_tier_val, 'value'):
                    signal_tier_val = signal_tier_val.value
                signal_tier = SignalTier(signal_tier_val) if isinstance(signal_tier_val, str) else signal_tier_val
                
                # Get probability
                probability = pred_data.get("probability") or (getattr(pred_obj, 'probability', 0.0) if hasattr(pred_obj, 'probability') else 0.0)
                
                # Get recommendation data (for kelly_fraction)
                recommendation = pred_data.get("recommendation", {})
                if hasattr(pred_obj, 'recommendation'):
                    rec_obj = pred_obj.recommendation
                    if hasattr(rec_obj, 'kelly_fraction'):
                        recommendation = {
                            'kelly_fraction': rec_obj.kelly_fraction,
                            'recommended_units': getattr(rec_obj, 'recommended_units', None)
                        }
                
                # Get values for insertion (only columns that exist in database)
                pred_id = uuid4()
                line_at_pred = pred_data.get("line_at_prediction") or pred_data.get("line") or (getattr(pred_obj, 'line', None) if hasattr(pred_obj, 'line') else None)
                odds_at_pred = pred_data.get("odds_at_prediction") or pred_data.get("odds") or (getattr(pred_obj, 'odds', None) if hasattr(pred_obj, 'odds') else None)
                edge_val = pred_data.get("edge") or (getattr(pred_obj, 'edge', None) if hasattr(pred_obj, 'edge') else None)
                kelly_frac = pred_data.get("kelly_fraction") or recommendation.get("kelly_fraction") if isinstance(recommendation, dict) else None
                
                # Use raw SQL INSERT to avoid non-existent columns (calibrated_probability, recommended_bet_size)
                insert_sql = text("""
                    INSERT INTO predictions (
                        id, game_id, model_id, bet_type, predicted_side, probability,
                        line_at_prediction, odds_at_prediction, edge, signal_tier,
                        kelly_fraction, prediction_hash, created_at
                    ) VALUES (
                        :id, :game_id, :model_id, :bet_type, :predicted_side, :probability,
                        :line_at_prediction, :odds_at_prediction, :edge, :signal_tier,
                        :kelly_fraction, :prediction_hash, :created_at
                    )
                """)
                
                insert_params = {
                    "id": pred_id,
                    "game_id": game_id,
                    "model_id": None,  # model_id from prediction if available
                    "bet_type": str(bet_type),
                    "predicted_side": str(predicted_side),
                    "probability": float(probability),
                    "line_at_prediction": line_at_pred,
                    "odds_at_prediction": odds_at_pred,
                    "edge": edge_val,
                    "signal_tier": signal_tier.value if hasattr(signal_tier, 'value') else str(signal_tier),
                    "kelly_fraction": kelly_frac,
                    "prediction_hash": str(pred_hash),
                    "created_at": datetime.utcnow()
                }
                
                await db.execute(insert_sql, insert_params)
                
                # Get sport_code for this prediction
                sport_code = game_sport_codes_map.get(game_id, None)
                created_at_time = insert_params["created_at"]
                
                # Create PredictionBase object for response
                saved_predictions.append(
                    PredictionBase(
                        id=str(pred_id),
                        game_id=str(game_id),
                        sport_code=sport_code,
                        bet_type=str(bet_type),
                        predicted_side=str(predicted_side),
                        probability=float(probability),
                        edge=edge_val,
                        signal_tier=signal_tier.value if hasattr(signal_tier, 'value') else str(signal_tier),
                        line_at_prediction=line_at_pred,
                        odds_at_prediction=odds_at_pred,
                        kelly_fraction=kelly_frac,
                        recommended_bet=None,  # Not in database
                        prediction_hash=str(pred_hash),
                        locked_at=created_at_time,  # Using created_at as locked_at
                        model_id=None,  # model_id not set yet
                        model_version="1.0.0",
                        is_graded=False,
                        result=None,
                        actual_outcome=None,
                        profit_loss=None,
                        clv=None
                    )
                )
                saved_count += 1
                
            except Exception as e:
                # Log error but continue with other predictions
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Error saving prediction to database: {e}", exc_info=True)
                errors.append(f"Error saving prediction: {str(e)}")
                continue
        
        # Commit all saved predictions
        if saved_count > 0:
            await db.commit()
            
            # Clear predictions cache to ensure fresh data is shown immediately
            try:
                import logging
                logger = logging.getLogger(__name__)
                deleted_count = await cache_manager.delete_pattern("*", prefix=CachePrefix.PREDICTIONS)
                logger.info(f"Cleared {deleted_count} prediction cache entries after generation")
            except Exception as cache_error:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Failed to clear prediction cache: {cache_error}")
                # Don't fail the request if cache clearing fails
        
        return GeneratePredictionsResponse(
            generated_count=saved_count,
            predictions=saved_predictions,
            errors=errors
        )
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate predictions: {str(e)}"
        )


@router.post("/{prediction_id}/verify")
async def verify_prediction_integrity(
    prediction_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Verify the SHA-256 hash integrity of a prediction.
    """
    query = text("""
        SELECT 
            id, game_id, bet_type, predicted_side, probability,
            line_at_prediction, odds_at_prediction, created_at, prediction_hash
        FROM predictions
        WHERE id = :prediction_id
    """)
    
    result = await db.execute(query, {"prediction_id": prediction_id})
    row = result.fetchone()
    
    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prediction {prediction_id} not found"
        )
    
    from app.core.security import SHA256Hasher
    
    # Reconstruct hash
    prediction_data = {
        "game_id": row.game_id,
        "bet_type": row.bet_type,
        "predicted_side": row.predicted_side,
        "probability": round(row.probability, 6),
        "line": row.line_at_prediction,
        "odds": row.odds_at_prediction,
        "timestamp": row.created_at.isoformat()  # Using created_at as locked_at since locked_at column doesn't exist
    }
    
    computed_hash = SHA256Hasher.hash_prediction(prediction_data)
    is_valid = SHA256Hasher.verify_prediction(prediction_data, row.prediction_hash)
    
    return {
        "prediction_id": prediction_id,
        "stored_hash": row.prediction_hash,
        "computed_hash": computed_hash,
        "is_valid": is_valid,
        "verified_at": datetime.utcnow().isoformat()
    }


@router.post("/grade")
async def grade_predictions(
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Grade all pending predictions with completed games.
    Requires admin role.
    """
    if current_user.get("role") not in ["admin", "system"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required to grade predictions"
        )
    
    from app.services.grading.auto_grader import auto_grader
    
    try:
        result = await auto_grader.grade_all_pending()
        
        return {
            "graded_count": result["graded_count"],
            "wins": result["wins"],
            "losses": result["losses"],
            "pushes": result["pushes"],
            "errors": result.get("errors", [])
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to grade predictions: {str(e)}"
        )
