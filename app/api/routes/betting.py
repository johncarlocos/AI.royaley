"""
LOYALEY - Betting API Routes
Enterprise-grade betting management with Kelly Criterion and CLV tracking
"""

from datetime import datetime, date, timedelta
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from app.core.database import get_db
from app.api.dependencies import get_current_user
from app.core.cache import cache_manager


router = APIRouter(tags=["betting"])


# ============================================================================
# SCHEMAS
# ============================================================================

class BankrollInfo(BaseModel):
    id: int
    user_id: int
    name: str
    initial_amount: float
    current_amount: float
    peak_amount: float
    low_amount: float
    total_wagered: float
    total_won: float
    total_lost: float
    roi: float
    win_rate: float
    max_drawdown: float
    kelly_fraction: float
    max_bet_percent: float
    created_at: datetime
    updated_at: datetime


class BankrollCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    initial_amount: float = Field(..., gt=0)
    kelly_fraction: float = Field(default=0.25, ge=0.1, le=1.0)
    max_bet_percent: float = Field(default=0.02, ge=0.01, le=0.1)


class BankrollUpdate(BaseModel):
    name: Optional[str] = None
    kelly_fraction: Optional[float] = Field(None, ge=0.1, le=1.0)
    max_bet_percent: Optional[float] = Field(None, ge=0.01, le=0.1)


class BetSizingRequest(BaseModel):
    probability: float = Field(..., ge=0, le=1)
    american_odds: int
    bankroll_id: Optional[int] = None


class BetSizingResponse(BaseModel):
    probability: float
    american_odds: int
    decimal_odds: float
    implied_probability: float
    edge: float
    full_kelly: float
    fractional_kelly: float
    recommended_bet: float
    recommended_units: float
    bankroll_amount: float
    max_bet: float
    should_bet: bool
    reason: Optional[str] = None


class BetCreate(BaseModel):
    prediction_id: int
    bankroll_id: int
    stake: float = Field(..., gt=0)
    odds_at_bet: int
    line_at_bet: Optional[float] = None
    sportsbook: str
    notes: Optional[str] = None


class BetBase(BaseModel):
    id: int
    prediction_id: int
    bankroll_id: int
    stake: float
    odds_at_bet: int
    line_at_bet: Optional[float] = None
    sportsbook: str
    placed_at: datetime
    is_graded: bool = False
    result: Optional[str] = None
    profit_loss: Optional[float] = None
    clv: Optional[float] = None
    notes: Optional[str] = None


class BetDetail(BetBase):
    sport_code: str
    bet_type: str
    predicted_side: str
    home_team: str
    away_team: str
    game_date: datetime
    closing_line: Optional[float] = None


class BetListResponse(BaseModel):
    bets: List[BetBase]
    total: int
    page: int
    per_page: int
    summary: dict


class CLVSummary(BaseModel):
    total_bets: int
    average_clv: float
    positive_clv_count: int
    negative_clv_count: int
    total_clv_cents: float
    clv_by_sport: dict
    clv_by_tier: dict
    clv_trend: List[dict]


class BettingStats(BaseModel):
    total_bets: int
    total_wagered: float
    total_profit_loss: float
    roi: float
    win_rate: float
    average_odds: float
    average_stake: float
    by_sport: dict
    by_bet_type: dict
    by_tier: dict
    by_month: List[dict]


class TransactionCreate(BaseModel):
    bankroll_id: int
    amount: float
    transaction_type: str = Field(..., pattern="^(deposit|withdrawal|adjustment)$")
    notes: Optional[str] = None


class Transaction(BaseModel):
    id: int
    bankroll_id: int
    amount: float
    transaction_type: str
    balance_after: float
    notes: Optional[str] = None
    created_at: datetime


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.get("/bankroll", response_model=List[BankrollInfo])
async def get_bankrolls(
    current_user = Depends(get_current_user),
    db: Optional[AsyncSession] = Depends(get_db)
):
    """
    Get all bankrolls for the current user.
    """
    # Check if demo user - return mock bankroll for demo mode (before DB access)
    if hasattr(current_user, 'id') and str(current_user.id) == "00000000-0000-0000-0000-000000000000":
        from datetime import datetime
        demo_bankroll = BankrollInfo(
            id=1,
            user_id=1,  # Use int for demo user
            name="Demo Bankroll",
            initial_amount=10000.0,
            current_amount=10000.0,
            peak_amount=10000.0,
            low_amount=10000.0,
            total_wagered=0.0,
            total_won=0.0,
            total_lost=0.0,
            roi=0.0,
            win_rate=0.0,
            max_drawdown=0.0,
            kelly_fraction=0.25,
            max_bet_percent=0.02,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        return [demo_bankroll]
    
    try:
        user_id = current_user.id if hasattr(current_user, 'id') else current_user["id"]
        query = text("""
            SELECT * FROM bankrolls
            WHERE user_id = :user_id
            ORDER BY created_at DESC
        """)
        
        result = await db.execute(query, {"user_id": user_id})
        rows = result.fetchall()
        
        return [
            BankrollInfo(
                id=row.id,
                user_id=row.user_id,
                name=row.name,
                initial_amount=row.initial_amount,
                current_amount=row.current_amount,
                peak_amount=row.peak_amount,
                low_amount=row.low_amount,
                total_wagered=row.total_wagered or 0,
                total_won=row.total_won or 0,
                total_lost=row.total_lost or 0,
                roi=row.roi or 0,
                win_rate=row.win_rate or 0,
                max_drawdown=row.max_drawdown or 0,
                kelly_fraction=row.kelly_fraction,
                max_bet_percent=row.max_bet_percent,
                created_at=row.created_at,
                updated_at=row.updated_at
            )
            for row in rows
        ]
    except Exception:
        # If database error, return mock bankroll for demo mode
        if hasattr(current_user, 'id') and str(current_user.id) == "00000000-0000-0000-0000-000000000000":
            from datetime import datetime
            demo_bankroll = BankrollInfo(
                id=1,
                user_id=1,  # Use int for demo user
                name="Demo Bankroll",
                initial_amount=10000.0,
                current_amount=10000.0,
                peak_amount=10000.0,
                low_amount=10000.0,
                total_wagered=0.0,
                total_won=0.0,
                total_lost=0.0,
                roi=0.0,
                win_rate=0.0,
                max_drawdown=0.0,
                kelly_fraction=0.25,
                max_bet_percent=0.02,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            return [demo_bankroll]
        raise


@router.post("/bankroll", response_model=BankrollInfo, status_code=status.HTTP_201_CREATED)
async def create_bankroll(
    bankroll: BankrollCreate,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Create a new bankroll for tracking.
    """
    query = """
        INSERT INTO bankrolls (
            user_id, name, initial_amount, current_amount,
            peak_amount, low_amount, kelly_fraction, max_bet_percent,
            created_at, updated_at
        ) VALUES (
            :user_id, :name, :initial_amount, :initial_amount,
            :initial_amount, :initial_amount, :kelly_fraction, :max_bet_percent,
            :now, :now
        ) RETURNING *
    """
    
    now = datetime.utcnow()
    result = await db.execute(query, {
        "user_id": current_user["id"],
        "name": bankroll.name,
        "initial_amount": bankroll.initial_amount,
        "kelly_fraction": bankroll.kelly_fraction,
        "max_bet_percent": bankroll.max_bet_percent,
        "now": now
    })
    await db.commit()
    
    row = result.fetchone()
    
    return BankrollInfo(
        id=row.id,
        user_id=row.user_id,
        name=row.name,
        initial_amount=row.initial_amount,
        current_amount=row.current_amount,
        peak_amount=row.peak_amount,
        low_amount=row.low_amount,
        total_wagered=0,
        total_won=0,
        total_lost=0,
        roi=0,
        win_rate=0,
        max_drawdown=0,
        kelly_fraction=row.kelly_fraction,
        max_bet_percent=row.max_bet_percent,
        created_at=row.created_at,
        updated_at=row.updated_at
    )


@router.patch("/bankroll/{bankroll_id}", response_model=BankrollInfo)
async def update_bankroll(
    bankroll_id: int,
    bankroll: BankrollUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Update bankroll settings.
    """
    # Verify ownership
    check_query = "SELECT id FROM bankrolls WHERE id = :id AND user_id = :user_id"
    check_result = await db.execute(check_query, {"id": bankroll_id, "user_id": current_user["id"]})
    
    if not check_result.fetchone():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Bankroll not found"
        )
    
    updates = []
    params = {"id": bankroll_id, "now": datetime.utcnow()}
    
    if bankroll.name is not None:
        updates.append("name = :name")
        params["name"] = bankroll.name
    if bankroll.kelly_fraction is not None:
        updates.append("kelly_fraction = :kelly_fraction")
        params["kelly_fraction"] = bankroll.kelly_fraction
    if bankroll.max_bet_percent is not None:
        updates.append("max_bet_percent = :max_bet_percent")
        params["max_bet_percent"] = bankroll.max_bet_percent
    
    updates.append("updated_at = :now")
    
    query = f"UPDATE bankrolls SET {', '.join(updates)} WHERE id = :id RETURNING *"
    
    result = await db.execute(query, params)
    await db.commit()
    
    row = result.fetchone()
    
    return BankrollInfo(
        id=row.id,
        user_id=row.user_id,
        name=row.name,
        initial_amount=row.initial_amount,
        current_amount=row.current_amount,
        peak_amount=row.peak_amount,
        low_amount=row.low_amount,
        total_wagered=row.total_wagered or 0,
        total_won=row.total_won or 0,
        total_lost=row.total_lost or 0,
        roi=row.roi or 0,
        win_rate=row.win_rate or 0,
        max_drawdown=row.max_drawdown or 0,
        kelly_fraction=row.kelly_fraction,
        max_bet_percent=row.max_bet_percent,
        created_at=row.created_at,
        updated_at=row.updated_at
    )


@router.post("/sizing", response_model=BetSizingResponse)
async def calculate_bet_sizing(
    request: BetSizingRequest,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Calculate optimal bet size using Kelly Criterion.
    """
    # Get bankroll info
    if request.bankroll_id:
        bankroll_query = """
            SELECT current_amount, kelly_fraction, max_bet_percent
            FROM bankrolls
            WHERE id = :id AND user_id = :user_id
        """
        bankroll_result = await db.execute(bankroll_query, {
            "id": request.bankroll_id,
            "user_id": current_user["id"]
        })
        bankroll = bankroll_result.fetchone()
        
        if not bankroll:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Bankroll not found"
            )
        
        bankroll_amount = bankroll.current_amount
        kelly_fraction = bankroll.kelly_fraction
        max_bet_percent = bankroll.max_bet_percent
    else:
        # Use defaults
        bankroll_amount = 10000
        kelly_fraction = 0.25
        max_bet_percent = 0.02
    
    # Convert American odds to decimal
    if request.american_odds > 0:
        decimal_odds = 1 + (request.american_odds / 100)
    else:
        decimal_odds = 1 + (100 / abs(request.american_odds))
    
    # Calculate implied probability
    implied_probability = 1 / decimal_odds
    
    # Calculate edge
    edge = request.probability - implied_probability
    
    # Full Kelly calculation
    b = decimal_odds - 1
    q = 1 - request.probability
    
    if b > 0 and edge > 0:
        full_kelly = (b * request.probability - q) / b
    else:
        full_kelly = 0
    
    # Fractional Kelly
    fractional_kelly = full_kelly * kelly_fraction
    
    # Max bet
    max_bet = bankroll_amount * max_bet_percent
    
    # Recommended bet
    if fractional_kelly > 0:
        recommended_bet = min(bankroll_amount * fractional_kelly, max_bet)
    else:
        recommended_bet = 0
    
    # Determine if should bet
    min_edge_threshold = 0.03  # 3% minimum edge
    should_bet = edge >= min_edge_threshold and recommended_bet > 0
    
    reason = None
    if not should_bet:
        if edge < min_edge_threshold:
            reason = f"Edge ({edge*100:.1f}%) below minimum threshold ({min_edge_threshold*100:.0f}%)"
        elif recommended_bet <= 0:
            reason = "Negative or zero Kelly suggests no bet"
    
    return BetSizingResponse(
        probability=request.probability,
        american_odds=request.american_odds,
        decimal_odds=round(decimal_odds, 4),
        implied_probability=round(implied_probability, 4),
        edge=round(edge, 4),
        full_kelly=round(full_kelly, 4),
        fractional_kelly=round(fractional_kelly, 4),
        recommended_bet=round(recommended_bet, 2),
        recommended_units=round(recommended_bet / (bankroll_amount / 100), 2),
        bankroll_amount=bankroll_amount,
        max_bet=round(max_bet, 2),
        should_bet=should_bet,
        reason=reason
    )


@router.post("/bet", response_model=BetBase, status_code=status.HTTP_201_CREATED)
async def place_bet(
    bet: BetCreate,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Record a placed bet.
    """
    # Verify prediction exists
    pred_query = "SELECT id, game_id FROM predictions WHERE id = :id"
    pred_result = await db.execute(pred_query, {"id": bet.prediction_id})
    prediction = pred_result.fetchone()
    
    if not prediction:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Prediction not found"
        )
    
    # Verify bankroll ownership
    bankroll_query = """
        SELECT id, current_amount FROM bankrolls
        WHERE id = :id AND user_id = :user_id
    """
    bankroll_result = await db.execute(bankroll_query, {
        "id": bet.bankroll_id,
        "user_id": current_user["id"]
    })
    bankroll = bankroll_result.fetchone()
    
    if not bankroll:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Bankroll not found"
        )
    
    if bet.stake > bankroll.current_amount:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Stake exceeds available bankroll"
        )
    
    # Create bet
    now = datetime.utcnow()
    
    bet_query = """
        INSERT INTO bets (
            prediction_id, bankroll_id, user_id, stake,
            odds_at_bet, line_at_bet, sportsbook, placed_at, notes
        ) VALUES (
            :prediction_id, :bankroll_id, :user_id, :stake,
            :odds_at_bet, :line_at_bet, :sportsbook, :placed_at, :notes
        ) RETURNING *
    """
    
    bet_result = await db.execute(bet_query, {
        "prediction_id": bet.prediction_id,
        "bankroll_id": bet.bankroll_id,
        "user_id": current_user["id"],
        "stake": bet.stake,
        "odds_at_bet": bet.odds_at_bet,
        "line_at_bet": bet.line_at_bet,
        "sportsbook": bet.sportsbook,
        "placed_at": now,
        "notes": bet.notes
    })
    
    # Update bankroll - subtract stake
    await db.execute(
        "UPDATE bankrolls SET current_amount = current_amount - :stake, total_wagered = COALESCE(total_wagered, 0) + :stake, updated_at = :now WHERE id = :id",
        {"stake": bet.stake, "id": bet.bankroll_id, "now": now}
    )
    
    await db.commit()
    
    row = bet_result.fetchone()
    
    return BetBase(
        id=row.id,
        prediction_id=row.prediction_id,
        bankroll_id=row.bankroll_id,
        stake=row.stake,
        odds_at_bet=row.odds_at_bet,
        line_at_bet=row.line_at_bet,
        sportsbook=row.sportsbook,
        placed_at=row.placed_at,
        is_graded=row.is_graded or False,
        result=row.result,
        profit_loss=row.profit_loss,
        clv=row.clv,
        notes=row.notes
    )


@router.get("/history", response_model=BetListResponse)
async def get_bet_history(
    sport: Optional[str] = Query(None),
    result: Optional[str] = Query(None, pattern="^(win|loss|push)$"),
    date_from: Optional[date] = Query(None),
    date_to: Optional[date] = Query(None),
    bankroll_id: Optional[int] = Query(None),
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Get bet history with filtering and pagination.
    """
    conditions = ["b.user_id = :user_id"]
    params = {"user_id": current_user["id"]}
    
    if sport:
        conditions.append("p.sport_code = :sport")
        params["sport"] = sport.upper()
    if result:
        conditions.append("b.result = :result")
        params["result"] = result
    if date_from:
        conditions.append("DATE(b.placed_at) >= :date_from")
        params["date_from"] = date_from
    if date_to:
        conditions.append("DATE(b.placed_at) <= :date_to")
        params["date_to"] = date_to
    if bankroll_id:
        conditions.append("b.bankroll_id = :bankroll_id")
        params["bankroll_id"] = bankroll_id
    
    where_clause = " AND ".join(conditions)
    
    # Count total
    count_query = f"""
        SELECT COUNT(*) FROM bets b
        JOIN predictions p ON b.prediction_id = p.id
        WHERE {where_clause}
    """
    count_result = await db.execute(count_query, params)
    total = count_result.scalar() or 0
    
    # Get summary
    summary_query = f"""
        SELECT 
            COUNT(*) as total_bets,
            SUM(stake) as total_wagered,
            SUM(COALESCE(profit_loss, 0)) as total_profit_loss,
            AVG(CASE WHEN result = 'win' THEN 1.0 WHEN result IS NOT NULL THEN 0.0 END) as win_rate,
            AVG(clv) as avg_clv
        FROM bets b
        JOIN predictions p ON b.prediction_id = p.id
        WHERE {where_clause}
    """
    summary_result = await db.execute(summary_query, params)
    summary_row = summary_result.fetchone()
    
    # Get paginated bets
    offset = (page - 1) * per_page
    params["limit"] = per_page
    params["offset"] = offset
    
    bets_query = f"""
        SELECT b.* FROM bets b
        JOIN predictions p ON b.prediction_id = p.id
        WHERE {where_clause}
        ORDER BY b.placed_at DESC
        LIMIT :limit OFFSET :offset
    """
    
    bets_result = await db.execute(bets_query, params)
    bet_rows = bets_result.fetchall()
    
    bets = [
        BetBase(
            id=row.id,
            prediction_id=row.prediction_id,
            bankroll_id=row.bankroll_id,
            stake=row.stake,
            odds_at_bet=row.odds_at_bet,
            line_at_bet=row.line_at_bet,
            sportsbook=row.sportsbook,
            placed_at=row.placed_at,
            is_graded=row.is_graded or False,
            result=row.result,
            profit_loss=row.profit_loss,
            clv=row.clv,
            notes=row.notes
        )
        for row in bet_rows
    ]
    
    summary = {
        "total_bets": summary_row.total_bets or 0,
        "total_wagered": round(summary_row.total_wagered or 0, 2),
        "total_profit_loss": round(summary_row.total_profit_loss or 0, 2),
        "win_rate": round((summary_row.win_rate or 0) * 100, 2),
        "avg_clv": round(summary_row.avg_clv or 0, 4)
    }
    
    return BetListResponse(
        bets=bets,
        total=total,
        page=page,
        per_page=per_page,
        summary=summary
    )


@router.get("/clv", response_model=CLVSummary)
async def get_clv_summary(
    days: int = Query(30, ge=1, le=365),
    sport: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Get CLV (Closing Line Value) summary.
    """
    from_date = date.today() - timedelta(days=days)
    
    conditions = [
        "b.user_id = :user_id",
        "b.is_graded = true",
        "b.clv IS NOT NULL",
        "DATE(b.placed_at) >= :from_date"
    ]
    params = {"user_id": current_user["id"], "from_date": from_date}
    
    if sport:
        conditions.append("p.sport_code = :sport")
        params["sport"] = sport.upper()
    
    where_clause = " AND ".join(conditions)
    
    # Overall CLV stats
    stats_query = f"""
        SELECT 
            COUNT(*) as total_bets,
            AVG(b.clv) as average_clv,
            SUM(CASE WHEN b.clv > 0 THEN 1 ELSE 0 END) as positive_clv_count,
            SUM(CASE WHEN b.clv < 0 THEN 1 ELSE 0 END) as negative_clv_count,
            SUM(b.clv * 100) as total_clv_cents
        FROM bets b
        JOIN predictions p ON b.prediction_id = p.id
        WHERE {where_clause}
    """
    
    stats_result = await db.execute(stats_query, params)
    stats = stats_result.fetchone()
    
    # CLV by sport
    sport_query = f"""
        SELECT 
            p.sport_code,
            COUNT(*) as bets,
            AVG(b.clv) as avg_clv
        FROM bets b
        JOIN predictions p ON b.prediction_id = p.id
        WHERE {where_clause}
        GROUP BY p.sport_code
    """
    
    sport_result = await db.execute(sport_query, params)
    sport_rows = sport_result.fetchall()
    
    clv_by_sport = {
        row.sport_code: {
            "bets": row.bets,
            "avg_clv": round(row.avg_clv, 4) if row.avg_clv else 0
        }
        for row in sport_rows
    }
    
    # CLV by tier
    tier_query = f"""
        SELECT 
            p.signal_tier,
            COUNT(*) as bets,
            AVG(b.clv) as avg_clv
        FROM bets b
        JOIN predictions p ON b.prediction_id = p.id
        WHERE {where_clause}
        GROUP BY p.signal_tier
    """
    
    tier_result = await db.execute(tier_query, params)
    tier_rows = tier_result.fetchall()
    
    clv_by_tier = {
        row.signal_tier: {
            "bets": row.bets,
            "avg_clv": round(row.avg_clv, 4) if row.avg_clv else 0
        }
        for row in tier_rows
    }
    
    # CLV trend
    trend_query = f"""
        SELECT 
            DATE(b.placed_at) as date,
            AVG(b.clv) as avg_clv,
            COUNT(*) as bets
        FROM bets b
        JOIN predictions p ON b.prediction_id = p.id
        WHERE {where_clause}
        GROUP BY DATE(b.placed_at)
        ORDER BY date
    """
    
    trend_result = await db.execute(trend_query, params)
    trend_rows = trend_result.fetchall()
    
    clv_trend = [
        {
            "date": row.date.isoformat(),
            "avg_clv": round(row.avg_clv, 4) if row.avg_clv else 0,
            "bets": row.bets
        }
        for row in trend_rows
    ]
    
    return CLVSummary(
        total_bets=stats.total_bets or 0,
        average_clv=round(stats.average_clv or 0, 4),
        positive_clv_count=stats.positive_clv_count or 0,
        negative_clv_count=stats.negative_clv_count or 0,
        total_clv_cents=round(stats.total_clv_cents or 0, 2),
        clv_by_sport=clv_by_sport,
        clv_by_tier=clv_by_tier,
        clv_trend=clv_trend
    )


@router.get("/stats", response_model=BettingStats)
async def get_betting_stats(
    days: int = Query(30, ge=1, le=365),
    bankroll_id: Optional[int] = Query(None),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Get comprehensive betting statistics.
    """
    from_date = date.today() - timedelta(days=days)
    
    conditions = ["b.user_id = :user_id", "DATE(b.placed_at) >= :from_date"]
    params = {"user_id": current_user["id"], "from_date": from_date}
    
    if bankroll_id:
        conditions.append("b.bankroll_id = :bankroll_id")
        params["bankroll_id"] = bankroll_id
    
    where_clause = " AND ".join(conditions)
    
    # Overall stats
    overall_query = f"""
        SELECT 
            COUNT(*) as total_bets,
            SUM(stake) as total_wagered,
            SUM(COALESCE(profit_loss, 0)) as total_profit_loss,
            AVG(CASE WHEN result = 'win' THEN 1.0 WHEN result IS NOT NULL THEN 0.0 END) as win_rate,
            AVG(odds_at_bet) as avg_odds,
            AVG(stake) as avg_stake
        FROM bets b
        WHERE {where_clause}
    """
    
    overall_result = await db.execute(overall_query, params)
    overall = overall_result.fetchone()
    
    roi = 0
    if overall.total_wagered and overall.total_wagered > 0:
        roi = (overall.total_profit_loss or 0) / overall.total_wagered * 100
    
    # By sport
    sport_query = f"""
        SELECT 
            p.sport_code,
            COUNT(*) as bets,
            SUM(b.stake) as wagered,
            SUM(COALESCE(b.profit_loss, 0)) as profit_loss,
            AVG(CASE WHEN b.result = 'win' THEN 1.0 WHEN b.result IS NOT NULL THEN 0.0 END) as win_rate
        FROM bets b
        JOIN predictions p ON b.prediction_id = p.id
        WHERE {where_clause}
        GROUP BY p.sport_code
    """
    
    sport_result = await db.execute(sport_query, params)
    by_sport = {
        row.sport_code: {
            "bets": row.bets,
            "wagered": round(row.wagered, 2),
            "profit_loss": round(row.profit_loss, 2),
            "win_rate": round((row.win_rate or 0) * 100, 2),
            "roi": round((row.profit_loss / row.wagered * 100) if row.wagered else 0, 2)
        }
        for row in sport_result.fetchall()
    }
    
    # By bet type
    type_query = f"""
        SELECT 
            p.bet_type,
            COUNT(*) as bets,
            SUM(b.stake) as wagered,
            SUM(COALESCE(b.profit_loss, 0)) as profit_loss,
            AVG(CASE WHEN b.result = 'win' THEN 1.0 WHEN b.result IS NOT NULL THEN 0.0 END) as win_rate
        FROM bets b
        JOIN predictions p ON b.prediction_id = p.id
        WHERE {where_clause}
        GROUP BY p.bet_type
    """
    
    type_result = await db.execute(type_query, params)
    by_bet_type = {
        row.bet_type: {
            "bets": row.bets,
            "wagered": round(row.wagered, 2),
            "profit_loss": round(row.profit_loss, 2),
            "win_rate": round((row.win_rate or 0) * 100, 2)
        }
        for row in type_result.fetchall()
    }
    
    # By tier
    tier_query = f"""
        SELECT 
            p.signal_tier,
            COUNT(*) as bets,
            SUM(b.stake) as wagered,
            SUM(COALESCE(b.profit_loss, 0)) as profit_loss,
            AVG(CASE WHEN b.result = 'win' THEN 1.0 WHEN b.result IS NOT NULL THEN 0.0 END) as win_rate
        FROM bets b
        JOIN predictions p ON b.prediction_id = p.id
        WHERE {where_clause}
        GROUP BY p.signal_tier
    """
    
    tier_result = await db.execute(tier_query, params)
    by_tier = {
        row.signal_tier: {
            "bets": row.bets,
            "wagered": round(row.wagered, 2),
            "profit_loss": round(row.profit_loss, 2),
            "win_rate": round((row.win_rate or 0) * 100, 2)
        }
        for row in tier_result.fetchall()
    }
    
    # By month
    month_query = f"""
        SELECT 
            DATE_TRUNC('month', b.placed_at) as month,
            COUNT(*) as bets,
            SUM(b.stake) as wagered,
            SUM(COALESCE(b.profit_loss, 0)) as profit_loss,
            AVG(CASE WHEN b.result = 'win' THEN 1.0 WHEN b.result IS NOT NULL THEN 0.0 END) as win_rate
        FROM bets b
        WHERE {where_clause}
        GROUP BY DATE_TRUNC('month', b.placed_at)
        ORDER BY month
    """
    
    month_result = await db.execute(month_query, params)
    by_month = [
        {
            "month": row.month.strftime("%Y-%m"),
            "bets": row.bets,
            "wagered": round(row.wagered, 2),
            "profit_loss": round(row.profit_loss, 2),
            "win_rate": round((row.win_rate or 0) * 100, 2)
        }
        for row in month_result.fetchall()
    ]
    
    return BettingStats(
        total_bets=overall.total_bets or 0,
        total_wagered=round(overall.total_wagered or 0, 2),
        total_profit_loss=round(overall.total_profit_loss or 0, 2),
        roi=round(roi, 2),
        win_rate=round((overall.win_rate or 0) * 100, 2),
        average_odds=round(overall.avg_odds or 0, 0),
        average_stake=round(overall.avg_stake or 0, 2),
        by_sport=by_sport,
        by_bet_type=by_bet_type,
        by_tier=by_tier,
        by_month=by_month
    )


@router.post("/transaction", response_model=Transaction, status_code=status.HTTP_201_CREATED)
async def create_transaction(
    transaction: TransactionCreate,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Create a bankroll transaction (deposit, withdrawal, adjustment).
    """
    # Verify bankroll ownership
    bankroll_query = """
        SELECT id, current_amount FROM bankrolls
        WHERE id = :id AND user_id = :user_id
    """
    bankroll_result = await db.execute(bankroll_query, {
        "id": transaction.bankroll_id,
        "user_id": current_user["id"]
    })
    bankroll = bankroll_result.fetchone()
    
    if not bankroll:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Bankroll not found"
        )
    
    # Calculate new balance
    if transaction.transaction_type == "withdrawal":
        if transaction.amount > bankroll.current_amount:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Withdrawal amount exceeds available balance"
            )
        balance_after = bankroll.current_amount - abs(transaction.amount)
        amount = -abs(transaction.amount)
    else:
        balance_after = bankroll.current_amount + abs(transaction.amount)
        amount = abs(transaction.amount)
    
    now = datetime.utcnow()
    
    # Create transaction
    trans_query = """
        INSERT INTO bankroll_transactions (
            bankroll_id, amount, transaction_type, balance_after, notes, created_at
        ) VALUES (
            :bankroll_id, :amount, :transaction_type, :balance_after, :notes, :now
        ) RETURNING *
    """
    
    trans_result = await db.execute(trans_query, {
        "bankroll_id": transaction.bankroll_id,
        "amount": amount,
        "transaction_type": transaction.transaction_type,
        "balance_after": balance_after,
        "notes": transaction.notes,
        "now": now
    })
    
    # Update bankroll
    update_query = """
        UPDATE bankrolls SET 
            current_amount = :balance_after,
            peak_amount = GREATEST(peak_amount, :balance_after),
            low_amount = LEAST(low_amount, :balance_after),
            updated_at = :now
        WHERE id = :id
    """
    
    await db.execute(update_query, {
        "balance_after": balance_after,
        "id": transaction.bankroll_id,
        "now": now
    })
    
    await db.commit()
    
    row = trans_result.fetchone()
    
    return Transaction(
        id=row.id,
        bankroll_id=row.bankroll_id,
        amount=row.amount,
        transaction_type=row.transaction_type,
        balance_after=row.balance_after,
        notes=row.notes,
        created_at=row.created_at
    )
