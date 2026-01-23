"""
ROYALEY - Backtesting API Routes
Historical simulation and strategy testing
"""

from datetime import datetime, timedelta, date
from typing import Optional, List
from enum import Enum

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from pydantic import BaseModel, Field
from sqlalchemy import select, func, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_db
from app.api.dependencies import get_current_user, require_roles
from app.core.cache import CacheManager
from app.core.config import settings
from app.models import User, BacktestRun, Sport

router = APIRouter()
cache = CacheManager()


# ============== Schemas ==============

class BacktestConfig(BaseModel):
    """Backtest configuration"""
    sport_codes: List[str] = Field(..., description="Sports to include")
    bet_types: List[str] = Field(default=["spread", "moneyline", "total"])
    start_date: date = Field(..., description="Backtest start date")
    end_date: date = Field(..., description="Backtest end date")
    initial_bankroll: float = Field(default=10000.0, ge=100)
    kelly_fraction: float = Field(default=0.25, ge=0.05, le=1.0)
    max_bet_percent: float = Field(default=0.02, ge=0.005, le=0.1)
    min_edge_threshold: float = Field(default=0.03, ge=0.0, le=0.2)
    tier_filter: Optional[List[str]] = Field(default=None, description="Filter by tiers (A, B, C, D)")
    model_id: Optional[int] = Field(default=None, description="Specific model to test")


class BacktestResult(BaseModel):
    """Backtest result summary"""
    id: int
    status: str
    config: dict
    started_at: datetime
    completed_at: Optional[datetime]
    duration_seconds: Optional[int]
    
    # Performance metrics
    total_predictions: Optional[int]
    total_bets: Optional[int]
    wins: Optional[int]
    losses: Optional[int]
    pushes: Optional[int]
    win_rate: Optional[float]
    
    # Financial metrics
    initial_bankroll: Optional[float]
    final_bankroll: Optional[float]
    total_wagered: Optional[float]
    total_profit: Optional[float]
    roi: Optional[float]
    max_drawdown: Optional[float]
    sharpe_ratio: Optional[float]
    
    # CLV metrics
    avg_clv: Optional[float]
    clv_positive_rate: Optional[float]
    
    class Config:
        from_attributes = True


class BacktestDetailedResult(BacktestResult):
    """Detailed backtest results with breakdowns"""
    by_sport: Optional[dict]
    by_bet_type: Optional[dict]
    by_tier: Optional[dict]
    by_month: Optional[dict]
    equity_curve: Optional[List[dict]]
    bet_history: Optional[List[dict]]


class BacktestComparisonRequest(BaseModel):
    """Compare multiple backtest strategies"""
    base_config: BacktestConfig
    variations: List[dict] = Field(..., description="Parameter variations to test")


class WalkForwardConfig(BaseModel):
    """Walk-forward validation configuration"""
    sport_code: str
    bet_type: str
    start_date: date
    end_date: date
    training_window_days: int = Field(default=365, ge=90)
    test_window_days: int = Field(default=30, ge=7)
    step_days: int = Field(default=30, ge=7)


class WalkForwardResult(BaseModel):
    """Walk-forward validation results"""
    id: int
    sport_code: str
    bet_type: str
    status: str
    folds: int
    avg_accuracy: Optional[float]
    avg_auc: Optional[float]
    std_accuracy: Optional[float]
    stability_score: Optional[float]
    fold_results: Optional[List[dict]]
    
    class Config:
        from_attributes = True


# ============== Endpoints ==============

@router.post("/run", response_model=BacktestResult, status_code=status.HTTP_202_ACCEPTED)
async def run_backtest(
    config: BacktestConfig,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Run a backtest simulation with the specified configuration.
    
    The backtest runs asynchronously. Use the returned ID to check status.
    
    - **sport_codes**: List of sports to include
    - **bet_types**: Bet types to test (spread, moneyline, total)
    - **start_date/end_date**: Date range for simulation
    - **initial_bankroll**: Starting bankroll amount
    - **kelly_fraction**: Fractional Kelly multiplier (0.25 = quarter Kelly)
    - **max_bet_percent**: Maximum bet as percent of bankroll
    - **min_edge_threshold**: Minimum edge required to place bet
    - **tier_filter**: Only include specific signal tiers
    """
    # Validate sports
    for code in config.sport_codes:
        result = await db.execute(
            select(Sport).where(Sport.code == code.upper())
        )
        if not result.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid sport code: {code}"
            )
    
    # Validate date range
    if config.end_date <= config.start_date:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="end_date must be after start_date"
        )
    
    if config.end_date > date.today():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="end_date cannot be in the future"
        )
    
    # Create backtest run record
    backtest_run = BacktestRun(
        user_id=current_user.id,
        status="pending",
        config=config.model_dump(mode="json"),
        started_at=datetime.utcnow()
    )
    
    db.add(backtest_run)
    await db.commit()
    await db.refresh(backtest_run)
    
    # Queue backtest execution
    background_tasks.add_task(
        execute_backtest,
        backtest_run.id,
        config.model_dump(mode="json")
    )
    
    return backtest_run


async def execute_backtest(backtest_id: int, config: dict):
    """Background task to execute backtest"""
    # Implementation in services/backtesting/backtest_engine.py
    pass


@router.get("/{backtest_id}", response_model=BacktestDetailedResult)
async def get_backtest_result(
    backtest_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get detailed results of a backtest run.
    
    Includes performance breakdowns by sport, bet type, tier, and month.
    """
    result = await db.execute(
        select(BacktestRun).where(
            BacktestRun.id == backtest_id,
            BacktestRun.user_id == current_user.id
        )
    )
    backtest = result.scalar_one_or_none()
    
    if not backtest:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Backtest not found"
        )
    
    return backtest


@router.get("/", response_model=List[BacktestResult])
async def list_backtests(
    status: Optional[str] = None,
    sport_code: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    List user's backtest runs.
    
    - **status**: Filter by status (pending, running, completed, failed)
    - **sport_code**: Filter by sport
    - **limit/offset**: Pagination
    """
    query = select(BacktestRun).where(BacktestRun.user_id == current_user.id)
    
    if status:
        query = query.where(BacktestRun.status == status)
    
    query = query.order_by(desc(BacktestRun.started_at)).offset(offset).limit(limit)
    
    result = await db.execute(query)
    return result.scalars().all()


@router.delete("/{backtest_id}")
async def delete_backtest(
    backtest_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Delete a backtest run.
    """
    result = await db.execute(
        select(BacktestRun).where(
            BacktestRun.id == backtest_id,
            BacktestRun.user_id == current_user.id
        )
    )
    backtest = result.scalar_one_or_none()
    
    if not backtest:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Backtest not found"
        )
    
    if backtest.status == "running":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete running backtest"
        )
    
    await db.delete(backtest)
    await db.commit()
    
    return {"message": "Backtest deleted"}


@router.get("/{backtest_id}/equity-curve")
async def get_equity_curve(
    backtest_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get equity curve data for charting.
    
    Returns daily bankroll values over the backtest period.
    """
    result = await db.execute(
        select(BacktestRun).where(
            BacktestRun.id == backtest_id,
            BacktestRun.user_id == current_user.id
        )
    )
    backtest = result.scalar_one_or_none()
    
    if not backtest:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Backtest not found"
        )
    
    if backtest.status != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Backtest not completed"
        )
    
    return {
        "backtest_id": backtest_id,
        "equity_curve": backtest.equity_curve or [],
        "initial_bankroll": backtest.initial_bankroll,
        "final_bankroll": backtest.final_bankroll
    }


@router.get("/{backtest_id}/bets")
async def get_backtest_bets(
    backtest_id: int,
    page: int = 1,
    per_page: int = 50,
    sport_code: Optional[str] = None,
    tier: Optional[str] = None,
    result_filter: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get individual bets from a backtest run.
    
    - **sport_code**: Filter by sport
    - **tier**: Filter by signal tier
    - **result_filter**: Filter by result (win, loss, push)
    """
    result = await db.execute(
        select(BacktestRun).where(
            BacktestRun.id == backtest_id,
            BacktestRun.user_id == current_user.id
        )
    )
    backtest = result.scalar_one_or_none()
    
    if not backtest:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Backtest not found"
        )
    
    if backtest.status != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Backtest not completed"
        )
    
    bets = backtest.bet_history or []
    
    # Apply filters
    if sport_code:
        bets = [b for b in bets if b.get("sport_code") == sport_code.upper()]
    if tier:
        bets = [b for b in bets if b.get("tier") == tier.upper()]
    if result_filter:
        bets = [b for b in bets if b.get("result") == result_filter.lower()]
    
    # Paginate
    total = len(bets)
    start = (page - 1) * per_page
    end = start + per_page
    
    return {
        "backtest_id": backtest_id,
        "total": total,
        "page": page,
        "per_page": per_page,
        "bets": bets[start:end]
    }


@router.post("/compare", response_model=dict)
async def compare_strategies(
    request: BacktestComparisonRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Compare multiple betting strategies by running parallel backtests.
    
    Provide a base configuration and variations to test.
    
    Example variations:
    ```json
    {
        "variations": [
            {"kelly_fraction": 0.1},
            {"kelly_fraction": 0.25},
            {"kelly_fraction": 0.5},
            {"min_edge_threshold": 0.02},
            {"min_edge_threshold": 0.05}
        ]
    }
    ```
    """
    if len(request.variations) > 10:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 10 variations allowed"
        )
    
    comparison_id = f"compare_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{current_user.id}"
    backtest_ids = []
    
    # Create backtest for each variation
    for idx, variation in enumerate(request.variations):
        config = request.base_config.model_dump(mode="json")
        config.update(variation)
        config["variation_index"] = idx
        config["comparison_id"] = comparison_id
        
        backtest_run = BacktestRun(
            user_id=current_user.id,
            status="pending",
            config=config,
            started_at=datetime.utcnow()
        )
        
        db.add(backtest_run)
        await db.flush()
        backtest_ids.append(backtest_run.id)
        
        background_tasks.add_task(execute_backtest, backtest_run.id, config)
    
    await db.commit()
    
    return {
        "comparison_id": comparison_id,
        "backtest_ids": backtest_ids,
        "variations_count": len(request.variations),
        "message": "Comparison backtests started"
    }


@router.get("/compare/{comparison_id}")
async def get_comparison_results(
    comparison_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get results of a strategy comparison.
    """
    result = await db.execute(
        select(BacktestRun).where(
            BacktestRun.user_id == current_user.id,
            func.json_extract_path_text(BacktestRun.config, "comparison_id") == comparison_id
        )
        .order_by(func.json_extract_path_text(BacktestRun.config, "variation_index"))
    )
    backtests = result.scalars().all()
    
    if not backtests:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Comparison not found"
        )
    
    # Check if all completed
    all_completed = all(b.status == "completed" for b in backtests)
    
    results = []
    for bt in backtests:
        results.append({
            "id": bt.id,
            "status": bt.status,
            "variation": {k: v for k, v in bt.config.items() if k not in ["comparison_id", "variation_index"]},
            "roi": bt.roi if bt.status == "completed" else None,
            "win_rate": bt.win_rate if bt.status == "completed" else None,
            "sharpe_ratio": bt.sharpe_ratio if bt.status == "completed" else None,
            "max_drawdown": bt.max_drawdown if bt.status == "completed" else None
        })
    
    # Find best strategy if all completed
    best_strategy = None
    if all_completed and results:
        best = max(results, key=lambda x: x.get("roi") or 0)
        best_strategy = {
            "id": best["id"],
            "variation": best["variation"],
            "roi": best["roi"]
        }
    
    return {
        "comparison_id": comparison_id,
        "all_completed": all_completed,
        "results": results,
        "best_strategy": best_strategy
    }


@router.post("/walk-forward", response_model=WalkForwardResult, status_code=status.HTTP_202_ACCEPTED)
async def run_walk_forward_validation(
    config: WalkForwardConfig,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_roles(["admin", "system"]))
):
    """
    Run walk-forward validation for a model.
    
    **Admin only**
    
    Walk-forward validation trains models on rolling windows and tests
    on subsequent periods to simulate real-world prediction scenarios.
    """
    # Validate sport
    result = await db.execute(
        select(Sport).where(Sport.code == config.sport_code.upper())
    )
    if not result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid sport code: {config.sport_code}"
        )
    
    # Calculate number of folds
    total_days = (config.end_date - config.start_date).days
    folds = max(1, (total_days - config.training_window_days) // config.step_days)
    
    # Create walk-forward run record
    from app.models import WalkForwardRun
    
    wf_run = WalkForwardRun(
        user_id=current_user.id,
        sport_code=config.sport_code.upper(),
        bet_type=config.bet_type.lower(),
        status="pending",
        config=config.model_dump(mode="json"),
        folds=folds,
        started_at=datetime.utcnow()
    )
    
    db.add(wf_run)
    await db.commit()
    await db.refresh(wf_run)
    
    # Queue execution
    background_tasks.add_task(
        execute_walk_forward,
        wf_run.id,
        config.model_dump(mode="json")
    )
    
    return wf_run


async def execute_walk_forward(run_id: int, config: dict):
    """Background task for walk-forward validation"""
    # Implementation in services/backtesting/walk_forward.py
    pass


@router.get("/walk-forward/{run_id}", response_model=WalkForwardResult)
async def get_walk_forward_result(
    run_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get walk-forward validation results.
    """
    from app.models import WalkForwardRun
    
    result = await db.execute(
        select(WalkForwardRun).where(WalkForwardRun.id == run_id)
    )
    wf_run = result.scalar_one_or_none()
    
    if not wf_run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Walk-forward run not found"
        )
    
    return wf_run


@router.get("/stats/summary")
async def get_backtest_stats_summary(
    days: int = 90,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get summary statistics across all user's backtests.
    """
    start_date = datetime.utcnow() - timedelta(days=days)
    
    result = await db.execute(
        select(BacktestRun).where(
            BacktestRun.user_id == current_user.id,
            BacktestRun.status == "completed",
            BacktestRun.started_at >= start_date
        )
    )
    backtests = result.scalars().all()
    
    if not backtests:
        return {
            "total_backtests": 0,
            "message": "No completed backtests found"
        }
    
    return {
        "total_backtests": len(backtests),
        "avg_roi": sum(b.roi or 0 for b in backtests) / len(backtests),
        "best_roi": max(b.roi or 0 for b in backtests),
        "worst_roi": min(b.roi or 0 for b in backtests),
        "avg_win_rate": sum(b.win_rate or 0 for b in backtests) / len(backtests),
        "avg_sharpe": sum(b.sharpe_ratio or 0 for b in backtests) / len(backtests),
        "profitable_backtests": sum(1 for b in backtests if (b.roi or 0) > 0),
        "profitable_rate": sum(1 for b in backtests if (b.roi or 0) > 0) / len(backtests)
    }
