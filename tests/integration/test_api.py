"""
ROYALEY - Integration Tests
API endpoint integration tests with database and services
"""

import pytest
from datetime import datetime, timedelta
from httpx import AsyncClient
from unittest.mock import patch, AsyncMock
import json

# Test configuration
pytestmark = pytest.mark.integration


class TestAuthEndpoints:
    """Test authentication API endpoints."""
    
    @pytest.mark.asyncio
    async def test_register_success(self, async_client: AsyncClient):
        """Test successful user registration."""
        response = await async_client.post(
            "/api/v1/auth/register",
            json={
                "email": "test@example.com",
                "username": "testuser",
                "password": "securepassword123",
                "full_name": "Test User"
            }
        )
        
        assert response.status_code == 201
        data = response.json()
        assert data["email"] == "test@example.com"
        assert data["username"] == "testuser"
        assert "password" not in data
    
    @pytest.mark.asyncio
    async def test_register_duplicate_email(self, async_client: AsyncClient, test_user):
        """Test registration with duplicate email."""
        response = await async_client.post(
            "/api/v1/auth/register",
            json={
                "email": test_user.email,
                "username": "newuser",
                "password": "securepassword123"
            }
        )
        
        assert response.status_code == 400
        assert "already registered" in response.json()["detail"]
    
    @pytest.mark.asyncio
    async def test_login_success(self, async_client: AsyncClient, test_user):
        """Test successful login."""
        response = await async_client.post(
            "/api/v1/auth/login",
            data={
                "username": test_user.username,
                "password": "testpassword"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
    
    @pytest.mark.asyncio
    async def test_login_wrong_password(self, async_client: AsyncClient, test_user):
        """Test login with wrong password."""
        response = await async_client.post(
            "/api/v1/auth/login",
            data={
                "username": test_user.username,
                "password": "wrongpassword"
            }
        )
        
        assert response.status_code == 401
    
    @pytest.mark.asyncio
    async def test_get_current_user(self, async_client: AsyncClient, auth_headers):
        """Test getting current user info."""
        response = await async_client.get(
            "/api/v1/auth/me",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert "email" in data
    
    @pytest.mark.asyncio
    async def test_refresh_token(self, async_client: AsyncClient, refresh_token):
        """Test token refresh."""
        response = await async_client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": refresh_token}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data


class TestPredictionEndpoints:
    """Test prediction API endpoints."""
    
    @pytest.mark.asyncio
    async def test_list_predictions(self, async_client: AsyncClient, auth_headers):
        """Test listing predictions."""
        response = await async_client.get(
            "/api/v1/predictions",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert "total" in data
        assert "page" in data
    
    @pytest.mark.asyncio
    async def test_list_predictions_with_filters(
        self, async_client: AsyncClient, auth_headers
    ):
        """Test listing predictions with filters."""
        response = await async_client.get(
            "/api/v1/predictions",
            params={
                "sport_code": "NBA",
                "bet_type": "spread",
                "tier": "A",
                "min_probability": 0.60
            },
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        for pred in data["predictions"]:
            assert pred["sport_code"] == "NBA"
            assert pred["bet_type"] == "spread"
    
    @pytest.mark.asyncio
    async def test_get_today_predictions(self, async_client: AsyncClient, auth_headers):
        """Test getting today's predictions."""
        response = await async_client.get(
            "/api/v1/predictions/today",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    @pytest.mark.asyncio
    async def test_get_prediction_by_id(
        self, async_client: AsyncClient, auth_headers, test_prediction
    ):
        """Test getting prediction by ID."""
        response = await async_client.get(
            f"/api/v1/predictions/{test_prediction.id}",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == test_prediction.id
        assert "shap_explanations" in data
    
    @pytest.mark.asyncio
    async def test_get_prediction_stats(self, async_client: AsyncClient, auth_headers):
        """Test getting prediction statistics."""
        response = await async_client.get(
            "/api/v1/predictions/stats",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "total_predictions" in data
        assert "win_rate" in data
        assert "by_sport" in data
    
    @pytest.mark.asyncio
    async def test_verify_prediction_hash(
        self, async_client: AsyncClient, auth_headers, test_prediction
    ):
        """Test prediction hash verification."""
        response = await async_client.post(
            f"/api/v1/predictions/{test_prediction.id}/verify",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "valid" in data
        assert "prediction_hash" in data


class TestGameEndpoints:
    """Test game API endpoints."""
    
    @pytest.mark.asyncio
    async def test_list_games(self, async_client: AsyncClient, auth_headers):
        """Test listing games."""
        response = await async_client.get(
            "/api/v1/games",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "games" in data
        assert "total" in data
    
    @pytest.mark.asyncio
    async def test_get_today_games(self, async_client: AsyncClient, auth_headers):
        """Test getting today's games."""
        response = await async_client.get(
            "/api/v1/games/today",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    @pytest.mark.asyncio
    async def test_get_game_by_id(
        self, async_client: AsyncClient, auth_headers, test_game
    ):
        """Test getting game by ID."""
        response = await async_client.get(
            f"/api/v1/games/{test_game.id}",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == test_game.id
    
    @pytest.mark.asyncio
    async def test_get_game_features(
        self, async_client: AsyncClient, auth_headers, test_game
    ):
        """Test getting game features."""
        response = await async_client.get(
            f"/api/v1/games/{test_game.id}/features",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "features" in data


class TestOddsEndpoints:
    """Test odds API endpoints."""
    
    @pytest.mark.asyncio
    async def test_get_game_odds(
        self, async_client: AsyncClient, auth_headers, test_game
    ):
        """Test getting game odds."""
        response = await async_client.get(
            f"/api/v1/odds/{test_game.id}",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "game_id" in data
        assert "sportsbooks" in data
    
    @pytest.mark.asyncio
    async def test_get_best_odds(
        self, async_client: AsyncClient, auth_headers, test_game
    ):
        """Test getting best available odds."""
        response = await async_client.get(
            f"/api/v1/odds/{test_game.id}/best",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "best_spread" in data
        assert "best_total" in data
    
    @pytest.mark.asyncio
    async def test_get_odds_movement(
        self, async_client: AsyncClient, auth_headers, test_game
    ):
        """Test getting odds movement."""
        response = await async_client.get(
            f"/api/v1/odds/{test_game.id}/movement",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    @pytest.mark.asyncio
    async def test_get_live_odds(self, async_client: AsyncClient, auth_headers):
        """Test getting live odds."""
        response = await async_client.get(
            "/api/v1/odds/live",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


class TestBettingEndpoints:
    """Test betting API endpoints."""
    
    @pytest.mark.asyncio
    async def test_get_bankroll(self, async_client: AsyncClient, auth_headers):
        """Test getting bankroll info."""
        response = await async_client.get(
            "/api/v1/betting/bankroll",
            headers=auth_headers
        )
        
        assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_create_bankroll(self, async_client: AsyncClient, auth_headers):
        """Test creating bankroll."""
        response = await async_client.post(
            "/api/v1/betting/bankroll",
            json={
                "name": "Main Bankroll",
                "initial_balance": 10000.0,
                "currency": "USD"
            },
            headers=auth_headers
        )
        
        assert response.status_code in [200, 201]
        data = response.json()
        assert data["name"] == "Main Bankroll"
    
    @pytest.mark.asyncio
    async def test_calculate_bet_sizing(self, async_client: AsyncClient, auth_headers):
        """Test Kelly Criterion bet sizing."""
        response = await async_client.post(
            "/api/v1/betting/sizing",
            json={
                "probability": 0.65,
                "odds": -110,
                "bankroll": 10000.0
            },
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "recommended_bet" in data
        assert "kelly_fraction" in data
        assert "edge" in data
    
    @pytest.mark.asyncio
    async def test_record_bet(
        self, async_client: AsyncClient, auth_headers, test_prediction
    ):
        """Test recording a bet."""
        response = await async_client.post(
            "/api/v1/betting/bet",
            json={
                "prediction_id": test_prediction.id,
                "stake": 100.0,
                "odds": -110,
                "sportsbook": "pinnacle"
            },
            headers=auth_headers
        )
        
        assert response.status_code in [200, 201]
        data = response.json()
        assert data["stake"] == 100.0
    
    @pytest.mark.asyncio
    async def test_get_bet_history(self, async_client: AsyncClient, auth_headers):
        """Test getting bet history."""
        response = await async_client.get(
            "/api/v1/betting/history",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "bets" in data
    
    @pytest.mark.asyncio
    async def test_get_clv_summary(self, async_client: AsyncClient, auth_headers):
        """Test getting CLV summary."""
        response = await async_client.get(
            "/api/v1/betting/clv",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "average_clv" in data
    
    @pytest.mark.asyncio
    async def test_get_betting_stats(self, async_client: AsyncClient, auth_headers):
        """Test getting betting statistics."""
        response = await async_client.get(
            "/api/v1/betting/stats",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "total_bets" in data
        assert "roi" in data


class TestPlayerPropsEndpoints:
    """Test player props API endpoints."""
    
    @pytest.mark.asyncio
    async def test_list_player_props(self, async_client: AsyncClient, auth_headers):
        """Test listing player props."""
        response = await async_client.get(
            "/api/v1/player-props",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "props" in data
    
    @pytest.mark.asyncio
    async def test_get_props_by_game(
        self, async_client: AsyncClient, auth_headers, test_game
    ):
        """Test getting props for specific game."""
        response = await async_client.get(
            f"/api/v1/player-props/game/{test_game.id}",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    @pytest.mark.asyncio
    async def test_get_today_props(self, async_client: AsyncClient, auth_headers):
        """Test getting today's props."""
        response = await async_client.get(
            "/api/v1/player-props/today",
            headers=auth_headers
        )
        
        assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_get_prop_types(self, async_client: AsyncClient, auth_headers):
        """Test getting available prop types."""
        response = await async_client.get(
            "/api/v1/player-props/prop-types",
            params={"sport_code": "NBA"},
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "NBA" in data
    
    @pytest.mark.asyncio
    async def test_search_players(self, async_client: AsyncClient, auth_headers):
        """Test player search."""
        response = await async_client.get(
            "/api/v1/player-props/players/search",
            params={"query": "James"},
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


class TestHealthEndpoints:
    """Test health check endpoints."""
    
    @pytest.mark.asyncio
    async def test_basic_health(self, async_client: AsyncClient):
        """Test basic health check."""
        response = await async_client.get("/api/v1/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    @pytest.mark.asyncio
    async def test_detailed_health(self, async_client: AsyncClient, auth_headers):
        """Test detailed health check."""
        response = await async_client.get(
            "/api/v1/health/detailed",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "components" in data
        assert "system" in data
    
    @pytest.mark.asyncio
    async def test_readiness_probe(self, async_client: AsyncClient):
        """Test readiness probe."""
        response = await async_client.get("/api/v1/health/ready")
        
        assert response.status_code in [200, 503]
    
    @pytest.mark.asyncio
    async def test_liveness_probe(self, async_client: AsyncClient):
        """Test liveness probe."""
        response = await async_client.get("/api/v1/health/live")
        
        assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_metrics_endpoint(self, async_client: AsyncClient, auth_headers):
        """Test metrics endpoint."""
        response = await async_client.get(
            "/api/v1/health/metrics",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "prediction_accuracy" in data or "requests_total" in data


class TestBacktestEndpoints:
    """Test backtesting endpoints."""
    
    @pytest.mark.asyncio
    async def test_run_backtest(self, async_client: AsyncClient, admin_headers):
        """Test running a backtest."""
        response = await async_client.post(
            "/api/v1/backtest/run",
            json={
                "sport_code": "NBA",
                "bet_type": "spread",
                "start_date": "2024-01-01",
                "end_date": "2024-06-30",
                "initial_bankroll": 10000.0
            },
            headers=admin_headers
        )
        
        assert response.status_code in [200, 202]
        data = response.json()
        assert "backtest_id" in data or "status" in data
    
    @pytest.mark.asyncio
    async def test_get_backtest_results(
        self, async_client: AsyncClient, admin_headers
    ):
        """Test getting backtest results."""
        response = await async_client.get(
            "/api/v1/backtest/1",
            headers=admin_headers
        )
        
        # May return 404 if no backtest exists
        assert response.status_code in [200, 404]


class TestAdminEndpoints:
    """Test admin-only endpoints."""
    
    @pytest.mark.asyncio
    async def test_generate_predictions_requires_admin(
        self, async_client: AsyncClient, auth_headers
    ):
        """Test that prediction generation requires admin role."""
        response = await async_client.post(
            "/api/v1/predictions/generate",
            json={"sport_code": "NBA"},
            headers=auth_headers  # Regular user
        )
        
        assert response.status_code == 403
    
    @pytest.mark.asyncio
    async def test_generate_predictions_admin(
        self, async_client: AsyncClient, admin_headers
    ):
        """Test prediction generation as admin."""
        response = await async_client.post(
            "/api/v1/predictions/generate",
            json={"sport_code": "NBA"},
            headers=admin_headers
        )
        
        assert response.status_code in [200, 202]
    
    @pytest.mark.asyncio
    async def test_grade_predictions_requires_admin(
        self, async_client: AsyncClient, auth_headers
    ):
        """Test that grading requires admin role."""
        response = await async_client.post(
            "/api/v1/predictions/grade",
            headers=auth_headers
        )
        
        assert response.status_code == 403
    
    @pytest.mark.asyncio
    async def test_refresh_odds_requires_admin(
        self, async_client: AsyncClient, auth_headers
    ):
        """Test that odds refresh requires admin role."""
        response = await async_client.post(
            "/api/v1/odds/refresh",
            headers=auth_headers
        )
        
        assert response.status_code == 403


# Test fixtures
@pytest.fixture
async def async_client(app):
    """Create async HTTP client for testing."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


@pytest.fixture
async def test_user(db_session):
    """Create test user."""
    from app.models import User
    from app.core.security import get_password_hash
    
    user = User(
        email="test@example.com",
        username="testuser",
        password_hash=get_password_hash("testpassword"),
        role="user",
        is_active=True,
        created_at=datetime.utcnow()
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    return user


@pytest.fixture
async def admin_user(db_session):
    """Create admin test user."""
    from app.models import User
    from app.core.security import get_password_hash
    
    user = User(
        email="admin@example.com",
        username="admin",
        password_hash=get_password_hash("adminpassword"),
        role="admin",
        is_active=True,
        created_at=datetime.utcnow()
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    return user


@pytest.fixture
def auth_headers(test_user):
    """Get auth headers for regular user."""
    from app.core.security import create_access_token
    
    token = create_access_token({"sub": str(test_user.id), "role": "user"})
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def admin_headers(admin_user):
    """Get auth headers for admin user."""
    from app.core.security import create_access_token
    
    token = create_access_token({"sub": str(admin_user.id), "role": "admin"})
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def refresh_token(test_user):
    """Get refresh token for test user."""
    from app.core.security import create_refresh_token
    
    return create_refresh_token({"sub": str(test_user.id)})


@pytest.fixture
async def test_game(db_session):
    """Create test game."""
    from app.models import Game
    
    game = Game(
        external_id="test_game_123",
        sport_code="NBA",
        home_team_id=1,
        away_team_id=2,
        scheduled_at=datetime.utcnow() + timedelta(hours=3),
        status="scheduled"
    )
    db_session.add(game)
    await db_session.commit()
    await db_session.refresh(game)
    return game


@pytest.fixture
async def test_prediction(db_session, test_game):
    """Create test prediction."""
    from app.models import Prediction
    import hashlib
    import json
    
    prediction = Prediction(
        game_id=test_game.id,
        sport_code="NBA",
        bet_type="spread",
        predicted_side="home",
        probability=0.65,
        edge=0.08,
        signal_tier="A",
        line=-3.5,
        odds=-110,
        created_at=datetime.utcnow()
    )
    
    # Generate hash
    hash_data = {
        "game_id": prediction.game_id,
        "bet_type": prediction.bet_type,
        "predicted_side": prediction.predicted_side,
        "probability": prediction.probability
    }
    prediction.prediction_hash = hashlib.sha256(
        json.dumps(hash_data, sort_keys=True).encode()
    ).hexdigest()
    
    db_session.add(prediction)
    await db_session.commit()
    await db_session.refresh(prediction)
    return prediction
