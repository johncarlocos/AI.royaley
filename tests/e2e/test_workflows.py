"""
ROYALEY - End-to-End Tests
Complete workflow tests simulating real user scenarios
"""

import pytest
from datetime import datetime, timedelta
from httpx import AsyncClient

pytestmark = pytest.mark.e2e


class TestCompleteUserWorkflow:
    """Test complete user journey from registration to betting."""
    
    @pytest.mark.asyncio
    async def test_full_user_journey(self, async_client: AsyncClient):
        """Test complete user workflow."""
        # Step 1: Register
        register_response = await async_client.post(
            "/api/v1/auth/register",
            json={
                "email": "newuser@example.com",
                "username": "newuser",
                "password": "securepassword123",
                "full_name": "New User"
            }
        )
        assert register_response.status_code == 201
        user_data = register_response.json()
        
        # Step 2: Login
        login_response = await async_client.post(
            "/api/v1/auth/login",
            data={
                "username": "newuser",
                "password": "securepassword123"
            }
        )
        assert login_response.status_code == 200
        tokens = login_response.json()
        headers = {"Authorization": f"Bearer {tokens['access_token']}"}
        
        # Step 3: Create bankroll
        bankroll_response = await async_client.post(
            "/api/v1/betting/bankroll",
            json={
                "name": "Main Bankroll",
                "initial_balance": 5000.0,
                "currency": "USD"
            },
            headers=headers
        )
        assert bankroll_response.status_code in [200, 201]
        
        # Step 4: View today's predictions
        predictions_response = await async_client.get(
            "/api/v1/predictions/today",
            headers=headers
        )
        assert predictions_response.status_code == 200
        
        # Step 5: View games
        games_response = await async_client.get(
            "/api/v1/games/today",
            headers=headers
        )
        assert games_response.status_code == 200
        
        # Step 6: Check prediction stats
        stats_response = await async_client.get(
            "/api/v1/predictions/stats",
            headers=headers
        )
        assert stats_response.status_code == 200
        
        # Step 7: View betting history
        history_response = await async_client.get(
            "/api/v1/betting/history",
            headers=headers
        )
        assert history_response.status_code == 200
        
        # Step 8: Logout
        logout_response = await async_client.post(
            "/api/v1/auth/logout",
            headers=headers
        )
        assert logout_response.status_code == 200


class TestPredictionToSettlementWorkflow:
    """Test prediction lifecycle from creation to settlement."""
    
    @pytest.mark.asyncio
    async def test_prediction_lifecycle(
        self, async_client: AsyncClient, admin_headers, test_game
    ):
        """Test complete prediction lifecycle."""
        # Step 1: Generate predictions (admin)
        generate_response = await async_client.post(
            "/api/v1/predictions/generate",
            json={"sport_code": "NBA"},
            headers=admin_headers
        )
        assert generate_response.status_code in [200, 202]
        
        # Step 2: View generated predictions
        predictions_response = await async_client.get(
            "/api/v1/predictions",
            params={"sport_code": "NBA", "graded": False},
            headers=admin_headers
        )
        assert predictions_response.status_code == 200
        predictions = predictions_response.json()
        
        if predictions.get("predictions"):
            prediction = predictions["predictions"][0]
            
            # Step 3: Verify prediction hash
            verify_response = await async_client.post(
                f"/api/v1/predictions/{prediction['id']}/verify",
                headers=admin_headers
            )
            assert verify_response.status_code == 200
            assert verify_response.json()["valid"] is True
            
            # Step 4: View SHAP explanations
            detail_response = await async_client.get(
                f"/api/v1/predictions/{prediction['id']}",
                headers=admin_headers
            )
            assert detail_response.status_code == 200
            assert "shap_explanations" in detail_response.json()
        
        # Step 5: Grade predictions (after game completion)
        grade_response = await async_client.post(
            "/api/v1/predictions/grade",
            headers=admin_headers
        )
        assert grade_response.status_code in [200, 202]


class TestBettingWorkflow:
    """Test complete betting workflow."""
    
    @pytest.mark.asyncio
    async def test_betting_workflow(
        self, async_client: AsyncClient, auth_headers, 
        test_prediction, test_bankroll
    ):
        """Test complete betting workflow."""
        # Step 1: Calculate bet sizing
        sizing_response = await async_client.post(
            "/api/v1/betting/sizing",
            json={
                "probability": 0.65,
                "odds": -110,
                "bankroll": 10000.0
            },
            headers=auth_headers
        )
        assert sizing_response.status_code == 200
        sizing = sizing_response.json()
        assert sizing["recommended_bet"] > 0
        
        # Step 2: Place bet
        bet_response = await async_client.post(
            "/api/v1/betting/bet",
            json={
                "prediction_id": test_prediction.id,
                "stake": sizing["recommended_bet"],
                "odds": -110,
                "sportsbook": "pinnacle"
            },
            headers=auth_headers
        )
        assert bet_response.status_code in [200, 201]
        
        # Step 3: View bet history
        history_response = await async_client.get(
            "/api/v1/betting/history",
            headers=auth_headers
        )
        assert history_response.status_code == 200
        
        # Step 4: Check CLV
        clv_response = await async_client.get(
            "/api/v1/betting/clv",
            headers=auth_headers
        )
        assert clv_response.status_code == 200
        
        # Step 5: View betting stats
        stats_response = await async_client.get(
            "/api/v1/betting/stats",
            headers=auth_headers
        )
        assert stats_response.status_code == 200


class TestOddsMonitoringWorkflow:
    """Test odds monitoring workflow."""
    
    @pytest.mark.asyncio
    async def test_odds_workflow(
        self, async_client: AsyncClient, auth_headers, test_game
    ):
        """Test odds monitoring workflow."""
        # Step 1: Get current odds
        odds_response = await async_client.get(
            f"/api/v1/odds/{test_game.id}",
            headers=auth_headers
        )
        assert odds_response.status_code == 200
        
        # Step 2: Get best odds
        best_response = await async_client.get(
            f"/api/v1/odds/{test_game.id}/best",
            headers=auth_headers
        )
        assert best_response.status_code == 200
        
        # Step 3: Get line movement
        movement_response = await async_client.get(
            f"/api/v1/odds/{test_game.id}/movement",
            headers=auth_headers
        )
        assert movement_response.status_code == 200
        
        # Step 4: Get live odds
        live_response = await async_client.get(
            "/api/v1/odds/live",
            headers=auth_headers
        )
        assert live_response.status_code == 200


class TestPlayerPropsWorkflow:
    """Test player props workflow."""
    
    @pytest.mark.asyncio
    async def test_props_workflow(
        self, async_client: AsyncClient, auth_headers, test_game
    ):
        """Test player props workflow."""
        # Step 1: Get today's props
        today_response = await async_client.get(
            "/api/v1/player-props/today",
            headers=auth_headers
        )
        assert today_response.status_code == 200
        
        # Step 2: Get props by game
        game_props_response = await async_client.get(
            f"/api/v1/player-props/game/{test_game.id}",
            headers=auth_headers
        )
        assert game_props_response.status_code == 200
        
        # Step 3: Search players
        search_response = await async_client.get(
            "/api/v1/player-props/players/search",
            params={"query": "James"},
            headers=auth_headers
        )
        assert search_response.status_code == 200
        
        # Step 4: Get prop types
        types_response = await async_client.get(
            "/api/v1/player-props/prop-types",
            params={"sport_code": "NBA"},
            headers=auth_headers
        )
        assert types_response.status_code == 200
        
        # Step 5: Get props stats
        stats_response = await async_client.get(
            "/api/v1/player-props/stats",
            headers=auth_headers
        )
        assert stats_response.status_code == 200


class TestSystemHealthWorkflow:
    """Test system health monitoring workflow."""
    
    @pytest.mark.asyncio
    async def test_health_workflow(
        self, async_client: AsyncClient, admin_headers
    ):
        """Test health monitoring workflow."""
        # Step 1: Basic health check
        health_response = await async_client.get("/api/v1/health")
        assert health_response.status_code == 200
        assert health_response.json()["status"] == "healthy"
        
        # Step 2: Readiness probe
        ready_response = await async_client.get("/api/v1/health/ready")
        assert ready_response.status_code in [200, 503]
        
        # Step 3: Liveness probe
        live_response = await async_client.get("/api/v1/health/live")
        assert live_response.status_code == 200
        
        # Step 4: Detailed health (authenticated)
        detailed_response = await async_client.get(
            "/api/v1/health/detailed",
            headers=admin_headers
        )
        assert detailed_response.status_code == 200
        
        # Step 5: Metrics
        metrics_response = await async_client.get(
            "/api/v1/health/metrics",
            headers=admin_headers
        )
        assert metrics_response.status_code == 200
        
        # Step 6: Components
        components_response = await async_client.get(
            "/api/v1/health/components",
            headers=admin_headers
        )
        assert components_response.status_code == 200
        
        # Step 7: Alerts
        alerts_response = await async_client.get(
            "/api/v1/health/alerts",
            headers=admin_headers
        )
        assert alerts_response.status_code == 200


class TestBacktestWorkflow:
    """Test backtesting workflow."""
    
    @pytest.mark.asyncio
    async def test_backtest_workflow(
        self, async_client: AsyncClient, admin_headers
    ):
        """Test backtesting workflow."""
        # Step 1: Run backtest
        run_response = await async_client.post(
            "/api/v1/backtest/run",
            json={
                "sport_code": "NBA",
                "bet_type": "spread",
                "start_date": "2024-01-01",
                "end_date": "2024-06-30",
                "initial_bankroll": 10000.0,
                "kelly_fraction": 0.25
            },
            headers=admin_headers
        )
        assert run_response.status_code in [200, 202]
        
        if run_response.status_code == 200:
            backtest_data = run_response.json()
            
            if "backtest_id" in backtest_data:
                # Step 2: Get backtest results
                results_response = await async_client.get(
                    f"/api/v1/backtest/{backtest_data['backtest_id']}",
                    headers=admin_headers
                )
                assert results_response.status_code in [200, 202]  # May be processing


class TestSecurityWorkflow:
    """Test security features workflow."""
    
    @pytest.mark.asyncio
    async def test_2fa_workflow(
        self, async_client: AsyncClient, auth_headers
    ):
        """Test 2FA setup workflow."""
        # Step 1: Setup 2FA
        setup_response = await async_client.post(
            "/api/v1/auth/2fa/setup",
            headers=auth_headers
        )
        assert setup_response.status_code in [200, 400]  # 400 if already enabled
        
        if setup_response.status_code == 200:
            setup_data = setup_response.json()
            assert "secret" in setup_data
            assert "qr_code" in setup_data
            assert "backup_codes" in setup_data
    
    @pytest.mark.asyncio
    async def test_session_management_workflow(
        self, async_client: AsyncClient, auth_headers
    ):
        """Test session management workflow."""
        # Step 1: List sessions
        sessions_response = await async_client.get(
            "/api/v1/auth/sessions",
            headers=auth_headers
        )
        assert sessions_response.status_code == 200
        
        sessions = sessions_response.json()
        if sessions.get("sessions"):
            session_id = sessions["sessions"][0]["id"]
            
            # Step 2: Revoke session
            revoke_response = await async_client.delete(
                f"/api/v1/auth/sessions/{session_id}",
                headers=auth_headers
            )
            assert revoke_response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_password_change_workflow(
        self, async_client: AsyncClient
    ):
        """Test password change workflow."""
        # First register and login
        await async_client.post(
            "/api/v1/auth/register",
            json={
                "email": "passchange@example.com",
                "username": "passchangeuser",
                "password": "oldpassword123"
            }
        )
        
        login_response = await async_client.post(
            "/api/v1/auth/login",
            data={
                "username": "passchangeuser",
                "password": "oldpassword123"
            }
        )
        tokens = login_response.json()
        headers = {"Authorization": f"Bearer {tokens['access_token']}"}
        
        # Change password
        change_response = await async_client.post(
            "/api/v1/auth/change-password",
            json={
                "current_password": "oldpassword123",
                "new_password": "newpassword456"
            },
            headers=headers
        )
        assert change_response.status_code == 200
        
        # Verify new password works
        new_login_response = await async_client.post(
            "/api/v1/auth/login",
            data={
                "username": "passchangeuser",
                "password": "newpassword456"
            }
        )
        assert new_login_response.status_code == 200


class TestDataRefreshWorkflow:
    """Test data refresh workflow (admin only)."""
    
    @pytest.mark.asyncio
    async def test_data_refresh_workflow(
        self, async_client: AsyncClient, admin_headers
    ):
        """Test data refresh workflow."""
        # Step 1: Refresh odds
        odds_refresh = await async_client.post(
            "/api/v1/odds/refresh",
            headers=admin_headers
        )
        assert odds_refresh.status_code in [200, 202]
        
        # Step 2: Refresh games
        games_refresh = await async_client.post(
            "/api/v1/games/refresh",
            headers=admin_headers
        )
        assert games_refresh.status_code in [200, 202]
        
        # Step 3: Generate predictions
        predict_response = await async_client.post(
            "/api/v1/predictions/generate",
            json={"sport_code": "NBA"},
            headers=admin_headers
        )
        assert predict_response.status_code in [200, 202]
        
        # Step 4: Grade predictions
        grade_response = await async_client.post(
            "/api/v1/predictions/grade",
            headers=admin_headers
        )
        assert grade_response.status_code in [200, 202]
