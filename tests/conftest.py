"""
LOYALEY - Test Configuration
Pytest fixtures and configuration for the test suite.
"""

import asyncio
import pytest
from typing import AsyncGenerator, Generator

@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def sample_game_data():
    """Sample game data for testing."""
    return {
        "game_id": "test-game-001",
        "sport": "NBA",
        "home_team": "Lakers",
        "away_team": "Celtics",
        "home_score": 105,
        "away_score": 98,
        "spread": -3.5,
        "total": 220.5
    }

@pytest.fixture
def sample_prediction_data():
    """Sample prediction data for testing."""
    return {
        "game_id": "test-game-001",
        "bet_type": "spread",
        "predicted_side": "home",
        "probability": 0.65,
        "line": -3.5,
        "odds": -110
    }
