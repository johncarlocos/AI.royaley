"""
LOYALEY - Data Validation Unit Tests
Phase 1: Test Data Validation Components
"""

import pytest
from datetime import datetime, timedelta

from app.services.data_quality.validator import (
    DataValidator,
    ValidationLevel,
    ValidationResult,
    ValidationReport,
)


class TestValidationResult:
    """Test validation result structure."""
    
    def test_create_error_result(self):
        """Test creating error validation result."""
        result = ValidationResult(
            level=ValidationLevel.ERROR,
            field="price",
            message="Price out of range",
            value=-15000,
        )
        
        assert result.level == ValidationLevel.ERROR
        assert result.field == "price"
        assert "out of range" in result.message
    
    def test_create_warning_result(self):
        """Test creating warning validation result."""
        result = ValidationResult(
            level=ValidationLevel.WARNING,
            field="spread",
            message="Unusual spread value",
            value=45.5,
        )
        
        assert result.level == ValidationLevel.WARNING


class TestValidationReport:
    """Test validation report functionality."""
    
    def test_valid_report(self):
        """Test valid report with no errors."""
        results = [
            ValidationResult(
                level=ValidationLevel.INFO,
                field="status",
                message="All checks passed",
            ),
            ValidationResult(
                level=ValidationLevel.WARNING,
                field="total",
                message="High total value",
                value=280,
            ),
        ]
        
        report = ValidationReport(
            timestamp=datetime.utcnow(),
            source="odds",
            results=results,
        )
        
        assert report.is_valid  # No errors means valid
    
    def test_invalid_report(self):
        """Test invalid report with errors."""
        results = [
            ValidationResult(
                level=ValidationLevel.ERROR,
                field="price",
                message="Invalid price",
                value=None,
            ),
        ]
        
        report = ValidationReport(
            timestamp=datetime.utcnow(),
            source="odds",
            results=results,
        )
        
        assert not report.is_valid


class TestDataValidator:
    """Test data validator functionality."""
    
    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        return DataValidator()
    
    def test_validate_odds_data_valid(self, validator: DataValidator):
        """Test validation of valid odds data."""
        odds_data = [
            {
                "game_id": "game123",
                "sportsbook": "fanduel",
                "market_type": "spread",
                "selection": "home",
                "price": -110,
                "line": -3.5,
                "recorded_at": datetime.utcnow().isoformat(),
            },
        ]
        
        results = validator._validate_odds_data_sync(odds_data)
        errors = [r for r in results if r.level == ValidationLevel.ERROR]
        
        assert len(errors) == 0
    
    def test_validate_odds_missing_fields(self, validator: DataValidator):
        """Test validation catches missing fields."""
        odds_data = [
            {
                "game_id": "game123",
                # Missing sportsbook, market_type, etc.
            },
        ]
        
        results = validator._validate_odds_data_sync(odds_data)
        errors = [r for r in results if r.level == ValidationLevel.ERROR]
        
        assert len(errors) > 0
    
    def test_validate_odds_price_range(self, validator: DataValidator):
        """Test price range validation."""
        # Test price too high
        odds_data = [
            {
                "game_id": "game123",
                "sportsbook": "fanduel",
                "market_type": "spread",
                "selection": "home",
                "price": 15000,  # Too high
                "line": -3.5,
                "recorded_at": datetime.utcnow().isoformat(),
            },
        ]
        
        results = validator._validate_odds_data_sync(odds_data)
        errors = [r for r in results if r.level == ValidationLevel.ERROR]
        
        assert any("price" in r.field.lower() for r in errors)
    
    def test_validate_spread_range(self, validator: DataValidator):
        """Test spread range validation."""
        odds_data = [
            {
                "game_id": "game123",
                "sportsbook": "fanduel",
                "market_type": "spread",
                "selection": "home",
                "price": -110,
                "line": -75.0,  # Too extreme
                "recorded_at": datetime.utcnow().isoformat(),
            },
        ]
        
        results = validator._validate_odds_data_sync(odds_data)
        errors = [r for r in results if r.level == ValidationLevel.ERROR]
        
        assert any("spread" in r.field.lower() or "line" in r.field.lower() for r in errors)
    
    def test_validate_total_range(self, validator: DataValidator):
        """Test total line range validation."""
        odds_data = [
            {
                "game_id": "game123",
                "sportsbook": "fanduel",
                "market_type": "total",
                "selection": "over",
                "price": -110,
                "line": 500.0,  # Too high for any sport
                "recorded_at": datetime.utcnow().isoformat(),
            },
        ]
        
        results = validator._validate_odds_data_sync(odds_data)
        errors = [r for r in results if r.level == ValidationLevel.ERROR]
        
        assert any("total" in r.field.lower() or "line" in r.field.lower() for r in errors)
    
    def test_validate_market_type(self, validator: DataValidator):
        """Test market type validation."""
        odds_data = [
            {
                "game_id": "game123",
                "sportsbook": "fanduel",
                "market_type": "invalid_market",
                "selection": "home",
                "price": -110,
                "line": -3.5,
                "recorded_at": datetime.utcnow().isoformat(),
            },
        ]
        
        results = validator._validate_odds_data_sync(odds_data)
        errors = [r for r in results if r.level == ValidationLevel.ERROR]
        
        assert any("market" in r.field.lower() for r in errors)
    
    def test_validate_games_data_valid(self, validator: DataValidator):
        """Test validation of valid game data."""
        game_data = [
            {
                "external_id": "game123",
                "home_team_id": "team1",
                "away_team_id": "team2",
                "game_date": (datetime.utcnow() + timedelta(days=1)).isoformat(),
                "sport_code": "NBA",
            },
        ]
        
        results = validator._validate_games_data_sync(game_data)
        errors = [r for r in results if r.level == ValidationLevel.ERROR]
        
        assert len(errors) == 0
    
    def test_validate_games_missing_teams(self, validator: DataValidator):
        """Test game validation catches missing teams."""
        game_data = [
            {
                "external_id": "game123",
                "game_date": datetime.utcnow().isoformat(),
                # Missing team IDs
            },
        ]
        
        results = validator._validate_games_data_sync(game_data)
        errors = [r for r in results if r.level == ValidationLevel.ERROR]
        
        assert len(errors) > 0
    
    def test_validate_games_same_team(self, validator: DataValidator):
        """Test game validation catches same team playing itself."""
        game_data = [
            {
                "external_id": "game123",
                "home_team_id": "team1",
                "away_team_id": "team1",  # Same team
                "game_date": datetime.utcnow().isoformat(),
                "sport_code": "NBA",
            },
        ]
        
        results = validator._validate_games_data_sync(game_data)
        errors = [r for r in results if r.level == ValidationLevel.ERROR]
        
        assert any("team" in r.message.lower() for r in errors)


class TestValidationThresholds:
    """Test validation threshold constants."""
    
    def test_price_thresholds(self):
        """Verify price threshold values."""
        from app.services.data_quality.validator import DataValidator
        
        assert DataValidator.ODDS_PRICE_MIN == -10000
        assert DataValidator.ODDS_PRICE_MAX == 10000
    
    def test_spread_thresholds(self):
        """Verify spread threshold values."""
        from app.services.data_quality.validator import DataValidator
        
        assert DataValidator.SPREAD_MIN == -50.0
        assert DataValidator.SPREAD_MAX == 50.0
    
    def test_total_thresholds(self):
        """Verify total threshold values."""
        from app.services.data_quality.validator import DataValidator
        
        assert DataValidator.TOTAL_MIN == 50.0
        assert DataValidator.TOTAL_MAX == 400.0
