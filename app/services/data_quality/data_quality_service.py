"""
LOYALEY - Phase 4 Data Quality Monitoring Service
Comprehensive data validation, anomaly detection, and quality scoring
"""

import asyncio
import logging
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from app.core.config import settings
from app.core.database import db_manager
from app.services.alerting.alerting_service import alerting_service, AlertSeverity

logger = logging.getLogger(__name__)


class QualityLevel(str, Enum):
    """Data quality levels"""
    EXCELLENT = "excellent"  # 95%+
    GOOD = "good"           # 85-95%
    ACCEPTABLE = "acceptable"  # 70-85%
    POOR = "poor"           # 50-70%
    CRITICAL = "critical"   # <50%


class ValidationRule(str, Enum):
    """Validation rule types"""
    REQUIRED = "required"
    RANGE = "range"
    TYPE = "type"
    PATTERN = "pattern"
    UNIQUENESS = "uniqueness"
    REFERENTIAL = "referential"
    CONSISTENCY = "consistency"
    FRESHNESS = "freshness"


@dataclass
class ValidationResult:
    """Result of a validation check"""
    rule: ValidationRule
    field: str
    passed: bool
    message: str = ""
    value: Any = None
    expected: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityReport:
    """Data quality report"""
    entity_type: str
    timestamp: datetime
    total_records: int
    valid_records: int
    quality_score: float
    quality_level: QualityLevel
    validation_results: List[ValidationResult]
    anomalies: List[Dict[str, Any]]
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_type": self.entity_type,
            "timestamp": self.timestamp.isoformat(),
            "total_records": self.total_records,
            "valid_records": self.valid_records,
            "quality_score": round(self.quality_score, 2),
            "quality_level": self.quality_level.value,
            "validation_summary": {
                "passed": sum(1 for r in self.validation_results if r.passed),
                "failed": sum(1 for r in self.validation_results if not r.passed)
            },
            "anomalies_count": len(self.anomalies),
            "recommendations": self.recommendations
        }


class DataValidator:
    """Validates data against defined rules"""
    
    def __init__(self):
        self._rules: Dict[str, List[Dict[str, Any]]] = {}
        self._register_default_rules()
    
    def _register_default_rules(self):
        """Register default validation rules for all entity types"""
        
        # Odds validation rules
        self._rules["odds"] = [
            {"rule": ValidationRule.REQUIRED, "field": "game_id"},
            {"rule": ValidationRule.REQUIRED, "field": "sportsbook"},
            {"rule": ValidationRule.REQUIRED, "field": "recorded_at"},
            {"rule": ValidationRule.RANGE, "field": "spread", "min": -50, "max": 50},
            {"rule": ValidationRule.RANGE, "field": "home_ml", "min": -10000, "max": 10000},
            {"rule": ValidationRule.RANGE, "field": "away_ml", "min": -10000, "max": 10000},
            {"rule": ValidationRule.RANGE, "field": "total", "min": 20, "max": 400},
            {"rule": ValidationRule.CONSISTENCY, "field": "moneyline", "check": "opposite_signs"},
        ]
        
        # Games validation rules
        self._rules["games"] = [
            {"rule": ValidationRule.REQUIRED, "field": "external_id"},
            {"rule": ValidationRule.REQUIRED, "field": "home_team_id"},
            {"rule": ValidationRule.REQUIRED, "field": "away_team_id"},
            {"rule": ValidationRule.REQUIRED, "field": "scheduled_at"},
            {"rule": ValidationRule.REFERENTIAL, "field": "home_team_id", "ref_table": "teams"},
            {"rule": ValidationRule.REFERENTIAL, "field": "away_team_id", "ref_table": "teams"},
            {"rule": ValidationRule.CONSISTENCY, "field": "teams", "check": "different_teams"},
        ]
        
        # Predictions validation rules
        self._rules["predictions"] = [
            {"rule": ValidationRule.REQUIRED, "field": "game_id"},
            {"rule": ValidationRule.REQUIRED, "field": "probability"},
            {"rule": ValidationRule.REQUIRED, "field": "bet_type"},
            {"rule": ValidationRule.RANGE, "field": "probability", "min": 0, "max": 1},
            {"rule": ValidationRule.RANGE, "field": "edge", "min": -1, "max": 1},
            {"rule": ValidationRule.FRESHNESS, "field": "created_at", "max_age_hours": 24},
        ]
        
        # Teams validation rules
        self._rules["teams"] = [
            {"rule": ValidationRule.REQUIRED, "field": "name"},
            {"rule": ValidationRule.REQUIRED, "field": "sport_code"},
            {"rule": ValidationRule.UNIQUENESS, "field": "external_id"},
            {"rule": ValidationRule.RANGE, "field": "elo_rating", "min": 800, "max": 2000},
        ]
    
    def validate_record(
        self,
        entity_type: str,
        record: Dict[str, Any]
    ) -> List[ValidationResult]:
        """Validate a single record against rules"""
        results = []
        rules = self._rules.get(entity_type, [])
        
        for rule_def in rules:
            result = self._apply_rule(rule_def, record)
            results.append(result)
        
        return results
    
    def _apply_rule(
        self,
        rule_def: Dict[str, Any],
        record: Dict[str, Any]
    ) -> ValidationResult:
        """Apply a validation rule to a record"""
        rule = rule_def["rule"]
        field = rule_def["field"]
        value = record.get(field)
        
        if rule == ValidationRule.REQUIRED:
            passed = value is not None and value != ""
            return ValidationResult(
                rule=rule,
                field=field,
                passed=passed,
                message=f"Field {field} is required" if not passed else "",
                value=value
            )
        
        elif rule == ValidationRule.RANGE:
            if value is None:
                return ValidationResult(rule=rule, field=field, passed=True)
            
            min_val = rule_def.get("min")
            max_val = rule_def.get("max")
            passed = True
            
            if min_val is not None and value < min_val:
                passed = False
            if max_val is not None and value > max_val:
                passed = False
            
            return ValidationResult(
                rule=rule,
                field=field,
                passed=passed,
                message=f"Value {value} out of range [{min_val}, {max_val}]" if not passed else "",
                value=value,
                expected={"min": min_val, "max": max_val}
            )
        
        elif rule == ValidationRule.FRESHNESS:
            if value is None:
                return ValidationResult(rule=rule, field=field, passed=False)
            
            max_age = timedelta(hours=rule_def.get("max_age_hours", 24))
            if isinstance(value, str):
                value = datetime.fromisoformat(value.replace("Z", "+00:00"))
            
            age = datetime.now(timezone.utc) - value.replace(tzinfo=timezone.utc)
            passed = age <= max_age
            
            return ValidationResult(
                rule=rule,
                field=field,
                passed=passed,
                message=f"Data is stale ({age.total_seconds() / 3600:.1f} hours old)" if not passed else "",
                value=value
            )
        
        elif rule == ValidationRule.CONSISTENCY:
            check = rule_def.get("check")
            
            if check == "opposite_signs":
                home_ml = record.get("home_ml", 0)
                away_ml = record.get("away_ml", 0)
                passed = (home_ml * away_ml) < 0 or (home_ml == 0 and away_ml == 0)
                return ValidationResult(
                    rule=rule,
                    field=field,
                    passed=passed,
                    message="Moneylines should have opposite signs" if not passed else ""
                )
            
            elif check == "different_teams":
                home_id = record.get("home_team_id")
                away_id = record.get("away_team_id")
                passed = home_id != away_id
                return ValidationResult(
                    rule=rule,
                    field=field,
                    passed=passed,
                    message="Home and away teams must be different" if not passed else ""
                )
        
        return ValidationResult(rule=rule, field=field, passed=True)


class AnomalyDetector:
    """Detects anomalies in data"""
    
    def __init__(self):
        self._thresholds = {
            "z_score": 3.0,
            "iqr_multiplier": 1.5,
            "min_sample_size": 10
        }
    
    def detect_outliers(
        self,
        values: List[float],
        method: str = "zscore"
    ) -> List[Tuple[int, float, str]]:
        """Detect outliers in numeric values"""
        if len(values) < self._thresholds["min_sample_size"]:
            return []
        
        outliers = []
        
        if method == "zscore":
            mean = statistics.mean(values)
            std = statistics.stdev(values)
            
            if std == 0:
                return []
            
            for i, val in enumerate(values):
                z = abs((val - mean) / std)
                if z > self._thresholds["z_score"]:
                    outliers.append((i, val, f"Z-score: {z:.2f}"))
        
        elif method == "iqr":
            sorted_vals = sorted(values)
            n = len(sorted_vals)
            q1 = sorted_vals[n // 4]
            q3 = sorted_vals[3 * n // 4]
            iqr = q3 - q1
            
            lower = q1 - self._thresholds["iqr_multiplier"] * iqr
            upper = q3 + self._thresholds["iqr_multiplier"] * iqr
            
            for i, val in enumerate(values):
                if val < lower or val > upper:
                    outliers.append((i, val, f"Outside IQR bounds [{lower:.2f}, {upper:.2f}]"))
        
        return outliers
    
    def detect_probability_bias(
        self,
        probabilities: List[float]
    ) -> Dict[str, Any]:
        """Detect bias in probability distributions"""
        if len(probabilities) < 20:
            return {"biased": False}
        
        mean_prob = statistics.mean(probabilities)
        std_prob = statistics.stdev(probabilities)
        
        # Check for bias toward 0.5
        center_count = sum(1 for p in probabilities if 0.45 <= p <= 0.55)
        center_ratio = center_count / len(probabilities)
        
        # Check for extreme clustering
        extreme_count = sum(1 for p in probabilities if p < 0.2 or p > 0.8)
        extreme_ratio = extreme_count / len(probabilities)
        
        biased = center_ratio > 0.5 or extreme_ratio > 0.3
        
        return {
            "biased": biased,
            "mean": mean_prob,
            "std": std_prob,
            "center_ratio": center_ratio,
            "extreme_ratio": extreme_ratio,
            "message": "Probability distribution appears biased" if biased else ""
        }
    
    def detect_edge_inflation(
        self,
        edges: List[float]
    ) -> Dict[str, Any]:
        """Detect unrealistic edge values"""
        if len(edges) < 10:
            return {"inflated": False}
        
        mean_edge = statistics.mean(edges)
        positive_count = sum(1 for e in edges if e > 0)
        high_edge_count = sum(1 for e in edges if e > 0.15)
        
        inflated = mean_edge > 0.05 or (positive_count / len(edges)) > 0.8 or high_edge_count > 5
        
        return {
            "inflated": inflated,
            "mean_edge": mean_edge,
            "positive_ratio": positive_count / len(edges),
            "high_edge_count": high_edge_count,
            "message": "Edge values appear inflated" if inflated else ""
        }
    
    def detect_line_movement_anomaly(
        self,
        movements: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Detect anomalous line movements"""
        anomalies = []
        
        for movement in movements:
            change = abs(movement.get("change", 0))
            time_minutes = movement.get("time_minutes", 60)
            
            # Rapid large movement
            if change > 2 and time_minutes < 30:
                anomalies.append({
                    "type": "rapid_movement",
                    "change": change,
                    "time_minutes": time_minutes,
                    "message": f"Rapid {change} point movement in {time_minutes} minutes"
                })
            
            # Suspicious steam move
            if movement.get("is_steam_move", False) and change > 1.5:
                anomalies.append({
                    "type": "suspicious_steam",
                    "change": change,
                    "message": f"Suspicious steam move of {change} points"
                })
        
        return anomalies


class DataQualityService:
    """Enterprise data quality monitoring service"""
    
    def __init__(self):
        self.validator = DataValidator()
        self.anomaly_detector = AnomalyDetector()
        self._quality_history: Dict[str, List[QualityReport]] = {}
        self._max_history = 100
    
    async def run_quality_check(
        self,
        entity_type: str,
        records: List[Dict[str, Any]]
    ) -> QualityReport:
        """Run comprehensive quality check on records"""
        if not records:
            return QualityReport(
                entity_type=entity_type,
                timestamp=datetime.now(timezone.utc),
                total_records=0,
                valid_records=0,
                quality_score=0,
                quality_level=QualityLevel.CRITICAL,
                validation_results=[],
                anomalies=[],
                recommendations=["No records to validate"]
            )
        
        # Run validation on all records
        all_results = []
        valid_count = 0
        
        for record in records:
            results = self.validator.validate_record(entity_type, record)
            all_results.extend(results)
            
            if all(r.passed for r in results):
                valid_count += 1
        
        # Calculate quality score
        passed_count = sum(1 for r in all_results if r.passed)
        total_checks = len(all_results)
        quality_score = (passed_count / total_checks * 100) if total_checks > 0 else 0
        
        # Determine quality level
        quality_level = self._get_quality_level(quality_score)
        
        # Detect anomalies
        anomalies = await self._detect_anomalies(entity_type, records)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            entity_type, all_results, anomalies, quality_score
        )
        
        # Create report
        report = QualityReport(
            entity_type=entity_type,
            timestamp=datetime.now(timezone.utc),
            total_records=len(records),
            valid_records=valid_count,
            quality_score=quality_score,
            quality_level=quality_level,
            validation_results=all_results,
            anomalies=anomalies,
            recommendations=recommendations
        )
        
        # Store in history
        self._store_report(report)
        
        # Send alerts for poor quality
        if quality_level in [QualityLevel.POOR, QualityLevel.CRITICAL]:
            await self._send_quality_alert(report)
        
        return report
    
    def _get_quality_level(self, score: float) -> QualityLevel:
        """Determine quality level from score"""
        if score >= 95:
            return QualityLevel.EXCELLENT
        elif score >= 85:
            return QualityLevel.GOOD
        elif score >= 70:
            return QualityLevel.ACCEPTABLE
        elif score >= 50:
            return QualityLevel.POOR
        else:
            return QualityLevel.CRITICAL
    
    async def _detect_anomalies(
        self,
        entity_type: str,
        records: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Detect anomalies in records"""
        anomalies = []
        
        if entity_type == "predictions":
            # Check probability distribution
            probabilities = [r.get("probability", 0.5) for r in records if "probability" in r]
            if probabilities:
                bias_result = self.anomaly_detector.detect_probability_bias(probabilities)
                if bias_result["biased"]:
                    anomalies.append({
                        "type": "probability_bias",
                        **bias_result
                    })
            
            # Check edge values
            edges = [r.get("edge", 0) for r in records if "edge" in r]
            if edges:
                inflation_result = self.anomaly_detector.detect_edge_inflation(edges)
                if inflation_result["inflated"]:
                    anomalies.append({
                        "type": "edge_inflation",
                        **inflation_result
                    })
        
        elif entity_type == "odds":
            # Check for spread outliers
            spreads = [r.get("spread", 0) for r in records if "spread" in r]
            if spreads:
                outliers = self.anomaly_detector.detect_outliers(spreads)
                for idx, val, msg in outliers:
                    anomalies.append({
                        "type": "spread_outlier",
                        "index": idx,
                        "value": val,
                        "message": msg
                    })
            
            # Check for total outliers
            totals = [r.get("total", 0) for r in records if "total" in r]
            if totals:
                outliers = self.anomaly_detector.detect_outliers(totals)
                for idx, val, msg in outliers:
                    anomalies.append({
                        "type": "total_outlier",
                        "index": idx,
                        "value": val,
                        "message": msg
                    })
        
        return anomalies
    
    def _generate_recommendations(
        self,
        entity_type: str,
        results: List[ValidationResult],
        anomalies: List[Dict[str, Any]],
        quality_score: float
    ) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        
        # Analyze failed validations
        failed_rules: Dict[ValidationRule, int] = {}
        for result in results:
            if not result.passed:
                failed_rules[result.rule] = failed_rules.get(result.rule, 0) + 1
        
        for rule, count in failed_rules.items():
            if rule == ValidationRule.REQUIRED and count > 5:
                recommendations.append(f"Fix {count} records with missing required fields")
            elif rule == ValidationRule.RANGE and count > 3:
                recommendations.append(f"Review {count} records with out-of-range values")
            elif rule == ValidationRule.FRESHNESS and count > 0:
                recommendations.append(f"Refresh stale data - {count} records are outdated")
        
        # Analyze anomalies
        for anomaly in anomalies:
            if anomaly["type"] == "probability_bias":
                recommendations.append("Review model calibration - probability distribution appears biased")
            elif anomaly["type"] == "edge_inflation":
                recommendations.append("Verify edge calculations - values appear inflated")
            elif anomaly["type"] in ["spread_outlier", "total_outlier"]:
                recommendations.append(f"Review outlier {anomaly['type']}: {anomaly['message']}")
        
        # Quality score recommendations
        if quality_score < 70:
            recommendations.append("Consider implementing additional data validation at source")
        if quality_score < 50:
            recommendations.append("CRITICAL: Data quality requires immediate attention")
        
        return recommendations
    
    def _store_report(self, report: QualityReport):
        """Store report in history"""
        entity_type = report.entity_type
        if entity_type not in self._quality_history:
            self._quality_history[entity_type] = []
        
        self._quality_history[entity_type].append(report)
        
        # Trim history
        if len(self._quality_history[entity_type]) > self._max_history:
            self._quality_history[entity_type] = self._quality_history[entity_type][-self._max_history:]
    
    async def _send_quality_alert(self, report: QualityReport):
        """Send alert for poor data quality"""
        severity = AlertSeverity.CRITICAL if report.quality_level == QualityLevel.CRITICAL else AlertSeverity.WARNING
        
        await alerting_service.send_alert(
            title=f"Data Quality Alert: {report.entity_type}",
            message=f"Data quality score: {report.quality_score:.1f}% ({report.quality_level.value})",
            severity=severity,
            source="data_quality",
            metadata={
                "entity_type": report.entity_type,
                "total_records": report.total_records,
                "valid_records": report.valid_records,
                "anomalies_count": len(report.anomalies),
                "recommendations": report.recommendations[:3]
            }
        )
    
    def get_quality_history(
        self,
        entity_type: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get quality check history for entity type"""
        history = self._quality_history.get(entity_type, [])
        return [r.to_dict() for r in history[-limit:]]
    
    def get_quality_trend(
        self,
        entity_type: str,
        days: int = 7
    ) -> Dict[str, Any]:
        """Get quality score trend over time"""
        history = self._quality_history.get(entity_type, [])
        
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        recent = [r for r in history if r.timestamp >= cutoff]
        
        if not recent:
            return {"trend": "unknown", "data_points": 0}
        
        scores = [r.quality_score for r in recent]
        
        return {
            "current_score": scores[-1] if scores else 0,
            "average_score": statistics.mean(scores),
            "min_score": min(scores),
            "max_score": max(scores),
            "trend": "improving" if len(scores) > 1 and scores[-1] > scores[0] else "declining",
            "data_points": len(scores)
        }


# Global data quality service instance
data_quality_service = DataQualityService()

def get_data_quality_service() -> DataQualityService:
    """
    Dependency-style accessor for data quality service.
    Keeps imports stable and avoids circular imports.
    """
    return data_quality_service