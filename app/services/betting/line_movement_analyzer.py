"""
ROYALEY - Line Movement Analyzer
Phase 2: Sharp Money Detection & Market Analysis

Analyzes line movements to detect sharp action, steam moves,
and reverse line movement patterns for betting edge.
"""

import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class LineMovementType(Enum):
    """Types of line movement."""
    STEAM = 'steam'                # Rapid movement from sharp action
    REVERSE = 'reverse'            # Against public betting
    ORGANIC = 'organic'            # Normal market adjustment
    STATIC = 'static'              # No movement
    BUYBACK = 'buyback'            # Line moves back after initial move


class MarketSentiment(Enum):
    """Market sentiment indicators."""
    SHARP_HOME = 'sharp_home'
    SHARP_AWAY = 'sharp_away'
    SHARP_OVER = 'sharp_over'
    SHARP_UNDER = 'sharp_under'
    PUBLIC_HOME = 'public_home'
    PUBLIC_AWAY = 'public_away'
    PUBLIC_OVER = 'public_over'
    PUBLIC_UNDER = 'public_under'
    NEUTRAL = 'neutral'


class AlertSeverity(Enum):
    """Alert severity levels."""
    HIGH = 'high'
    MEDIUM = 'medium'
    LOW = 'low'


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class OddsSnapshot:
    """Point-in-time odds data."""
    game_id: str
    timestamp: datetime
    
    # Spread
    spread_home: float
    spread_home_odds: int
    spread_away_odds: int
    
    # Total
    total_line: float
    total_over_odds: int
    total_under_odds: int
    
    # Moneyline
    ml_home: int
    ml_away: int
    
    # Betting percentages
    public_spread_home_pct: Optional[float] = None
    public_total_over_pct: Optional[float] = None
    money_spread_home_pct: Optional[float] = None
    money_total_over_pct: Optional[float] = None
    
    # Sportsbook
    sportsbook: str = 'consensus'


@dataclass
class LineMovement:
    """Detected line movement."""
    game_id: str
    bet_type: str  # 'spread' or 'total'
    
    # Movement details
    from_line: float
    to_line: float
    movement: float
    
    # Movement type
    movement_type: LineMovementType
    direction: str  # 'home', 'away', 'over', 'under'
    
    # Context
    time_elapsed_minutes: float
    public_pct_at_move: Optional[float] = None
    money_pct_at_move: Optional[float] = None
    
    # Significance
    significance: float = 0.0  # 0-1 scale
    
    # Timestamps
    detected_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'game_id': self.game_id,
            'bet_type': self.bet_type,
            'from': self.from_line,
            'to': self.to_line,
            'movement': self.movement,
            'type': self.movement_type.value,
            'direction': self.direction,
            'significance': round(self.significance, 2),
        }


@dataclass
class SharpIndicator:
    """Sharp money indicator."""
    game_id: str
    bet_type: str
    side: str  # 'home', 'away', 'over', 'under'
    
    # Evidence
    indicators: List[str]
    confidence: float  # 0-1
    
    # Details
    public_vs_money_split: Optional[float] = None  # Difference between ticket and money %
    line_moved_against_public: bool = False
    steam_move_count: int = 0
    
    # Recommendation
    fade_public: bool = False
    follow_sharp: bool = False
    
    detected_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class MarketAlert:
    """Real-time market alert."""
    alert_id: str
    game_id: str
    
    # Alert details
    alert_type: str
    severity: AlertSeverity
    message: str
    
    # Context
    bet_type: str
    side: str
    
    # Action
    suggested_action: str
    confidence: float
    
    # Timing
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None


@dataclass 
class MarketAnalysis:
    """Complete market analysis for a game."""
    game_id: str
    sport: str
    
    # Line movements
    spread_movements: List[LineMovement]
    total_movements: List[LineMovement]
    
    # Sharp indicators
    spread_sharp_indicator: Optional[SharpIndicator] = None
    total_sharp_indicator: Optional[SharpIndicator] = None
    
    # Current state
    current_spread: float = 0.0
    opening_spread: float = 0.0
    spread_movement_total: float = 0.0
    
    current_total: float = 0.0
    opening_total: float = 0.0
    total_movement_total: float = 0.0
    
    # Sentiment
    spread_sentiment: MarketSentiment = MarketSentiment.NEUTRAL
    total_sentiment: MarketSentiment = MarketSentiment.NEUTRAL
    
    # Alerts
    alerts: List[MarketAlert] = field(default_factory=list)
    
    # Timestamps
    analyzed_at: datetime = field(default_factory=datetime.utcnow)
    
    def has_sharp_action(self) -> bool:
        """Check if any sharp action detected."""
        return (
            (self.spread_sharp_indicator and self.spread_sharp_indicator.confidence > 0.6) or
            (self.total_sharp_indicator and self.total_sharp_indicator.confidence > 0.6)
        )
    
    def get_sharp_side(self, bet_type: str) -> Optional[str]:
        """Get the sharp side for a bet type."""
        if bet_type == 'spread' and self.spread_sharp_indicator:
            return self.spread_sharp_indicator.side
        elif bet_type == 'total' and self.total_sharp_indicator:
            return self.total_sharp_indicator.side
        return None


# =============================================================================
# LINE MOVEMENT ANALYZER
# =============================================================================

class LineMovementAnalyzer:
    """
    Analyzes line movements to detect sharp action.
    
    Features:
    - Steam move detection
    - Reverse line movement identification
    - Public vs money split analysis
    - Sharp indicator calculation
    - Real-time alerts
    """
    
    # Thresholds
    STEAM_MOVE_THRESHOLD = 1.0  # Points for spread
    STEAM_MOVE_TOTAL_THRESHOLD = 1.5  # Points for total
    STEAM_TIME_WINDOW = 10  # Minutes
    
    RLM_PUBLIC_THRESHOLD = 0.60  # 60%+ on one side
    RLM_LINE_THRESHOLD = 0.5  # Half point minimum move
    
    MONEY_SPLIT_THRESHOLD = 0.10  # 10% difference triggers alert
    
    def __init__(self):
        """Initialize line movement analyzer."""
        # Store odds history by game
        self._odds_history: Dict[str, List[OddsSnapshot]] = defaultdict(list)
        self._movements: Dict[str, List[LineMovement]] = defaultdict(list)
        self._alerts: List[MarketAlert] = []
        
        self._alert_counter = 0
        
        logger.info("Line Movement Analyzer initialized")
    
    def record_odds(self, snapshot: OddsSnapshot):
        """
        Record an odds snapshot for tracking.
        
        Args:
            snapshot: Point-in-time odds data
        """
        self._odds_history[snapshot.game_id].append(snapshot)
        
        # Check for movements
        history = self._odds_history[snapshot.game_id]
        if len(history) >= 2:
            self._detect_movements(snapshot.game_id, history[-2], history[-1])
    
    def _detect_movements(
        self,
        game_id: str,
        prev: OddsSnapshot,
        curr: OddsSnapshot,
    ):
        """Detect line movements between snapshots."""
        time_diff = (curr.timestamp - prev.timestamp).total_seconds() / 60
        
        # Check spread movement
        spread_move = curr.spread_home - prev.spread_home
        if abs(spread_move) >= 0.5:  # Half point minimum
            movement = self._classify_spread_movement(
                game_id, prev, curr, spread_move, time_diff
            )
            self._movements[game_id].append(movement)
            
            # Check for alert
            self._check_alert(movement, curr)
        
        # Check total movement
        total_move = curr.total_line - prev.total_line
        if abs(total_move) >= 0.5:
            movement = self._classify_total_movement(
                game_id, prev, curr, total_move, time_diff
            )
            self._movements[game_id].append(movement)
            
            # Check for alert
            self._check_alert(movement, curr)
    
    def _classify_spread_movement(
        self,
        game_id: str,
        prev: OddsSnapshot,
        curr: OddsSnapshot,
        move: float,
        time_minutes: float,
    ) -> LineMovement:
        """Classify a spread movement."""
        # Determine direction
        direction = 'home' if move < 0 else 'away'  # Negative = home favored more
        
        # Determine type
        if abs(move) >= self.STEAM_MOVE_THRESHOLD and time_minutes <= self.STEAM_TIME_WINDOW:
            move_type = LineMovementType.STEAM
            significance = min(1.0, abs(move) / 2.0)
        elif self._is_reverse_line_movement(prev, curr, 'spread'):
            move_type = LineMovementType.REVERSE
            significance = 0.8
        elif abs(move) >= 0.5:
            move_type = LineMovementType.ORGANIC
            significance = min(0.6, abs(move) / 3.0)
        else:
            move_type = LineMovementType.STATIC
            significance = 0.0
        
        return LineMovement(
            game_id=game_id,
            bet_type='spread',
            from_line=prev.spread_home,
            to_line=curr.spread_home,
            movement=move,
            movement_type=move_type,
            direction=direction,
            time_elapsed_minutes=time_minutes,
            public_pct_at_move=curr.public_spread_home_pct,
            money_pct_at_move=curr.money_spread_home_pct,
            significance=significance,
        )
    
    def _classify_total_movement(
        self,
        game_id: str,
        prev: OddsSnapshot,
        curr: OddsSnapshot,
        move: float,
        time_minutes: float,
    ) -> LineMovement:
        """Classify a total movement."""
        direction = 'over' if move > 0 else 'under'
        
        if abs(move) >= self.STEAM_MOVE_TOTAL_THRESHOLD and time_minutes <= self.STEAM_TIME_WINDOW:
            move_type = LineMovementType.STEAM
            significance = min(1.0, abs(move) / 3.0)
        elif self._is_reverse_line_movement(prev, curr, 'total'):
            move_type = LineMovementType.REVERSE
            significance = 0.8
        elif abs(move) >= 0.5:
            move_type = LineMovementType.ORGANIC
            significance = min(0.6, abs(move) / 4.0)
        else:
            move_type = LineMovementType.STATIC
            significance = 0.0
        
        return LineMovement(
            game_id=game_id,
            bet_type='total',
            from_line=prev.total_line,
            to_line=curr.total_line,
            movement=move,
            movement_type=move_type,
            direction=direction,
            time_elapsed_minutes=time_minutes,
            public_pct_at_move=curr.public_total_over_pct,
            money_pct_at_move=curr.money_total_over_pct,
            significance=significance,
        )
    
    def _is_reverse_line_movement(
        self,
        prev: OddsSnapshot,
        curr: OddsSnapshot,
        bet_type: str,
    ) -> bool:
        """
        Check if movement is reverse line movement.
        
        RLM = Line moves opposite to public betting direction
        """
        if bet_type == 'spread':
            public_pct = curr.public_spread_home_pct
            line_move = curr.spread_home - prev.spread_home
            
            if public_pct is None:
                return False
            
            # Public heavy on home but line moving toward away
            if public_pct > self.RLM_PUBLIC_THRESHOLD and line_move > self.RLM_LINE_THRESHOLD:
                return True
            # Public heavy on away but line moving toward home
            if public_pct < (1 - self.RLM_PUBLIC_THRESHOLD) and line_move < -self.RLM_LINE_THRESHOLD:
                return True
        
        elif bet_type == 'total':
            public_pct = curr.public_total_over_pct
            line_move = curr.total_line - prev.total_line
            
            if public_pct is None:
                return False
            
            # Public heavy on over but line moving down
            if public_pct > self.RLM_PUBLIC_THRESHOLD and line_move < -self.RLM_LINE_THRESHOLD:
                return True
            # Public heavy on under but line moving up
            if public_pct < (1 - self.RLM_PUBLIC_THRESHOLD) and line_move > self.RLM_LINE_THRESHOLD:
                return True
        
        return False
    
    def _check_alert(self, movement: LineMovement, snapshot: OddsSnapshot):
        """Check if movement warrants an alert."""
        if movement.movement_type == LineMovementType.STEAM:
            self._alert_counter += 1
            alert = MarketAlert(
                alert_id=f"alert_{self._alert_counter}",
                game_id=movement.game_id,
                alert_type='steam_move',
                severity=AlertSeverity.HIGH,
                message=f"Steam move detected: {movement.bet_type} {movement.direction} "
                        f"({movement.movement:+.1f} points in {movement.time_elapsed_minutes:.0f} min)",
                bet_type=movement.bet_type,
                side=movement.direction,
                suggested_action=f"Consider {movement.direction} as sharp side",
                confidence=movement.significance,
            )
            self._alerts.append(alert)
            logger.info(f"ALERT: {alert.message}")
        
        elif movement.movement_type == LineMovementType.REVERSE:
            self._alert_counter += 1
            alert = MarketAlert(
                alert_id=f"alert_{self._alert_counter}",
                game_id=movement.game_id,
                alert_type='reverse_line_movement',
                severity=AlertSeverity.MEDIUM,
                message=f"RLM detected: {movement.bet_type} moving toward {movement.direction} "
                        f"against public betting",
                bet_type=movement.bet_type,
                side=movement.direction,
                suggested_action=f"Sharp money likely on {movement.direction}",
                confidence=0.7,
            )
            self._alerts.append(alert)
            logger.info(f"ALERT: {alert.message}")
    
    def detect_sharp_indicators(
        self,
        game_id: str,
    ) -> Tuple[Optional[SharpIndicator], Optional[SharpIndicator]]:
        """
        Detect sharp money indicators for a game.
        
        Returns:
            Tuple of (spread_indicator, total_indicator)
        """
        movements = self._movements.get(game_id, [])
        history = self._odds_history.get(game_id, [])
        
        if not movements and not history:
            return None, None
        
        # Analyze spread
        spread_moves = [m for m in movements if m.bet_type == 'spread']
        spread_indicator = self._analyze_sharp_indicators(
            game_id, 'spread', spread_moves, history
        )
        
        # Analyze total
        total_moves = [m for m in movements if m.bet_type == 'total']
        total_indicator = self._analyze_sharp_indicators(
            game_id, 'total', total_moves, history
        )
        
        return spread_indicator, total_indicator
    
    def _analyze_sharp_indicators(
        self,
        game_id: str,
        bet_type: str,
        movements: List[LineMovement],
        history: List[OddsSnapshot],
    ) -> Optional[SharpIndicator]:
        """Analyze movements for sharp indicators."""
        if not movements and not history:
            return None
        
        indicators = []
        confidence = 0.0
        steam_count = 0
        rlm_detected = False
        money_split = None
        side = None
        
        # Count steam moves
        for m in movements:
            if m.movement_type == LineMovementType.STEAM:
                steam_count += 1
                indicators.append(f"Steam move: {m.direction}")
                confidence += 0.2
                if side is None:
                    side = m.direction
            elif m.movement_type == LineMovementType.REVERSE:
                rlm_detected = True
                indicators.append(f"Reverse line movement: {m.direction}")
                confidence += 0.25
                if side is None:
                    side = m.direction
        
        # Check money vs ticket split
        if history:
            latest = history[-1]
            if bet_type == 'spread':
                if latest.public_spread_home_pct and latest.money_spread_home_pct:
                    split = latest.money_spread_home_pct - latest.public_spread_home_pct
                    money_split = abs(split)
                    if money_split > self.MONEY_SPLIT_THRESHOLD:
                        indicators.append(f"Money split: {money_split:.1%} difference")
                        confidence += 0.15
                        if side is None:
                            side = 'home' if split > 0 else 'away'
            elif bet_type == 'total':
                if latest.public_total_over_pct and latest.money_total_over_pct:
                    split = latest.money_total_over_pct - latest.public_total_over_pct
                    money_split = abs(split)
                    if money_split > self.MONEY_SPLIT_THRESHOLD:
                        indicators.append(f"Money split: {money_split:.1%} difference")
                        confidence += 0.15
                        if side is None:
                            side = 'over' if split > 0 else 'under'
        
        if not indicators:
            return None
        
        confidence = min(1.0, confidence)
        
        return SharpIndicator(
            game_id=game_id,
            bet_type=bet_type,
            side=side or 'unknown',
            indicators=indicators,
            confidence=confidence,
            public_vs_money_split=money_split,
            line_moved_against_public=rlm_detected,
            steam_move_count=steam_count,
            fade_public=rlm_detected or (money_split and money_split > 0.15),
            follow_sharp=confidence > 0.5,
        )
    
    def analyze_game(
        self,
        game_id: str,
        sport: str,
    ) -> MarketAnalysis:
        """
        Perform complete market analysis for a game.
        
        Args:
            game_id: Game identifier
            sport: Sport code
            
        Returns:
            MarketAnalysis with all insights
        """
        history = self._odds_history.get(game_id, [])
        movements = self._movements.get(game_id, [])
        
        # Get movements by type
        spread_moves = [m for m in movements if m.bet_type == 'spread']
        total_moves = [m for m in movements if m.bet_type == 'total']
        
        # Calculate line movement totals
        opening_spread = history[0].spread_home if history else 0.0
        current_spread = history[-1].spread_home if history else 0.0
        
        opening_total = history[0].total_line if history else 0.0
        current_total = history[-1].total_line if history else 0.0
        
        # Detect sharp indicators
        spread_indicator, total_indicator = self.detect_sharp_indicators(game_id)
        
        # Determine sentiment
        spread_sentiment = self._determine_sentiment('spread', spread_indicator, history)
        total_sentiment = self._determine_sentiment('total', total_indicator, history)
        
        # Get relevant alerts
        game_alerts = [a for a in self._alerts if a.game_id == game_id]
        
        return MarketAnalysis(
            game_id=game_id,
            sport=sport,
            spread_movements=spread_moves,
            total_movements=total_moves,
            spread_sharp_indicator=spread_indicator,
            total_sharp_indicator=total_indicator,
            current_spread=current_spread,
            opening_spread=opening_spread,
            spread_movement_total=current_spread - opening_spread,
            current_total=current_total,
            opening_total=opening_total,
            total_movement_total=current_total - opening_total,
            spread_sentiment=spread_sentiment,
            total_sentiment=total_sentiment,
            alerts=game_alerts,
        )
    
    def _determine_sentiment(
        self,
        bet_type: str,
        indicator: Optional[SharpIndicator],
        history: List[OddsSnapshot],
    ) -> MarketSentiment:
        """Determine market sentiment."""
        if indicator and indicator.confidence > 0.5:
            if bet_type == 'spread':
                if indicator.side == 'home':
                    return MarketSentiment.SHARP_HOME
                elif indicator.side == 'away':
                    return MarketSentiment.SHARP_AWAY
            elif bet_type == 'total':
                if indicator.side == 'over':
                    return MarketSentiment.SHARP_OVER
                elif indicator.side == 'under':
                    return MarketSentiment.SHARP_UNDER
        
        # Check public sentiment from history
        if history:
            latest = history[-1]
            if bet_type == 'spread' and latest.public_spread_home_pct:
                if latest.public_spread_home_pct > 0.65:
                    return MarketSentiment.PUBLIC_HOME
                elif latest.public_spread_home_pct < 0.35:
                    return MarketSentiment.PUBLIC_AWAY
            elif bet_type == 'total' and latest.public_total_over_pct:
                if latest.public_total_over_pct > 0.65:
                    return MarketSentiment.PUBLIC_OVER
                elif latest.public_total_over_pct < 0.35:
                    return MarketSentiment.PUBLIC_UNDER
        
        return MarketSentiment.NEUTRAL
    
    def get_recent_alerts(
        self,
        hours: int = 24,
        severity: Optional[AlertSeverity] = None,
    ) -> List[MarketAlert]:
        """
        Get recent market alerts.
        
        Args:
            hours: Look back hours
            severity: Filter by severity
            
        Returns:
            List of recent alerts
        """
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        alerts = [a for a in self._alerts if a.created_at >= cutoff]
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        return sorted(alerts, key=lambda a: a.created_at, reverse=True)
    
    def get_games_with_sharp_action(self) -> List[str]:
        """Get list of game IDs with detected sharp action."""
        sharp_games = []
        
        for game_id in self._odds_history.keys():
            spread_ind, total_ind = self.detect_sharp_indicators(game_id)
            
            if ((spread_ind and spread_ind.confidence > 0.5) or
                (total_ind and total_ind.confidence > 0.5)):
                sharp_games.append(game_id)
        
        return sharp_games


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == '__main__':
    analyzer = LineMovementAnalyzer()
    
    # Simulate odds snapshots over time
    base_time = datetime.utcnow()
    
    # Opening odds
    snap1 = OddsSnapshot(
        game_id='NBA_LAL_GSW',
        timestamp=base_time - timedelta(hours=4),
        spread_home=-3.5,
        spread_home_odds=-110,
        spread_away_odds=-110,
        total_line=220.5,
        total_over_odds=-110,
        total_under_odds=-110,
        ml_home=-150,
        ml_away=+130,
        public_spread_home_pct=0.55,
        public_total_over_pct=0.60,
    )
    analyzer.record_odds(snap1)
    
    # Movement #1: Line moves to -4 with 65% on home
    snap2 = OddsSnapshot(
        game_id='NBA_LAL_GSW',
        timestamp=base_time - timedelta(hours=2),
        spread_home=-4.0,
        spread_home_odds=-110,
        spread_away_odds=-110,
        total_line=221.5,
        total_over_odds=-110,
        total_under_odds=-110,
        ml_home=-165,
        ml_away=+140,
        public_spread_home_pct=0.65,  # Public heavy on home
        money_spread_home_pct=0.55,   # But money more balanced
        public_total_over_pct=0.70,   # Public on over
        money_total_over_pct=0.45,    # Sharp money on under
    )
    analyzer.record_odds(snap2)
    
    # Movement #2: Steam move - line moves to -5 in 5 minutes!
    snap3 = OddsSnapshot(
        game_id='NBA_LAL_GSW',
        timestamp=base_time - timedelta(minutes=5),
        spread_home=-5.0,
        spread_home_odds=-105,
        spread_away_odds=-115,
        total_line=220.0,  # Total moves DOWN despite public on over
        total_over_odds=-115,
        total_under_odds=-105,
        ml_home=-180,
        ml_away=+155,
        public_spread_home_pct=0.68,
        money_spread_home_pct=0.50,
        public_total_over_pct=0.72,
        money_total_over_pct=0.40,
    )
    analyzer.record_odds(snap3)
    
    # Analyze the game
    analysis = analyzer.analyze_game('NBA_LAL_GSW', 'NBA')
    
    print("=== Market Analysis ===")
    print(f"Game: {analysis.game_id}")
    print(f"\nSpread Movement: {analysis.opening_spread} → {analysis.current_spread}")
    print(f"Total Movement: {analysis.opening_total} → {analysis.current_total}")
    
    print(f"\nSpread Sentiment: {analysis.spread_sentiment.value}")
    print(f"Total Sentiment: {analysis.total_sentiment.value}")
    
    if analysis.spread_sharp_indicator:
        print(f"\nSpread Sharp Indicator:")
        print(f"  Side: {analysis.spread_sharp_indicator.side}")
        print(f"  Confidence: {analysis.spread_sharp_indicator.confidence:.1%}")
        print(f"  Indicators: {', '.join(analysis.spread_sharp_indicator.indicators)}")
    
    if analysis.total_sharp_indicator:
        print(f"\nTotal Sharp Indicator:")
        print(f"  Side: {analysis.total_sharp_indicator.side}")
        print(f"  Confidence: {analysis.total_sharp_indicator.confidence:.1%}")
        print(f"  Indicators: {', '.join(analysis.total_sharp_indicator.indicators)}")
    
    print(f"\nAlerts: {len(analysis.alerts)}")
    for alert in analysis.alerts:
        print(f"  [{alert.severity.value.upper()}] {alert.message}")
