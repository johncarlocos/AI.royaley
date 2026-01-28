"""
Collector #24: Polymarket (Simplified)
Prediction market data stored in EXISTING tables

Uses:
- sportsbooks: Polymarket as a sportsbook
- odds: Price data (probability → American odds)
- odds_movements: Price changes over time
- consensus_lines: Crowd wisdom metrics
- polymarket_game_map: Minimal mapping table (only new table)
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from decimal import Decimal
from uuid import uuid4
import httpx
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.services.collectors.base_collector import BaseCollector, CollectorResult
from app.models.models import Game, Sportsbook, Odds, OddsMovement, ConsensusLine

logger = logging.getLogger(__name__)


class PolymarketCollector(BaseCollector):
    """
    Polymarket prediction market collector.
    Saves data to EXISTING tables (odds, odds_movements, consensus_lines).
    Only requires 1 small mapping table: polymarket_game_map
    """
    
    COLLECTOR_NAME = "polymarket"
    COLLECTOR_ID = 24
    
    # API Endpoints
    GAMMA_BASE_URL = "https://gamma-api.polymarket.com"
    CLOB_BASE_URL = "https://clob.polymarket.com"
    
    # Sports categories on Polymarket
    SPORTS_CATEGORIES = [
        "NFL", "NBA", "MLB", "NHL", "Soccer", "Tennis",
        "College Football", "College Basketball", "MMA"
    ]
    
    # League mappings
    LEAGUE_MAPPING = {
        "NFL": {"sport": "football", "league": "NFL"},
        "NBA": {"sport": "basketball", "league": "NBA"},
        "MLB": {"sport": "baseball", "league": "MLB"},
        "NHL": {"sport": "hockey", "league": "NHL"},
        "Soccer": {"sport": "soccer", "league": "SOCCER"},
        "Tennis": {"sport": "tennis", "league": "ATP"},
        "College Football": {"sport": "football", "league": "NCAAF"},
        "College Basketball": {"sport": "basketball", "league": "NCAAB"},
        "MMA": {"sport": "mma", "league": "UFC"},
    }
    
    def __init__(self, db: Session, config: Optional[Dict] = None):
        # Initialize base collector
        super().__init__(
            name="polymarket",
            base_url=self.GAMMA_BASE_URL,
            rate_limit=60,
            rate_window=60,
            timeout=30.0,
            max_retries=3
        )
        self.db = db
        self.config = config or {}
        self._polymarket_sportsbook_id = None
        
    async def initialize(self):
        """Initialize and get Polymarket sportsbook ID"""
        self._polymarket_sportsbook_id = await self._get_polymarket_sportsbook_id()
        
    async def _get_polymarket_sportsbook_id(self):
        """Get Polymarket sportsbook ID from database"""
        result = self.db.execute(
            text("SELECT id FROM sportsbooks WHERE key = 'polymarket'")
        ).fetchone()
        
        if result:
            return result[0]
        
        # Create if not exists
        new_id = uuid4()
        self.db.execute(text("""
            INSERT INTO sportsbooks (id, name, key, is_sharp, is_active, priority, created_at)
            VALUES (:id, 'Polymarket', 'polymarket', true, true, 5, NOW())
            ON CONFLICT (key) DO NOTHING
        """), {"id": new_id})
        self.db.commit()
        
        result = self.db.execute(
            text("SELECT id FROM sportsbooks WHERE key = 'polymarket'")
        ).fetchone()
        return result[0] if result else new_id
    
    async def collect(self, **kwargs) -> CollectorResult:
        """
        Required abstract method implementation.
        Main collection method.
        """
        return await self._collect_impl(**kwargs)
    
    async def validate(self, data: Any) -> bool:
        """
        Required abstract method implementation.
        Validate collected data.
        """
        if data is None:
            return False
        if isinstance(data, dict):
            return "events_found" in data or "odds_saved" in data
        return True
            
    async def collect_all(self) -> Dict[str, Any]:
        """Main collection method - wrapper for collect()"""
        result = await self.collect()
        return result.data if result.data else {
            "events_found": 0,
            "games_linked": 0,
            "odds_saved": 0,
            "movements_saved": 0,
            "errors": [result.error] if result.error else []
        }
    
    async def _collect_impl(self, **kwargs) -> CollectorResult:
        """Implementation of data collection"""
        await self.initialize()
        
        stats = {
            "events_found": 0,
            "games_linked": 0,
            "odds_saved": 0,
            "movements_saved": 0,
            "errors": []
        }
        
        try:
            for category in self.SPORTS_CATEGORIES:
                try:
                    await asyncio.sleep(0.5)  # Rate limiting
                    events = await self._fetch_events_by_category(category)
                    stats["events_found"] += len(events)
                    
                    for event_data in events:
                        try:
                            result = await self._process_event(event_data, category)
                            if result.get("linked"):
                                stats["games_linked"] += 1
                            stats["odds_saved"] += result.get("odds_count", 0)
                            stats["movements_saved"] += result.get("movements_count", 0)
                        except Exception as e:
                            logger.warning(f"Error processing event: {e}")
                            continue
                            
                    logger.info(f"Processed {len(events)} events for {category}")
                    
                except Exception as e:
                    logger.error(f"Error fetching {category}: {e}")
                    stats["errors"].append(f"{category}: {str(e)}")
                    continue
                    
            self.db.commit()
            
            return CollectorResult(
                success=len(stats["errors"]) == 0,
                data=stats,
                records_count=stats["odds_saved"] + stats["games_linked"],
                error="; ".join(stats["errors"]) if stats["errors"] else None
            )
            
        except Exception as e:
            logger.error(f"Polymarket collection error: {e}")
            stats["errors"].append(str(e))
            return CollectorResult(
                success=False,
                data=stats,
                records_count=0,
                error=str(e)
            )
    
    async def _fetch_events_by_category(self, category: str) -> List[Dict]:
        """Fetch events from Polymarket Gamma API"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                params = {
                    "tag": category,
                    "active": "true",
                    "closed": "false",
                    "limit": 100
                }
                
                response = await client.get(
                    f"{self.GAMMA_BASE_URL}/events",
                    params=params
                )
                response.raise_for_status()
                return response.json() or []
                
        except Exception as e:
            logger.error(f"Error fetching events for {category}: {e}")
            return []
    
    async def _process_event(self, event_data: Dict, category: str) -> Dict:
        """Process a single Polymarket event and save to existing tables"""
        result = {"linked": False, "odds_count": 0, "movements_count": 0}
        
        condition_id = event_data.get("conditionId") or event_data.get("id")
        if not condition_id:
            return result
            
        title = event_data.get("title", "")
        league_info = self.LEAGUE_MAPPING.get(category, {})
        sport = league_info.get("sport")
        league = league_info.get("league")
        
        # Parse teams from title
        home_team, away_team = self._parse_teams_from_title(title)
        
        # Find matching game in our database
        game = await self._find_matching_game(sport, league, home_team, away_team, event_data)
        
        if not game:
            return result
            
        # Save/update mapping
        await self._save_game_mapping(
            condition_id, game.id, title,
            event_data.get("volume", 0),
            event_data.get("liquidity", 0)
        )
        result["linked"] = True
        
        # Get markets (outcomes) for this event
        markets = event_data.get("markets", [])
        if len(markets) >= 2:
            # Extract probabilities
            home_prob, away_prob = self._extract_probabilities(markets, home_team, away_team)
            
            if home_prob and away_prob:
                # Convert to American odds
                home_odds = self._prob_to_american_odds(home_prob)
                away_odds = self._prob_to_american_odds(away_prob)
                
                # Check for movement
                previous_odds = await self._get_previous_odds(game.id)
                
                # Save to odds table
                odds_record = Odds(
                    id=uuid4(),
                    game_id=game.id,
                    sportsbook_id=self._polymarket_sportsbook_id,
                    sportsbook_key="polymarket",
                    bet_type="moneyline",
                    home_odds=int(home_odds),
                    away_odds=int(away_odds),
                    is_opening=previous_odds is None,
                    recorded_at=datetime.utcnow()
                )
                self.db.add(odds_record)
                result["odds_count"] += 1
                
                # Save movement if changed
                if previous_odds and (
                    abs(home_odds - previous_odds.get("home", 0)) > 5 or
                    abs(away_odds - previous_odds.get("away", 0)) > 5
                ):
                    movement = OddsMovement(
                        id=uuid4(),
                        game_id=game.id,
                        bet_type="moneyline",
                        previous_line=previous_odds.get("home"),
                        current_line=home_odds,
                        movement=home_odds - previous_odds.get("home", 0),
                        is_steam=abs(home_odds - previous_odds.get("home", 0)) > 20,
                        detected_at=datetime.utcnow()
                    )
                    self.db.add(movement)
                    result["movements_count"] += 1
                
                # Update consensus_lines with crowd probability
                await self._update_consensus(game.id, home_prob, away_prob, event_data)
                
        return result
    
    async def _save_game_mapping(self, condition_id: str, game_id, title: str, volume, liquidity):
        """Save or update polymarket_game_map"""
        self.db.execute(text("""
            INSERT INTO polymarket_game_map (condition_id, game_id, event_title, volume, liquidity, updated_at)
            VALUES (:cid, :gid, :title, :vol, :liq, NOW())
            ON CONFLICT (condition_id) DO UPDATE SET
                game_id = :gid,
                event_title = :title,
                volume = :vol,
                liquidity = :liq,
                updated_at = NOW()
        """), {
            "cid": str(condition_id),
            "gid": game_id,
            "title": title,
            "vol": float(volume or 0),
            "liq": float(liquidity or 0)
        })
    
    async def _find_matching_game(self, sport: str, league: str, home_team: str, 
                                   away_team: str, event_data: Dict) -> Optional[Game]:
        """Find matching game in our database"""
        if not home_team or not away_team:
            return None
            
        # Parse event date
        start_date = event_data.get("startDate")
        if start_date:
            try:
                event_dt = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
                start_window = event_dt - timedelta(days=1)
                end_window = event_dt + timedelta(days=3)
            except:
                start_window = datetime.utcnow() - timedelta(days=1)
                end_window = datetime.utcnow() + timedelta(days=7)
        else:
            start_window = datetime.utcnow() - timedelta(days=1)
            end_window = datetime.utcnow() + timedelta(days=7)
        
        # Try to find game
        game = self.db.query(Game).filter(
            Game.sport == sport,
            Game.league == league,
            Game.home_team.ilike(f"%{home_team}%"),
            Game.away_team.ilike(f"%{away_team}%"),
            Game.game_datetime >= start_window,
            Game.game_datetime <= end_window
        ).first()
        
        if game:
            return game
            
        # Try swapped teams
        game = self.db.query(Game).filter(
            Game.sport == sport,
            Game.league == league,
            Game.home_team.ilike(f"%{away_team}%"),
            Game.away_team.ilike(f"%{home_team}%"),
            Game.game_datetime >= start_window,
            Game.game_datetime <= end_window
        ).first()
        
        return game
    
    async def _get_previous_odds(self, game_id) -> Optional[Dict]:
        """Get previous Polymarket odds for comparison"""
        result = self.db.execute(text("""
            SELECT home_odds, away_odds FROM odds
            WHERE game_id = :gid AND sportsbook_key = 'polymarket'
            ORDER BY recorded_at DESC LIMIT 1
        """), {"gid": game_id}).fetchone()
        
        if result:
            return {"home": result[0], "away": result[1]}
        return None
    
    async def _update_consensus(self, game_id, home_prob: float, away_prob: float, event_data: Dict):
        """Update consensus_lines with Polymarket crowd probability"""
        volume = float(event_data.get("volume", 0) or 0)
        
        # Volume tier for weighting
        if volume >= 100000:
            confidence = 0.9
        elif volume >= 25000:
            confidence = 0.7
        elif volume >= 5000:
            confidence = 0.5
        else:
            confidence = 0.3
        
        # Check if consensus exists for this game
        existing = self.db.execute(text("""
            SELECT id FROM consensus_lines 
            WHERE game_id = :gid AND bet_type = 'polymarket_crowd'
        """), {"gid": game_id}).fetchone()
        
        if existing:
            self.db.execute(text("""
                UPDATE consensus_lines SET
                    consensus_line = :home_prob,
                    public_bet_pct = :away_prob,
                    sharp_bet_pct = :confidence,
                    calculated_at = NOW()
                WHERE game_id = :gid AND bet_type = 'polymarket_crowd'
            """), {
                "gid": game_id,
                "home_prob": home_prob * 100,
                "away_prob": away_prob * 100,
                "confidence": confidence * 100
            })
        else:
            self.db.execute(text("""
                INSERT INTO consensus_lines 
                (id, game_id, bet_type, consensus_line, public_bet_pct, sharp_bet_pct, calculated_at)
                VALUES (gen_random_uuid(), :gid, 'polymarket_crowd', :home_prob, :away_prob, :confidence, NOW())
            """), {
                "gid": game_id,
                "home_prob": home_prob * 100,
                "away_prob": away_prob * 100,
                "confidence": confidence * 100
            })
    
    def _extract_probabilities(self, markets: List[Dict], home_team: str, away_team: str) -> Tuple[float, float]:
        """Extract home/away probabilities from markets"""
        home_prob = None
        away_prob = None
        
        for market in markets:
            outcome = (market.get("outcome") or "").lower()
            price = market.get("price")
            
            if price is None:
                continue
                
            try:
                price = float(price)
            except:
                continue
            
            if home_team and home_team.lower() in outcome:
                home_prob = price
            elif away_team and away_team.lower() in outcome:
                away_prob = price
        
        # If not found by team name, use by index
        if home_prob is None or away_prob is None:
            if len(markets) >= 2:
                sorted_markets = sorted(markets, key=lambda m: m.get("outcomeIndex", 0))
                try:
                    home_prob = float(sorted_markets[0].get("price", 0.5) or 0.5)
                    away_prob = float(sorted_markets[1].get("price", 0.5) or 0.5)
                except:
                    pass
        
        # Normalize
        if home_prob and away_prob:
            total = home_prob + away_prob
            if total > 0 and total != 1.0:
                home_prob = home_prob / total
                away_prob = away_prob / total
                
        return home_prob or 0.5, away_prob or 0.5
    
    def _parse_teams_from_title(self, title: str) -> Tuple[Optional[str], Optional[str]]:
        """Parse team names from event title"""
        if not title:
            return None, None
            
        title_lower = title.lower()
        
        for separator in [" vs ", " vs. ", " v ", " @ ", " at "]:
            if separator in title_lower:
                idx = title_lower.find(separator)
                parts = [title[:idx], title[idx + len(separator):]]
                if len(parts) == 2:
                    home = parts[0].strip().split(" to ")[0].strip()
                    away = parts[1].strip().split(" to ")[0].strip()
                    # Remove common prefixes
                    for prefix in ["will ", "can "]:
                        if home.lower().startswith(prefix):
                            home = home[len(prefix):]
                        if away.lower().startswith(prefix):
                            away = away[len(prefix):]
                    return home, away
                    
        return None, None
    
    def _prob_to_american_odds(self, prob: float) -> float:
        """Convert probability (0-1) to American odds"""
        if prob <= 0 or prob >= 1:
            return -110  # Default
            
        if prob >= 0.5:
            return -(prob / (1 - prob)) * 100
        else:
            return ((1 - prob) / prob) * 100


# Factory function
def get_collector(db: Session, config: Optional[Dict] = None) -> PolymarketCollector:
    return PolymarketCollector(db, config)