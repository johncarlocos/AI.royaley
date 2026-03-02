
import json
import logging
import httpx
from typing import Optional, Tuple, Dict, Any

logger = logging.getLogger("royaley.claude_predictor")

# Sports where Claude API replaces ML models due to insufficient training data
CLAUDE_API_SPORTS = {"CFL", "WNBA", "NCAAB"}

# Cache to avoid duplicate API calls for same game/bet_type
_prediction_cache: Dict[str, Tuple[float, float]] = {}


async def claude_predict(
    sport_code: str,
    bet_type: str,
    home_team: str,
    away_team: str,
    odds_data: Dict[str, Any],
    api_key: str,
) -> Optional[Tuple[float, float]]:
    """
    Use Claude API to predict probability for a game.
    
    Args:
        sport_code: CFL, WNBA, or NCAAB
        bet_type: spread, moneyline, or total
        home_team: Home team name
        away_team: Away team name
        odds_data: Dict with consensus_spread, consensus_total, moneyline_home_close, etc.
        api_key: Anthropic API key
        
    Returns:
        (prob_positive, prob_negative) where positive = home_cover/home_win/over
        None if API call fails
    """
    if not api_key:
        logger.warning("No ANTHROPIC_API_KEY set, skipping Claude prediction")
        return None

    # Check cache
    cache_key = f"{sport_code}:{home_team}:{away_team}:{bet_type}"
    if cache_key in _prediction_cache:
        cached = _prediction_cache[cache_key]
        logger.info(f"  🤖 Claude cached: {sport_code}/{bet_type} = {cached[0]:.3f}")
        return cached

    # Build context for Claude
    spread = odds_data.get("consensus_spread", 0)
    total = odds_data.get("consensus_total", 0)
    home_ml = odds_data.get("moneyline_home_close", -110)
    away_ml = odds_data.get("moneyline_away_close", -110)
    num_books = odds_data.get("num_books", 0)

    if bet_type == "spread":
        bet_context = f"Spread: {home_team} {'+' if spread > 0 else ''}{spread}"
    elif bet_type == "total":
        bet_context = f"Total: {total} points"
    else:
        bet_context = f"Moneyline: {home_team} {home_ml} / {away_team} {away_ml}"

    prompt = f"""You are an expert sports analyst. Analyze this {sport_code} game and estimate the probability.

GAME: {home_team} (home) vs {away_team} (away)
SPORT: {sport_code}
BET TYPE: {bet_type}

MARKET DATA:
- Spread: {home_team} {'+' if spread > 0 else ''}{spread}
- Total: {total}
- Moneyline: {home_team} {home_ml} / {away_team} {away_ml}
- Number of sportsbooks: {num_books}

TASK: Estimate the probability for this bet type.

For {bet_type}:
{_get_bet_type_instruction(bet_type, home_team, away_team, spread, total, home_ml, away_ml)}

CRITICAL RULES:
1. Output ONLY valid JSON, no other text
2. Probability must be between 0.40 and 0.65 (realistic sports betting range)
3. Use market odds as your anchor - they contain most of the information
4. Only deviate from market-implied probability by 1-5% based on your analysis
5. Consider home court/field advantage, league-specific factors

Output format:
{{"probability": 0.XX, "confidence": "low|medium|high", "reasoning": "brief 1-sentence explanation"}}"""

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-haiku-4-5-20251001",
                    "max_tokens": 200,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            
            if response.status_code != 200:
                logger.error(f"Claude API error {response.status_code}: {response.text[:200]}")
                return None

            data = response.json()
            text = data["content"][0]["text"].strip()
            
            # Parse JSON response (handle markdown code blocks)
            text = text.replace("```json", "").replace("```", "").strip()
            result = json.loads(text)
            
            prob = float(result["probability"])
            
            # Safety clamp to realistic range
            prob = max(0.40, min(0.65, prob))
            
            # Apply slight shrinkage toward market-implied
            market_implied = _get_market_implied(bet_type, odds_data)
            if market_implied:
                # Blend: 60% Claude + 40% market (Claude should add info, not replace market)
                prob = prob * 0.6 + market_implied * 0.4
            
            prob_positive = prob
            prob_negative = 1.0 - prob_positive

            reasoning = result.get("reasoning", "")
            confidence = result.get("confidence", "medium")
            logger.info(
                f"  🤖 Claude {sport_code}/{bet_type}: P={prob_positive:.3f} "
                f"[{confidence}] {reasoning[:80]}"
            )

            # Cache result
            _prediction_cache[cache_key] = (prob_positive, prob_negative)
            
            return (prob_positive, prob_negative)

    except json.JSONDecodeError as e:
        logger.error(f"Claude response not valid JSON: {e}")
        return None
    except httpx.TimeoutException:
        logger.error(f"Claude API timeout for {sport_code}/{bet_type}")
        return None
    except Exception as e:
        logger.error(f"Claude prediction failed: {e}")
        return None


def _get_bet_type_instruction(
    bet_type: str, home: str, away: str, 
    spread: float, total: float, home_ml: float, away_ml: float
) -> str:
    """Generate bet-type specific instruction."""
    if bet_type == "spread":
        return (
            f"What is the probability that {home} covers the spread of {'+' if spread > 0 else ''}{spread}?\n"
            f"probability = P({home} wins by more than {abs(spread)} points)"
            if spread < 0 else
            f"What is the probability that {home} covers the spread of +{spread}?\n"
            f"probability = P({home} loses by fewer than {spread} points OR wins)"
        )
    elif bet_type == "total":
        return (
            f"What is the probability the game total goes OVER {total}?\n"
            f"probability = P(combined score > {total})"
        )
    else:  # moneyline
        return (
            f"What is the probability that {home} wins the game?\n"
            f"probability = P({home} wins outright)\n"
            f"Market-implied: {home} {_implied_prob_str(home_ml)}, {away} {_implied_prob_str(away_ml)}"
        )


def _implied_prob_str(american_odds: float) -> str:
    """Convert American odds to implied probability string."""
    try:
        american_odds = float(american_odds)
        if american_odds > 0:
            prob = 100 / (american_odds + 100)
        else:
            prob = abs(american_odds) / (abs(american_odds) + 100)
        return f"{prob:.1%}"
    except (ValueError, ZeroDivisionError):
        return "N/A"


def _get_market_implied(bet_type: str, odds_data: Dict) -> Optional[float]:
    """Get market-implied probability for blending with Claude's estimate."""
    try:
        if bet_type == "moneyline":
            home_ml = float(odds_data.get("moneyline_home_close", -110))
            away_ml = float(odds_data.get("moneyline_away_close", -110))
            if home_ml > 0:
                home_prob = 100 / (home_ml + 100)
            else:
                home_prob = abs(home_ml) / (abs(home_ml) + 100)
            if away_ml > 0:
                away_prob = 100 / (away_ml + 100)
            else:
                away_prob = abs(away_ml) / (abs(away_ml) + 100)
            total = home_prob + away_prob
            return home_prob / total if total > 0 else 0.5
        elif bet_type == "spread":
            # Spread is generally priced at ~50/50 (-110 each side)
            return 0.50
        elif bet_type == "total":
            return 0.50
    except (ValueError, ZeroDivisionError):
        pass
    return None


def clear_cache():
    """Clear prediction cache (called daily)."""
    global _prediction_cache
    _prediction_cache.clear()
    logger.info("Claude prediction cache cleared")