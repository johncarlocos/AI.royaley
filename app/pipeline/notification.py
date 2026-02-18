"""
ROYALEY - Notification Dispatcher
Sends Telegram/Email notifications based on user preferences stored in DB.

Events:
  tier_a          → New Tier A prediction (58%+)
  tier_b          → New Tier B prediction (55-58%)
  tier_c          → New Tier C prediction (52-55%)
  tier_d          → New Tier D prediction (<52%)
  grading_complete→ Batch grading finished
  clv_alert       → High CLV detected (>5%)
  daily_summary   → Daily P/L summary
  system_errors   → Pipeline errors
  model_training  → Model retrained
  live_game       → Game started/in progress

Usage:
    from app.pipeline.notifications import notify

    # Single prediction alert
    await notify("tier_a", "🔥 Tier A: Lakers ML @ -150", {"sport": "NBA", "confidence": "59.2%"})

    # Grading summary
    await notify("grading_complete", "Graded 12 games: 8W 4L (+$340)", {"roi": "+4.2%"})
"""

import logging
import smtplib
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional

import httpx
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

logger = logging.getLogger("royaley.notifications")

# Cache settings to avoid DB hit every call (refreshed every 5 min)
_settings_cache: Optional[dict] = None
_cache_ts: float = 0


async def _load_settings(db: AsyncSession) -> dict:
    """Load notification settings from user_preferences."""
    global _settings_cache, _cache_ts
    import time

    # Return cache if fresh (< 5 min old)
    if _settings_cache and (time.time() - _cache_ts) < 300:
        return _settings_cache

    result = await db.execute(text(
        "SELECT notification_settings FROM user_preferences LIMIT 1"
    ))
    row = result.fetchone()

    if row and row.notification_settings:
        _settings_cache = row.notification_settings
    else:
        _settings_cache = {}

    _cache_ts = time.time()
    return _settings_cache


def _should_notify(settings: dict, event: str, channel: str) -> bool:
    """Check if user wants notifications for this event on this channel."""
    for pref in settings.get("preferences", []):
        if pref.get("event") == event:
            return pref.get(channel, False)
    return False


def _get_telegram_config(settings: dict) -> List[dict]:
    """Get list of enabled Telegram accounts with token."""
    token = settings.get("telegram_token", "")
    if not token:
        return []

    accounts = settings.get("telegram_accounts", [])
    return [
        {"token": token, "chat_id": acc["chat_id"], "name": acc.get("name", "")}
        for acc in accounts
        if acc.get("enabled", True) and acc.get("chat_id")
    ]


def _get_email_config(settings: dict) -> List[str]:
    """Get list of enabled email addresses."""
    accounts = settings.get("email_accounts", [])
    return [
        acc["email"]
        for acc in accounts
        if acc.get("enabled", True) and acc.get("email")
    ]


async def _send_telegram(token: str, chat_id: str, message: str) -> bool:
    """Send a Telegram message."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={
                    "chat_id": chat_id,
                    "text": message,
                    "parse_mode": "HTML",
                },
            )
            if resp.status_code == 200:
                return True
            else:
                logger.warning(f"Telegram send failed ({resp.status_code}): {resp.text[:200]}")
                return False
    except Exception as e:
        logger.warning(f"Telegram error: {e}")
        return False


async def _send_email(to_email: str, subject: str, body: str) -> bool:
    """Send an email via SMTP."""
    try:
        from app.core.config import settings as app_settings

        host = getattr(app_settings, "EMAIL_SMTP_HOST", "")
        port = int(getattr(app_settings, "EMAIL_SMTP_PORT", 587))
        user = getattr(app_settings, "EMAIL_SMTP_USER", "")
        password = getattr(app_settings, "EMAIL_SMTP_PASSWORD", "")
        from_addr = getattr(app_settings, "EMAIL_FROM_ADDRESS", user)

        if not all([host, user, password]):
            return False

        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = from_addr
        msg["To"] = to_email

        with smtplib.SMTP(host, port) as server:
            server.starttls()
            server.login(user, password)
            server.send_message(msg)

        return True
    except Exception as e:
        logger.warning(f"Email error: {e}")
        return False


def _format_telegram_message(event: str, message: str, metadata: Dict[str, Any]) -> str:
    """Format a notification for Telegram."""
    emoji_map = {
        "tier_a": "🔥",
        "tier_b": "⭐",
        "tier_c": "📊",
        "tier_d": "📉",
        "grading_complete": "✅",
        "clv_alert": "💰",
        "daily_summary": "📋",
        "system_errors": "🚨",
        "model_training": "🧠",
        "live_game": "🏟️",
    }

    emoji = emoji_map.get(event, "📢")
    label_map = {
        "tier_a": "Tier A Prediction",
        "tier_b": "Tier B Prediction",
        "tier_c": "Tier C Prediction",
        "tier_d": "Tier D Prediction",
        "grading_complete": "Grading Complete",
        "clv_alert": "High CLV Alert",
        "daily_summary": "Daily Summary",
        "system_errors": "System Error",
        "model_training": "Model Training",
        "live_game": "Live Game",
    }

    title = label_map.get(event, event.replace("_", " ").title())
    lines = [f"{emoji} <b>ROYALEY — {title}</b>", "", message]

    if metadata:
        lines.append("")
        for k, v in metadata.items():
            lines.append(f"<b>{k}:</b> {v}")

    return "\n".join(lines)


# =========================================================================
# PUBLIC API
# =========================================================================

async def notify(
    event: str,
    message: str,
    metadata: Optional[Dict[str, Any]] = None,
    db: Optional[AsyncSession] = None,
) -> int:
    """
    Send notification for an event. Returns number of messages sent.

    Args:
        event: Event type (tier_a, grading_complete, etc.)
        message: Human-readable message
        metadata: Optional key-value pairs for extra detail
        db: Optional DB session (creates one if not provided)
    """
    metadata = metadata or {}
    sent = 0

    # Get or create DB session
    own_engine = None
    if db is None:
        from app.core.config import settings as app_settings
        own_engine = create_async_engine(app_settings.DATABASE_URL, echo=False)
        session_maker = async_sessionmaker(own_engine, class_=AsyncSession, expire_on_commit=False)
        db_session = session_maker()
    else:
        db_session = db

    try:
        settings = await _load_settings(db_session)

        if not settings:
            return 0

        # Telegram
        if _should_notify(settings, event, "telegram"):
            tg_accounts = _get_telegram_config(settings)
            if tg_accounts:
                tg_message = _format_telegram_message(event, message, metadata)
                for acc in tg_accounts:
                    ok = await _send_telegram(acc["token"], acc["chat_id"], tg_message)
                    if ok:
                        sent += 1
                        logger.info(f"[Notify] Telegram sent: {event} → {acc.get('name', acc['chat_id'])}")

        # Email
        if _should_notify(settings, event, "email"):
            emails = _get_email_config(settings)
            if emails:
                subject = f"ROYALEY: {event.replace('_', ' ').title()}"
                body = f"{message}\n\n" + "\n".join(
                    f"{k}: {v}" for k, v in metadata.items()
                )
                for email in emails:
                    ok = await _send_email(email, subject, body)
                    if ok:
                        sent += 1
                        logger.info(f"[Notify] Email sent: {event} → {email}")

    except Exception as e:
        logger.error(f"[Notify] Error dispatching {event}: {e}")
    finally:
        if own_engine:
            await db_session.close()
            await own_engine.dispose()

    return sent


async def notify_predictions(predictions: list, db: Optional[AsyncSession] = None) -> int:
    """
    Send notifications for new predictions based on their tier.

    Args:
        predictions: List of dicts with keys: sport, bet_type, team, confidence, odds
    """
    if not predictions:
        return 0

    # Group by tier
    tiers = {"tier_a": [], "tier_b": [], "tier_c": [], "tier_d": []}
    for p in predictions:
        conf = float(p.get("confidence", 0.5)) * 100
        if conf >= 58:
            tiers["tier_a"].append(p)
        elif conf >= 55:
            tiers["tier_b"].append(p)
        elif conf >= 52:
            tiers["tier_c"].append(p)
        else:
            tiers["tier_d"].append(p)

    sent = 0
    for tier, preds in tiers.items():
        if not preds:
            continue

        lines = []
        for p in preds[:10]:  # Max 10 per tier message
            conf = float(p.get("confidence", 0.5)) * 100
            lines.append(
                f"• {p.get('sport', '?')} | {p.get('team', '?')} "
                f"({p.get('bet_type', '?')}) — {conf:.1f}% @ {p.get('odds', '?')}"
            )

        message = "\n".join(lines)
        meta = {"Count": len(preds)}
        if len(preds) > 10:
            meta["Note"] = f"Showing 10 of {len(preds)}"

        sent += await notify(tier, message, meta, db=db)

    return sent


async def notify_grading(stats: dict, db: Optional[AsyncSession] = None) -> int:
    """
    Send notification after grading completes.

    Args:
        stats: Dict from grade_predictions() with games_graded, predictions_graded, etc.
    """
    games = stats.get("games_graded", 0)
    preds = stats.get("predictions_graded", 0)

    if games == 0:
        return 0

    message = f"Graded {games} games, {preds} predictions"
    metadata = {
        "Phase 1 (cached scores)": stats.get("already_scored", 0),
        "Phase 2 (API fetched)": stats.get("api_requests", 0),
        "ESPN fallback": stats.get("espn_graded", 0),
    }

    return await notify("grading_complete", message, metadata, db=db)