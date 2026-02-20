"""
ROYALEY — Professional Notification System
Telegram + Email notifications with rich formatting.

Events:
  tier_a           → High-value prediction (58%+)
  tier_b           → Strong prediction (55-58%)
  tier_c           → Standard prediction (52-55%)
  tier_d           → Low-edge prediction (<52%)
  grading_complete → Batch grading finished with P&L
  clv_alert        → Closing Line Value detected
  daily_summary    → End-of-day performance report
  system_errors    → Pipeline/system errors
  model_training   → Model retrained/updated
  live_game        → Game started / score update
  backup_complete  → Backup finished successfully
"""

import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

import httpx
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

logger = logging.getLogger("royaley.notifications")

# Cache settings to avoid DB hit every call (refreshed every 5 min)
_settings_cache: Optional[dict] = None
_cache_ts: float = 0


# =============================================================================
# SETTINGS LOADERS
# =============================================================================

async def _load_settings(db: AsyncSession) -> dict:
    """Load notification settings from user_preferences."""
    global _settings_cache, _cache_ts
    import time

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


# =============================================================================
# SEND FUNCTIONS
# =============================================================================

async def _send_telegram(token: str, chat_id: str, message: str) -> bool:
    """Send a Telegram message with HTML formatting."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={
                    "chat_id": chat_id,
                    "text": message,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True,
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


async def _send_email(to_email: str, subject: str, html_body: str, text_body: str = "") -> bool:
    """Send an HTML email via SMTP."""
    try:
        from app.core.config import settings as app_settings

        host = getattr(app_settings, "EMAIL_SMTP_HOST", "")
        port = int(getattr(app_settings, "EMAIL_SMTP_PORT", 587))
        user = getattr(app_settings, "EMAIL_SMTP_USER", "")
        password = getattr(app_settings, "EMAIL_SMTP_PASSWORD", "")
        from_addr = getattr(app_settings, "EMAIL_FROM_ADDRESS", user)

        if not all([host, user, password]):
            return False

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = f"Royaley <{from_addr}>"
        msg["To"] = to_email

        # Plain text fallback
        if text_body:
            msg.attach(MIMEText(text_body, "plain"))

        # HTML version
        msg.attach(MIMEText(html_body, "html"))

        with smtplib.SMTP(host, port) as server:
            server.starttls()
            server.login(user, password)
            server.send_message(msg)

        return True
    except Exception as e:
        logger.warning(f"Email error: {e}")
        return False


# =============================================================================
# TELEGRAM FORMATTERS
# =============================================================================

def _format_prediction_telegram(predictions: list, tier: str) -> str:
    """Format prediction alerts for Telegram with professional layout."""
    tier_config = {
        "tier_a": {"emoji": "🔴", "label": "TIER A", "desc": "High Value"},
        "tier_b": {"emoji": "🟠", "label": "TIER B", "desc": "Strong Edge"},
        "tier_c": {"emoji": "🟡", "label": "TIER C", "desc": "Standard"},
        "tier_d": {"emoji": "⚪", "label": "TIER D", "desc": "Low Edge"},
    }
    cfg = tier_config.get(tier, tier_config["tier_c"])
    now = datetime.now(timezone.utc).strftime("%b %d, %I:%M %p UTC")

    lines = [
        f"{cfg['emoji']} <b>ROYALEY — {cfg['label']} ALERT</b>",
        f"<i>{cfg['desc']} • {now}</i>",
        "━━━━━━━━━━━━━━━━━━━━━━",
    ]

    for p in predictions[:8]:
        conf = float(p.get("confidence", 0.5)) * 100
        edge = float(p.get("edge", 0)) * 100
        sport = p.get("sport", "?")
        team = p.get("team", "?")
        bet_type = p.get("bet_type", "?").upper()
        odds = p.get("odds", "?")
        line = p.get("line", "")
        game_time = p.get("game_time", "")

        # Sport emoji
        sport_emoji = {
            "NBA": "🏀", "NCAAB": "🏀", "NFL": "🏈", "NCAAF": "🏈",
            "NHL": "🏒", "MLB": "⚾", "ATP": "🎾", "WTA": "🎾",
            "WNBA": "🏀", "CFL": "🏈",
        }.get(sport, "🏅")

        # Format bet type display
        if bet_type == "MONEYLINE":
            pick_display = f"{team} ML"
        elif bet_type == "SPREAD":
            pick_display = f"{team} {'+' if line and float(line) > 0 else ''}{line}"
        elif bet_type == "TOTAL":
            side = p.get("predicted_side", "over").upper()
            pick_display = f"{side} {line}"
        else:
            pick_display = f"{team} {bet_type}"

        # Matchup display
        home = p.get("home", "")
        away = p.get("away", "")
        matchup = f"{home} vs {away}" if home and away else ""

        lines.append(f"")
        lines.append(f"{sport_emoji} <b>{sport}</b> {'• ' + game_time if game_time else ''}")
        if matchup:
            lines.append(f"   {matchup}")
        lines.append(f"   📌 <b>{pick_display}</b> ({odds})")
        lines.append(f"   📊 {conf:.1f}% conf • {edge:+.1f}% edge")

    if len(predictions) > 8:
        lines.append(f"\n<i>+{len(predictions) - 8} more predictions</i>")

    lines.extend([
        "",
        "━━━━━━━━━━━━━━━━━━━━━━",
        f"📊 <b>{len(predictions)} prediction{'s' if len(predictions) != 1 else ''}</b> • royaley.com",
    ])

    return "\n".join(lines)


def _format_grading_telegram(stats: dict) -> str:
    """Format grading results for Telegram with game details."""
    games = stats.get("games_graded", 0)
    preds = stats.get("predictions_graded", 0)
    wins = stats.get("wins", 0)
    losses = stats.get("losses", 0)
    pushes = stats.get("pushes", 0)
    pnl = stats.get("total_pnl", 0)
    now = datetime.now(timezone.utc).strftime("%b %d, %I:%M %p UTC")

    # Win rate
    total_decided = wins + losses
    win_rate = (wins / total_decided * 100) if total_decided > 0 else 0

    # P&L emoji
    pnl_emoji = "🟢" if pnl >= 0 else "🔴"
    pnl_display = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"

    sport_emoji = {
        "NBA": "🏀", "NCAAB": "🏀", "NFL": "🏈", "NCAAF": "🏈",
        "NHL": "🏒", "MLB": "⚾", "ATP": "🎾", "WTA": "🎾",
        "WNBA": "🏀", "CFL": "🏈",
    }

    lines = [
        f"✅ <b>ROYALEY — GRADING COMPLETE</b>",
        f"<i>{now}</i>",
        "━━━━━━━━━━━━━━━━━━━━━━",
    ]

    # Per-game breakdown FIRST (most important info)
    game_results = stats.get("game_results", [])
    if game_results:
        for gr in game_results[:8]:
            sport = gr.get("sport", "?")
            home = gr.get("home", "?")
            away = gr.get("away", "?")
            h_score = gr.get("home_score", "?")
            a_score = gr.get("away_score", "?")
            gw = gr.get("wins", 0)
            gl = gr.get("losses", 0)
            gpnl = gr.get("pnl", 0)
            emoji = sport_emoji.get(sport, "🏅")

            gpnl_str = f"+${gpnl:.0f}" if gpnl >= 0 else f"-${abs(gpnl):.0f}"
            gpnl_icon = "🟢" if gpnl >= 0 else "🔴"

            lines.append(f"")
            lines.append(f"{emoji} <b>{sport}</b>")
            lines.append(f"   {home} <b>{h_score}</b> - <b>{a_score}</b> {away}")
            lines.append(f"   {gw}W-{gl}L • {gpnl_icon} {gpnl_str}")

        if len(game_results) > 8:
            lines.append(f"\n<i>+{len(game_results) - 8} more games</i>")

    lines.extend([
        "",
        "━━━━━━━━━━━━━━━━━━━━━━",
        f"",
        f"📊 <b>SUMMARY</b>",
        f"   🏟️ Games: <b>{games}</b> • Picks: <b>{preds}</b>",
        f"   ✅ <b>{wins}W</b> - ❌ <b>{losses}L</b>{f' - ➖ {pushes}P' if pushes else ''} ({win_rate:.1f}%)",
        f"   {pnl_emoji} P&L: <b>{pnl_display}</b>",
        f"",
        "━━━━━━━━━━━━━━━━━━━━━━",
        f"📊 royaley.com/predictions",
    ])

    return "\n".join(lines)


def _format_daily_summary_telegram(stats: dict) -> str:
    """Format daily performance summary for Telegram."""
    date = stats.get("date", datetime.now(timezone.utc).strftime("%b %d, %Y"))
    total_preds = stats.get("total_predictions", 0)
    graded = stats.get("graded", 0)
    pending = stats.get("pending", 0)
    wins = stats.get("wins", 0)
    losses = stats.get("losses", 0)
    pushes = stats.get("pushes", 0)
    pnl = stats.get("pnl", 0)
    roi = stats.get("roi", 0)
    clv_avg = stats.get("avg_clv", 0)
    best_pick = stats.get("best_pick", "")
    worst_pick = stats.get("worst_pick", "")

    total_decided = wins + losses
    win_rate = (wins / total_decided * 100) if total_decided > 0 else 0
    pnl_emoji = "🟢" if pnl >= 0 else "🔴"
    pnl_display = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"

    lines = [
        f"📋 <b>ROYALEY — DAILY REPORT</b>",
        f"<i>{date}</i>",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "",
        f"📊 <b>OVERVIEW</b>",
        f"   Predictions: {total_preds} ({graded} graded, {pending} pending)",
        f"   Record: <b>{wins}W - {losses}L{f' - {pushes}P' if pushes else ''}</b>",
        f"   Win Rate: <b>{win_rate:.1f}%</b>",
        "",
        f"{pnl_emoji} <b>FINANCIALS</b>",
        f"   P&L: <b>{pnl_display}</b>",
        f"   ROI: <b>{'+' if roi >= 0 else ''}{roi:.1f}%</b>",
        f"   Avg CLV: <b>{'+' if clv_avg >= 0 else ''}{clv_avg:.1f}%</b>",
    ]

    # By-tier breakdown
    tier_stats = stats.get("by_tier", {})
    if tier_stats:
        lines.extend(["", f"🎯 <b>BY TIER</b>"])
        tier_emojis = {"A": "🔴", "B": "🟠", "C": "🟡", "D": "⚪"}
        for tier_name in ["A", "B", "C", "D"]:
            ts = tier_stats.get(tier_name, {})
            if ts:
                tw = ts.get("wins", 0)
                tl = ts.get("losses", 0)
                tpnl = ts.get("pnl", 0)
                emoji = tier_emojis.get(tier_name, "⚪")
                pnl_str = f"+${tpnl:.0f}" if tpnl >= 0 else f"-${abs(tpnl):.0f}"
                lines.append(f"   {emoji} Tier {tier_name}: {tw}W-{tl}L ({pnl_str})")

    # By-sport breakdown
    sport_stats = stats.get("by_sport", {})
    if sport_stats:
        lines.extend(["", f"🏅 <b>BY SPORT</b>"])
        sport_emojis = {
            "NBA": "🏀", "NCAAB": "🏀", "NFL": "🏈", "NCAAF": "🏈",
            "NHL": "🏒", "MLB": "⚾", "ATP": "🎾", "WTA": "🎾",
        }
        for sport, ss in sorted(sport_stats.items()):
            sw = ss.get("wins", 0)
            sl = ss.get("losses", 0)
            emoji = sport_emojis.get(sport, "🏅")
            lines.append(f"   {emoji} {sport}: {sw}W-{sl}L")

    # Best/worst picks
    if best_pick or worst_pick:
        lines.extend(["", f"💡 <b>HIGHLIGHTS</b>"])
        if best_pick:
            lines.append(f"   🏆 Best: {best_pick}")
        if worst_pick:
            lines.append(f"   📉 Worst: {worst_pick}")

    lines.extend([
        "",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"📊 Full details: royaley.com/analytics",
    ])

    return "\n".join(lines)


def _format_clv_alert_telegram(data: dict) -> str:
    """Format CLV alert for Telegram."""
    sport = data.get("sport", "?")
    team = data.get("team", "?")
    bet_type = data.get("bet_type", "?").upper()
    clv = data.get("clv", 0)
    open_odds = data.get("open_odds", "?")
    close_odds = data.get("close_odds", "?")
    now = datetime.now(timezone.utc).strftime("%I:%M %p UTC")

    sport_emoji = {
        "NBA": "🏀", "NCAAB": "🏀", "NFL": "🏈", "NHL": "🏒",
        "MLB": "⚾", "ATP": "🎾", "WTA": "🎾",
    }.get(sport, "🏅")

    clv_emoji = "💰" if clv >= 5 else "📈"

    lines = [
        f"{clv_emoji} <b>ROYALEY — CLV ALERT</b>",
        f"<i>{now}</i>",
        "━━━━━━━━━━━━━━━━━━━━━━",
        f"",
        f"{sport_emoji} <b>{sport}</b> • {bet_type}",
        f"📌 <b>{team}</b>",
        f"",
        f"   Open:  <code>{open_odds}</code>",
        f"   Close: <code>{close_odds}</code>",
        f"   CLV:   <b>{'+' if clv >= 0 else ''}{clv:.1f}%</b>",
        f"",
        "━━━━━━━━━━━━━━━━━━━━━━",
        f"📊 royaley.com/analytics",
    ]

    return "\n".join(lines)


def _format_system_error_telegram(message: str, metadata: dict) -> str:
    """Format system error alert for Telegram."""
    now = datetime.now(timezone.utc).strftime("%b %d, %I:%M %p UTC")
    component = metadata.get("component", "Unknown")
    severity = metadata.get("severity", "warning").upper()

    severity_emoji = {"CRITICAL": "🔴", "ERROR": "🟠", "WARNING": "🟡"}.get(severity, "🟡")

    lines = [
        f"🚨 <b>ROYALEY — SYSTEM ALERT</b>",
        f"<i>{now}</i>",
        "━━━━━━━━━━━━━━━━━━━━━━",
        f"",
        f"{severity_emoji} <b>{severity}</b> • {component}",
        f"",
        f"<code>{message[:500]}</code>",
    ]

    # Add extra metadata
    for k, v in metadata.items():
        if k not in ("component", "severity"):
            lines.append(f"<b>{k}:</b> {v}")

    lines.extend([
        "",
        "━━━━━━━━━━━━━━━━━━━━━━",
        f"🔧 royaley.com/system-health",
    ])

    return "\n".join(lines)


def _format_model_training_telegram(data: dict) -> str:
    """Format model training notification for Telegram."""
    sport = data.get("sport", "?")
    bet_type = data.get("bet_type", "?")
    accuracy = data.get("accuracy", 0)
    samples = data.get("samples", 0)
    duration = data.get("duration", "?")
    now = datetime.now(timezone.utc).strftime("%b %d, %I:%M %p UTC")

    lines = [
        f"🧠 <b>ROYALEY — MODEL UPDATED</b>",
        f"<i>{now}</i>",
        "━━━━━━━━━━━━━━━━━━━━━━",
        f"",
        f"🏅 <b>{sport} / {bet_type}</b>",
        f"   Accuracy: <b>{accuracy:.1f}%</b>",
        f"   Samples: <b>{samples:,}</b>",
        f"   Duration: <b>{duration}</b>",
    ]

    # Algorithm details if available
    algorithms = data.get("algorithms", {})
    if algorithms:
        lines.extend(["", "📊 <b>Ensemble Weights</b>"])
        for algo, weight in sorted(algorithms.items(), key=lambda x: -x[1]):
            if weight > 0:
                bar = "█" * int(weight * 20) + "░" * (20 - int(weight * 20))
                lines.append(f"   {algo}: <code>{bar}</code> {weight:.0%}")

    lines.extend([
        "",
        "━━━━━━━━━━━━━━━━━━━━━━",
        f"🧠 royaley.com/models",
    ])

    return "\n".join(lines)


def _format_generic_telegram(event: str, message: str, metadata: dict) -> str:
    """Fallback formatter for any event type."""
    emoji_map = {
        "backup_complete": "💾",
        "live_game": "🏟️",
    }
    label_map = {
        "backup_complete": "BACKUP COMPLETE",
        "live_game": "LIVE GAME",
    }
    emoji = emoji_map.get(event, "📢")
    label = label_map.get(event, event.replace("_", " ").upper())
    now = datetime.now(timezone.utc).strftime("%b %d, %I:%M %p UTC")

    lines = [
        f"{emoji} <b>ROYALEY — {label}</b>",
        f"<i>{now}</i>",
        "━━━━━━━━━━━━━━━━━━━━━━",
        f"",
        message,
    ]

    if metadata:
        lines.append("")
        for k, v in metadata.items():
            lines.append(f"<b>{k}:</b> {v}")

    lines.extend([
        "",
        "━━━━━━━━━━━━━━━━━━━━━━",
        f"📊 royaley.com",
    ])

    return "\n".join(lines)


# =============================================================================
# EMAIL FORMATTERS
# =============================================================================

def _email_wrapper(title: str, body_html: str) -> str:
    """Wrap email body in professional HTML template."""
    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="margin:0; padding:0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; background-color: #0d1117; color: #e6edf3;">
    <table width="100%" cellpadding="0" cellspacing="0" style="background-color: #0d1117; padding: 20px 0;">
        <tr>
            <td align="center">
                <table width="600" cellpadding="0" cellspacing="0" style="background-color: #161b22; border-radius: 12px; overflow: hidden; border: 1px solid #30363d;">
                    <!-- Header -->
                    <tr>
                        <td style="background: linear-gradient(135deg, #1a5276 0%, #0e7490 100%); padding: 24px 32px;">
                            <h1 style="margin:0; color: #ffffff; font-size: 20px; font-weight: 600; letter-spacing: 1px;">
                                ♛ ROYALEY
                            </h1>
                            <p style="margin: 4px 0 0 0; color: #94d2e8; font-size: 13px;">
                                {title}
                            </p>
                        </td>
                    </tr>
                    <!-- Body -->
                    <tr>
                        <td style="padding: 28px 32px;">
                            {body_html}
                        </td>
                    </tr>
                    <!-- Footer -->
                    <tr>
                        <td style="padding: 16px 32px; background-color: #0d1117; border-top: 1px solid #30363d;">
                            <p style="margin:0; color: #6e7681; font-size: 12px; text-align: center;">
                                <a href="https://royaley.com" style="color: #58a6ff; text-decoration: none;">royaley.com</a>
                                &nbsp;•&nbsp; Enterprise Sports Prediction Platform
                            </p>
                        </td>
                    </tr>
                </table>
            </td>
        </tr>
    </table>
</body>
</html>"""


def _format_prediction_email(predictions: list, tier: str) -> tuple:
    """Format prediction alert email. Returns (subject, html_body, text_body)."""
    tier_labels = {
        "tier_a": ("🔴 Tier A — High Value", "#e74c3c"),
        "tier_b": ("🟠 Tier B — Strong Edge", "#f39c12"),
        "tier_c": ("🟡 Tier C — Standard", "#f1c40f"),
        "tier_d": ("⚪ Tier D — Low Edge", "#95a5a6"),
    }
    label, color = tier_labels.get(tier, ("Prediction", "#95a5a6"))

    subject = f"Royaley: {len(predictions)} {label.split('—')[0].strip()} Prediction{'s' if len(predictions) != 1 else ''}"

    rows = ""
    for p in predictions[:12]:
        conf = float(p.get("confidence", 0.5)) * 100
        edge = float(p.get("edge", 0)) * 100
        sport = p.get("sport", "?")
        team = p.get("team", "?")
        bet_type = p.get("bet_type", "?").upper()
        odds = p.get("odds", "?")

        rows += f"""
        <tr style="border-bottom: 1px solid #30363d;">
            <td style="padding: 10px 8px; color: #8b949e; font-size: 12px;">{sport}</td>
            <td style="padding: 10px 8px; color: #e6edf3; font-weight: 600;">{team}</td>
            <td style="padding: 10px 8px; color: #8b949e;">{bet_type}</td>
            <td style="padding: 10px 8px; color: #e6edf3;">{odds}</td>
            <td style="padding: 10px 8px; color: #58a6ff; font-weight: 600;">{conf:.1f}%</td>
            <td style="padding: 10px 8px; color: {'#3fb950' if edge > 0 else '#f85149'};">{edge:+.1f}%</td>
        </tr>"""

    body = f"""
    <div style="margin-bottom: 20px;">
        <span style="display: inline-block; background-color: {color}; color: #fff; padding: 4px 12px; border-radius: 4px; font-size: 13px; font-weight: 600;">
            {label}
        </span>
        <span style="color: #8b949e; font-size: 13px; margin-left: 8px;">
            {len(predictions)} prediction{'s' if len(predictions) != 1 else ''}
        </span>
    </div>
    <table width="100%" cellpadding="0" cellspacing="0" style="font-size: 13px;">
        <tr style="border-bottom: 2px solid #30363d;">
            <th style="text-align:left; padding: 8px; color: #8b949e; font-weight: 500;">Sport</th>
            <th style="text-align:left; padding: 8px; color: #8b949e; font-weight: 500;">Pick</th>
            <th style="text-align:left; padding: 8px; color: #8b949e; font-weight: 500;">Type</th>
            <th style="text-align:left; padding: 8px; color: #8b949e; font-weight: 500;">Odds</th>
            <th style="text-align:left; padding: 8px; color: #8b949e; font-weight: 500;">Conf</th>
            <th style="text-align:left; padding: 8px; color: #8b949e; font-weight: 500;">Edge</th>
        </tr>
        {rows}
    </table>
    """

    text_body = f"{label} - {len(predictions)} predictions\n\n"
    for p in predictions[:12]:
        conf = float(p.get("confidence", 0.5)) * 100
        text_body += f"  {p.get('sport','?')} | {p.get('team','?')} ({p.get('bet_type','?')}) @ {p.get('odds','?')} — {conf:.1f}%\n"

    return subject, _email_wrapper(label, body), text_body


def _format_grading_email(stats: dict) -> tuple:
    """Format grading summary email."""
    games = stats.get("games_graded", 0)
    preds = stats.get("predictions_graded", 0)
    wins = stats.get("wins", 0)
    losses = stats.get("losses", 0)
    pnl = stats.get("total_pnl", 0)

    total_decided = wins + losses
    win_rate = (wins / total_decided * 100) if total_decided > 0 else 0
    pnl_color = "#3fb950" if pnl >= 0 else "#f85149"
    pnl_display = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"

    subject = f"Royaley: {games} Games Graded — {wins}W {losses}L ({pnl_display})"

    body = f"""
    <div style="display: flex; gap: 12px; margin-bottom: 24px;">
        <div style="flex:1; background: #0d1117; border-radius: 8px; padding: 16px; border: 1px solid #30363d; text-align: center;">
            <div style="color: #8b949e; font-size: 11px; text-transform: uppercase;">Games</div>
            <div style="color: #e6edf3; font-size: 28px; font-weight: 700;">{games}</div>
        </div>
    </div>

    <table width="100%" cellpadding="12" cellspacing="0" style="margin-bottom: 20px;">
        <tr>
            <td style="background: #0d1117; border-radius: 8px; border: 1px solid #30363d; text-align: center; width: 25%;">
                <div style="color: #8b949e; font-size: 11px;">RECORD</div>
                <div style="color: #e6edf3; font-size: 22px; font-weight: 700;">{wins}W-{losses}L</div>
            </td>
            <td style="background: #0d1117; border-radius: 8px; border: 1px solid #30363d; text-align: center; width: 25%;">
                <div style="color: #8b949e; font-size: 11px;">WIN RATE</div>
                <div style="color: #58a6ff; font-size: 22px; font-weight: 700;">{win_rate:.1f}%</div>
            </td>
            <td style="background: #0d1117; border-radius: 8px; border: 1px solid #30363d; text-align: center; width: 25%;">
                <div style="color: #8b949e; font-size: 11px;">P&L</div>
                <div style="color: {pnl_color}; font-size: 22px; font-weight: 700;">{pnl_display}</div>
            </td>
            <td style="background: #0d1117; border-radius: 8px; border: 1px solid #30363d; text-align: center; width: 25%;">
                <div style="color: #8b949e; font-size: 11px;">PREDICTIONS</div>
                <div style="color: #e6edf3; font-size: 22px; font-weight: 700;">{preds}</div>
            </td>
        </tr>
    </table>
    """

    text_body = f"Grading Complete: {games} games, {preds} predictions\nRecord: {wins}W-{losses}L ({win_rate:.1f}%)\nP&L: {pnl_display}"

    return subject, _email_wrapper("Grading Complete", body), text_body


def _format_daily_summary_email(stats: dict) -> tuple:
    """Format daily summary email."""
    date = stats.get("date", datetime.now(timezone.utc).strftime("%b %d, %Y"))
    wins = stats.get("wins", 0)
    losses = stats.get("losses", 0)
    pnl = stats.get("pnl", 0)
    roi = stats.get("roi", 0)

    total_decided = wins + losses
    win_rate = (wins / total_decided * 100) if total_decided > 0 else 0
    pnl_color = "#3fb950" if pnl >= 0 else "#f85149"
    pnl_display = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"

    subject = f"Royaley Daily: {wins}W-{losses}L ({pnl_display}) — {date}"

    body = f"""
    <h2 style="color: #e6edf3; margin: 0 0 20px 0; font-size: 18px;">Daily Performance Report</h2>
    <p style="color: #8b949e; margin: 0 0 24px 0;">{date}</p>

    <table width="100%" cellpadding="12" cellspacing="8" style="margin-bottom: 24px;">
        <tr>
            <td style="background: #0d1117; border-radius: 8px; border: 1px solid #30363d; text-align: center;">
                <div style="color: #8b949e; font-size: 11px;">RECORD</div>
                <div style="color: #e6edf3; font-size: 24px; font-weight: 700;">{wins}W-{losses}L</div>
                <div style="color: #58a6ff; font-size: 13px;">{win_rate:.1f}% win rate</div>
            </td>
            <td style="background: #0d1117; border-radius: 8px; border: 1px solid #30363d; text-align: center;">
                <div style="color: #8b949e; font-size: 11px;">P&L</div>
                <div style="color: {pnl_color}; font-size: 24px; font-weight: 700;">{pnl_display}</div>
                <div style="color: #8b949e; font-size: 13px;">ROI: {'+' if roi >= 0 else ''}{roi:.1f}%</div>
            </td>
        </tr>
    </table>
    """

    text_body = f"Royaley Daily Report — {date}\nRecord: {wins}W-{losses}L ({win_rate:.1f}%)\nP&L: {pnl_display}, ROI: {roi:.1f}%"

    return subject, _email_wrapper(f"Daily Report — {date}", body), text_body


# =============================================================================
# PUBLIC API
# =============================================================================

async def notify(
    event: str,
    message: str,
    metadata: Optional[Dict[str, Any]] = None,
    db: Optional[AsyncSession] = None,
) -> int:
    """
    Send notification for an event. Returns number of messages sent.
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

        # ── Telegram ────────────────────────────────────────────────
        if _should_notify(settings, event, "telegram"):
            tg_accounts = _get_telegram_config(settings)
            if tg_accounts:
                # Use specialized formatters based on event
                if event in ("tier_a", "tier_b", "tier_c", "tier_d"):
                    tg_message = _format_generic_telegram(event, message, metadata)
                elif event == "grading_complete":
                    tg_message = _format_grading_telegram(metadata)
                elif event == "daily_summary":
                    tg_message = _format_daily_summary_telegram(metadata)
                elif event == "clv_alert":
                    tg_message = _format_clv_alert_telegram(metadata)
                elif event == "system_errors":
                    tg_message = _format_system_error_telegram(message, metadata)
                elif event == "model_training":
                    tg_message = _format_model_training_telegram(metadata)
                else:
                    tg_message = _format_generic_telegram(event, message, metadata)

                for acc in tg_accounts:
                    ok = await _send_telegram(acc["token"], acc["chat_id"], tg_message)
                    if ok:
                        sent += 1
                        logger.info(f"[Notify] Telegram sent: {event} → {acc.get('name', acc['chat_id'])}")

        # ── Email ───────────────────────────────────────────────────
        if _should_notify(settings, event, "email"):
            emails = _get_email_config(settings)
            if emails:
                # Use specialized email formatters
                if event == "grading_complete":
                    subject, html_body, text_body = _format_grading_email(metadata)
                elif event == "daily_summary":
                    subject, html_body, text_body = _format_daily_summary_email(metadata)
                elif event in ("tier_a", "tier_b", "tier_c", "tier_d"):
                    # Build prediction list from metadata
                    preds = metadata.get("predictions", [])
                    if preds:
                        subject, html_body, text_body = _format_prediction_email(preds, event)
                    else:
                        subject = f"Royaley: {event.replace('_', ' ').title()}"
                        html_body = _email_wrapper(subject, f"<p style='color:#e6edf3;'>{message}</p>")
                        text_body = message
                else:
                    subject = f"Royaley: {event.replace('_', ' ').title()}"
                    body_html = f"<p style='color:#e6edf3;'>{message}</p>"
                    if metadata:
                        body_html += "<table style='margin-top:16px; font-size:13px;'>"
                        for k, v in metadata.items():
                            body_html += f"<tr><td style='color:#8b949e; padding:4px 12px 4px 0;'>{k}</td><td style='color:#e6edf3;'>{v}</td></tr>"
                        body_html += "</table>"
                    html_body = _email_wrapper(event.replace("_", " ").title(), body_html)
                    text_body = f"{message}\n\n" + "\n".join(f"{k}: {v}" for k, v in metadata.items())

                for email in emails:
                    ok = await _send_email(email, subject, html_body, text_body)
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
    Send notifications for new predictions grouped by tier.
    """
    if not predictions:
        return 0

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

        # Telegram gets rich formatting
        tg_message = _format_prediction_telegram(preds, tier)

        # Build generic message for the notify() dispatcher
        lines = []
        for p in preds[:10]:
            conf = float(p.get("confidence", 0.5)) * 100
            lines.append(
                f"• {p.get('sport', '?')} | {p.get('team', '?')} "
                f"({p.get('bet_type', '?')}) — {conf:.1f}% @ {p.get('odds', '?')}"
            )

        metadata = {"Count": len(preds), "predictions": preds}
        if len(preds) > 10:
            metadata["Note"] = f"Showing 10 of {len(preds)}"

        # Override telegram message with rich version
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
                continue

            # Telegram
            if _should_notify(settings, tier, "telegram"):
                tg_accounts = _get_telegram_config(settings)
                for acc in tg_accounts:
                    ok = await _send_telegram(acc["token"], acc["chat_id"], tg_message)
                    if ok:
                        sent += 1
                        logger.info(f"[Notify] Telegram sent: {tier} ({len(preds)} picks) → {acc.get('name', acc['chat_id'])}")

            # Email
            if _should_notify(settings, tier, "email"):
                emails = _get_email_config(settings)
                if emails:
                    subject, html_body, text_body = _format_prediction_email(preds, tier)
                    for email in emails:
                        ok = await _send_email(email, subject, html_body, text_body)
                        if ok:
                            sent += 1
                            logger.info(f"[Notify] Email sent: {tier} ({len(preds)} picks) → {email}")

        except Exception as e:
            logger.error(f"[Notify] Error dispatching {tier}: {e}")
        finally:
            if own_engine:
                await db_session.close()
                await own_engine.dispose()

    return sent


async def notify_grading(stats: dict, db: Optional[AsyncSession] = None) -> int:
    """Send notification after grading completes."""
    games = stats.get("games_graded", 0)
    if games == 0:
        return 0

    message = f"Graded {games} games, {stats.get('predictions_graded', 0)} predictions"
    return await notify("grading_complete", message, stats, db=db)