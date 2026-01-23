"""
LOYALEY - Phase 4 Enterprise Alerting Service
Multi-channel alerting: Telegram, Slack, Email, PagerDuty, Datadog
"""

import asyncio
import json
import logging
import smtplib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from typing import Any, Dict, List, Optional

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertChannel(str, Enum):
    """Available alert channels"""
    TELEGRAM = "telegram"
    SLACK = "slack"
    EMAIL = "email"
    PAGERDUTY = "pagerduty"
    DATADOG = "datadog"


@dataclass
class Alert:
    """Alert data structure"""
    title: str
    message: str
    severity: AlertSeverity
    source: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)
    alert_id: str = ""
    
    def __post_init__(self):
        if not self.alert_id:
            self.alert_id = f"{self.source}_{self.timestamp.timestamp()}"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "title": self.title,
            "message": self.message,
            "severity": self.severity.value,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }
    
    def get_emoji(self) -> str:
        """Get emoji based on severity"""
        emojis = {
            AlertSeverity.INFO: "ℹ️",
            AlertSeverity.WARNING: "⚠️",
            AlertSeverity.ERROR: "🔴",
            AlertSeverity.CRITICAL: "🚨"
        }
        return emojis.get(self.severity, "📢")


class AlertProvider(ABC):
    """Abstract base class for alert providers"""
    
    @abstractmethod
    async def send(self, alert: Alert) -> bool:
        """Send alert through this channel"""
        pass
    
    @abstractmethod
    def is_configured(self) -> bool:
        """Check if provider is properly configured"""
        pass


class TelegramProvider(AlertProvider):
    """Telegram alert provider"""
    
    def __init__(self):
        self.bot_token = settings.TELEGRAM_BOT_TOKEN
        self.chat_id = settings.TELEGRAM_CHAT_ID
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
    
    def is_configured(self) -> bool:
        return bool(self.bot_token and self.chat_id)
    
    async def send(self, alert: Alert) -> bool:
        if not self.is_configured():
            return False
        
        message = self._format_message(alert)
        
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.post(
                    f"{self.base_url}/sendMessage",
                    json={
                        "chat_id": self.chat_id,
                        "text": message,
                        "parse_mode": "HTML"
                    }
                )
                
                if response.status_code == 200:
                    logger.info(f"Telegram alert sent: {alert.alert_id}")
                    return True
                else:
                    logger.error(f"Telegram send failed: {response.text}")
                    return False
        except Exception as e:
            logger.error(f"Telegram alert error: {e}")
            return False
    
    def _format_message(self, alert: Alert) -> str:
        """Format alert for Telegram"""
        return f"""
{alert.get_emoji()} <b>{alert.title}</b>

<b>Severity:</b> {alert.severity.value.upper()}
<b>Source:</b> {alert.source}
<b>Time:</b> {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}

{alert.message}

{self._format_metadata(alert.metadata)}
        """.strip()
    
    def _format_metadata(self, metadata: Dict[str, Any]) -> str:
        if not metadata:
            return ""
        lines = [f"<b>{k}:</b> {v}" for k, v in metadata.items()]
        return "\n".join(lines)


class SlackProvider(AlertProvider):
    """Slack alert provider via webhook"""
    
    def __init__(self):
        self.webhook_url = settings.SLACK_WEBHOOK_URL
    
    def is_configured(self) -> bool:
        return bool(self.webhook_url)
    
    async def send(self, alert: Alert) -> bool:
        if not self.is_configured():
            return False
        
        payload = self._format_payload(alert)
        
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.post(
                    self.webhook_url,
                    json=payload
                )
                
                if response.status_code == 200:
                    logger.info(f"Slack alert sent: {alert.alert_id}")
                    return True
                else:
                    logger.error(f"Slack send failed: {response.text}")
                    return False
        except Exception as e:
            logger.error(f"Slack alert error: {e}")
            return False
    
    def _format_payload(self, alert: Alert) -> Dict[str, Any]:
        """Format alert for Slack"""
        color_map = {
            AlertSeverity.INFO: "#36a64f",
            AlertSeverity.WARNING: "#ffcc00",
            AlertSeverity.ERROR: "#ff0000",
            AlertSeverity.CRITICAL: "#8b0000"
        }
        
        fields = [
            {"title": "Severity", "value": alert.severity.value.upper(), "short": True},
            {"title": "Source", "value": alert.source, "short": True},
            {"title": "Time", "value": alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC'), "short": True}
        ]
        
        for key, value in alert.metadata.items():
            fields.append({"title": key, "value": str(value), "short": True})
        
        return {
            "attachments": [{
                "color": color_map.get(alert.severity, "#808080"),
                "title": f"{alert.get_emoji()} {alert.title}",
                "text": alert.message,
                "fields": fields,
                "footer": "LOYALEY Alerting",
                "ts": int(alert.timestamp.timestamp())
            }]
        }


class EmailProvider(AlertProvider):
    """Email alert provider via SMTP"""
    
    def __init__(self):
        self.smtp_host = settings.EMAIL_SMTP_HOST
        self.smtp_port = settings.EMAIL_SMTP_PORT
        self.smtp_user = settings.EMAIL_SMTP_USER
        self.smtp_password = settings.EMAIL_SMTP_PASSWORD
        self.from_address = settings.EMAIL_FROM_ADDRESS
        self.recipients = settings.ALERT_EMAIL_RECIPIENTS
    
    def is_configured(self) -> bool:
        return bool(
            self.smtp_host and
            self.smtp_user and
            self.smtp_password and
            self.from_address and
            self.recipients
        )
    
    async def send(self, alert: Alert) -> bool:
        if not self.is_configured():
            return False
        
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._send_sync, alert)
            logger.info(f"Email alert sent: {alert.alert_id}")
            return True
        except Exception as e:
            logger.error(f"Email alert error: {e}")
            return False
    
    def _send_sync(self, alert: Alert):
        """Synchronous email send"""
        msg = MIMEMultipart('alternative')
        msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
        msg['From'] = self.from_address
        msg['To'] = ', '.join(self.recipients)
        
        # Plain text
        text = f"""
{alert.title}

Severity: {alert.severity.value.upper()}
Source: {alert.source}
Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}

{alert.message}

Metadata:
{json.dumps(alert.metadata, indent=2)}
        """
        
        # HTML
        html = f"""
        <html>
        <head>
            <style>
                .alert-box {{
                    padding: 20px;
                    border-radius: 8px;
                    font-family: Arial, sans-serif;
                }}
                .info {{ background-color: #e7f3fe; border-left: 6px solid #2196F3; }}
                .warning {{ background-color: #ffffcc; border-left: 6px solid #ffeb3b; }}
                .error {{ background-color: #ffebee; border-left: 6px solid #f44336; }}
                .critical {{ background-color: #ffcdd2; border-left: 6px solid #b71c1c; }}
            </style>
        </head>
        <body>
            <div class="alert-box {alert.severity.value}">
                <h2>{alert.get_emoji()} {alert.title}</h2>
                <p><strong>Severity:</strong> {alert.severity.value.upper()}</p>
                <p><strong>Source:</strong> {alert.source}</p>
                <p><strong>Time:</strong> {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
                <hr>
                <p>{alert.message}</p>
                <hr>
                <h3>Metadata</h3>
                <pre>{json.dumps(alert.metadata, indent=2)}</pre>
            </div>
        </body>
        </html>
        """
        
        msg.attach(MIMEText(text, 'plain'))
        msg.attach(MIMEText(html, 'html'))
        
        with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
            server.starttls()
            server.login(self.smtp_user, self.smtp_password)
            server.sendmail(self.from_address, self.recipients, msg.as_string())


class PagerDutyProvider(AlertProvider):
    """PagerDuty alert provider"""
    
    def __init__(self):
        self.api_key = settings.PAGERDUTY_API_KEY
        self.events_url = "https://events.pagerduty.com/v2/enqueue"
    
    def is_configured(self) -> bool:
        return bool(self.api_key)
    
    async def send(self, alert: Alert) -> bool:
        if not self.is_configured():
            return False
        
        # Only send critical/error alerts to PagerDuty
        if alert.severity not in [AlertSeverity.ERROR, AlertSeverity.CRITICAL]:
            return True
        
        payload = {
            "routing_key": self.api_key,
            "event_action": "trigger",
            "dedup_key": alert.alert_id,
            "payload": {
                "summary": f"[{alert.severity.value.upper()}] {alert.title}",
                "source": alert.source,
                "severity": "critical" if alert.severity == AlertSeverity.CRITICAL else "error",
                "timestamp": alert.timestamp.isoformat(),
                "custom_details": {
                    "message": alert.message,
                    **alert.metadata
                }
            }
        }
        
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.post(self.events_url, json=payload)
                
                if response.status_code == 202:
                    logger.info(f"PagerDuty alert sent: {alert.alert_id}")
                    return True
                else:
                    logger.error(f"PagerDuty send failed: {response.text}")
                    return False
        except Exception as e:
            logger.error(f"PagerDuty alert error: {e}")
            return False


class DatadogProvider(AlertProvider):
    """Datadog event provider"""
    
    def __init__(self):
        self.api_key = settings.DATADOG_API_KEY
        self.events_url = "https://api.datadoghq.com/api/v1/events"
    
    def is_configured(self) -> bool:
        return bool(self.api_key)
    
    async def send(self, alert: Alert) -> bool:
        if not self.is_configured():
            return False
        
        alert_type_map = {
            AlertSeverity.INFO: "info",
            AlertSeverity.WARNING: "warning",
            AlertSeverity.ERROR: "error",
            AlertSeverity.CRITICAL: "error"
        }
        
        payload = {
            "title": f"[{alert.severity.value.upper()}] {alert.title}",
            "text": f"{alert.message}\n\nMetadata:\n```\n{json.dumps(alert.metadata, indent=2)}\n```",
            "alert_type": alert_type_map.get(alert.severity, "info"),
            "source_type_name": "LOYALEY",
            "tags": [
                f"source:{alert.source}",
                f"severity:{alert.severity.value}",
                f"env:{settings.ENVIRONMENT}"
            ]
        }
        
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.post(
                    self.events_url,
                    json=payload,
                    headers={"DD-API-KEY": self.api_key}
                )
                
                if response.status_code in [200, 202]:
                    logger.info(f"Datadog alert sent: {alert.alert_id}")
                    return True
                else:
                    logger.error(f"Datadog send failed: {response.text}")
                    return False
        except Exception as e:
            logger.error(f"Datadog alert error: {e}")
            return False


class AlertingService:
    """Enterprise alerting service managing all channels"""
    
    def __init__(self):
        self.providers: Dict[AlertChannel, AlertProvider] = {
            AlertChannel.TELEGRAM: TelegramProvider(),
            AlertChannel.SLACK: SlackProvider(),
            AlertChannel.EMAIL: EmailProvider(),
            AlertChannel.PAGERDUTY: PagerDutyProvider(),
            AlertChannel.DATADOG: DatadogProvider()
        }
        
        self._alert_history: List[Alert] = []
        self._max_history = 1000
        self._cooldowns: Dict[str, float] = {}
        self._cooldown_seconds = 300  # 5 minutes between duplicate alerts
    
    def get_configured_channels(self) -> List[AlertChannel]:
        """Get list of properly configured channels"""
        return [
            channel for channel, provider in self.providers.items()
            if provider.is_configured()
        ]
    
    async def send_alert(
        self,
        title: str,
        message: str,
        severity: AlertSeverity = AlertSeverity.INFO,
        source: str = "system",
        channels: Optional[List[AlertChannel]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        respect_cooldown: bool = True
    ) -> Dict[str, bool]:
        """
        Send alert to specified channels
        Returns dict of channel -> success status
        """
        alert = Alert(
            title=title,
            message=message,
            severity=severity,
            source=source,
            metadata=metadata or {}
        )
        
        # Check cooldown
        cooldown_key = f"{source}:{title}"
        if respect_cooldown and self._is_in_cooldown(cooldown_key):
            logger.debug(f"Alert skipped due to cooldown: {cooldown_key}")
            return {}
        
        # Use all configured channels if none specified
        if channels is None:
            channels = self.get_configured_channels()
        
        # Send to all channels concurrently
        results = {}
        tasks = []
        
        for channel in channels:
            provider = self.providers.get(channel)
            if provider and provider.is_configured():
                tasks.append(self._send_with_retry(channel, provider, alert))
        
        if tasks:
            channel_results = await asyncio.gather(*tasks, return_exceptions=True)
            for channel, result in zip(channels, channel_results):
                if isinstance(result, Exception):
                    results[channel.value] = False
                    logger.error(f"Alert failed for {channel}: {result}")
                else:
                    results[channel.value] = result
        
        # Record alert
        self._record_alert(alert)
        
        # Update cooldown
        if respect_cooldown:
            self._set_cooldown(cooldown_key)
        
        return results
    
    async def _send_with_retry(
        self,
        channel: AlertChannel,
        provider: AlertProvider,
        alert: Alert,
        max_retries: int = 3
    ) -> bool:
        """Send alert with retry logic"""
        for attempt in range(max_retries):
            try:
                result = await provider.send(alert)
                if result:
                    return True
            except Exception as e:
                logger.warning(f"Alert retry {attempt + 1} failed for {channel}: {e}")
            
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        return False
    
    def _is_in_cooldown(self, key: str) -> bool:
        """Check if alert is in cooldown period"""
        import time
        last_sent = self._cooldowns.get(key, 0)
        return (time.time() - last_sent) < self._cooldown_seconds
    
    def _set_cooldown(self, key: str):
        """Set cooldown for alert key"""
        import time
        self._cooldowns[key] = time.time()
    
    def _record_alert(self, alert: Alert):
        """Record alert in history"""
        self._alert_history.append(alert)
        if len(self._alert_history) > self._max_history:
            self._alert_history = self._alert_history[-self._max_history:]
    
    def get_alert_history(
        self,
        limit: int = 100,
        severity: Optional[AlertSeverity] = None,
        source: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get alert history with optional filters"""
        alerts = self._alert_history
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        if source:
            alerts = [a for a in alerts if a.source == source]
        
        return [a.to_dict() for a in alerts[-limit:]]
    
    # Convenience methods for common alert types
    async def info(self, title: str, message: str, source: str = "system", **kwargs):
        """Send info alert"""
        return await self.send_alert(title, message, AlertSeverity.INFO, source, **kwargs)
    
    async def warning(self, title: str, message: str, source: str = "system", **kwargs):
        """Send warning alert"""
        return await self.send_alert(title, message, AlertSeverity.WARNING, source, **kwargs)
    
    async def error(self, title: str, message: str, source: str = "system", **kwargs):
        """Send error alert"""
        return await self.send_alert(title, message, AlertSeverity.ERROR, source, **kwargs)
    
    async def critical(self, title: str, message: str, source: str = "system", **kwargs):
        """Send critical alert"""
        return await self.send_alert(title, message, AlertSeverity.CRITICAL, source, **kwargs)
    
    # Prediction-specific alerts
    async def tier_a_prediction(
        self,
        sport: str,
        game: str,
        prediction: str,
        probability: float,
        edge: float
    ):
        """Alert for Tier A prediction"""
        await self.send_alert(
            title=f"🎯 Tier A Prediction: {sport}",
            message=f"High-confidence prediction generated",
            severity=AlertSeverity.INFO,
            source="predictions",
            metadata={
                "sport": sport,
                "game": game,
                "prediction": prediction,
                "probability": f"{probability:.1%}",
                "edge": f"{edge:.2%}"
            }
        )
    
    async def model_performance_alert(
        self,
        sport: str,
        current_accuracy: float,
        threshold: float
    ):
        """Alert for model performance degradation"""
        await self.send_alert(
            title=f"⚠️ Model Performance Degraded: {sport}",
            message=f"Model accuracy has dropped below threshold",
            severity=AlertSeverity.WARNING,
            source="ml",
            metadata={
                "sport": sport,
                "current_accuracy": f"{current_accuracy:.1%}",
                "threshold": f"{threshold:.1%}",
                "difference": f"{(threshold - current_accuracy):.1%}"
            }
        )
    
    async def system_health_alert(
        self,
        component: str,
        status: str,
        details: Dict[str, Any]
    ):
        """Alert for system health issues"""
        severity = AlertSeverity.CRITICAL if status == "critical" else AlertSeverity.WARNING
        
        await self.send_alert(
            title=f"🔧 System Health: {component}",
            message=f"Component health status: {status}",
            severity=severity,
            source="health",
            metadata=details
        )


# Global alerting service instance
alerting_service = AlertingService()

def get_alerting_service() -> AlertingService:
    return alerting_service