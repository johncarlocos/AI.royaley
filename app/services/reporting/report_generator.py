"""
ROYALEY - Report Generator
Comprehensive report generation for performance analysis
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum
import json

logger = logging.getLogger(__name__)


class ReportFormat(Enum):
    JSON = "json"
    HTML = "html"
    MARKDOWN = "markdown"
    CSV = "csv"


class ReportType(Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    CUSTOM = "custom"


@dataclass
class ReportConfig:
    """Report configuration"""
    report_type: ReportType = ReportType.DAILY
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    sports: Optional[List[str]] = None
    bet_types: Optional[List[str]] = None
    signal_tiers: Optional[List[str]] = None
    include_details: bool = True
    include_charts: bool = True
    format: ReportFormat = ReportFormat.JSON


@dataclass
class DailyReport:
    """Daily performance report"""
    date: datetime
    
    # Summary metrics
    total_predictions: int
    total_bets: int
    bets_won: int
    bets_lost: int
    bets_pushed: int
    bets_pending: int
    
    # Financial metrics
    total_wagered: float
    total_profit_loss: float
    roi_percent: float
    win_rate: float
    
    # Accuracy metrics
    tier_a_accuracy: float
    tier_b_accuracy: float
    overall_accuracy: float
    
    # CLV metrics
    avg_clv: float
    clv_positive_rate: float
    
    # By sport breakdown
    by_sport: Dict[str, Dict] = field(default_factory=dict)
    
    # By bet type breakdown
    by_bet_type: Dict[str, Dict] = field(default_factory=dict)
    
    # Top performers
    best_predictions: List[Dict] = field(default_factory=list)
    worst_predictions: List[Dict] = field(default_factory=list)


@dataclass
class WeeklyReport:
    """Weekly performance report"""
    week_start: datetime
    week_end: datetime
    
    # Daily summaries
    daily_reports: List[DailyReport] = field(default_factory=list)
    
    # Week totals
    total_predictions: int = 0
    total_bets: int = 0
    bets_won: int = 0
    bets_lost: int = 0
    total_wagered: float = 0.0
    total_profit_loss: float = 0.0
    
    # Performance metrics
    roi_percent: float = 0.0
    win_rate: float = 0.0
    avg_clv: float = 0.0
    
    # Trends
    daily_roi_trend: List[float] = field(default_factory=list)
    cumulative_profit: List[float] = field(default_factory=list)
    
    # Highlights
    best_day: Optional[datetime] = None
    worst_day: Optional[datetime] = None
    longest_win_streak: int = 0
    longest_loss_streak: int = 0


@dataclass 
class MonthlyReport:
    """Monthly performance report"""
    month: int
    year: int
    
    # Weekly summaries
    weekly_reports: List[WeeklyReport] = field(default_factory=list)
    
    # Month totals
    total_predictions: int = 0
    total_bets: int = 0
    bets_won: int = 0
    bets_lost: int = 0
    total_wagered: float = 0.0
    total_profit_loss: float = 0.0
    
    # Performance metrics
    roi_percent: float = 0.0
    win_rate: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    
    # CLV metrics
    avg_clv: float = 0.0
    clv_positive_rate: float = 0.0
    
    # By sport analysis
    sport_performance: Dict[str, Dict] = field(default_factory=dict)
    
    # By tier analysis
    tier_performance: Dict[str, Dict] = field(default_factory=dict)
    
    # Model performance
    model_accuracy_trend: List[float] = field(default_factory=list)


class ReportGenerator:
    """
    Generate comprehensive performance reports
    """
    
    def __init__(self, data_source: Any):
        """
        Initialize report generator
        
        Args:
            data_source: Database session or data access object
        """
        self.data_source = data_source
    
    def generate_daily_report(
        self,
        date: datetime,
        config: Optional[ReportConfig] = None
    ) -> DailyReport:
        """
        Generate daily performance report
        
        Args:
            date: Report date
            config: Optional configuration
            
        Returns:
            DailyReport with all metrics
        """
        config = config or ReportConfig()
        logger.info(f"Generating daily report for {date.date()}")
        
        # Get predictions for the day
        predictions = self._get_predictions_for_date(date, config)
        
        # Get bets for the day
        bets = self._get_bets_for_date(date, config)
        
        # Calculate metrics
        settled_bets = [b for b in bets if b.get('result') in ['win', 'loss', 'push']]
        pending_bets = [b for b in bets if b.get('result') == 'pending']
        
        bets_won = sum(1 for b in settled_bets if b.get('result') == 'win')
        bets_lost = sum(1 for b in settled_bets if b.get('result') == 'loss')
        bets_pushed = sum(1 for b in settled_bets if b.get('result') == 'push')
        
        total_wagered = sum(b.get('stake', 0) for b in settled_bets)
        total_profit_loss = sum(b.get('profit_loss', 0) for b in settled_bets)
        
        roi_percent = (total_profit_loss / total_wagered * 100) if total_wagered > 0 else 0
        win_rate = (bets_won / (bets_won + bets_lost) * 100) if (bets_won + bets_lost) > 0 else 0
        
        # Tier accuracy
        tier_a_preds = [p for p in predictions if p.get('signal_tier') == 'A' and p.get('graded')]
        tier_b_preds = [p for p in predictions if p.get('signal_tier') == 'B' and p.get('graded')]
        all_graded = [p for p in predictions if p.get('graded')]
        
        tier_a_accuracy = self._calculate_accuracy(tier_a_preds)
        tier_b_accuracy = self._calculate_accuracy(tier_b_preds)
        overall_accuracy = self._calculate_accuracy(all_graded)
        
        # CLV metrics
        clv_values = [b.get('clv', 0) for b in settled_bets if b.get('clv') is not None]
        avg_clv = sum(clv_values) / len(clv_values) if clv_values else 0
        clv_positive_rate = (sum(1 for c in clv_values if c > 0) / len(clv_values) * 100) if clv_values else 0
        
        # By sport breakdown
        by_sport = self._calculate_breakdown(settled_bets, 'sport')
        
        # By bet type breakdown
        by_bet_type = self._calculate_breakdown(settled_bets, 'bet_type')
        
        # Best/worst predictions
        sorted_by_profit = sorted(settled_bets, key=lambda x: x.get('profit_loss', 0), reverse=True)
        best_predictions = sorted_by_profit[:5] if config.include_details else []
        worst_predictions = sorted_by_profit[-5:][::-1] if config.include_details else []
        
        return DailyReport(
            date=date,
            total_predictions=len(predictions),
            total_bets=len(bets),
            bets_won=bets_won,
            bets_lost=bets_lost,
            bets_pushed=bets_pushed,
            bets_pending=len(pending_bets),
            total_wagered=total_wagered,
            total_profit_loss=total_profit_loss,
            roi_percent=roi_percent,
            win_rate=win_rate,
            tier_a_accuracy=tier_a_accuracy,
            tier_b_accuracy=tier_b_accuracy,
            overall_accuracy=overall_accuracy,
            avg_clv=avg_clv,
            clv_positive_rate=clv_positive_rate,
            by_sport=by_sport,
            by_bet_type=by_bet_type,
            best_predictions=best_predictions,
            worst_predictions=worst_predictions
        )
    
    def generate_weekly_report(
        self,
        week_start: datetime,
        config: Optional[ReportConfig] = None
    ) -> WeeklyReport:
        """
        Generate weekly performance report
        
        Args:
            week_start: Start of week (Monday)
            config: Optional configuration
            
        Returns:
            WeeklyReport with all metrics
        """
        config = config or ReportConfig()
        week_end = week_start + timedelta(days=6)
        logger.info(f"Generating weekly report for {week_start.date()} to {week_end.date()}")
        
        # Generate daily reports
        daily_reports = []
        current_date = week_start
        
        while current_date <= week_end:
            daily_report = self.generate_daily_report(current_date, config)
            daily_reports.append(daily_report)
            current_date += timedelta(days=1)
        
        # Aggregate totals
        total_predictions = sum(d.total_predictions for d in daily_reports)
        total_bets = sum(d.total_bets for d in daily_reports)
        bets_won = sum(d.bets_won for d in daily_reports)
        bets_lost = sum(d.bets_lost for d in daily_reports)
        total_wagered = sum(d.total_wagered for d in daily_reports)
        total_profit_loss = sum(d.total_profit_loss for d in daily_reports)
        
        roi_percent = (total_profit_loss / total_wagered * 100) if total_wagered > 0 else 0
        win_rate = (bets_won / (bets_won + bets_lost) * 100) if (bets_won + bets_lost) > 0 else 0
        
        # CLV
        clv_values = [d.avg_clv for d in daily_reports if d.avg_clv != 0]
        avg_clv = sum(clv_values) / len(clv_values) if clv_values else 0
        
        # Trends
        daily_roi_trend = [d.roi_percent for d in daily_reports]
        cumulative_profit = []
        running_total = 0
        for d in daily_reports:
            running_total += d.total_profit_loss
            cumulative_profit.append(running_total)
        
        # Best/worst days
        profitable_days = [(d.date, d.total_profit_loss) for d in daily_reports if d.total_profit_loss != 0]
        if profitable_days:
            best_day = max(profitable_days, key=lambda x: x[1])[0]
            worst_day = min(profitable_days, key=lambda x: x[1])[0]
        else:
            best_day = None
            worst_day = None
        
        # Streaks
        longest_win_streak, longest_loss_streak = self._calculate_streaks(daily_reports)
        
        return WeeklyReport(
            week_start=week_start,
            week_end=week_end,
            daily_reports=daily_reports,
            total_predictions=total_predictions,
            total_bets=total_bets,
            bets_won=bets_won,
            bets_lost=bets_lost,
            total_wagered=total_wagered,
            total_profit_loss=total_profit_loss,
            roi_percent=roi_percent,
            win_rate=win_rate,
            avg_clv=avg_clv,
            daily_roi_trend=daily_roi_trend,
            cumulative_profit=cumulative_profit,
            best_day=best_day,
            worst_day=worst_day,
            longest_win_streak=longest_win_streak,
            longest_loss_streak=longest_loss_streak
        )
    
    def generate_monthly_report(
        self,
        year: int,
        month: int,
        config: Optional[ReportConfig] = None
    ) -> MonthlyReport:
        """
        Generate monthly performance report
        
        Args:
            year: Report year
            month: Report month
            config: Optional configuration
            
        Returns:
            MonthlyReport with all metrics
        """
        config = config or ReportConfig()
        logger.info(f"Generating monthly report for {year}-{month:02d}")
        
        # Calculate month boundaries
        month_start = datetime(year, month, 1)
        if month == 12:
            month_end = datetime(year + 1, 1, 1) - timedelta(days=1)
        else:
            month_end = datetime(year, month + 1, 1) - timedelta(days=1)
        
        # Generate weekly reports
        weekly_reports = []
        current_week_start = month_start
        
        # Align to Monday
        while current_week_start.weekday() != 0:
            current_week_start -= timedelta(days=1)
        
        while current_week_start <= month_end:
            weekly_report = self.generate_weekly_report(current_week_start, config)
            weekly_reports.append(weekly_report)
            current_week_start += timedelta(days=7)
        
        # Get all data for the month
        all_predictions = self._get_predictions_for_range(month_start, month_end, config)
        all_bets = self._get_bets_for_range(month_start, month_end, config)
        
        settled_bets = [b for b in all_bets if b.get('result') in ['win', 'loss', 'push']]
        
        # Calculate metrics
        total_predictions = len(all_predictions)
        total_bets = len(all_bets)
        bets_won = sum(1 for b in settled_bets if b.get('result') == 'win')
        bets_lost = sum(1 for b in settled_bets if b.get('result') == 'loss')
        total_wagered = sum(b.get('stake', 0) for b in settled_bets)
        total_profit_loss = sum(b.get('profit_loss', 0) for b in settled_bets)
        
        roi_percent = (total_profit_loss / total_wagered * 100) if total_wagered > 0 else 0
        win_rate = (bets_won / (bets_won + bets_lost) * 100) if (bets_won + bets_lost) > 0 else 0
        
        # Risk metrics
        daily_returns = [w.roi_percent for w in weekly_reports for d in [w] if d.total_wagered > 0]
        if len(daily_returns) > 1:
            import statistics
            mean_return = statistics.mean(daily_returns)
            std_return = statistics.stdev(daily_returns)
            sharpe_ratio = mean_return / std_return if std_return > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Max drawdown
        max_drawdown = self._calculate_max_drawdown(weekly_reports)
        
        # CLV
        clv_values = [b.get('clv', 0) for b in settled_bets if b.get('clv') is not None]
        avg_clv = sum(clv_values) / len(clv_values) if clv_values else 0
        clv_positive_rate = (sum(1 for c in clv_values if c > 0) / len(clv_values) * 100) if clv_values else 0
        
        # By sport analysis
        sport_performance = self._calculate_breakdown(settled_bets, 'sport')
        
        # By tier analysis
        tier_performance = self._calculate_tier_breakdown(all_predictions)
        
        # Model accuracy trend
        model_accuracy_trend = [w.daily_reports[0].overall_accuracy for w in weekly_reports if w.daily_reports]
        
        return MonthlyReport(
            month=month,
            year=year,
            weekly_reports=weekly_reports,
            total_predictions=total_predictions,
            total_bets=total_bets,
            bets_won=bets_won,
            bets_lost=bets_lost,
            total_wagered=total_wagered,
            total_profit_loss=total_profit_loss,
            roi_percent=roi_percent,
            win_rate=win_rate,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            avg_clv=avg_clv,
            clv_positive_rate=clv_positive_rate,
            sport_performance=sport_performance,
            tier_performance=tier_performance,
            model_accuracy_trend=model_accuracy_trend
        )
    
    def export_report(
        self,
        report: Any,
        format: ReportFormat,
        output_path: Optional[str] = None
    ) -> str:
        """
        Export report to specified format
        
        Args:
            report: Report object (Daily/Weekly/Monthly)
            format: Output format
            output_path: Optional file path
            
        Returns:
            Formatted report string or file path
        """
        if format == ReportFormat.JSON:
            return self._export_json(report, output_path)
        elif format == ReportFormat.HTML:
            return self._export_html(report, output_path)
        elif format == ReportFormat.MARKDOWN:
            return self._export_markdown(report, output_path)
        elif format == ReportFormat.CSV:
            return self._export_csv(report, output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _get_predictions_for_date(self, date: datetime, config: ReportConfig) -> List[Dict]:
        """Get predictions for a specific date"""
        # This would query the database
        # Placeholder implementation
        return []
    
    def _get_bets_for_date(self, date: datetime, config: ReportConfig) -> List[Dict]:
        """Get bets for a specific date"""
        # This would query the database
        return []
    
    def _get_predictions_for_range(self, start: datetime, end: datetime, config: ReportConfig) -> List[Dict]:
        """Get predictions for date range"""
        return []
    
    def _get_bets_for_range(self, start: datetime, end: datetime, config: ReportConfig) -> List[Dict]:
        """Get bets for date range"""
        return []
    
    def _calculate_accuracy(self, predictions: List[Dict]) -> float:
        """Calculate prediction accuracy"""
        if not predictions:
            return 0.0
        correct = sum(1 for p in predictions if p.get('correct'))
        return (correct / len(predictions)) * 100
    
    def _calculate_breakdown(self, bets: List[Dict], field: str) -> Dict[str, Dict]:
        """Calculate breakdown by field"""
        breakdown = {}
        
        for bet in bets:
            key = bet.get(field, 'unknown')
            if key not in breakdown:
                breakdown[key] = {
                    'count': 0, 'won': 0, 'lost': 0,
                    'wagered': 0, 'profit_loss': 0
                }
            
            breakdown[key]['count'] += 1
            breakdown[key]['wagered'] += bet.get('stake', 0)
            breakdown[key]['profit_loss'] += bet.get('profit_loss', 0)
            
            if bet.get('result') == 'win':
                breakdown[key]['won'] += 1
            elif bet.get('result') == 'loss':
                breakdown[key]['lost'] += 1
        
        # Calculate rates
        for key in breakdown:
            data = breakdown[key]
            decided = data['won'] + data['lost']
            data['win_rate'] = (data['won'] / decided * 100) if decided > 0 else 0
            data['roi'] = (data['profit_loss'] / data['wagered'] * 100) if data['wagered'] > 0 else 0
        
        return breakdown
    
    def _calculate_tier_breakdown(self, predictions: List[Dict]) -> Dict[str, Dict]:
        """Calculate breakdown by signal tier"""
        breakdown = {}
        
        for pred in predictions:
            tier = pred.get('signal_tier', 'unknown')
            if tier not in breakdown:
                breakdown[tier] = {
                    'count': 0, 'correct': 0, 'graded': 0
                }
            
            breakdown[tier]['count'] += 1
            if pred.get('graded'):
                breakdown[tier]['graded'] += 1
                if pred.get('correct'):
                    breakdown[tier]['correct'] += 1
        
        # Calculate accuracy
        for tier in breakdown:
            data = breakdown[tier]
            data['accuracy'] = (data['correct'] / data['graded'] * 100) if data['graded'] > 0 else 0
        
        return breakdown
    
    def _calculate_streaks(self, daily_reports: List[DailyReport]) -> tuple:
        """Calculate win/loss streaks"""
        current_win_streak = 0
        current_loss_streak = 0
        longest_win = 0
        longest_loss = 0
        
        for report in daily_reports:
            if report.total_profit_loss > 0:
                current_win_streak += 1
                current_loss_streak = 0
                longest_win = max(longest_win, current_win_streak)
            elif report.total_profit_loss < 0:
                current_loss_streak += 1
                current_win_streak = 0
                longest_loss = max(longest_loss, current_loss_streak)
            else:
                current_win_streak = 0
                current_loss_streak = 0
        
        return longest_win, longest_loss
    
    def _calculate_max_drawdown(self, weekly_reports: List[WeeklyReport]) -> float:
        """Calculate maximum drawdown"""
        if not weekly_reports:
            return 0.0
        
        peak = 0
        max_dd = 0
        cumulative = 0
        
        for week in weekly_reports:
            cumulative += week.total_profit_loss
            if cumulative > peak:
                peak = cumulative
            dd = (peak - cumulative) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)
        
        return max_dd * 100
    
    def _export_json(self, report: Any, output_path: Optional[str]) -> str:
        """Export report to JSON"""
        from dataclasses import asdict
        
        def convert_datetime(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            return obj
        
        data = asdict(report)
        
        # Convert datetimes
        def recursive_convert(d):
            if isinstance(d, dict):
                return {k: recursive_convert(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [recursive_convert(i) for i in d]
            elif isinstance(d, datetime):
                return d.isoformat()
            return d
        
        data = recursive_convert(data)
        json_str = json.dumps(data, indent=2, default=str)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(json_str)
            return output_path
        
        return json_str
    
    def _export_html(self, report: Any, output_path: Optional[str]) -> str:
        """Export report to HTML"""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>ROYALEY Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background: #1e3a5f; color: white; padding: 20px; }}
        .metric {{ display: inline-block; margin: 10px; padding: 15px; background: #f0f0f0; border-radius: 5px; }}
        .metric .value {{ font-size: 24px; font-weight: bold; }}
        .metric .label {{ font-size: 12px; color: #666; }}
        .positive {{ color: #28a745; }}
        .negative {{ color: #dc3545; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 10px; border: 1px solid #ddd; text-align: left; }}
        th {{ background: #f5f5f5; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>ROYALEY Performance Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
"""
        
        if hasattr(report, 'total_bets'):
            html += f"""
    <div class="metrics">
        <div class="metric">
            <div class="value">{report.total_bets}</div>
            <div class="label">Total Bets</div>
        </div>
        <div class="metric">
            <div class="value">{report.bets_won}</div>
            <div class="label">Wins</div>
        </div>
        <div class="metric">
            <div class="value">{report.bets_lost}</div>
            <div class="label">Losses</div>
        </div>
        <div class="metric">
            <div class="value {'positive' if report.win_rate >= 50 else 'negative'}">{report.win_rate:.1f}%</div>
            <div class="label">Win Rate</div>
        </div>
        <div class="metric">
            <div class="value {'positive' if report.roi_percent >= 0 else 'negative'}">{report.roi_percent:.1f}%</div>
            <div class="label">ROI</div>
        </div>
        <div class="metric">
            <div class="value">${report.total_wagered:,.2f}</div>
            <div class="label">Total Wagered</div>
        </div>
        <div class="metric">
            <div class="value {'positive' if report.total_profit_loss >= 0 else 'negative'}">${report.total_profit_loss:,.2f}</div>
            <div class="label">Profit/Loss</div>
        </div>
    </div>
"""
        
        html += """
</body>
</html>"""
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(html)
            return output_path
        
        return html
    
    def _export_markdown(self, report: Any, output_path: Optional[str]) -> str:
        """Export report to Markdown"""
        md = f"""# ROYALEY Performance Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

"""
        
        if hasattr(report, 'total_bets'):
            md += f"""| Metric | Value |
|--------|-------|
| Total Bets | {report.total_bets} |
| Wins | {report.bets_won} |
| Losses | {report.bets_lost} |
| Win Rate | {report.win_rate:.1f}% |
| ROI | {report.roi_percent:.1f}% |
| Total Wagered | ${report.total_wagered:,.2f} |
| Profit/Loss | ${report.total_profit_loss:,.2f} |

"""
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(md)
            return output_path
        
        return md
    
    def _export_csv(self, report: Any, output_path: Optional[str]) -> str:
        """Export report to CSV"""
        import csv
        import io
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Headers
        writer.writerow(['Metric', 'Value'])
        
        if hasattr(report, 'total_bets'):
            writer.writerow(['Total Bets', report.total_bets])
            writer.writerow(['Wins', report.bets_won])
            writer.writerow(['Losses', report.bets_lost])
            writer.writerow(['Win Rate', f'{report.win_rate:.1f}%'])
            writer.writerow(['ROI', f'{report.roi_percent:.1f}%'])
            writer.writerow(['Total Wagered', f'${report.total_wagered:.2f}'])
            writer.writerow(['Profit/Loss', f'${report.total_profit_loss:.2f}'])
        
        csv_str = output.getvalue()
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(csv_str)
            return output_path
        
        return csv_str
