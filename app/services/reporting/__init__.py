"""Reporting service module."""

from .report_generator import (
    ReportFormat,
    ReportType,
    ReportConfig,
    DailyReport,
    WeeklyReport,
    MonthlyReport,
    ReportGenerator,
)

# Try to create a getter function
_report_generator = None

def get_report_generator() -> ReportGenerator:
    """Get the global report generator instance."""
    global _report_generator
    if _report_generator is None:
        _report_generator = ReportGenerator()
    return _report_generator

__all__ = [
    "ReportFormat",
    "ReportType",
    "ReportConfig",
    "DailyReport",
    "WeeklyReport",
    "MonthlyReport",
    "ReportGenerator",
    "get_report_generator",
]
