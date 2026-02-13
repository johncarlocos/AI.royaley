// src/utils/formatters.ts — Global formatting utilities for timezone, time format, odds

/**
 * Format a date string or Date object to the user's timezone + time format.
 * Returns { date, time, dateTime } strings.
 */
export function formatDateTime(
  input: string | Date | null | undefined,
  timezone: string = 'America/New_York',
  timeFormat: '12h' | '24h' = '12h',
): { date: string; time: string; dateTime: string } {
  if (!input) return { date: '-', time: '-', dateTime: '-' };
  const d = typeof input === 'string' ? new Date(input) : input;
  if (isNaN(d.getTime())) return { date: '-', time: '-', dateTime: '-' };

  const dateStr = d.toLocaleDateString('en-US', {
    month: 'numeric',
    day: 'numeric',
    year: 'numeric',
    timeZone: timezone,
  });

  const timeStr = d.toLocaleTimeString('en-US', {
    hour: 'numeric',
    minute: '2-digit',
    hour12: timeFormat === '12h',
    timeZone: timezone,
  });

  return { date: dateStr, time: timeStr, dateTime: `${dateStr} ${timeStr}` };
}

/**
 * Format just the time portion (for "Updated: ..." displays).
 */
export function formatTime(
  input: string | Date | null | undefined,
  timezone: string = 'America/New_York',
  timeFormat: '12h' | '24h' = '12h',
): string {
  if (!input) return '-';
  const d = typeof input === 'string' ? new Date(input) : input;
  if (isNaN(d.getTime())) return '-';

  return d.toLocaleTimeString('en-US', {
    hour: 'numeric',
    minute: '2-digit',
    hour12: timeFormat === '12h',
    timeZone: timezone,
  });
}

/**
 * Format a short date for charts (e.g., "Jan 5").
 */
export function formatShortDate(
  input: string | Date | null | undefined,
  timezone: string = 'America/New_York',
): string {
  if (!input) return '?';
  const d = typeof input === 'string' ? new Date(input) : input;
  if (isNaN(d.getTime())) return '?';
  return d.toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    timeZone: timezone,
  });
}

/**
 * Convert American odds to the selected format.
 * Input: American odds number (e.g., -110, +150)
 * Output: formatted string in selected format
 */
export function formatOdds(
  americanOdds: number | string | null | undefined,
  oddsFormat: 'american' | 'decimal' | 'fractional' = 'american',
): string {
  if (americanOdds == null || americanOdds === '-' || americanOdds === '') return '-';

  const odds = typeof americanOdds === 'string' ? parseFloat(americanOdds) : americanOdds;
  if (isNaN(odds)) return String(americanOdds);

  switch (oddsFormat) {
    case 'american':
      return odds > 0 ? `+${odds}` : `${odds}`;

    case 'decimal': {
      let decimal: number;
      if (odds > 0) {
        decimal = (odds / 100) + 1;
      } else {
        decimal = (100 / Math.abs(odds)) + 1;
      }
      return decimal.toFixed(2);
    }

    case 'fractional': {
      let numerator: number;
      let denominator: number;
      if (odds > 0) {
        numerator = odds;
        denominator = 100;
      } else {
        numerator = 100;
        denominator = Math.abs(odds);
      }
      // Simplify the fraction
      const gcd = (a: number, b: number): number => b === 0 ? a : gcd(b, a % b);
      const divisor = gcd(Math.round(numerator), Math.round(denominator));
      return `${Math.round(numerator / divisor)}/${Math.round(denominator / divisor)}`;
    }

    default:
      return String(odds);
  }
}

/**
 * Get timezone abbreviation for display (e.g., "ET", "PT", "UTC").
 */
export function getTimezoneAbbr(timezone: string): string {
  const map: Record<string, string> = {
    'America/New_York': 'ET',
    'America/Chicago': 'CT',
    'America/Denver': 'MT',
    'America/Los_Angeles': 'PT',
    'UTC': 'UTC',
    'Europe/London': 'GMT',
    'Europe/Paris': 'CET',
    'Asia/Tokyo': 'JST',
    'Asia/Seoul': 'KST',
    'Australia/Sydney': 'AEST',
  };
  return map[timezone] || timezone.split('/').pop()?.replace('_', ' ') || timezone;
}