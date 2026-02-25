// src/pages/Live/Live.tsx
import React, { useState, useEffect, useCallback } from 'react';
import {
  Box, Typography, Select, MenuItem, FormControl, InputLabel, Button,
  LinearProgress, Table, TableBody, TableCell, TableContainer, TableHead,
  TableRow, Paper, Chip, useTheme, alpha, Tooltip
} from '@mui/material';
import { PlayCircle, Refresh, FiberManualRecord, SportsScore } from '@mui/icons-material';
import { api } from '../../api/client';
import { useSettingsStore } from '../../store';
import { SPORTS } from '../../types';
import { formatTime, getTimezoneAbbr } from '../../utils/formatters';

interface LiveGame {
  id: string;
  sport: string;
  date: string;
  time: string;
  gameNumber: number;
  homeTeam: string;
  awayTeam: string;
  homeRecord: string;
  awayRecord: string;
  homeScore: number | null;
  awayScore: number | null;
  period: string;
  status: 'live' | 'upcoming' | 'halftime' | 'final' | 'scheduled';
  spread: { away: string; home: string };
  total: string;
  prediction: {
    pick: string;
    type: string;
    probability: number;
    edge: number;
    tier: string;
  } | null;
}

interface LiveResponse {
  games: LiveGame[];
  counts: {
    live: number;
    halftime: number;
    upcoming: number;
    final: number;
    with_predictions: number;
  };
  updated_at: string;
}

// Tier badge component with distinct colors
const TierBadge: React.FC<{ tier: string }> = ({ tier }) => {
  const colors: Record<string, string> = {
    A: '#16a34a',  // Green
    B: '#2563eb',  // Blue
    C: '#d97706',  // Amber
    D: '#6b7280',  // Gray
  };
  const labels: Record<string, string> = {
    A: '\u{1F525} Tier A',
    B: '\u2705 Tier B',
    C: 'Tier C',
    D: 'Tier D',
  };
  return (
    <Chip
      label={labels[tier] || `Tier ${tier}`}
      size="small"
      sx={{
        bgcolor: colors[tier] || colors.D,
        color: 'white',
        fontWeight: 600,
        fontSize: '0.7rem',
        height: 22,
      }}
    />
  );
};

// Status indicator with better visuals
const StatusBadge: React.FC<{ status: string; period?: string }> = ({ status, period }) => {
  if (status === 'live') {
    return (
      <Box display="flex" alignItems="center" gap={0.5}>
        <FiberManualRecord sx={{ fontSize: 10, color: 'error.main', animation: 'pulse 1.5s infinite' }} />
        <Typography variant="caption" color="error.main" fontWeight={600}>
          {period || 'LIVE'}
        </Typography>
      </Box>
    );
  }
  if (status === 'halftime') {
    return <Chip label="HALF" size="small" sx={{ bgcolor: 'warning.main', color: 'white', height: 20, fontSize: '0.65rem' }} />;
  }
  if (status === 'final') {
    return <Chip label="FINAL" size="small" sx={{ bgcolor: 'grey.600', color: 'white', height: 20, fontSize: '0.65rem' }} />;
  }
  // upcoming / scheduled — show time
  return <Typography variant="caption" color="text.secondary">{period}</Typography>;
};

// Edge display with color coding — edge is now in percentage (e.g. 6.0 = 6%)
const EdgeDisplay: React.FC<{ edge: number }> = ({ edge }) => {
  if (edge < 1.0) return <Typography sx={{ fontSize: '0.8rem', color: 'text.disabled' }}>-</Typography>;

  let color = 'text.secondary';
  if (edge >= 8) color = '#16a34a';       // Strong green
  else if (edge >= 5) color = '#22c55e';  // Green
  else if (edge >= 3) color = '#84cc16';  // Light green
  else if (edge >= 1) color = '#eab308';  // Yellow

  return (
    <Typography sx={{ fontSize: '0.8rem', fontWeight: 600, color }}>
      +{edge.toFixed(1)}%
    </Typography>
  );
};

// Empty state component
const EmptyState: React.FC = () => {
  const theme = useTheme();
  const isDark = theme.palette.mode === 'dark';
  return (
    <Box sx={{ textAlign: 'center', py: 8, px: 4 }}>
      <SportsScore sx={{ fontSize: 56, color: isDark ? '#475569' : '#94a3b8', mb: 2 }} />
      <Typography variant="h6" color="text.secondary" sx={{ mb: 1 }}>
        No Games Available
      </Typography>
      <Typography variant="body2" color="text.disabled" sx={{ maxWidth: 500, mx: 'auto', lineHeight: 1.6 }}>
        There are no games scheduled in the current window.
        Games appear here within 24 hours of their start time, along with live scores and predictions.
        Check back when games are approaching — NBA, NHL, NCAAB, NFL, and MLB are all supported.
      </Typography>
    </Box>
  );
};

const Live: React.FC = () => {
  const theme = useTheme();
  const [games, setGames] = useState<LiveGame[]>([]);
  const [counts, setCounts] = useState({ live: 0, halftime: 0, upcoming: 0, final: 0, with_predictions: 0 });
  const [loading, setLoading] = useState(true);
  const [sportFilter, setSportFilter] = useState('all');
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());
  const { timezone, timeFormat } = useSettingsStore();

  const isDark = theme.palette.mode === 'dark';
  const headerBg = isDark ? '#1a2332' : '#f5f5f5';
  const grayColBg = isDark ? alpha('#64748b', 0.15) : alpha('#94a3b8', 0.15);
  const rowHoverBg = isDark ? alpha('#3b82f6', 0.08) : alpha('#3b82f6', 0.04);

  const loadGames = useCallback(async () => {
    setLoading(true);
    try {
      const data: LiveResponse = await api.getLiveGames(
        sportFilter !== 'all' ? sportFilter : undefined
      );
      setGames(Array.isArray(data.games) ? data.games : []);
      if (data.counts) setCounts(data.counts);
      if (data.updated_at) setLastUpdate(new Date(data.updated_at));
    } catch {
      setGames([]);
    }
    setLoading(false);
  }, [sportFilter]);

  useEffect(() => {
    loadGames();
    // Refresh every 30 seconds when there are live games, otherwise 90 seconds
    const hasLive = counts.live > 0 || counts.halftime > 0;
    const interval = setInterval(loadGames, hasLive ? 30000 : 90000);
    return () => clearInterval(interval);
  }, [loadGames, counts.live, counts.halftime]);

  const filtered = games;
  const liveCount = counts.live + counts.halftime;

  // Common cell style
  const cellSx = { py: 1, px: 1.5, fontSize: '0.8rem', borderRight: `1px solid ${isDark ? '#2d3748' : '#e2e8f0'}` };
  const grayCellSx = { ...cellSx, bgcolor: grayColBg };
  const headerCellSx = { py: 1, px: 1.5, fontWeight: 600, fontSize: '0.75rem', borderRight: `1px solid ${isDark ? '#2d3748' : '#e2e8f0'}`, bgcolor: headerBg };

  return (
    <Box>
      {/* Header */}
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
        <Box display="flex" alignItems="center" gap={2}>
          <Typography variant="h5" fontWeight={700} sx={{ fontSize: 20 }}>
            <PlayCircle sx={{ verticalAlign: 'middle', mr: 1, color: 'error.main' }} />
            LIVE
          </Typography>
          {liveCount > 0 && (
            <Chip
              icon={<FiberManualRecord sx={{ fontSize: '10px !important' }} />}
              label={`${liveCount} Live`}
              color="error"
              size="small"
              sx={{ animation: 'pulse 2s infinite' }}
            />
          )}
          <Typography variant="caption" color="text.secondary">
            Updated: {formatTime(lastUpdate, timezone, timeFormat)} {getTimezoneAbbr(timezone)}
          </Typography>
        </Box>
        <Box display="flex" gap={2}>
          <FormControl size="small" sx={{ minWidth: 120 }}>
            <InputLabel>Sport</InputLabel>
            <Select value={sportFilter} label="Sport" onChange={(e) => setSportFilter(e.target.value)}>
              <MenuItem value="all">All Sports</MenuItem>
              {SPORTS.map(s => <MenuItem key={s.code} value={s.code}>{s.code}</MenuItem>)}
            </Select>
          </FormControl>
          <Button variant="outlined" startIcon={<Refresh />} onClick={loadGames} size="small">
            Refresh
          </Button>
        </Box>
      </Box>

      {loading && <LinearProgress sx={{ mb: 2 }} />}

      {/* Live Games Table */}
      <Paper variant="outlined" sx={{ overflow: 'hidden' }}>
        {filtered.length > 0 ? (
          <TableContainer sx={{ maxHeight: 'calc(100vh - 200px)' }}>
            <Table size="small" stickyHeader>
              <TableHead>
                <TableRow>
                  <TableCell sx={{ ...headerCellSx, minWidth: 60 }}>Sport</TableCell>
                  <TableCell sx={{ ...headerCellSx, minWidth: 70 }}>Time</TableCell>
                  <TableCell sx={{ ...headerCellSx, minWidth: 50, textAlign: 'center' }}>Game #</TableCell>
                  <TableCell sx={{ ...headerCellSx, minWidth: 140 }}>Team</TableCell>
                  <TableCell sx={{ ...headerCellSx, minWidth: 50, textAlign: 'center' }}>Record</TableCell>
                  <TableCell sx={{ ...headerCellSx, minWidth: 50, textAlign: 'center' }}>Score</TableCell>
                  <TableCell sx={{ ...headerCellSx, minWidth: 70, textAlign: 'center' }}>Status</TableCell>
                  <TableCell sx={{ ...headerCellSx, minWidth: 60, textAlign: 'center' }}>Spread</TableCell>
                  <TableCell sx={{ ...headerCellSx, minWidth: 60, textAlign: 'center' }}>Total</TableCell>
                  <TableCell sx={{ ...headerCellSx, minWidth: 100 }}>Pick</TableCell>
                  <TableCell sx={{ ...headerCellSx, minWidth: 50, textAlign: 'center' }}>%</TableCell>
                  <TableCell sx={{ ...headerCellSx, minWidth: 50, textAlign: 'center' }}>Edge</TableCell>
                  <TableCell sx={{ ...headerCellSx, minWidth: 70, textAlign: 'center', borderRight: 'none' }}>Tier</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {filtered.map((game, gameIdx) => {
                  const isLive = game.status === 'live';
                  const isFinal = game.status === 'final';
                  const isTennis = game.sport === 'ATP' || game.sport === 'WTA';

                  // Row background based on status
                  let rowBg = 'transparent';
                  if (isLive) rowBg = alpha('#ef4444', 0.08);
                  else if (isFinal) rowBg = isDark ? alpha('#64748b', 0.06) : alpha('#94a3b8', 0.04);

                  // Determine if team name should be bold (picked team)
                  const pickText = game.prediction?.pick || '';
                  const awayLastWord = game.awayTeam.split(' ').pop() || '';
                  const homeLastWord = game.homeTeam.split(' ').pop() || '';
                  const isAwayPicked = pickText.includes(awayLastWord) || pickText.includes('Away');
                  const isHomePicked = pickText.includes(homeLastWord) || pickText.includes('Home');

                  return (
                    <React.Fragment key={game.id}>
                      {/* Away Team Row */}
                      <TableRow
                        sx={{
                          bgcolor: rowBg,
                          '&:hover': { bgcolor: rowHoverBg },
                          borderTop: gameIdx > 0 ? `2px solid ${isDark ? '#4a5568' : '#cbd5e1'}` : undefined,
                          opacity: isFinal ? 0.7 : 1,
                        }}
                      >
                        <TableCell rowSpan={2} sx={{ ...grayCellSx, verticalAlign: 'middle', fontWeight: 600 }}>
                          {game.sport}
                        </TableCell>
                        <TableCell rowSpan={2} sx={{ ...grayCellSx, verticalAlign: 'middle' }}>
                          {game.time}
                        </TableCell>
                        <TableCell sx={{ ...cellSx, textAlign: 'center', bgcolor: isDark ? '#2a3441' : '#f0f4f8' }}>
                          {game.gameNumber}
                        </TableCell>
                        <TableCell sx={{ ...cellSx, fontWeight: isAwayPicked ? 700 : 400 }}>
                          {game.awayTeam}
                        </TableCell>
                        <TableCell sx={{ ...cellSx, textAlign: 'center', color: 'text.secondary' }}>
                          {game.awayRecord || (isTennis ? '' : '-')}
                        </TableCell>
                        <TableCell sx={{ ...cellSx, textAlign: 'center', fontWeight: 700, fontSize: '0.9rem' }}>
                          {game.awayScore !== null ? game.awayScore : '-'}
                        </TableCell>
                        <TableCell rowSpan={2} sx={{ ...cellSx, verticalAlign: 'middle', textAlign: 'center' }}>
                          <StatusBadge status={game.status} period={game.period} />
                        </TableCell>
                        <TableCell sx={{ ...cellSx, textAlign: 'center' }}>
                          {game.spread.away || (isTennis ? '' : '-')}
                        </TableCell>
                        <TableCell rowSpan={2} sx={{ ...cellSx, textAlign: 'center', verticalAlign: 'middle' }}>
                          {game.total || (isTennis ? '' : '-')}
                        </TableCell>
                        <TableCell rowSpan={2} sx={{ ...cellSx, verticalAlign: 'middle' }}>
                          {game.prediction ? (
                            <Box>
                              <Typography sx={{ fontSize: 11, fontWeight: 600, color: 'success.main', lineHeight: 1.3 }}>
                                {game.prediction.pick}
                              </Typography>
                              <Typography sx={{ fontSize: 10, color: 'text.secondary' }}>
                                {game.prediction.type}
                              </Typography>
                            </Box>
                          ) : (
                            <Tooltip title={isTennis ? 'Tennis predictions coming soon' : 'No qualifying prediction'}>
                              <Typography sx={{ fontSize: 10, color: 'text.disabled' }}>
                                {isTennis ? 'N/A' : '-'}
                              </Typography>
                            </Tooltip>
                          )}
                        </TableCell>
                        <TableCell rowSpan={2} sx={{ ...cellSx, textAlign: 'center', verticalAlign: 'middle', fontWeight: 600 }}>
                          {game.prediction ? `${(game.prediction.probability * 100).toFixed(0)}%` : '-'}
                        </TableCell>
                        <TableCell rowSpan={2} sx={{ ...cellSx, textAlign: 'center', verticalAlign: 'middle' }}>
                          {game.prediction ? <EdgeDisplay edge={game.prediction.edge} /> : (
                            <Typography sx={{ fontSize: '0.8rem', color: 'text.disabled' }}>-</Typography>
                          )}
                        </TableCell>
                        <TableCell rowSpan={2} sx={{ ...cellSx, textAlign: 'center', verticalAlign: 'middle', borderRight: 'none' }}>
                          {game.prediction ? <TierBadge tier={game.prediction.tier} /> : '-'}
                        </TableCell>
                      </TableRow>

                      {/* Home Team Row */}
                      <TableRow
                        sx={{
                          bgcolor: rowBg,
                          '&:hover': { bgcolor: rowHoverBg },
                          opacity: isFinal ? 0.7 : 1,
                        }}
                      >
                        <TableCell sx={{ ...cellSx, textAlign: 'center', bgcolor: isDark ? '#2a3441' : '#f0f4f8' }}>
                          {game.gameNumber + 1}
                        </TableCell>
                        <TableCell sx={{ ...cellSx, fontWeight: isHomePicked ? 700 : 400 }}>
                          {game.homeTeam}
                        </TableCell>
                        <TableCell sx={{ ...cellSx, textAlign: 'center', color: 'text.secondary' }}>
                          {game.homeRecord || (isTennis ? '' : '-')}
                        </TableCell>
                        <TableCell sx={{ ...cellSx, textAlign: 'center', fontWeight: 700, fontSize: '0.9rem' }}>
                          {game.homeScore !== null ? game.homeScore : '-'}
                        </TableCell>
                        <TableCell sx={{ ...cellSx, textAlign: 'center' }}>
                          {game.spread.home || (isTennis ? '' : '-')}
                        </TableCell>
                      </TableRow>
                    </React.Fragment>
                  );
                })}
              </TableBody>
            </Table>
          </TableContainer>
        ) : (
          !loading && <EmptyState />
        )}
      </Paper>

      {/* Summary Stats */}
      {filtered.length > 0 && (
        <Box display="flex" gap={3} mt={2} alignItems="center">
          <Typography variant="body2" color="text.secondary">
            <strong>{counts.live}</strong> Live
          </Typography>
          <Typography variant="body2" color="text.secondary">
            <strong>{counts.halftime}</strong> Halftime
          </Typography>
          <Typography variant="body2" color="text.secondary">
            <strong>{counts.upcoming}</strong> Upcoming
          </Typography>
          <Typography variant="body2" color="text.secondary">
            <strong>{counts.final}</strong> Final
          </Typography>
          <Typography variant="body2" color="text.secondary">
            <strong>{counts.with_predictions}</strong> With Predictions
          </Typography>
        </Box>
      )}

      {/* CSS for pulse animation */}
      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }
      `}</style>
    </Box>
  );
};

export default Live;
