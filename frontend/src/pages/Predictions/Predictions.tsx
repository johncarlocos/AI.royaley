// src/pages/Predictions/Predictions.tsx - All stats computed from real data
import React, { useState, useEffect, useMemo } from 'react';
import {
  Box, Card, CardContent, Typography, Table, TableBody, TableCell, TableContainer,
  TableHead, TableRow, Chip, IconButton, Button, Select, MenuItem,
  FormControl, InputLabel, Grid, LinearProgress, Dialog,
  DialogTitle, DialogContent, DialogActions, Tabs, Tab, TablePagination,
  TableSortLabel, useTheme, Tooltip
} from '@mui/material';
import { Refresh, ExpandMore, CheckCircle, Cancel, Schedule, Remove, TrendingUp, TrendingDown, Assessment, EmojiEvents, Casino, SportsSoccer, Timer, CalendarToday, RestartAlt } from '@mui/icons-material';
import { api } from '../../api/client';
import { useFilterStore } from '../../store';
import { SPORTS } from '../../types';
import { formatPercent, formatOdds } from '../../utils';

// Types
interface FlatRow {
  id: string;
  game_id: string;
  sport: string;
  date: string;
  time: string;
  datetime: Date;
  away_team: string;
  home_team: string;
  bet_type: string;
  bet_type_label: string;
  line: number | null;
  odds: number | null;
  system_pick: string;
  pick_team: 'away' | 'home' | null;
  probability: number;
  edge: number;
  clv: number | null;
  signal_tier: 'A' | 'B' | 'C' | 'D';
  result: 'pending' | 'won' | 'lost' | 'push';
  reason: string;
}

interface PerformanceRow {
  label: string;
  total: number;
  wins: number;
  losses: number;
  pushes: number;
  pending: number;
  winPct: number;
  avgProb: number;
  avgEdge: number;
}

type SortField = 'sport' | 'datetime' | 'probability' | 'edge' | 'clv' | 'tier' | null;
type SortOrder = 'asc' | 'desc';

// Tier Badge
const TierBadge: React.FC<{ tier: string }> = ({ tier }) => {
  const colors: Record<string, { bg: string; color: string }> = {
    'A': { bg: '#4caf50', color: '#fff' },
    'B': { bg: '#2196f3', color: '#fff' },
    'C': { bg: '#ff9800', color: '#fff' },
    'D': { bg: '#9e9e9e', color: '#fff' },
  };
  const style = colors[tier] || colors['D'];
  return <Chip label={tier} size="small" sx={{ bgcolor: style.bg, color: style.color, fontWeight: 700, minWidth: 28, height: 22, fontSize: 11 }} />;
};

// Stat Card Component (no fake trends)
const StatCard: React.FC<{ title: string; value: string | number; subtitle?: string; icon: React.ReactNode; color?: string }> =
  ({ title, value, subtitle, icon, color = 'primary' }) => {
  const theme = useTheme();
  return (
    <Card sx={{ height: '100%' }}>
      <CardContent sx={{ py: 1.5, px: 2, '&:last-child': { pb: 1.5 } }}>
        <Box display="flex" justifyContent="space-between" alignItems="flex-start">
          <Box>
            <Typography variant="caption" color="text.secondary" sx={{ fontSize: 11 }}>{title}</Typography>
            <Typography variant="h5" fontWeight={700} color={`${color}.main`} sx={{ fontSize: 20, lineHeight: 1.2 }}>{value}</Typography>
            {subtitle && <Typography variant="caption" color="text.secondary" sx={{ fontSize: 10 }}>{subtitle}</Typography>}
          </Box>
          <Box sx={{ p: 0.75, borderRadius: 1.5, bgcolor: theme.palette.mode === 'dark' ? `${color}.900` : `${color}.50`, color: `${color}.main` }}>
            {icon}
          </Box>
        </Box>
      </CardContent>
    </Card>
  );
};

// Performance Table Component (computed from real data)
const PerformanceTable: React.FC<{ data: PerformanceRow[]; isDark: boolean }> = ({ data, isDark }) => {
  const perfHdr = { fontWeight: 600, fontSize: 11, py: 0.75, bgcolor: isDark ? 'grey.900' : 'grey.100', color: isDark ? 'grey.100' : 'grey.800', borderBottom: 2, borderColor: isDark ? 'grey.700' : 'grey.300' };
  const hasGraded = data.some(r => r.wins > 0 || r.losses > 0 || r.pushes > 0);

  return (
    <TableContainer>
      <Table size="small">
        <TableHead>
          <TableRow>
            <TableCell sx={perfHdr}>Category</TableCell>
            <TableCell sx={perfHdr} align="center">Total</TableCell>
            <TableCell sx={perfHdr} align="center">Pending</TableCell>
            {hasGraded && <>
              <TableCell sx={perfHdr} align="center">W</TableCell>
              <TableCell sx={perfHdr} align="center">L</TableCell>
              <TableCell sx={perfHdr} align="center">P</TableCell>
              <TableCell sx={perfHdr} align="center">Win %</TableCell>
            </>}
            <TableCell sx={perfHdr} align="center">Avg Prob</TableCell>
            <TableCell sx={perfHdr} align="center">Avg Edge</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {data.map((row, idx) => (
            <TableRow key={idx} sx={{ '& td': { py: 0.75, fontSize: 12 } }}>
              <TableCell sx={{ fontWeight: 500 }}>{row.label}</TableCell>
              <TableCell align="center"><Chip label={row.total} size="small" color="primary" variant="outlined" sx={{ fontSize: 11, height: 22, minWidth: 40 }} /></TableCell>
              <TableCell align="center"><Chip label={row.pending} size="small" sx={{ fontSize: 11, height: 22, minWidth: 40 }} /></TableCell>
              {hasGraded && <>
                <TableCell align="center"><Chip label={row.wins} size="small" color="success" sx={{ fontSize: 11, height: 22, minWidth: 40 }} /></TableCell>
                <TableCell align="center"><Chip label={row.losses} size="small" color="error" sx={{ fontSize: 11, height: 22, minWidth: 40 }} /></TableCell>
                <TableCell align="center"><Chip label={row.pushes} size="small" sx={{ fontSize: 11, height: 22, minWidth: 32 }} /></TableCell>
                <TableCell align="center" sx={{ color: row.winPct >= 60 ? 'success.main' : row.winPct >= 55 ? 'warning.main' : 'text.secondary', fontWeight: 600 }}>
                  {row.wins + row.losses > 0 ? `${row.winPct.toFixed(1)}%` : '-'}
                </TableCell>
              </>}
              <TableCell align="center" sx={{ fontWeight: 600 }}>{(row.avgProb * 100).toFixed(1)}%</TableCell>
              <TableCell align="center" sx={{ color: row.avgEdge > 0 ? 'success.main' : 'text.secondary' }}>
                {row.avgEdge > 0 ? `+${row.avgEdge.toFixed(1)}%` : `${row.avgEdge.toFixed(1)}%`}
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  );
};

// Helper: build PerformanceRow[] from rows grouped by a key function
const buildPerformance = (rows: FlatRow[], keyFn: (r: FlatRow) => string, labelFn?: (key: string) => string): PerformanceRow[] => {
  const groups: Record<string, FlatRow[]> = {};
  rows.forEach(r => {
    const key = keyFn(r);
    if (!groups[key]) groups[key] = [];
    groups[key].push(r);
  });
  return Object.entries(groups)
    .map(([key, items]) => {
      const wins = items.filter(r => r.result === 'won').length;
      const losses = items.filter(r => r.result === 'lost').length;
      const pushes = items.filter(r => r.result === 'push').length;
      const pending = items.filter(r => r.result === 'pending').length;
      const graded = wins + losses;
      return {
        label: labelFn ? labelFn(key) : key,
        total: items.length,
        wins,
        losses,
        pushes,
        pending,
        winPct: graded > 0 ? (wins / graded) * 100 : 0,
        avgProb: items.reduce((s, r) => s + r.probability, 0) / items.length,
        avgEdge: items.reduce((s, r) => s + r.edge, 0) / items.length,
      };
    })
    .sort((a, b) => b.total - a.total);
};

const Predictions: React.FC = () => {
  const theme = useTheme();
  const isDark = theme.palette.mode === 'dark';

  const [rows, setRows] = useState<FlatRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [tab, setTab] = useState(0);
  const [perfTab, setPerfTab] = useState(0);
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(50);
  const [sortField, setSortField] = useState<SortField>(null);
  const [sortOrder, setSortOrder] = useState<SortOrder>('asc');
  const [reasonDialog, setReasonDialog] = useState<{ open: boolean; row: FlatRow | null }>({ open: false, row: null });

  const { selectedSport, setSelectedSport, selectedTier, setSelectedTier } = useFilterStore();

  // Summary stats computed from real data
  const graded = rows.filter(r => r.result !== 'pending');
  const wins = rows.filter(r => r.result === 'won').length;
  const losses = rows.filter(r => r.result === 'lost').length;
  const pushes = rows.filter(r => r.result === 'push').length;
  const pending = rows.filter(r => r.result === 'pending').length;
  const summaryStats = {
    totalPredictions: rows.length,
    wins,
    losses,
    pushes,
    pending,
    winRate: graded.length > 0 ? Math.round(wins / graded.length * 1000) / 10 : 0,
    avgEdge: rows.length > 0 ? Math.round(rows.reduce((s, r) => s + r.edge, 0) / rows.length * 10) / 10 : 0,
    avgProb: rows.length > 0 ? Math.round(rows.reduce((s, r) => s + r.probability, 0) / rows.length * 1000) / 10 : 0,
  };

  // Performance data computed from actual rows
  const tierPerformance = useMemo(() =>
    buildPerformance(rows, r => r.signal_tier, key => {
      const labels: Record<string, string> = { A: 'Tier A (65%+)', B: 'Tier B (60-65%)', C: 'Tier C (55-60%)', D: 'Tier D (<55%)' };
      return labels[key] || `Tier ${key}`;
    }), [rows]);

  const sportPerformance = useMemo(() => buildPerformance(rows, r => r.sport), [rows]);

  const betTypePerformance = useMemo(() =>
    buildPerformance(rows, r => r.bet_type, key => {
      const labels: Record<string, string> = { spread: 'Spreads', total: 'Totals', moneyline: 'Moneyline' };
      return labels[key] || key;
    }), [rows]);

  const getPerformanceData = () => {
    switch (perfTab) {
      case 0: return tierPerformance;
      case 1: return sportPerformance;
      case 2: return betTypePerformance;
      default: return tierPerformance;
    }
  };

  const loadPredictions = async (showLoading = true) => {
    if (showLoading) setLoading(true);
    try {
      const data = await api.getPublicPredictions({ sport: selectedSport !== 'all' ? selectedSport : undefined, per_page: 200 });
      const preds = data?.predictions || (Array.isArray(data) ? data : []);
      setRows(transformToFlatRows(preds));
    } catch (err) {
      console.error('Load predictions error:', err);
      setRows([]);
    }
    if (showLoading) setLoading(false);
  };

  useEffect(() => { loadPredictions(); }, [selectedSport]);
  useEffect(() => { const iv = setInterval(() => loadPredictions(false), 60000); return () => clearInterval(iv); }, [selectedSport]);

  const filteredAndSorted = useMemo(() => {
    let filtered = [...rows];
    if (tab === 1) filtered = filtered.filter(r => r.result === 'pending');
    else if (tab === 2) filtered = filtered.filter(r => r.result !== 'pending');
    if (selectedTier !== 'all') filtered = filtered.filter(r => r.signal_tier === selectedTier);

    if (sortField) {
      const multiplier = sortOrder === 'asc' ? 1 : -1;
      filtered.sort((a, b) => {
        switch (sortField) {
          case 'sport': return multiplier * a.sport.localeCompare(b.sport);
          case 'datetime': return multiplier * (a.datetime.getTime() - b.datetime.getTime());
          case 'probability': return multiplier * (a.probability - b.probability);
          case 'edge': return multiplier * (a.edge - b.edge);
          case 'clv': return multiplier * ((b.clv || 0) - (a.clv || 0));
          case 'tier': return multiplier * a.signal_tier.localeCompare(b.signal_tier);
          default: return 0;
        }
      });
    }
    return filtered;
  }, [rows, tab, selectedTier, sortField, sortOrder]);

  const totalRows = filteredAndSorted.length;
  const paginated = useMemo(() => {
    if (rowsPerPage === -1) return filteredAndSorted;
    return filteredAndSorted.slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage);
  }, [filteredAndSorted, page, rowsPerPage]);

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortOrder('desc');
    }
  };

  const handleResetSort = () => {
    setSortField(null);
    setSortOrder('asc');
  };

  const fmtLine = (value: number | null | undefined) => {
    if (value == null) return '-';
    return value > 0 ? `+${value}` : `${value}`;
  };

  const fmtOdds = (value: number | null | undefined) => {
    if (value == null) return '-';
    return value > 0 ? `+${value}` : `${value}`;
  };

  const getStatusChip = (result: string) => {
    switch (result) {
      case 'won': return <Chip icon={<CheckCircle />} label="" size="small" color="success" sx={{ '& .MuiChip-icon': { fontSize: 16, ml: 0.5, mr: -0.5 }, minWidth: 28, height: 22 }} />;
      case 'lost': return <Chip icon={<Cancel />} label="" size="small" color="error" sx={{ '& .MuiChip-icon': { fontSize: 16, ml: 0.5, mr: -0.5 }, minWidth: 28, height: 22 }} />;
      case 'push': return <Chip icon={<Remove />} label="" size="small" sx={{ '& .MuiChip-icon': { fontSize: 16, ml: 0.5, mr: -0.5 }, minWidth: 28, height: 22 }} />;
      default: return <Chip icon={<Schedule />} label="" size="small" sx={{ '& .MuiChip-icon': { fontSize: 16, ml: 0.5, mr: -0.5 }, minWidth: 28, height: 22 }} />;
    }
  };

  const hdr = { fontWeight: 600, fontSize: 11, py: 0.75, bgcolor: isDark ? 'grey.900' : 'grey.100', color: isDark ? 'grey.100' : 'grey.800', whiteSpace: 'nowrap', borderBottom: 1, borderColor: 'divider' };
  const gameCount = useMemo(() => new Set(rows.map(r => r.game_id)).size, [rows]);

  return (
    <Box>
      {/* Header */}
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={1.5}>
        <Typography variant="h5" fontWeight={700} sx={{ fontSize: 20 }}>Predictions</Typography>
        <Button variant="contained" size="small" startIcon={<Refresh sx={{ fontSize: 14 }} />} onClick={() => loadPredictions()} sx={{ fontSize: 11 }}>Refresh</Button>
      </Box>

      {/* System Performance Dashboard */}
      <Box sx={{ mb: 1.5, p: 1.5, borderRadius: 2, bgcolor: isDark ? 'rgba(30, 41, 59, 0.5)' : 'rgba(248, 250, 252, 0.8)' }}>
        <Typography variant="subtitle1" fontWeight={600} sx={{ fontSize: 14, mb: 1 }}>System Performance Dashboard</Typography>
        <Grid container spacing={1.5}>
          <Grid item xs={6} sm={3}>
            <StatCard
              title="Total Predictions"
              value={summaryStats.totalPredictions.toLocaleString()}
              subtitle={graded.length > 0 ? `${wins}W - ${losses}L - ${pushes}P` : `${pending} pending`}
              icon={<Assessment sx={{ fontSize: 18 }} />}
              color="primary"
            />
          </Grid>
          <Grid item xs={6} sm={3}>
            <StatCard
              title="Win Rate"
              value={graded.length > 0 ? `${summaryStats.winRate}%` : '-'}
              subtitle={graded.length > 0 ? `${graded.length} graded` : 'No graded picks yet'}
              icon={<EmojiEvents sx={{ fontSize: 18 }} />}
              color="success"
            />
          </Grid>
          <Grid item xs={6} sm={3}>
            <StatCard
              title="Avg Probability"
              value={rows.length > 0 ? `${summaryStats.avgProb}%` : '-'}
              subtitle={rows.length > 0 ? `Avg Edge: ${summaryStats.avgEdge > 0 ? '+' : ''}${summaryStats.avgEdge}%` : 'No predictions'}
              icon={<TrendingUp sx={{ fontSize: 18 }} />}
              color="info"
            />
          </Grid>
          <Grid item xs={6} sm={3}>
            <StatCard
              title="Games Covered"
              value={gameCount}
              subtitle={rows.length > 0 ? `${(rows.length / gameCount).toFixed(1)} picks/game` : 'No games'}
              icon={<Casino sx={{ fontSize: 18 }} />}
              color="warning"
            />
          </Grid>
        </Grid>
      </Box>

      {/* Performance Tabs - computed from real data */}
      <Card sx={{ mb: 1.5 }}>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs value={perfTab} onChange={(_, v) => setPerfTab(v)} sx={{ minHeight: 40 }}>
            <Tab icon={<EmojiEvents sx={{ fontSize: 16 }} />} iconPosition="start" label="By Tier" sx={{ fontSize: 11, minHeight: 40, py: 0.5, textTransform: 'uppercase' }} />
            <Tab icon={<SportsSoccer sx={{ fontSize: 16 }} />} iconPosition="start" label="By Sport" sx={{ fontSize: 11, minHeight: 40, py: 0.5, textTransform: 'uppercase' }} />
            <Tab icon={<Assessment sx={{ fontSize: 16 }} />} iconPosition="start" label="By Bet Type" sx={{ fontSize: 11, minHeight: 40, py: 0.5, textTransform: 'uppercase' }} />
          </Tabs>
        </Box>
        <CardContent sx={{ py: 1, px: 1.5, '&:last-child': { pb: 1 } }}>
          {rows.length > 0 ? (
            <PerformanceTable data={getPerformanceData()} isDark={isDark} />
          ) : (
            <Typography variant="body2" color="text.secondary" sx={{ py: 2, textAlign: 'center' }}>
              No prediction data available yet
            </Typography>
          )}
        </CardContent>
      </Card>

      {/* Tabs + Filters Row */}
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', borderBottom: 1, borderColor: 'divider', mb: 1.5 }}>
        <Tabs value={tab} onChange={(_, v) => { setTab(v); setPage(0); }} sx={{ minHeight: 40 }}>
          <Tab label={`All (${rows.length})`} sx={{ fontSize: 12, minHeight: 40, py: 0.5 }} />
          <Tab label={`Pending (${pending})`} sx={{ fontSize: 12, minHeight: 40, py: 0.5 }} />
          <Tab label={`Graded (${graded.length})`} sx={{ fontSize: 12, minHeight: 40, py: 0.5 }} />
        </Tabs>
        <Box display="flex" alignItems="center" gap={1.5}>
          <FormControl size="small" sx={{ minWidth: 100 }}>
            <InputLabel sx={{ fontSize: 12 }}>Sport</InputLabel>
            <Select value={selectedSport} label="Sport" onChange={(e) => setSelectedSport(e.target.value)} sx={{ fontSize: 12, height: 34 }}>
              <MenuItem value="all" sx={{ fontSize: 12 }}>All Sports</MenuItem>
              {SPORTS.map(s => <MenuItem key={s.code} value={s.code} sx={{ fontSize: 12 }}>{s.code}</MenuItem>)}
            </Select>
          </FormControl>
          <FormControl size="small" sx={{ minWidth: 100 }}>
            <InputLabel sx={{ fontSize: 12 }}>Tier</InputLabel>
            <Select value={selectedTier} label="Tier" onChange={(e) => setSelectedTier(e.target.value)} sx={{ fontSize: 12, height: 34 }}>
              <MenuItem value="all" sx={{ fontSize: 12 }}>All Tiers</MenuItem>
              <MenuItem value="A" sx={{ fontSize: 12 }}>Tier A</MenuItem>
              <MenuItem value="B" sx={{ fontSize: 12 }}>Tier B</MenuItem>
              <MenuItem value="C" sx={{ fontSize: 12 }}>Tier C</MenuItem>
              <MenuItem value="D" sx={{ fontSize: 12 }}>Tier D</MenuItem>
            </Select>
          </FormControl>
          <Typography variant="caption" color="text.secondary" sx={{ fontSize: 11 }}>{totalRows} rows • {gameCount} games</Typography>
        </Box>
      </Box>

      {/* Predictions Table */}
      <Card>
        {loading && <LinearProgress />}
        <TableContainer>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell sx={hdr}><TableSortLabel active={sortField === 'sport'} direction={sortField === 'sport' ? sortOrder : 'asc'} onClick={() => handleSort('sport')}>Sport</TableSortLabel></TableCell>
                <TableCell sx={hdr}><TableSortLabel active={sortField === 'datetime'} direction={sortField === 'datetime' ? sortOrder : 'asc'} onClick={() => handleSort('datetime')}>Date</TableSortLabel></TableCell>
                <TableCell sx={hdr}>Time</TableCell>
                <TableCell sx={hdr}>Team</TableCell>
                <TableCell sx={hdr} align="center">Line</TableCell>
                <TableCell sx={hdr} align="center">Odds</TableCell>
                <TableCell sx={hdr}>Pick</TableCell>
                <TableCell sx={hdr} align="center"><TableSortLabel active={sortField === 'probability'} direction={sortField === 'probability' ? sortOrder : 'asc'} onClick={() => handleSort('probability')}>Prob</TableSortLabel></TableCell>
                <TableCell sx={hdr} align="center"><TableSortLabel active={sortField === 'edge'} direction={sortField === 'edge' ? sortOrder : 'asc'} onClick={() => handleSort('edge')}>Edge</TableSortLabel></TableCell>
                <TableCell sx={hdr} align="center"><TableSortLabel active={sortField === 'clv'} direction={sortField === 'clv' ? sortOrder : 'asc'} onClick={() => handleSort('clv')}>CLV</TableSortLabel></TableCell>
                <TableCell sx={hdr}><TableSortLabel active={sortField === 'tier'} direction={sortField === 'tier' ? sortOrder : 'asc'} onClick={() => handleSort('tier')}>Tier</TableSortLabel></TableCell>
                <TableCell sx={hdr}>W/L</TableCell>
                <TableCell sx={hdr}>
                  <Tooltip title="Reset Sort">
                    <IconButton size="small" onClick={handleResetSort} sx={{ p: 0.25 }}>
                      <RestartAlt sx={{ fontSize: 14 }} />
                    </IconButton>
                  </Tooltip>
                </TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {paginated.map((row) => {
                // Line display logic per bet type
                let awayLine = '-';
                let homeLine = '-';
                if (row.bet_type === 'spread' && row.line != null) {
                  if (row.pick_team === 'away') {
                    awayLine = fmtLine(row.line);
                    homeLine = fmtLine(row.line * -1);
                  } else if (row.pick_team === 'home') {
                    homeLine = fmtLine(row.line);
                    awayLine = fmtLine(row.line * -1);
                  }
                } else if (row.bet_type === 'total' && row.line != null) {
                  awayLine = `O/U ${row.line}`;
                  homeLine = `O/U ${row.line}`;
                }
                // For ML: line stays "-", odds column shows the value

                return (
                <React.Fragment key={row.id}>
                  {/* Away Row */}
                  <TableRow>
                    <TableCell rowSpan={2} sx={{ py: 0.75, fontSize: 11, verticalAlign: 'middle', fontWeight: 600, borderBottom: 1, borderColor: 'divider' }}>{row.sport}</TableCell>
                    <TableCell rowSpan={2} sx={{ py: 0.75, fontSize: 11, verticalAlign: 'middle', borderBottom: 1, borderColor: 'divider' }}>{row.date}</TableCell>
                    <TableCell rowSpan={2} sx={{ py: 0.75, fontSize: 11, verticalAlign: 'middle', borderBottom: 1, borderColor: 'divider' }}>{row.time}</TableCell>
                    <TableCell sx={{ py: 0.75, fontSize: 11, fontWeight: row.pick_team === 'away' ? 700 : 400, borderBottom: 0 }}>{row.away_team}</TableCell>
                    <TableCell align="center" sx={{ py: 0.75, fontSize: 11, fontFamily: 'monospace', borderBottom: 0 }}>{awayLine}</TableCell>
                    <TableCell rowSpan={2} align="center" sx={{ py: 0.75, fontSize: 11, fontFamily: 'monospace', verticalAlign: 'middle', borderBottom: 1, borderColor: 'divider' }}>{fmtOdds(row.odds)}</TableCell>
                    <TableCell rowSpan={2} sx={{ py: 0.75, fontSize: 11, verticalAlign: 'middle', borderBottom: 1, borderColor: 'divider' }}>
                      <Typography sx={{ fontSize: 11, fontWeight: 600, color: 'success.main', lineHeight: 1.3 }}>{row.system_pick}</Typography>
                      <Typography sx={{ fontSize: 10, color: 'text.secondary' }}>{row.bet_type_label}</Typography>
                    </TableCell>
                    <TableCell rowSpan={2} align="center" sx={{ py: 0.75, fontSize: 11, verticalAlign: 'middle', borderBottom: 1, borderColor: 'divider' }}>{formatPercent(row.probability)}</TableCell>
                    <TableCell rowSpan={2} align="center" sx={{ py: 0.75, fontSize: 11, verticalAlign: 'middle', borderBottom: 1, borderColor: 'divider', color: row.edge >= 3 ? 'success.main' : row.edge >= 1 ? 'warning.main' : 'text.secondary' }}>
                      {row.edge > 0 ? `+${row.edge.toFixed(1)}%` : `${row.edge.toFixed(1)}%`}
                    </TableCell>
                    <TableCell rowSpan={2} align="center" sx={{ py: 0.75, fontSize: 11, verticalAlign: 'middle', borderBottom: 1, borderColor: 'divider', color: row.clv != null ? (row.clv > 0 ? 'success.main' : row.clv < 0 ? 'error.main' : 'inherit') : 'inherit' }}>
                      {row.clv != null ? `${row.clv > 0 ? '+' : ''}${row.clv.toFixed(1)}%` : '-'}
                    </TableCell>
                    <TableCell rowSpan={2} sx={{ py: 0.75, fontSize: 11, verticalAlign: 'middle', borderBottom: 1, borderColor: 'divider' }}><TierBadge tier={row.signal_tier} /></TableCell>
                    <TableCell rowSpan={2} sx={{ py: 0.75, fontSize: 11, verticalAlign: 'middle', borderBottom: 1, borderColor: 'divider' }}>{getStatusChip(row.result)}</TableCell>
                    <TableCell rowSpan={2} sx={{ py: 0.75, fontSize: 11, verticalAlign: 'middle', borderBottom: 1, borderColor: 'divider' }}>
                      <IconButton size="small" onClick={() => setReasonDialog({ open: true, row })}><ExpandMore /></IconButton>
                    </TableCell>
                  </TableRow>
                  {/* Home Row */}
                  <TableRow>
                    <TableCell sx={{ py: 0.75, fontSize: 11, fontWeight: row.pick_team === 'home' ? 700 : 400, borderBottom: 1, borderColor: 'divider' }}>{row.home_team}</TableCell>
                    <TableCell align="center" sx={{ py: 0.75, fontSize: 11, fontFamily: 'monospace', borderBottom: 1, borderColor: 'divider' }}>{homeLine}</TableCell>
                  </TableRow>
                </React.Fragment>
                );
              })}
              {paginated.length === 0 && !loading && (
                <TableRow>
                  <TableCell colSpan={13} align="center" sx={{ py: 4 }}>
                    <Typography color="text.secondary">No predictions found</Typography>
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </TableContainer>
        <TablePagination
          component="div"
          count={totalRows}
          page={page}
          onPageChange={(_, p) => setPage(p)}
          rowsPerPage={rowsPerPage}
          onRowsPerPageChange={(e) => { setRowsPerPage(parseInt(e.target.value)); setPage(0); }}
          rowsPerPageOptions={[50, 100, 200, 500, { value: -1, label: 'All' }]}
          labelRowsPerPage="Rows per page:"
        />
      </Card>

      {/* Detail Dialog */}
      <Dialog open={reasonDialog.open} onClose={() => setReasonDialog({ open: false, row: null })} maxWidth="sm" fullWidth>
        <DialogTitle>
          <Box display="flex" alignItems="center" gap={2}>
            <Typography variant="h6" fontWeight={700}>📊 Prediction Details</Typography>
            {reasonDialog.row && <TierBadge tier={reasonDialog.row.signal_tier} />}
          </Box>
        </DialogTitle>
        <DialogContent>
          {reasonDialog.row && (
            <Box>
              <Box sx={{ mb: 2, p: 2, bgcolor: 'action.hover', borderRadius: 1 }}>
                <Typography variant="subtitle2" color="text.secondary" sx={{ mb: 0.5 }}>
                  {reasonDialog.row.away_team} vs {reasonDialog.row.home_team}
                </Typography>
                <Typography variant="subtitle1" fontWeight={700} color="success.main">{reasonDialog.row.system_pick}</Typography>
                <Chip label={reasonDialog.row.bet_type_label} size="small" sx={{ mt: 1 }} />
              </Box>
              <Grid container spacing={2} sx={{ mb: 2 }}>
                <Grid item xs={3}>
                  <Typography variant="caption" color="text.secondary">Probability</Typography>
                  <Typography variant="h6" fontWeight={700}>{formatPercent(reasonDialog.row.probability)}</Typography>
                </Grid>
                <Grid item xs={3}>
                  <Typography variant="caption" color="text.secondary">Edge</Typography>
                  <Typography variant="h6" fontWeight={700} color={reasonDialog.row.edge > 0 ? 'success.main' : 'text.secondary'}>
                    {reasonDialog.row.edge > 0 ? '+' : ''}{reasonDialog.row.edge.toFixed(1)}%
                  </Typography>
                </Grid>
                <Grid item xs={3}>
                  <Typography variant="caption" color="text.secondary">Line</Typography>
                  <Typography variant="h6" fontWeight={700}>
                    {reasonDialog.row.line != null ? fmtLine(reasonDialog.row.line) : '-'}
                  </Typography>
                </Grid>
                <Grid item xs={3}>
                  <Typography variant="caption" color="text.secondary">Odds</Typography>
                  <Typography variant="h6" fontWeight={700}>
                    {reasonDialog.row.odds != null ? fmtOdds(reasonDialog.row.odds) : '-'}
                  </Typography>
                </Grid>
              </Grid>
              <Typography variant="subtitle2" fontWeight={700} gutterBottom>Analysis</Typography>
              <Typography variant="body2" color="text.secondary" sx={{ lineHeight: 1.8 }}>{reasonDialog.row.reason}</Typography>
            </Box>
          )}
        </DialogContent>
        <DialogActions><Button onClick={() => setReasonDialog({ open: false, row: null })}>Close</Button></DialogActions>
      </Dialog>
    </Box>
  );
};

// Transform API response to flat rows
const transformToFlatRows = (data: any[]): FlatRow[] => {
  if (!data || data.length === 0) return [];
  return data.map((pred: any, idx: number) => {
    const gameTime = pred.game_time ? new Date(pred.game_time) : new Date();
    const dateStr = gameTime.toLocaleDateString('en-US', { month: 'numeric', day: 'numeric', year: 'numeric' });
    const timeStr = gameTime.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', hour12: true });
    const side = pred.predicted_side || '';
    const line = pred.line_at_prediction != null ? pred.line_at_prediction : null;
    const odds = pred.odds_at_prediction != null ? pred.odds_at_prediction : null;
    const bt = pred.bet_type || 'spread';

    let pickStr = side;
    let pickTeam: 'away' | 'home' | null = null;
    if (bt === 'spread') {
      const team = side === 'home' ? pred.home_team : pred.away_team;
      pickStr = line != null ? `${team} ${line > 0 ? '+' : ''}${line}` : (team || side);
      pickTeam = side === 'home' ? 'home' : side === 'away' ? 'away' : null;
    } else if (bt === 'total') {
      pickStr = line != null ? `${side === 'over' ? 'Over' : 'Under'} ${line}` : (side === 'over' ? 'Over' : 'Under');
      pickTeam = null;
    } else if (bt === 'moneyline') {
      const team = side === 'home' ? pred.home_team : pred.away_team;
      pickStr = odds != null ? `${team} ML (${odds > 0 ? '+' : ''}${odds})` : `${team} ML`;
      pickTeam = side === 'home' ? 'home' : side === 'away' ? 'away' : null;
    }

    let result: 'pending' | 'won' | 'lost' | 'push' = 'pending';
    if (pred.result === 'win') result = 'won';
    else if (pred.result === 'loss') result = 'lost';
    else if (pred.result === 'push') result = 'push';

    const btLabel = bt === 'spread' ? 'Spread' : bt === 'total' ? 'Total' : bt === 'moneyline' ? 'ML' : bt;
    const prob = pred.probability || 0.5;
    const edge = pred.edge != null ? pred.edge : 0;

    return {
      id: pred.id || `pred_${idx}`,
      game_id: pred.game_id || `game_${idx}`,
      sport: pred.sport || pred.sport_code || 'UNK',
      date: dateStr,
      time: timeStr,
      datetime: gameTime,
      away_team: pred.away_team || 'TBD',
      home_team: pred.home_team || 'TBD',
      bet_type: bt,
      bet_type_label: btLabel,
      line,
      odds,
      system_pick: pickStr,
      pick_team: pickTeam,
      probability: prob,
      edge,
      clv: pred.clv != null ? pred.clv : null,
      signal_tier: (pred.signal_tier || 'D') as 'A' | 'B' | 'C' | 'D',
      result,
      reason: `Model probability: ${(prob * 100).toFixed(1)}% | Edge: ${edge > 0 ? '+' : ''}${edge.toFixed(1)}%${line != null ? ` | Line: ${line > 0 ? '+' : ''}${line}` : ''}${odds != null ? ` | Odds: ${odds > 0 ? '+' : ''}${odds}` : ''}`,
    };
  });
};

export default Predictions;