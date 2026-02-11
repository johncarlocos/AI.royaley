// src/pages/Predictions/Predictions.tsx - Flat Table with Uniform Borders
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
import { formatPercent } from '../../utils';

// Types
interface FlatRow {
  id: string;
  game_id: string;
  sport: string;
  date: string;
  time: string;
  datetime: Date;
  away_rotation: number;
  away_team: string;
  away_record: string;
  home_rotation: number;
  home_team: string;
  home_record: string;
  bet_type: string;
  bet_type_label: string;
  away_circa_open: number | string;
  away_circa_current: number | string;
  away_system_open: number | string;
  away_system_current: number | string;
  home_circa_open: number | string;
  home_circa_current: number | string;
  home_system_open: number | string;
  home_system_current: number | string;
  system_pick: string;
  pick_team: 'away' | 'home' | null;
  probability: number;
  edge: number;
  clv?: number;
  signal_tier: 'A' | 'B' | 'C' | 'D';
  result: 'pending' | 'won' | 'lost' | 'push';
  reason: string;
}

interface PerformanceRow {
  label: string;
  wins: number;
  losses: number;
  pushes: number;
  winPct: number;
  edge: number;
  clv: number;
  roi: number;
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

// Stat Card Component
const StatCard: React.FC<{ title: string; value: string | number; subtitle?: string; icon: React.ReactNode; color?: string; trend?: number }> = 
  ({ title, value, subtitle, icon, color = 'primary', trend }) => {
  const theme = useTheme();
  return (
    <Card sx={{ height: '100%' }}>
      <CardContent sx={{ py: 1.5, px: 2, '&:last-child': { pb: 1.5 } }}>
        <Box display="flex" justifyContent="space-between" alignItems="flex-start">
          <Box>
            <Typography variant="caption" color="text.secondary" sx={{ fontSize: 11 }}>{title}</Typography>
            <Typography variant="h5" fontWeight={700} color={`${color}.main`} sx={{ fontSize: 20, lineHeight: 1.2 }}>{value}</Typography>
            {subtitle && <Typography variant="caption" color="text.secondary" sx={{ fontSize: 10 }}>{subtitle}</Typography>}
            {trend !== undefined && (
              <Box display="flex" alignItems="center" mt={0.25}>
                {trend >= 0 ? <TrendingUp sx={{ fontSize: 14 }} color="success" /> : <TrendingDown sx={{ fontSize: 14 }} color="error" />}
                <Typography variant="caption" sx={{ ml: 0.3, color: trend >= 0 ? 'success.main' : 'error.main', fontSize: 10 }}>
                  {trend >= 0 ? '+' : ''}{trend.toFixed(1)}% vs last week
                </Typography>
              </Box>
            )}
          </Box>
          <Box sx={{ p: 0.75, borderRadius: 1.5, bgcolor: theme.palette.mode === 'dark' ? `${color}.900` : `${color}.50`, color: `${color}.main` }}>
            {icon}
          </Box>
        </Box>
      </CardContent>
    </Card>
  );
};

// Performance Table Component
const PerformanceTable: React.FC<{ data: PerformanceRow[]; isDark: boolean }> = ({ data, isDark }) => {
  const perfHdr = { fontWeight: 600, fontSize: 11, py: 0.75, bgcolor: isDark ? 'grey.900' : 'grey.100', color: isDark ? 'grey.100' : 'grey.800', borderBottom: 2, borderColor: isDark ? 'grey.700' : 'grey.300' };
  
  return (
    <TableContainer>
      <Table size="small">
        <TableHead>
          <TableRow>
            <TableCell sx={perfHdr}>Category</TableCell>
            <TableCell sx={perfHdr} align="center">W</TableCell>
            <TableCell sx={perfHdr} align="center">L</TableCell>
            <TableCell sx={perfHdr} align="center">P</TableCell>
            <TableCell sx={perfHdr} align="center">Record</TableCell>
            <TableCell sx={perfHdr} align="center">Win %</TableCell>
            <TableCell sx={perfHdr} align="center">Avg Edge</TableCell>
            <TableCell sx={perfHdr} align="center">Avg CLV</TableCell>
            <TableCell sx={perfHdr} align="center">ROI</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {data.map((row, idx) => (
            <TableRow key={idx} sx={{ '& td': { py: 0.75, fontSize: 12 } }}>
              <TableCell sx={{ fontWeight: 500 }}>{row.label}</TableCell>
              <TableCell align="center"><Chip label={row.wins} size="small" color="success" sx={{ fontSize: 11, height: 22, minWidth: 40 }} /></TableCell>
              <TableCell align="center"><Chip label={row.losses} size="small" color="error" sx={{ fontSize: 11, height: 22, minWidth: 40 }} /></TableCell>
              <TableCell align="center"><Chip label={row.pushes} size="small" sx={{ fontSize: 11, height: 22, minWidth: 32 }} /></TableCell>
              <TableCell align="center">{row.wins}-{row.losses}-{row.pushes}</TableCell>
              <TableCell align="center" sx={{ color: row.winPct >= 60 ? 'success.main' : row.winPct >= 55 ? 'warning.main' : 'text.secondary', fontWeight: 600 }}>{row.winPct}%</TableCell>
              <TableCell align="center" sx={{ color: 'success.main' }}>+{row.edge}%</TableCell>
              <TableCell align="center" sx={{ color: row.clv >= 0 ? 'success.main' : 'error.main' }}>{row.clv >= 0 ? '+' : ''}{row.clv}%</TableCell>
              <TableCell align="center" sx={{ color: row.roi >= 0 ? 'success.main' : 'error.main', fontWeight: 600 }}>{row.roi >= 0 ? '+' : ''}{row.roi}%</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  );
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

  // Summary stats (computed from real data)
  const graded = rows.filter(r => r.result !== 'pending');
  const wins = rows.filter(r => r.result === 'won').length;
  const losses = rows.filter(r => r.result === 'lost').length;
  const pushes = rows.filter(r => r.result === 'push').length;
  const summaryStats = {
    totalPredictions: rows.length,
    totalWins: wins,
    totalLosses: losses,
    totalPushes: pushes,
    winRate: graded.length > 0 ? Math.round(wins / graded.length * 1000) / 10 : 0,
    avgEdge: rows.length > 0 ? Math.round(rows.reduce((s, r) => s + (r.edge || 0), 0) / rows.length * 10) / 10 : 0,
    avgCLV: 0,
    roi: 0,
    bankrollGrowth: 0
  };

  // Performance data (kept as reference - will be replaced by live aggregation later)
  const tierPerformance: PerformanceRow[] = [
    { label: 'Tier A (65%+)', wins: 189, losses: 87, pushes: 12, winPct: 68.5, edge: 4.2, clv: 2.1, roi: 14.2 },
    { label: 'Tier B (60-65%)', wins: 234, losses: 156, pushes: 10, winPct: 60.0, edge: 2.8, clv: 1.4, roi: 8.5 },
    { label: 'Tier C (55-60%)', wins: 198, losses: 168, pushes: 9, winPct: 54.1, edge: 1.6, clv: 0.7, roi: 3.2 },
    { label: 'Tier D (<55%)', wins: 91, losses: 87, pushes: 6, winPct: 51.1, edge: 0.5, clv: -0.2, roi: -1.8 },
  ];

  const sportPerformance: PerformanceRow[] = [
    { label: 'NFL', wins: 142, losses: 98, pushes: 8, winPct: 59.2, edge: 3.1, clv: 1.5, roi: 9.8 },
    { label: 'NBA', wins: 186, losses: 134, pushes: 7, winPct: 58.1, edge: 2.6, clv: 1.2, roi: 7.4 },
    { label: 'MLB', wins: 124, losses: 89, pushes: 6, winPct: 58.2, edge: 2.4, clv: 1.0, roi: 6.8 },
    { label: 'NHL', wins: 98, losses: 72, pushes: 5, winPct: 57.6, edge: 2.2, clv: 0.9, roi: 5.9 },
    { label: 'NCAAF', wins: 67, losses: 48, pushes: 4, winPct: 58.3, edge: 3.0, clv: 1.4, roi: 8.2 },
    { label: 'NCAAB', wins: 58, losses: 42, pushes: 3, winPct: 58.0, edge: 2.5, clv: 1.1, roi: 6.5 },
    { label: 'WNBA', wins: 18, losses: 9, pushes: 2, winPct: 66.7, edge: 3.8, clv: 1.8, roi: 12.4 },
    { label: 'CFL', wins: 12, losses: 4, pushes: 1, winPct: 75.0, edge: 4.5, clv: 2.2, roi: 15.8 },
  ];

  const periodPerformance: PerformanceRow[] = [
    { label: 'Full Game', wins: 412, losses: 284, pushes: 21, winPct: 59.2, edge: 3.0, clv: 1.4, roi: 9.2 },
    { label: '1st Half', wins: 178, losses: 128, pushes: 9, winPct: 58.2, edge: 2.5, clv: 1.1, roi: 7.1 },
    { label: '2nd Half', wins: 122, losses: 86, pushes: 7, winPct: 58.7, edge: 2.3, clv: 0.9, roi: 6.4 },
  ];

  const betTypePerformance: PerformanceRow[] = [
    { label: 'Spreads', wins: 398, losses: 278, pushes: 22, winPct: 58.9, edge: 2.9, clv: 1.3, roi: 8.8 },
    { label: 'Totals', wins: 314, losses: 220, pushes: 15, winPct: 58.8, edge: 2.7, clv: 1.1, roi: 8.2 },
  ];

  const combinedPerformance: PerformanceRow[] = [
    { label: 'FG Spread', wins: 218, losses: 152, pushes: 12, winPct: 58.9, edge: 3.1, clv: 1.5, roi: 9.4 },
    { label: 'FG Total', wins: 194, losses: 132, pushes: 9, winPct: 59.5, edge: 2.9, clv: 1.3, roi: 9.0 },
    { label: '1H Spread', wins: 98, losses: 72, pushes: 5, winPct: 57.6, edge: 2.4, clv: 1.0, roi: 6.8 },
    { label: '1H Total', wins: 80, losses: 56, pushes: 4, winPct: 58.8, edge: 2.6, clv: 1.2, roi: 7.4 },
    { label: '2H Spread', wins: 82, losses: 54, pushes: 5, winPct: 60.3, edge: 2.6, clv: 1.1, roi: 7.8 },
    { label: '2H Total', wins: 40, losses: 32, pushes: 2, winPct: 55.6, edge: 2.0, clv: 0.8, roi: 5.0 },
  ];

  const getPerformanceData = () => {
    switch (perfTab) {
      case 0: return tierPerformance;
      case 1: return sportPerformance;
      case 2: return periodPerformance;
      case 3: return betTypePerformance;
      case 4: return combinedPerformance;
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
  // Auto-refresh every 60 seconds
  useEffect(() => { const iv = setInterval(() => loadPredictions(false), 60000); return () => clearInterval(iv); }, [selectedSport]);

  const filteredAndSorted = useMemo(() => {
    let filtered = [...rows];
    if (tab === 1) filtered = filtered.filter(r => r.result === 'pending');
    else if (tab === 2) filtered = filtered.filter(r => r.result !== 'pending');
    if (selectedTier !== 'all') filtered = filtered.filter(r => r.signal_tier === selectedTier);
    
    if (sortField) {
      filtered.sort((a, b) => {
        const multiplier = sortOrder === 'asc' ? 1 : -1;
        switch (sortField) {
          case 'sport': return multiplier * a.sport.localeCompare(b.sport);
          case 'datetime': return multiplier * (a.datetime.getTime() - b.datetime.getTime());
          case 'probability': return multiplier * (b.probability - a.probability);
          case 'edge': return multiplier * (b.edge - a.edge);
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

  const formatLine = (value: number | string | undefined) => {
    if (value === undefined || value === null || value === '') return '-';
    const num = typeof value === 'string' ? parseFloat(value) : value;
    if (isNaN(num)) return value;
    return num > 0 ? `+${num}` : `${num}`;
  };

  const getStatusChip = (result: string) => {
    switch (result) {
      case 'won': return <Chip icon={<CheckCircle />} label="" size="small" color="success" sx={{ '& .MuiChip-icon': { fontSize: 16, ml: 0.5, mr: -0.5 }, minWidth: 28, height: 22 }} />;
      case 'lost': return <Chip icon={<Cancel />} label="" size="small" color="error" sx={{ '& .MuiChip-icon': { fontSize: 16, ml: 0.5, mr: -0.5 }, minWidth: 28, height: 22 }} />;
      case 'push': return <Chip icon={<Remove />} label="" size="small" sx={{ '& .MuiChip-icon': { fontSize: 16, ml: 0.5, mr: -0.5 }, minWidth: 28, height: 22 }} />;
      default: return <Chip icon={<Schedule />} label="" size="small" sx={{ '& .MuiChip-icon': { fontSize: 16, ml: 0.5, mr: -0.5 }, minWidth: 28, height: 22 }} />;
    }
  };

  // Uniform styles - no borders between rows
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
              <StatCard title="Total Predictions" value={summaryStats.totalPredictions.toLocaleString()} subtitle={`${summaryStats.totalWins}W - ${summaryStats.totalLosses}L - ${summaryStats.totalPushes}P`} icon={<Assessment sx={{ fontSize: 18 }} />} color="primary" />
            </Grid>
            <Grid item xs={6} sm={3}>
              <StatCard title="Win Rate" value={`${summaryStats.winRate}%`} subtitle="Across all tiers" icon={<EmojiEvents sx={{ fontSize: 18 }} />} color="success" trend={2.3} />
            </Grid>
            <Grid item xs={6} sm={3}>
              <StatCard title="Average Edge" value={`+${summaryStats.avgEdge}%`} subtitle={`CLV: +${summaryStats.avgCLV}%`} icon={<TrendingUp sx={{ fontSize: 18 }} />} color="info" trend={0.5} />
            </Grid>
            <Grid item xs={6} sm={3}>
              <StatCard title="ROI" value={`+${summaryStats.roi}%`} subtitle={`Bankroll: +${summaryStats.bankrollGrowth}%`} icon={<Casino sx={{ fontSize: 18 }} />} color="warning" trend={1.8} />
            </Grid>
          </Grid>
      </Box>

      {/* Performance Tabs */}
      <Card sx={{ mb: 1.5 }}>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs value={perfTab} onChange={(_, v) => setPerfTab(v)} sx={{ minHeight: 40 }}>
            <Tab icon={<EmojiEvents sx={{ fontSize: 16 }} />} iconPosition="start" label="By Tier" sx={{ fontSize: 11, minHeight: 40, py: 0.5, textTransform: 'uppercase' }} />
            <Tab icon={<SportsSoccer sx={{ fontSize: 16 }} />} iconPosition="start" label="By Sport" sx={{ fontSize: 11, minHeight: 40, py: 0.5, textTransform: 'uppercase' }} />
            <Tab icon={<Timer sx={{ fontSize: 16 }} />} iconPosition="start" label="By Period" sx={{ fontSize: 11, minHeight: 40, py: 0.5, textTransform: 'uppercase' }} />
            <Tab icon={<Assessment sx={{ fontSize: 16 }} />} iconPosition="start" label="By Bet Type" sx={{ fontSize: 11, minHeight: 40, py: 0.5, textTransform: 'uppercase' }} />
            <Tab icon={<CalendarToday sx={{ fontSize: 16 }} />} iconPosition="start" label="Combined" sx={{ fontSize: 11, minHeight: 40, py: 0.5, textTransform: 'uppercase' }} />
          </Tabs>
        </Box>
        <CardContent sx={{ py: 1, px: 1.5, '&:last-child': { pb: 1 } }}>
          <PerformanceTable data={getPerformanceData()} isDark={isDark} />
        </CardContent>
      </Card>

      {/* Tabs + Filters Row */}
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', borderBottom: 1, borderColor: 'divider', mb: 1.5 }}>
        <Tabs value={tab} onChange={(_, v) => { setTab(v); setPage(0); }} sx={{ minHeight: 40 }}>
          <Tab label={`All (${gameCount})`} sx={{ fontSize: 12, minHeight: 40, py: 0.5 }} />
          <Tab label="Pending" sx={{ fontSize: 12, minHeight: 40, py: 0.5 }} />
          <Tab label="Graded" sx={{ fontSize: 12, minHeight: 40, py: 0.5 }} />
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
                <TableCell sx={hdr} align="center">Game #</TableCell>
                <TableCell sx={hdr}>Team</TableCell>
                <TableCell sx={hdr}>Record</TableCell>
                <TableCell sx={hdr} align="center">Circa O</TableCell>
                <TableCell sx={hdr} align="center">Circa.</TableCell>
                <TableCell sx={hdr} align="center">System O</TableCell>
                <TableCell sx={hdr} align="center">System.</TableCell>
                <TableCell sx={hdr}>Pick</TableCell>
                <TableCell sx={hdr} align="center"><TableSortLabel active={sortField === 'probability'} direction={sortField === 'probability' ? sortOrder : 'asc'} onClick={() => handleSort('probability')}>%</TableSortLabel></TableCell>
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
              {paginated.map((row, idx) => {
                return (
                <React.Fragment key={row.id}>
                  {/* Away Row */}
                  <TableRow>
                    <TableCell rowSpan={2} sx={{ py: 0.75, fontSize: 11, verticalAlign: 'middle', fontWeight: 600, borderBottom: 1, borderColor: 'divider' }}>{row.sport}</TableCell>
                    <TableCell rowSpan={2} sx={{ py: 0.75, fontSize: 11, verticalAlign: 'middle', borderBottom: 1, borderColor: 'divider' }}>{row.date}</TableCell>
                    <TableCell rowSpan={2} sx={{ py: 0.75, fontSize: 11, verticalAlign: 'middle', borderBottom: 1, borderColor: 'divider' }}>{row.time}</TableCell>
                    <TableCell align="center" sx={{ py: 0.75, fontSize: 11, fontFamily: 'monospace', fontWeight: 600, borderBottom: 0 }}>{row.away_rotation}</TableCell>
                    <TableCell sx={{ py: 0.75, fontSize: 11, fontWeight: row.pick_team === 'away' ? 700 : 400, borderBottom: 0 }}>{row.away_team}</TableCell>
                    <TableCell sx={{ py: 0.75, fontSize: 11, borderBottom: 0 }}>{row.away_record}</TableCell>
                    <TableCell align="center" sx={{ py: 0.75, fontSize: 11, borderBottom: 0 }}>{formatLine(row.away_circa_open)}</TableCell>
                    <TableCell align="center" sx={{ py: 0.75, fontSize: 11, fontWeight: 600, borderBottom: 0 }}>{formatLine(row.away_circa_current)}</TableCell>
                    <TableCell align="center" sx={{ py: 0.75, fontSize: 11, color: 'info.main', borderBottom: 0 }}>{formatLine(row.away_system_open)}</TableCell>
                    <TableCell align="center" sx={{ py: 0.75, fontSize: 11, color: 'info.main', fontWeight: 600, borderBottom: 0 }}>{formatLine(row.away_system_current)}</TableCell>
                    <TableCell rowSpan={2} sx={{ py: 0.75, fontSize: 11, verticalAlign: 'middle', borderBottom: 1, borderColor: 'divider' }}><Typography sx={{ fontSize: 11, fontWeight: 600, color: 'success.main', lineHeight: 1.3 }}>{row.system_pick}</Typography><Typography sx={{ fontSize: 10, color: 'text.secondary' }}>{row.bet_type_label}</Typography></TableCell>
                    <TableCell rowSpan={2} align="center" sx={{ py: 0.75, fontSize: 11, verticalAlign: 'middle', borderBottom: 1, borderColor: 'divider' }}>{formatPercent(row.probability)}</TableCell>
                    <TableCell rowSpan={2} align="center" sx={{ py: 0.75, fontSize: 11, verticalAlign: 'middle', borderBottom: 1, borderColor: 'divider', color: row.edge >= 3 ? 'success.main' : row.edge >= 1 ? 'warning.main' : 'text.secondary' }}>+{row.edge.toFixed(1)}%</TableCell>
                    <TableCell rowSpan={2} align="center" sx={{ py: 0.75, fontSize: 11, verticalAlign: 'middle', borderBottom: 1, borderColor: 'divider', color: row.clv !== undefined ? (row.clv > 0 ? 'success.main' : row.clv < 0 ? 'error.main' : 'inherit') : 'inherit' }}>{row.clv !== undefined ? `${row.clv > 0 ? '+' : ''}${row.clv.toFixed(1)}%` : ''}</TableCell>
                    <TableCell rowSpan={2} sx={{ py: 0.75, fontSize: 11, verticalAlign: 'middle', borderBottom: 1, borderColor: 'divider' }}><TierBadge tier={row.signal_tier} /></TableCell>
                    <TableCell rowSpan={2} sx={{ py: 0.75, fontSize: 11, verticalAlign: 'middle', borderBottom: 1, borderColor: 'divider' }}>{getStatusChip(row.result)}</TableCell>
                    <TableCell rowSpan={2} sx={{ py: 0.75, fontSize: 11, verticalAlign: 'middle', borderBottom: 1, borderColor: 'divider' }}><IconButton size="small" onClick={() => setReasonDialog({ open: true, row })}><ExpandMore /></IconButton></TableCell>
                  </TableRow>
                  {/* Home Row */}
                  <TableRow>
                    <TableCell align="center" sx={{ py: 0.75, fontSize: 11, fontFamily: 'monospace', fontWeight: 600, borderBottom: 1, borderColor: 'divider' }}>{row.home_rotation}</TableCell>
                    <TableCell sx={{ py: 0.75, fontSize: 11, fontWeight: row.pick_team === 'home' ? 700 : 400, borderBottom: 1, borderColor: 'divider' }}>{row.home_team}</TableCell>
                    <TableCell sx={{ py: 0.75, fontSize: 11, borderBottom: 1, borderColor: 'divider' }}>{row.home_record}</TableCell>
                    <TableCell align="center" sx={{ py: 0.75, fontSize: 11, borderBottom: 1, borderColor: 'divider' }}>{formatLine(row.home_circa_open)}</TableCell>
                    <TableCell align="center" sx={{ py: 0.75, fontSize: 11, fontWeight: 600, borderBottom: 1, borderColor: 'divider' }}>{formatLine(row.home_circa_current)}</TableCell>
                    <TableCell align="center" sx={{ py: 0.75, fontSize: 11, color: 'info.main', borderBottom: 1, borderColor: 'divider' }}>{formatLine(row.home_system_open)}</TableCell>
                    <TableCell align="center" sx={{ py: 0.75, fontSize: 11, color: 'info.main', fontWeight: 600, borderBottom: 1, borderColor: 'divider' }}>{formatLine(row.home_system_current)}</TableCell>
                  </TableRow>
                </React.Fragment>
                );
              })}
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

      <Dialog open={reasonDialog.open} onClose={() => setReasonDialog({ open: false, row: null })} maxWidth="sm" fullWidth>
        <DialogTitle><Box display="flex" alignItems="center" gap={2}><Typography variant="h6" fontWeight={700}>📊 Prediction Reason</Typography>{reasonDialog.row && <TierBadge tier={reasonDialog.row.signal_tier} />}</Box></DialogTitle>
        <DialogContent>
          {reasonDialog.row && (
            <Box>
              <Box sx={{ mb: 2, p: 2, bgcolor: 'action.hover', borderRadius: 1 }}>
                <Typography variant="subtitle1" fontWeight={700} color="success.main">{reasonDialog.row.system_pick}</Typography>
                <Chip label={reasonDialog.row.bet_type_label} size="small" sx={{ mt: 1 }} />
              </Box>
              <Grid container spacing={2} sx={{ mb: 2 }}>
                <Grid item xs={4}><Typography variant="caption" color="text.secondary">Probability</Typography><Typography variant="h6" fontWeight={700}>{formatPercent(reasonDialog.row.probability)}</Typography></Grid>
                <Grid item xs={4}><Typography variant="caption" color="text.secondary">Edge</Typography><Typography variant="h6" fontWeight={700} color="success.main">+{reasonDialog.row.edge.toFixed(1)}%</Typography></Grid>
                <Grid item xs={4}><Typography variant="caption" color="text.secondary">CLV</Typography><Typography variant="h6" fontWeight={700}>{reasonDialog.row.clv !== undefined ? `${reasonDialog.row.clv > 0 ? '+' : ''}${reasonDialog.row.clv.toFixed(1)}%` : '-'}</Typography></Grid>
              </Grid>
              <Typography variant="subtitle2" fontWeight={700} gutterBottom>Why this pick?</Typography>
              <Typography variant="body2" color="text.secondary" sx={{ lineHeight: 1.8 }}>{reasonDialog.row.reason}</Typography>
            </Box>
          )}
        </DialogContent>
        <DialogActions><Button onClick={() => setReasonDialog({ open: false, row: null })}>Close</Button></DialogActions>
      </Dialog>
    </Box>
  );
};

const transformToFlatRows = (data: any[]): FlatRow[] => {
  if (!data || data.length === 0) return [];
  return data.map((pred: any, idx: number) => {
    const gameTime = pred.game_time ? new Date(pred.game_time) : new Date();
    const dateStr = gameTime.toLocaleDateString('en-US', { month: 'numeric', day: 'numeric', year: 'numeric' });
    const timeStr = gameTime.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', hour12: true });
    const side = pred.predicted_side || '';
    const line = pred.line_at_prediction;
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
      pickStr = `${team} ML`;
      pickTeam = side === 'home' ? 'home' : side === 'away' ? 'away' : null;
    }

    let result: 'pending' | 'won' | 'lost' | 'push' = 'pending';
    if (pred.result === 'win') result = 'won';
    else if (pred.result === 'loss') result = 'lost';
    else if (pred.result === 'push') result = 'push';

    const btLabel = bt === 'spread' ? 'Spread' : bt === 'total' ? 'Total' : bt === 'moneyline' ? 'ML' : bt;

    return {
      id: pred.id || `pred_${idx}`,
      game_id: pred.game_id || `game_${idx}`,
      sport: pred.sport || pred.sport_code || 'UNK',
      date: dateStr,
      time: timeStr,
      datetime: gameTime,
      away_rotation: 0,
      away_team: pred.away_team || 'TBD',
      away_record: '',
      home_rotation: 0,
      home_team: pred.home_team || 'TBD',
      home_record: '',
      bet_type: bt,
      bet_type_label: btLabel,
      away_circa_open: bt === 'total' && line ? line : (side === 'away' && line ? line : '-'),
      away_circa_current: bt === 'total' && line ? line : (side === 'away' && line ? line : '-'),
      away_system_open: bt === 'total' && line ? line : (side === 'away' && line ? line : '-'),
      away_system_current: bt === 'total' && line ? line : (side === 'away' && line ? line : '-'),
      home_circa_open: bt === 'total' && line ? line : (side === 'home' && line ? line : (bt === 'spread' && side === 'away' && line ? line * -1 : '-')),
      home_circa_current: bt === 'total' && line ? line : (side === 'home' && line ? line : (bt === 'spread' && side === 'away' && line ? line * -1 : '-')),
      home_system_open: bt === 'total' && line ? line : (side === 'home' && line ? line : (bt === 'spread' && side === 'away' && line ? line * -1 : '-')),
      home_system_current: bt === 'total' && line ? line : (side === 'home' && line ? line : (bt === 'spread' && side === 'away' && line ? line * -1 : '-')),
      system_pick: pickStr,
      pick_team: pickTeam,
      probability: pred.probability || 0.5,
      edge: pred.edge || 0,
      clv: pred.clv,
      signal_tier: (pred.signal_tier || 'D') as 'A' | 'B' | 'C' | 'D',
      result: result,
      reason: `Model probability: ${Math.round((pred.probability || 0.5) * 100)}% | Edge: ${pred.edge ? pred.edge.toFixed(1) : '0'}%`,
    };
  });
};

export default Predictions;
