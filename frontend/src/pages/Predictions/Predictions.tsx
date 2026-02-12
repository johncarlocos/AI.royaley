// src/pages/Predictions/Predictions.tsx - Original design with real opening/current line data
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
  away_rotation: string;
  away_team: string;
  away_record: string;
  home_rotation: string;
  home_team: string;
  home_record: string;
  bet_type: string;
  bet_type_label: string;
  // Opening line (snapshot at prediction time) - "Circa O" + "System O"
  away_circa_open: number | string;
  home_circa_open: number | string;
  // Current line (latest from market) - "Circa." + "System."
  away_circa_current: number | string;
  home_circa_current: number | string;
  // System columns (will differ from Circa when ML model generates own lines)
  away_system_open: number | string;
  home_system_open: number | string;
  away_system_current: number | string;
  home_system_current: number | string;
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
              <TableCell align="center" sx={{ color: row.winPct >= 60 ? 'success.main' : row.winPct >= 55 ? 'warning.main' : 'text.secondary', fontWeight: 600 }}>
                {row.wins + row.losses > 0 ? `${row.winPct.toFixed(1)}%` : '-'}
              </TableCell>
              <TableCell align="center" sx={{ color: row.edge > 0 ? 'success.main' : row.edge < 0 ? 'error.main' : 'text.secondary' }}>
                {row.edge > 0 ? '+' : ''}{(row.edge * 100).toFixed(1)}%
              </TableCell>
              <TableCell align="center" sx={{ color: row.clv > 0 ? 'success.main' : row.clv < 0 ? 'error.main' : 'text.secondary' }}>
                {row.clv > 0 ? '+' : ''}{row.clv.toFixed(1)}%
              </TableCell>
              <TableCell align="center" sx={{ color: row.roi > 0 ? 'success.main' : row.roi < 0 ? 'error.main' : 'text.secondary', fontWeight: 600 }}>
                {row.roi > 0 ? '+' : ''}{row.roi.toFixed(1)}%
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  );
};

// Build PerformanceRow from actual prediction rows
const buildPerf = (rows: FlatRow[], keyFn: (r: FlatRow) => string, labelFn?: (key: string) => string, order?: string[]): PerformanceRow[] => {
  const groups: Record<string, FlatRow[]> = {};
  rows.forEach(r => { const k = keyFn(r); if (!groups[k]) groups[k] = []; groups[k].push(r); });
  const result = Object.entries(groups).map(([key, items]) => {
    const w = items.filter(r => r.result === 'won').length;
    const l = items.filter(r => r.result === 'lost').length;
    const p = items.filter(r => r.result === 'push').length;
    const g = w + l;
    const withClv = items.filter(r => r.clv != null);
    return {
      label: labelFn ? labelFn(key) : key,
      wins: w, losses: l, pushes: p,
      winPct: g > 0 ? (w / g) * 100 : 0,
      edge: items.reduce((s, r) => s + r.edge, 0) / items.length,
      clv: withClv.length > 0 ? withClv.reduce((s, r) => s + (r.clv || 0), 0) / withClv.length : 0,
      roi: 0,
      _key: key,
    };
  });
  if (order) result.sort((a, b) => {
    const ai = order.indexOf((a as any)._key); const bi = order.indexOf((b as any)._key);
    return (ai === -1 ? 999 : ai) - (bi === -1 ? 999 : bi);
  });
  else result.sort((a, b) => (b.wins + b.losses + b.pushes) - (a.wins + a.losses + a.pushes));
  return result;
};

const Predictions: React.FC = () => {
  const theme = useTheme();
  const isDark = theme.palette.mode === 'dark';
  const [rows, setRows] = useState<FlatRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [tab, setTab] = useState(0);
  const [perfTab, setPerfTab] = useState(0);
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(20);
  const [sortField, setSortField] = useState<SortField>('datetime');
  const [sortOrder, setSortOrder] = useState<SortOrder>('asc');
  const [reasonDialog, setReasonDialog] = useState<{ open: boolean; row: FlatRow | null }>({ open: false, row: null });
  const [positiveEdgeOnly, setPositiveEdgeOnly] = useState(false);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const { selectedSport, setSelectedSport, selectedTier, setSelectedTier } = useFilterStore();

  // Summary stats computed from real data
  const graded = rows.filter(r => r.result !== 'pending');
  const wins = rows.filter(r => r.result === 'won').length;
  const losses = rows.filter(r => r.result === 'lost').length;
  const pushes = rows.filter(r => r.result === 'push').length;
  const pending = rows.filter(r => r.result === 'pending').length;
  const summaryStats = {
    totalPredictions: rows.length,
    totalWins: wins, totalLosses: losses, totalPushes: pushes,
    winRate: graded.length > 0 ? Math.round(wins / graded.length * 1000) / 10 : 0,
    avgEdge: rows.length > 0 ? Math.round(rows.reduce((s, r) => s + r.edge, 0) / rows.length * 1000) / 10 : 0,
    avgCLV: 0, roi: 0, bankrollGrowth: 0,
  };

  // Performance data computed from actual rows
  const tierPerf = useMemo(() => buildPerf(rows, r => r.signal_tier,
    k => ({ A: 'Tier A (65%+)', B: 'Tier B (60-65%)', C: 'Tier C (55-60%)', D: 'Tier D (<55%)' }[k] || `Tier ${k}`),
    ['A', 'B', 'C', 'D']), [rows]);
  const sportPerf = useMemo(() => buildPerf(rows, r => r.sport), [rows]);
  const periodPerf = useMemo(() => buildPerf(rows, () => 'Full Game'), [rows]);
  const betTypePerf = useMemo(() => buildPerf(rows, r => r.bet_type,
    k => ({ spread: 'Spreads', total: 'Totals', moneyline: 'Moneyline' }[k] || k)), [rows]);
  const combinedPerf = useMemo(() => buildPerf(rows, r => {
    const bt = r.bet_type === 'spread' ? 'Spread' : r.bet_type === 'total' ? 'Total' : 'ML';
    return `FG ${bt}`;
  }), [rows]);

  const getPerfData = () => [tierPerf, sportPerf, periodPerf, betTypePerf, combinedPerf][perfTab] || tierPerf;

  const loadPredictions = async (showLoading = true) => {
    if (showLoading) setLoading(true);
    try {
      const data = await api.getAllPublicPredictions({ sport: selectedSport !== 'all' ? selectedSport : undefined });
      setRows(transformToFlatRows(data?.predictions || (Array.isArray(data) ? data : [])));
      setLastUpdated(new Date());
    } catch (err) { console.error('Load predictions error:', err); setRows([]); }
    if (showLoading) setLoading(false);
  };

  useEffect(() => { loadPredictions(); }, [selectedSport]);
  useEffect(() => { const iv = setInterval(() => loadPredictions(false), 60000); return () => clearInterval(iv); }, [selectedSport]);

  // Group rows by game_id, sort bet types within each game: spread → total → moneyline
  const BET_ORDER: Record<string, number> = { spread: 0, total: 1, moneyline: 2 };

  interface GameGroup {
    game_id: string;
    sport: string;
    date: string;
    time: string;
    datetime: Date;
    away_team: string;
    home_team: string;
    bets: FlatRow[];
  }

  const groupedGames = useMemo((): GameGroup[] => {
    let f = [...rows];
    if (tab === 1) f = f.filter(r => r.result === 'pending');
    else if (tab === 2) f = f.filter(r => r.result !== 'pending');
    if (selectedTier !== 'all') f = f.filter(r => r.signal_tier === selectedTier);
    if (positiveEdgeOnly) f = f.filter(r => r.edge > 0);

    // Group by game_id
    const map = new Map<string, GameGroup>();
    f.forEach(r => {
      if (!map.has(r.game_id)) {
        map.set(r.game_id, {
          game_id: r.game_id, sport: r.sport, date: r.date, time: r.time,
          datetime: r.datetime, away_team: r.away_team, home_team: r.home_team, bets: [],
        });
      }
      map.get(r.game_id)!.bets.push(r);
    });

    // Sort bet types within each game
    map.forEach(g => g.bets.sort((a, b) => (BET_ORDER[a.bet_type] ?? 9) - (BET_ORDER[b.bet_type] ?? 9)));

    // Sort games
    let games = Array.from(map.values());
    if (sortField) {
      const m = sortOrder === 'asc' ? 1 : -1;
      games.sort((a, b) => {
        switch (sortField) {
          case 'sport': return m * a.sport.localeCompare(b.sport);
          case 'datetime': {
            return m * (a.datetime.getTime() - b.datetime.getTime());
          }
          case 'probability': return m * (Math.max(...a.bets.map(x => x.probability)) - Math.max(...b.bets.map(x => x.probability)));
          case 'edge': return m * (Math.max(...a.bets.map(x => x.edge)) - Math.max(...b.bets.map(x => x.edge)));
          case 'clv': {
            const ac = a.bets.find(x => x.clv != null)?.clv || 0;
            const bc = b.bets.find(x => x.clv != null)?.clv || 0;
            return m * (ac - bc);
          }
          case 'tier': {
            const at = a.bets.reduce((best, x) => x.signal_tier < best ? x.signal_tier : best, 'Z');
            const bt2 = b.bets.reduce((best, x) => x.signal_tier < best ? x.signal_tier : best, 'Z');
            return m * at.localeCompare(bt2);
          }
          default: return 0;
        }
      });
    }
    return games;
  }, [rows, tab, selectedTier, sortField, sortOrder, positiveEdgeOnly]);

  const totalGames = groupedGames.length;
  const totalRows = groupedGames.reduce((s, g) => s + g.bets.length, 0);
  const paginatedGames = useMemo(() => rowsPerPage === -1 ? groupedGames : groupedGames.slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage), [groupedGames, page, rowsPerPage]);
  const handleSort = (field: SortField) => { if (sortField === field) setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc'); else { setSortField(field); setSortOrder('desc'); } };
  const handleResetSort = () => { setSortField('datetime'); setSortOrder('asc'); };

  const formatLine = (value: number | string | undefined | null) => {
    if (value == null || value === '' || value === '-') return '-';
    const num = typeof value === 'string' ? parseFloat(value) : value;
    if (isNaN(num)) return String(value);
    // Round to clean number: integers stay integer, halves stay .5
    const rounded = Math.round(num * 2) / 2;  // Snap to nearest 0.5
    const display = Number.isInteger(rounded) ? rounded.toString() : rounded.toFixed(1);
    return rounded > 0 ? `+${display}` : display;
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
  const lineHdr = { ...hdr, minWidth: 60, textAlign: 'center' };
  const lineCell = { py: 0.75, fontSize: 11, fontFamily: 'monospace', textAlign: 'center', minWidth: 60 };


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
              <StatCard title="Total Predictions" value={summaryStats.totalPredictions.toLocaleString()} subtitle={graded.length > 0 ? `${wins}W - ${losses}L - ${pushes}P` : `${pending} pending`} icon={<Assessment sx={{ fontSize: 18 }} />} color="primary" />
            </Grid>
            <Grid item xs={6} sm={3}>
              <StatCard title="Win Rate" value={graded.length > 0 ? `${summaryStats.winRate}%` : '0%'} subtitle="Across all tiers" icon={<EmojiEvents sx={{ fontSize: 18 }} />} color="success" />
            </Grid>
            <Grid item xs={6} sm={3}>
              <StatCard title="Average Edge" value={`${summaryStats.avgEdge >= 0 ? '+' : ''}${summaryStats.avgEdge}%`} subtitle={`CLV: +${summaryStats.avgCLV}%`} icon={<TrendingUp sx={{ fontSize: 18 }} />} color="info" />
            </Grid>
            <Grid item xs={6} sm={3}>
              <StatCard title="ROI" value={`+${summaryStats.roi}%`} subtitle={`Bankroll: +${summaryStats.bankrollGrowth}%`} icon={<Casino sx={{ fontSize: 18 }} />} color="warning" />
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
          <PerformanceTable data={getPerfData()} isDark={isDark} />
        </CardContent>
      </Card>

      {/* Tabs + Filters Row */}
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', borderBottom: 1, borderColor: 'divider', mb: 1.5 }}>
        <Tabs value={tab} onChange={(_, v) => { setTab(v); setPage(0); }} sx={{ minHeight: 40 }}>
          <Tab label={`All (${totalGames})`} sx={{ fontSize: 12, minHeight: 40, py: 0.5 }} />
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
          <Chip 
            label="+Edge Only" 
            size="small" 
            variant={positiveEdgeOnly ? 'filled' : 'outlined'}
            color={positiveEdgeOnly ? 'success' : 'default'}
            onClick={() => { setPositiveEdgeOnly(!positiveEdgeOnly); setPage(0); }}
            sx={{ fontSize: 11, height: 28, cursor: 'pointer' }}
          />
          <Box sx={{ textAlign: 'right' }}>
            <Typography variant="caption" color="text.secondary" sx={{ fontSize: 11, display: 'block' }}>{totalRows} predictions • {totalGames} games</Typography>
            {lastUpdated && <Typography variant="caption" color="text.secondary" sx={{ fontSize: 10, display: 'block' }}>Updated {lastUpdated.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', timeZone: 'America/Los_Angeles' })} PST</Typography>}
          </Box>
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
                <TableCell sx={hdr}>Time (PST)</TableCell>
                <TableCell sx={hdr} align="center">Game #</TableCell>
                <TableCell sx={hdr}>Team</TableCell>
                <TableCell sx={hdr}>Record</TableCell>
                <TableCell sx={lineHdr}>Circa O</TableCell>
                <TableCell sx={lineHdr}>Circa.</TableCell>
                <TableCell sx={lineHdr}>System O</TableCell>
                <TableCell sx={lineHdr}>System.</TableCell>
                <TableCell sx={hdr}>Pick</TableCell>
                <TableCell sx={hdr} align="center"><TableSortLabel active={sortField === 'probability'} direction={sortField === 'probability' ? sortOrder : 'asc'} onClick={() => handleSort('probability')}>%</TableSortLabel></TableCell>
                <TableCell sx={hdr} align="center"><TableSortLabel active={sortField === 'edge'} direction={sortField === 'edge' ? sortOrder : 'asc'} onClick={() => handleSort('edge')}>Edge</TableSortLabel></TableCell>
                <TableCell sx={hdr} align="center"><TableSortLabel active={sortField === 'clv'} direction={sortField === 'clv' ? sortOrder : 'asc'} onClick={() => handleSort('clv')}>CLV</TableSortLabel></TableCell>
                <TableCell sx={hdr}><TableSortLabel active={sortField === 'tier'} direction={sortField === 'tier' ? sortOrder : 'asc'} onClick={() => handleSort('tier')}>Tier</TableSortLabel></TableCell>
                <TableCell sx={hdr}>W/L</TableCell>
                <TableCell sx={hdr}>
                  <Tooltip title="Reset Sort"><IconButton size="small" onClick={handleResetSort} sx={{ p: 0.25 }}><RestartAlt sx={{ fontSize: 14 }} /></IconButton></Tooltip>
                </TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {paginatedGames.map((game, gameIdx) => {
                const gameNum = page * rowsPerPage + gameIdx + 1;
                const totalBetRows = game.bets.length * 2; // each bet has away + home row
                const bestTier = game.bets.reduce((best, b) => b.signal_tier < best ? b.signal_tier : best, 'Z' as string);
                // Subtle tier background tint
                const tierBg = bestTier === 'A' ? (isDark ? 'rgba(46, 125, 50, 0.06)' : 'rgba(46, 125, 50, 0.04)')
                  : bestTier === 'B' ? (isDark ? 'rgba(2, 136, 209, 0.06)' : 'rgba(2, 136, 209, 0.04)')
                  : undefined;

                return (
                  <React.Fragment key={game.game_id}>
                    {game.bets.map((row, betIdx) => {
                      const isFirstBet = betIdx === 0;
                      const isLastBet = betIdx === game.bets.length - 1;
                      const gameBorderSx = isLastBet ? { borderBottom: 2, borderColor: 'divider' } : {};
                      const betDivider = !isLastBet ? { borderBottom: 1, borderColor: 'action.hover' } : gameBorderSx;
                      const isNegEdge = row.edge < 0;
                      const pickHighlight = (team: 'away' | 'home') => row.pick_team === team ? {
                        fontWeight: 700,
                        bgcolor: isDark ? 'rgba(46, 125, 50, 0.12)' : 'rgba(46, 125, 50, 0.08)',
                      } : { fontWeight: 400 };

                      return (
                        <React.Fragment key={row.id}>
                          {/* Away Row */}
                          <TableRow sx={{ bgcolor: tierBg }}>
                            {isFirstBet && <>
                              <TableCell rowSpan={totalBetRows} sx={{ py: 0.75, fontSize: 11, verticalAlign: 'middle', fontWeight: 600, borderBottom: 2, borderColor: 'divider' }}>{row.sport}</TableCell>
                              <TableCell rowSpan={totalBetRows} sx={{ py: 0.75, fontSize: 11, verticalAlign: 'middle', borderBottom: 2, borderColor: 'divider' }}>{row.date}</TableCell>
                              <TableCell rowSpan={totalBetRows} sx={{ py: 0.75, fontSize: 11, verticalAlign: 'middle', borderBottom: 2, borderColor: 'divider' }}>{row.time}</TableCell>
                              <TableCell align="center" rowSpan={totalBetRows} sx={{ py: 0.75, fontSize: 11, fontFamily: 'monospace', fontWeight: 600, verticalAlign: 'middle', borderBottom: 2, borderColor: 'divider' }}>{gameNum}</TableCell>
                            </>}
                            <TableCell sx={{ py: 0.75, fontSize: 11, borderBottom: 0, ...pickHighlight('away') }}>{row.away_team}</TableCell>
                            <TableCell sx={{ py: 0.75, fontSize: 11, color: 'text.secondary', borderBottom: 0 }}>{row.away_record}</TableCell>
                            <TableCell align="center" sx={{ ...lineCell, borderBottom: 0 }}>{formatLine(row.away_circa_open)}</TableCell>
                            <TableCell align="center" sx={{ ...lineCell, fontWeight: 600, borderBottom: 0 }}>{formatLine(row.away_circa_current)}</TableCell>
                            <TableCell align="center" sx={{ ...lineCell, color: 'info.main', borderBottom: 0 }}>{formatLine(row.away_system_open)}</TableCell>
                            <TableCell align="center" sx={{ ...lineCell, color: 'info.main', fontWeight: 600, borderBottom: 0 }}>{formatLine(row.away_system_current)}</TableCell>
                            <TableCell rowSpan={2} sx={{ py: 0.75, fontSize: 11, verticalAlign: 'middle', ...betDivider }}>
                              <Typography sx={{ fontSize: 11, fontWeight: 600, color: isNegEdge ? 'text.secondary' : 'success.main', lineHeight: 1.3 }}>{row.system_pick}</Typography>
                              <Typography sx={{ fontSize: 10, color: 'text.secondary' }}>{row.bet_type_label}</Typography>
                            </TableCell>
                            <TableCell rowSpan={2} align="center" sx={{ py: 0.75, fontSize: 11, verticalAlign: 'middle', ...betDivider }}>{formatPercent(row.probability)}</TableCell>
                            <TableCell rowSpan={2} align="center" sx={{ py: 0.75, fontSize: 11, verticalAlign: 'middle', ...betDivider, color: isNegEdge ? 'error.main' : (row.edge * 100) >= 3 ? 'success.main' : (row.edge * 100) >= 1 ? 'warning.main' : 'text.secondary' }}>
                              {row.edge >= 0 ? '+' : ''}{(row.edge * 100).toFixed(1)}%
                            </TableCell>
                            <TableCell rowSpan={2} align="center" sx={{ py: 0.75, fontSize: 11, verticalAlign: 'middle', ...betDivider, color: row.clv != null ? (row.clv > 0 ? 'success.main' : row.clv < 0 ? 'error.main' : 'inherit') : 'inherit' }}>{row.clv != null ? `${row.clv > 0 ? '+' : ''}${row.clv.toFixed(1)}%` : '-'}</TableCell>
                            <TableCell rowSpan={2} sx={{ py: 0.75, fontSize: 11, verticalAlign: 'middle', ...betDivider }}><TierBadge tier={row.signal_tier} /></TableCell>
                            <TableCell rowSpan={2} sx={{ py: 0.75, fontSize: 11, verticalAlign: 'middle', ...betDivider }}>{getStatusChip(row.result)}</TableCell>
                            <TableCell rowSpan={2} sx={{ py: 0.75, fontSize: 11, verticalAlign: 'middle', ...betDivider }}><IconButton size="small" onClick={() => setReasonDialog({ open: true, row })}><ExpandMore /></IconButton></TableCell>
                          </TableRow>
                          {/* Home Row */}
                          <TableRow sx={{ bgcolor: tierBg }}>
                            <TableCell sx={{ py: 0.75, fontSize: 11, ...betDivider, ...pickHighlight('home') }}>{row.home_team}</TableCell>
                            <TableCell sx={{ py: 0.75, fontSize: 11, color: 'text.secondary', ...betDivider }}>{row.home_record}</TableCell>
                            <TableCell align="center" sx={{ ...lineCell, ...betDivider }}>{formatLine(row.home_circa_open)}</TableCell>
                            <TableCell align="center" sx={{ ...lineCell, fontWeight: 600, ...betDivider }}>{formatLine(row.home_circa_current)}</TableCell>
                            <TableCell align="center" sx={{ ...lineCell, color: 'info.main', ...betDivider }}>{formatLine(row.home_system_open)}</TableCell>
                            <TableCell align="center" sx={{ ...lineCell, color: 'info.main', fontWeight: 600, ...betDivider }}>{formatLine(row.home_system_current)}</TableCell>
                          </TableRow>
                        </React.Fragment>
                      );
                    })}
                  </React.Fragment>
                );
              })}
              {paginatedGames.length === 0 && !loading && (
                <TableRow><TableCell colSpan={17} align="center" sx={{ py: 4 }}><Typography color="text.secondary">No predictions found</Typography></TableCell></TableRow>
              )}
            </TableBody>
          </Table>
        </TableContainer>
        <TablePagination component="div" count={totalGames} page={page} onPageChange={(_, p) => setPage(p)} rowsPerPage={rowsPerPage}
          onRowsPerPageChange={(e) => { setRowsPerPage(parseInt(e.target.value)); setPage(0); }}
          rowsPerPageOptions={[20, 50, 100, { value: -1, label: 'All' }]} labelRowsPerPage="Games per page:" />
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
                <Grid item xs={4}><Typography variant="caption" color="text.secondary">Edge</Typography><Typography variant="h6" fontWeight={700} color={reasonDialog.row.edge >= 0 ? 'success.main' : 'error.main'}>{reasonDialog.row.edge >= 0 ? '+' : ''}{(reasonDialog.row.edge * 100).toFixed(1)}%</Typography></Grid>
                <Grid item xs={4}><Typography variant="caption" color="text.secondary">CLV</Typography><Typography variant="h6" fontWeight={700}>{reasonDialog.row.clv != null ? `${reasonDialog.row.clv > 0 ? '+' : ''}${reasonDialog.row.clv.toFixed(1)}%` : '-'}</Typography></Grid>
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

// =============================================================================
// Transform API response → FlatRows with opening vs current line mapping
// =============================================================================
const transformToFlatRows = (data: any[]): FlatRow[] => {
  if (!data || data.length === 0) return [];
  return data.map((pred: any, idx: number) => {
    const gameTime = pred.game_time ? new Date(pred.game_time) : new Date();
    const dateStr = gameTime.toLocaleDateString('en-US', { month: 'numeric', day: 'numeric', year: 'numeric', timeZone: 'America/Los_Angeles' });
    const timeStr = gameTime.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', hour12: true, timeZone: 'America/Los_Angeles' });
    const side = pred.predicted_side || '';
    const bt = pred.bet_type || 'spread';
    const line = pred.line_at_prediction;
    const odds = pred.odds_at_prediction;

    // Helper: snap to nearest 0.5 for clean display
    const snapLine = (v: number | null | undefined): number | null => {
      if (v == null) return null;
      return Math.round(v * 2) / 2;
    };
    const fmtNum = (v: number) => Number.isInteger(v) ? v.toString() : v.toFixed(1);

    // Build pick string
    let pickStr = side;
    let pickTeam: 'away' | 'home' | null = null;
    if (bt === 'spread') {
      const team = side === 'home' ? pred.home_team : pred.away_team;
      const snapped = snapLine(line);
      pickStr = snapped != null ? `${team} ${snapped > 0 ? '+' : ''}${fmtNum(snapped)}` : (team || side);
      pickTeam = side === 'home' ? 'home' : side === 'away' ? 'away' : null;
    } else if (bt === 'total') {
      const snapped = snapLine(line);
      pickStr = snapped != null ? `${side === 'over' ? 'Over' : 'Under'} ${fmtNum(snapped)}` : (side === 'over' ? 'Over' : 'Under');
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
    const prob = pred.probability || 0.5;
    const edge = pred.edge != null ? pred.edge : 0;

    // ---- MAP LINE COLUMNS: Opening (Circa O) vs Current (Circa.) ----
    // Circa O / System O = opening snapshot (from predictions table)
    // Circa. / System. = current consensus (from upcoming_odds via API)
    let awayOpen: number | string = '-';
    let homeOpen: number | string = '-';
    let awayCurrent: number | string = '-';
    let homeCurrent: number | string = '-';

    if (bt === 'spread') {
      // Opening: from predictions snapshot columns
      awayOpen = pred.away_line_open ?? '-';
      homeOpen = pred.home_line_open ?? '-';
      // Current: from upcoming_odds consensus
      awayCurrent = pred.current_away_line ?? '-';
      homeCurrent = pred.current_home_line ?? '-';
      // Fallback: if opening snapshot missing, derive from line_at_prediction
      if (awayOpen === '-' && line != null) {
        if (side === 'away') { awayOpen = line; homeOpen = line * -1; }
        else if (side === 'home') { homeOpen = line; awayOpen = line * -1; }
      }
      // Fallback: if current missing, use opening
      if (awayCurrent === '-') awayCurrent = awayOpen;
      if (homeCurrent === '-') homeCurrent = homeOpen;
    } else if (bt === 'total') {
      // Opening
      awayOpen = pred.total_open ?? line ?? '-';
      homeOpen = awayOpen;
      // Current
      awayCurrent = pred.current_total ?? awayOpen;
      homeCurrent = awayCurrent;
    } else if (bt === 'moneyline') {
      // Opening: ML odds for each side
      awayOpen = pred.away_ml_open ?? '-';
      homeOpen = pred.home_ml_open ?? '-';
      // Current
      awayCurrent = pred.current_away_ml ?? awayOpen;
      homeCurrent = pred.current_home_ml ?? homeOpen;
      // Fallback: if opening missing, use odds_at_prediction for predicted side
      if (awayOpen === '-' && side === 'away' && odds != null) awayOpen = odds;
      if (homeOpen === '-' && side === 'home' && odds != null) homeOpen = odds;
    }

    // ---- SYSTEM COLUMNS: Model-projected fair lines/odds ----
    // Convert model probability → system-implied lines/odds
    // Moneyline: probability → American odds directly
    // Spread: shift market line by model edge (each ~3% edge ≈ 1 point)
    // Total: shift market total by model edge
    const probToAmerican = (p: number): number => {
      if (p <= 0 || p >= 1) return 0;
      if (p >= 0.5) return Math.round(-100 * p / (1 - p));
      else return Math.round(100 * (1 - p) / p);
    };

    let awaySystemOpen: number | string = awayOpen;
    let homeSystemOpen: number | string = homeOpen;
    let awaySystemCurrent: number | string = awayCurrent;
    let homeSystemCurrent: number | string = homeCurrent;

    if (edge !== 0 && prob > 0 && prob < 1) {
      if (bt === 'moneyline') {
        // Predicted side gets model prob, other side gets 1-prob
        const predictedOdds = probToAmerican(prob);
        const otherOdds = probToAmerican(1 - prob);
        if (side === 'home') {
          homeSystemOpen = predictedOdds;
          homeSystemCurrent = predictedOdds;
          awaySystemOpen = otherOdds;
          awaySystemCurrent = otherOdds;
        } else {
          awaySystemOpen = predictedOdds;
          awaySystemCurrent = predictedOdds;
          homeSystemOpen = otherOdds;
          homeSystemCurrent = otherOdds;
        }
      } else if (bt === 'spread') {
        // Shift spread: positive edge for predicted side = line moves in their favor
        // ~3% probability per point of spread
        let lineShift = Math.round((edge * 100 / 3) * 2) / 2; // snap to 0.5
        lineShift = Math.max(-5, Math.min(5, lineShift)); // cap ±5 points
        const hOpen = typeof homeOpen === 'number' ? homeOpen : 0;
        const aOpen = typeof awayOpen === 'number' ? awayOpen : 0;
        if (side === 'home') {
          // Model thinks home is stronger → home spread more negative
          homeSystemOpen = hOpen - lineShift;
          awaySystemOpen = -(homeSystemOpen as number);
        } else {
          // Model thinks away is stronger → away spread more negative (home more positive)
          awaySystemOpen = aOpen - lineShift;
          homeSystemOpen = -(awaySystemOpen as number);
        }
        homeSystemCurrent = homeSystemOpen;
        awaySystemCurrent = awaySystemOpen;
      } else if (bt === 'total') {
        // Shift total: over edge = model thinks total should be higher
        let totalShift = Math.round((edge * 100 / 3) * 2) / 2; // snap to 0.5
        totalShift = Math.max(-5, Math.min(5, totalShift)); // cap ±5 points
        const tOpen = typeof homeOpen === 'number' ? homeOpen : 0;
        if (side === 'over') {
          // Model thinks over → fair total is higher
          const systemTotal = tOpen + totalShift;
          awaySystemOpen = systemTotal;
          homeSystemOpen = systemTotal;
        } else {
          // Model thinks under → fair total is lower
          const systemTotal = tOpen - totalShift;
          awaySystemOpen = systemTotal;
          homeSystemOpen = systemTotal;
        }
        awaySystemCurrent = awaySystemOpen;
        homeSystemCurrent = homeSystemOpen;
      }
    }

    return {
      id: pred.id || `pred_${idx}`,
      game_id: pred.game_id || `game_${idx}`,
      sport: pred.sport || pred.sport_code || 'UNK',
      date: dateStr,
      time: timeStr,
      datetime: gameTime,
      away_rotation: '',
      away_team: pred.away_team || 'TBD',
      away_record: pred.away_record || '',
      home_rotation: '',
      home_team: pred.home_team || 'TBD',
      home_record: pred.home_record || '',
      bet_type: bt,
      bet_type_label: btLabel,
      // Circa columns (market)
      away_circa_open: awayOpen,
      away_circa_current: awayCurrent,
      home_circa_open: homeOpen,
      home_circa_current: homeCurrent,
      // System columns (model-projected, same as Circa for now)
      away_system_open: awaySystemOpen,
      away_system_current: awaySystemCurrent,
      home_system_open: homeSystemOpen,
      home_system_current: homeSystemCurrent,
      system_pick: pickStr,
      pick_team: pickTeam,
      probability: prob,
      edge,
      clv: pred.clv != null ? pred.clv : null,
      signal_tier: (pred.signal_tier || 'D') as 'A' | 'B' | 'C' | 'D',
      result,
      reason: (() => {
        const edgePct = (edge * 100).toFixed(1);
        const probPct = Math.round(prob * 100);
        const marketProb = Math.round((prob - edge) * 100);
        if (bt === 'moneyline') {
          return `Model assigns ${probPct}% win probability vs market-implied ${marketProb}%. Edge: ${edgePct}% → System fair odds: ${typeof homeSystemCurrent === 'number' ? homeSystemCurrent : '?'} / ${typeof awaySystemCurrent === 'number' ? awaySystemCurrent : '?'}`;
        } else if (bt === 'spread') {
          const marketLine = side === 'home' ? homeOpen : awayOpen;
          const systemLine = side === 'home' ? homeSystemCurrent : awaySystemCurrent;
          return `Model gives ${probPct}% to cover ${typeof marketLine === 'number' ? (marketLine > 0 ? '+' : '') + marketLine : '?'} (market implies ${marketProb}%). System fair line: ${typeof systemLine === 'number' ? (systemLine > 0 ? '+' : '') + systemLine : '?'}. Edge: ${edgePct}%`;
        } else {
          const marketTotal = typeof homeOpen === 'number' ? homeOpen : 0;
          const systemTotal = typeof homeSystemCurrent === 'number' ? homeSystemCurrent : 0;
          return `Model gives ${probPct}% for ${side === 'over' ? 'Over' : 'Under'} ${marketTotal} (market implies ${marketProb}%). System fair total: ${systemTotal}. Edge: ${edgePct}%`;
        }
      })(),
    };
  });
};

export default Predictions;