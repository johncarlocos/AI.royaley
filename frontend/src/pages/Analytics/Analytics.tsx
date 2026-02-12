// src/pages/Analytics/Analytics.tsx - Real data from betting-summary endpoint
import React, { useState, useEffect, useCallback, useMemo } from 'react';
import {
  Box, Card, CardContent, Typography, Grid, Table, TableBody, TableCell,
  TableContainer, TableHead, TableRow, Chip, Select, MenuItem, FormControl,
  InputLabel, Button, TableSortLabel, LinearProgress, useTheme, TablePagination
} from '@mui/material';
import { Refresh, CheckCircle, Cancel, Schedule, Remove } from '@mui/icons-material';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, LineChart, Line, ReferenceLine
} from 'recharts';
import { api } from '../../api/client';

// ─── Types ───────────────────────────────────────────────────────────
interface Bet {
  id: string;
  game_id: string;
  sport: string | null;
  home_team: string | null;
  away_team: string | null;
  game_time: string | null;
  bet_type: string;
  predicted_side: string;
  pick_team: string | null;
  line: number | null;
  odds: number | null;
  probability: number;
  edge: number | null;
  signal_tier: string | null;
  stake: number;
  result: string;
  profit_loss: number | null;
  clv: number | null;
}

type SortField = 'sport' | 'bet_type' | 'probability' | 'edge' | 'clv' | 'profit_loss';

// ─── Helpers ─────────────────────────────────────────────────────────
const TierBadge: React.FC<{ tier: string }> = ({ tier }) => {
  const colors: Record<string, { bg: string; color: string }> = {
    'A': { bg: '#4caf50', color: '#fff' },
    'B': { bg: '#2196f3', color: '#fff' },
    'C': { bg: '#ff9800', color: '#fff' },
    'D': { bg: '#9e9e9e', color: '#fff' },
  };
  const style = colors[tier] || colors['D'];
  return <Chip label={tier} size="small" sx={{ bgcolor: style.bg, color: style.color, fontWeight: 700, minWidth: 28 }} />;
};

const TIER_COLORS: Record<string, string> = { A: '#4caf50', B: '#2196f3', C: '#ff9800', D: '#9e9e9e' };

const formatPnL = (v: number) => {
  const sign = v > 0 ? '+' : '';
  return `${sign}$${Math.abs(v).toFixed(2)}`;
};

const formatGameTime = (gt: string | null) => {
  if (!gt) return { date: '-', time: '-' };
  const d = new Date(gt);
  return {
    date: d.toLocaleDateString('en-US', { month: 'numeric', day: 'numeric', year: 'numeric', timeZone: 'America/Los_Angeles' }),
    time: d.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', timeZone: 'America/Los_Angeles' }),
  };
};

// ─── Component ───────────────────────────────────────────────────────
const Analytics: React.FC = () => {
  const theme = useTheme();
  const isDark = theme.palette.mode === 'dark';

  const [allBets, setAllBets] = useState<Bet[]>([]);
  const [equityCurve, setEquityCurve] = useState<{ date: string; value: number }[]>([]);
  const [loading, setLoading] = useState(true);

  // Filters
  const [sportFilter, setSportFilter] = useState('all');
  const [typeFilter, setTypeFilter] = useState('all');
  const [tierFilter, setTierFilter] = useState('all');
  const [resultFilter, setResultFilter] = useState('all');
  const [edgeFilter, setEdgeFilter] = useState('all');

  // Sort & pagination
  const [sortField, setSortField] = useState<SortField>('edge');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(25);

  // ─── Load Data ──────────────────────────────────────────────────
  const loadData = useCallback(async () => {
    setLoading(true);
    try {
      const data = await api.getBettingSummary({ tiers: 'A,B,C,D', stake: 100, initial_bankroll: 10000 });
      if (data.bets) setAllBets(data.bets);
      if (data.equity_curve) setEquityCurve(data.equity_curve);
    } catch (err) {
      console.error('Analytics load error:', err);
    }
    setLoading(false);
  }, []);

  useEffect(() => { loadData(); }, [loadData]);

  // ─── Derived: Available filter options ──────────────────────────
  const availableSports = useMemo(() =>
    Array.from(new Set(allBets.map(b => b.sport).filter(Boolean))).sort() as string[],
    [allBets]
  );

  const availableTypes = useMemo(() =>
    Array.from(new Set(allBets.map(b => b.bet_type).filter(Boolean))).sort() as string[],
    [allBets]
  );

  // ─── Filtered bets ─────────────────────────────────────────────
  const filtered = useMemo(() => allBets.filter(b => {
    if (sportFilter !== 'all' && b.sport !== sportFilter) return false;
    if (typeFilter !== 'all' && b.bet_type !== typeFilter) return false;
    if (tierFilter !== 'all' && b.signal_tier !== tierFilter) return false;
    if (resultFilter === 'graded' && b.result === 'pending') return false;
    if (resultFilter === 'wins' && b.result !== 'win') return false;
    if (resultFilter === 'losses' && b.result !== 'loss') return false;
    if (resultFilter === 'pending' && b.result !== 'pending') return false;
    if (edgeFilter === '3+' && ((b.edge || 0) * 100) < 3) return false;
    if (edgeFilter === '5+' && ((b.edge || 0) * 100) < 5) return false;
    if (edgeFilter === '10+' && ((b.edge || 0) * 100) < 10) return false;
    return true;
  }), [allBets, sportFilter, typeFilter, tierFilter, resultFilter, edgeFilter]);

  // ─── Filtered stats ────────────────────────────────────────────
  const fStats = useMemo(() => {
    const graded = filtered.filter(b => b.result !== 'pending');
    const wins = graded.filter(b => b.result === 'win').length;
    const losses = graded.filter(b => b.result === 'loss').length;
    const pushes = graded.filter(b => b.result === 'push').length;
    const decided = wins + losses;
    const totalPnl = graded.reduce((s, b) => s + (b.profit_loss || 0), 0);
    const avgEdge = filtered.length > 0
      ? filtered.reduce((s, b) => s + (b.edge || 0), 0) / filtered.length * 100 : 0;
    const clvBets = graded.filter(b => b.clv != null);
    const avgClv = clvBets.length > 0
      ? clvBets.reduce((s, b) => s + (b.clv || 0), 0) / clvBets.length * 100 : 0;
    return {
      total: filtered.length, graded: graded.length, pending: filtered.length - graded.length,
      wins, losses, pushes,
      winRate: decided > 0 ? (wins / decided * 100) : 0,
      roi: graded.length > 0 ? (totalPnl / (graded.length * 100) * 100) : 0,
      avgEdge, avgClv, totalPnl,
    };
  }, [filtered]);

  // ─── Chart Data ────────────────────────────────────────────────
  const sportPerf = useMemo(() => {
    const sports = Array.from(new Set(filtered.map(b => b.sport).filter(Boolean))) as string[];
    return sports.map(s => {
      const sb = filtered.filter(b => b.sport === s);
      const gr = sb.filter(b => b.result !== 'pending');
      const w = gr.filter(b => b.result === 'win').length;
      const d = gr.filter(b => b.result === 'win' || b.result === 'loss').length;
      const pnl = gr.reduce((sum, b) => sum + (b.profit_loss || 0), 0);
      return { sport: s, bets: sb.length, graded: gr.length, wins: w, winRate: d > 0 ? Math.round(w / d * 100) : 0, pnl: Math.round(pnl * 100) / 100, avgEdge: sb.length > 0 ? Math.round(sb.reduce((sum, b) => sum + (b.edge || 0), 0) / sb.length * 1000) / 10 : 0 };
    }).sort((a, b) => b.bets - a.bets);
  }, [filtered]);

  const tierDist = useMemo(() =>
    ['A', 'B', 'C', 'D'].map(t => ({ name: `Tier ${t}`, value: filtered.filter(b => b.signal_tier === t).length, tier: t })).filter(d => d.value > 0),
    [filtered]
  );

  const tierPerf = useMemo(() =>
    ['A', 'B', 'C', 'D'].map(t => {
      const tb = filtered.filter(b => b.signal_tier === t);
      const gr = tb.filter(b => b.result !== 'pending');
      const w = gr.filter(b => b.result === 'win').length;
      const d = gr.filter(b => b.result === 'win' || b.result === 'loss').length;
      const pnl = gr.reduce((sum, b) => sum + (b.profit_loss || 0), 0);
      return { tier: t, bets: tb.length, graded: gr.length, wins: w, winRate: d > 0 ? Math.round(w / d * 100) : 0, pnl: Math.round(pnl * 100) / 100, avgEdge: tb.length > 0 ? Math.round(tb.reduce((sum, b) => sum + (b.edge || 0), 0) / tb.length * 1000) / 10 : 0 };
    }).filter(d => d.bets > 0),
    [filtered]
  );

  const typePerf = useMemo(() => {
    const types = Array.from(new Set(filtered.map(b => b.bet_type).filter(Boolean))) as string[];
    return types.map(t => {
      const tb = filtered.filter(b => b.bet_type === t);
      const gr = tb.filter(b => b.result !== 'pending');
      const w = gr.filter(b => b.result === 'win').length;
      const d = gr.filter(b => b.result === 'win' || b.result === 'loss').length;
      const pnl = gr.reduce((sum, b) => sum + (b.profit_loss || 0), 0);
      return { type: t === 'moneyline' ? 'ML' : t === 'spread' ? 'Spread' : t === 'total' ? 'Total' : t, bets: tb.length, wins: w, winRate: d > 0 ? Math.round(w / d * 100) : 0, pnl: Math.round(pnl * 100) / 100 };
    });
  }, [filtered]);

  // ─── Sort & paginate ───────────────────────────────────────────
  const sorted = useMemo(() => [...filtered].sort((a, b) => {
    let aV: string | number, bV: string | number;
    switch (sortField) {
      case 'sport': aV = a.sport || ''; bV = b.sport || ''; break;
      case 'bet_type': aV = a.bet_type || ''; bV = b.bet_type || ''; break;
      case 'probability': aV = a.probability; bV = b.probability; break;
      case 'edge': aV = a.edge || 0; bV = b.edge || 0; break;
      case 'clv': aV = a.clv || 0; bV = b.clv || 0; break;
      case 'profit_loss': aV = a.profit_loss || 0; bV = b.profit_loss || 0; break;
      default: aV = a.edge || 0; bV = b.edge || 0;
    }
    if (typeof aV === 'string') return sortOrder === 'asc' ? aV.localeCompare(bV as string) : (bV as string).localeCompare(aV);
    return sortOrder === 'asc' ? (aV as number) - (bV as number) : (bV as number) - (aV as number);
  }), [filtered, sortField, sortOrder]);

  const paginatedBets = sorted.slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage);

  const handleSort = (field: SortField) => {
    if (sortField === field) setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
    else { setSortField(field); setSortOrder('desc'); }
  };

  const getStatusChip = (status: string) => {
    switch (status) {
      case 'win': return <Chip icon={<CheckCircle />} label="W" size="small" color="success" />;
      case 'loss': return <Chip icon={<Cancel />} label="L" size="small" color="error" />;
      case 'push': return <Chip icon={<Remove />} label="P" size="small" />;
      default: return <Chip icon={<Schedule />} label="-" size="small" color="primary" />;
    }
  };

  const formatBetType = (bt: string) => bt === 'spread' ? 'Spread' : bt === 'total' ? 'Total' : bt === 'moneyline' ? 'ML' : bt;

  const hdr = { fontWeight: 600, fontSize: 11, py: 0.75, bgcolor: isDark ? 'grey.900' : 'grey.100', color: isDark ? 'grey.100' : 'grey.800', whiteSpace: 'nowrap', borderBottom: 1, borderColor: 'divider' };
  const csx = { height: '100%' };
  const ccsx = { textAlign: 'center', py: 1.5, '&:last-child': { pb: 1.5 }, height: '100%', display: 'flex', flexDirection: 'column', justifyContent: 'center' };

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
        <Typography variant="h5" fontWeight={700} sx={{ fontSize: 20 }}>Analytics</Typography>
        <Button variant="outlined" size="small" startIcon={<Refresh />} onClick={loadData} sx={{ fontSize: 11 }}>Refresh</Button>
      </Box>

      {/* ─── Stat Cards ─────────────────────────────────────────── */}
      <Grid container spacing={2} mb={2} sx={{ alignItems: 'stretch' }}>
        <Grid item xs={6} sm={4} md={2}>
          <Card sx={csx}><CardContent sx={ccsx}>
            <Typography sx={{ fontSize: 10, color: 'text.secondary' }}>Total Predictions</Typography>
            <Typography sx={{ fontSize: 18, fontWeight: 700 }}>{fStats.total}</Typography>
            <Typography sx={{ fontSize: 9, color: 'text.secondary' }}>{fStats.graded} graded • {fStats.pending} pending</Typography>
          </CardContent></Card>
        </Grid>
        <Grid item xs={6} sm={4} md={2}>
          <Card sx={csx}><CardContent sx={ccsx}>
            <Typography sx={{ fontSize: 10, color: 'text.secondary' }}>Win Rate</Typography>
            <Typography sx={{ fontSize: 18, fontWeight: 700, color: fStats.winRate >= 55 ? 'success.main' : fStats.winRate > 0 ? 'warning.main' : 'text.primary' }}>
              {fStats.graded > 0 ? `${fStats.winRate.toFixed(1)}%` : '-'}
            </Typography>
            <Typography sx={{ fontSize: 9, color: 'text.secondary' }}>{fStats.wins}W-{fStats.losses}L-{fStats.pushes}P</Typography>
          </CardContent></Card>
        </Grid>
        <Grid item xs={6} sm={4} md={2}>
          <Card sx={csx}><CardContent sx={ccsx}>
            <Typography sx={{ fontSize: 10, color: 'text.secondary' }}>ROI</Typography>
            <Typography sx={{ fontSize: 18, fontWeight: 700, color: fStats.roi >= 0 ? 'success.main' : 'error.main' }}>
              {fStats.graded > 0 ? `${fStats.roi >= 0 ? '+' : ''}${fStats.roi.toFixed(1)}%` : '-'}
            </Typography>
          </CardContent></Card>
        </Grid>
        <Grid item xs={6} sm={4} md={2}>
          <Card sx={csx}><CardContent sx={ccsx}>
            <Typography sx={{ fontSize: 10, color: 'text.secondary' }}>Avg Edge</Typography>
            <Typography sx={{ fontSize: 18, fontWeight: 700, color: fStats.avgEdge > 0 ? 'success.main' : 'text.primary' }}>
              {fStats.avgEdge >= 0 ? '+' : ''}{fStats.avgEdge.toFixed(1)}%
            </Typography>
          </CardContent></Card>
        </Grid>
        <Grid item xs={6} sm={4} md={2}>
          <Card sx={csx}><CardContent sx={ccsx}>
            <Typography sx={{ fontSize: 10, color: 'text.secondary' }}>Avg CLV</Typography>
            <Typography sx={{ fontSize: 18, fontWeight: 700, color: fStats.avgClv > 0 ? 'success.main' : fStats.avgClv < 0 ? 'error.main' : 'text.primary' }}>
              {fStats.avgClv !== 0 ? `${fStats.avgClv >= 0 ? '+' : ''}${fStats.avgClv.toFixed(2)}%` : '-'}
            </Typography>
          </CardContent></Card>
        </Grid>
        <Grid item xs={6} sm={4} md={2}>
          <Card sx={csx}><CardContent sx={ccsx}>
            <Typography sx={{ fontSize: 10, color: 'text.secondary' }}>P/L</Typography>
            <Typography sx={{ fontSize: 18, fontWeight: 700, color: fStats.totalPnl >= 0 ? 'success.main' : 'error.main' }}>
              {fStats.graded > 0 ? `${fStats.totalPnl >= 0 ? '+' : ''}$${Math.abs(fStats.totalPnl).toFixed(2)}` : '-'}
            </Typography>
          </CardContent></Card>
        </Grid>
      </Grid>

      {/* ─── Filters ────────────────────────────────────────────── */}
      <Card sx={{ mb: 2 }}>
        <CardContent sx={{ py: 1.5, px: 2 }}>
          <Typography variant="body2" fontWeight={500} gutterBottom sx={{ fontSize: 12 }}>Filter & Sort</Typography>
          <Grid container spacing={2}>
            <Grid item xs={6} sm={4} md>
              <FormControl fullWidth size="small">
                <InputLabel sx={{ fontSize: 12 }}>Sport</InputLabel>
                <Select value={sportFilter} label="Sport" onChange={(e) => { setSportFilter(e.target.value); setPage(0); }} sx={{ fontSize: 12 }}>
                  <MenuItem value="all" sx={{ fontSize: 12 }}>All Sports</MenuItem>
                  {availableSports.map(s => <MenuItem key={s} value={s} sx={{ fontSize: 12 }}>{s}</MenuItem>)}
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={6} sm={4} md>
              <FormControl fullWidth size="small">
                <InputLabel sx={{ fontSize: 12 }}>Type</InputLabel>
                <Select value={typeFilter} label="Type" onChange={(e) => { setTypeFilter(e.target.value); setPage(0); }} sx={{ fontSize: 12 }}>
                  <MenuItem value="all" sx={{ fontSize: 12 }}>All Types</MenuItem>
                  {availableTypes.map(t => <MenuItem key={t} value={t} sx={{ fontSize: 12 }}>{formatBetType(t)}</MenuItem>)}
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={6} sm={4} md>
              <FormControl fullWidth size="small">
                <InputLabel sx={{ fontSize: 12 }}>Tier</InputLabel>
                <Select value={tierFilter} label="Tier" onChange={(e) => { setTierFilter(e.target.value); setPage(0); }} sx={{ fontSize: 12 }}>
                  <MenuItem value="all" sx={{ fontSize: 12 }}>All Tiers</MenuItem>
                  <MenuItem value="A" sx={{ fontSize: 12 }}>Tier A (65%+)</MenuItem>
                  <MenuItem value="B" sx={{ fontSize: 12 }}>Tier B (60-65%)</MenuItem>
                  <MenuItem value="C" sx={{ fontSize: 12 }}>Tier C (55-60%)</MenuItem>
                  <MenuItem value="D" sx={{ fontSize: 12 }}>Tier D (&lt;55%)</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={6} sm={4} md>
              <FormControl fullWidth size="small">
                <InputLabel sx={{ fontSize: 12 }}>Result</InputLabel>
                <Select value={resultFilter} label="Result" onChange={(e) => { setResultFilter(e.target.value); setPage(0); }} sx={{ fontSize: 12 }}>
                  <MenuItem value="all" sx={{ fontSize: 12 }}>All</MenuItem>
                  <MenuItem value="graded" sx={{ fontSize: 12 }}>Graded Only</MenuItem>
                  <MenuItem value="wins" sx={{ fontSize: 12 }}>Wins</MenuItem>
                  <MenuItem value="losses" sx={{ fontSize: 12 }}>Losses</MenuItem>
                  <MenuItem value="pending" sx={{ fontSize: 12 }}>Pending</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={6} sm={4} md>
              <FormControl fullWidth size="small">
                <InputLabel sx={{ fontSize: 12 }}>Edge</InputLabel>
                <Select value={edgeFilter} label="Edge" onChange={(e) => { setEdgeFilter(e.target.value); setPage(0); }} sx={{ fontSize: 12 }}>
                  <MenuItem value="all" sx={{ fontSize: 12 }}>All</MenuItem>
                  <MenuItem value="3+" sx={{ fontSize: 12 }}>3%+</MenuItem>
                  <MenuItem value="5+" sx={{ fontSize: 12 }}>5%+</MenuItem>
                  <MenuItem value="10+" sx={{ fontSize: 12 }}>10%+</MenuItem>
                </Select>
              </FormControl>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* ─── Charts Row 1: Sport Performance + Tier Distribution ── */}
      <Grid container spacing={2} mb={2}>
        <Grid item xs={12} md={8}>
          <Card sx={{ height: '100%' }}>
            <CardContent sx={{ py: 1.5, px: 2 }}>
              <Typography variant="subtitle1" fontWeight={600} gutterBottom sx={{ fontSize: 13 }}>Performance by Sport</Typography>
              {sportPerf.length > 0 ? (
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart data={sportPerf} margin={{ top: 5, right: 5, bottom: 0, left: -10 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                    <XAxis dataKey="sport" stroke="#666" fontSize={10} />
                    <YAxis stroke="#666" fontSize={10} />
                    <Tooltip contentStyle={{ backgroundColor: isDark ? '#1e1e1e' : '#fff', border: '1px solid #333', fontSize: 11 }} />
                    <Bar dataKey="bets" fill="#3b82f6" name="Total Bets" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="wins" fill="#4caf50" name="Wins" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <Box sx={{ height: 220, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <Typography color="text.secondary" fontSize={12}>No data yet</Typography>
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={4}>
          <Card sx={{ height: '100%' }}>
            <CardContent sx={{ py: 1.5, px: 2 }}>
              <Typography variant="subtitle1" fontWeight={600} gutterBottom sx={{ fontSize: 13 }}>Distribution by Tier</Typography>
              {tierDist.length > 0 ? (
                <ResponsiveContainer width="100%" height={220}>
                  <PieChart>
                    <Pie data={tierDist} dataKey="value" nameKey="name" cx="50%" cy="50%" innerRadius={40} outerRadius={70}
                      label={({ name, value }) => `${name}: ${value}`} labelLine={{ strokeWidth: 1 }} fontSize={10}>
                      {tierDist.map((d) => <Cell key={d.tier} fill={TIER_COLORS[d.tier] || '#9e9e9e'} />)}
                    </Pie>
                    <Tooltip contentStyle={{ backgroundColor: isDark ? '#1e1e1e' : '#fff', border: '1px solid #333', fontSize: 11 }} />
                  </PieChart>
                </ResponsiveContainer>
              ) : (
                <Box sx={{ height: 220, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <Typography color="text.secondary" fontSize={12}>No data yet</Typography>
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* ─── Charts Row 2: Tier Performance + Bet Type Breakdown ── */}
      <Grid container spacing={2} mb={2}>
        <Grid item xs={12} md={6}>
          <Card sx={{ height: '100%' }}>
            <CardContent sx={{ py: 1.5, px: 2 }}>
              <Typography variant="subtitle1" fontWeight={600} gutterBottom sx={{ fontSize: 13 }}>Tier Performance</Typography>
              <TableContainer>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell sx={{ fontWeight: 600, fontSize: 11, py: 0.5 }}>Tier</TableCell>
                      <TableCell align="center" sx={{ fontWeight: 600, fontSize: 11, py: 0.5 }}>Bets</TableCell>
                      <TableCell align="center" sx={{ fontWeight: 600, fontSize: 11, py: 0.5 }}>Graded</TableCell>
                      <TableCell align="center" sx={{ fontWeight: 600, fontSize: 11, py: 0.5 }}>Win Rate</TableCell>
                      <TableCell align="center" sx={{ fontWeight: 600, fontSize: 11, py: 0.5 }}>Avg Edge</TableCell>
                      <TableCell align="right" sx={{ fontWeight: 600, fontSize: 11, py: 0.5 }}>P/L</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {tierPerf.map(t => (
                      <TableRow key={t.tier}>
                        <TableCell sx={{ py: 0.5, fontSize: 11 }}><TierBadge tier={t.tier} /></TableCell>
                        <TableCell align="center" sx={{ py: 0.5, fontSize: 11 }}>{t.bets}</TableCell>
                        <TableCell align="center" sx={{ py: 0.5, fontSize: 11 }}>{t.graded}</TableCell>
                        <TableCell align="center" sx={{ py: 0.5, fontSize: 11, color: t.winRate >= 55 ? 'success.main' : t.graded > 0 ? 'warning.main' : 'text.secondary' }}>
                          {t.graded > 0 ? `${t.winRate}%` : '-'}
                        </TableCell>
                        <TableCell align="center" sx={{ py: 0.5, fontSize: 11, color: 'success.main' }}>+{t.avgEdge}%</TableCell>
                        <TableCell align="right" sx={{ py: 0.5, fontSize: 11, fontWeight: 600, color: t.pnl >= 0 ? 'success.main' : 'error.main' }}>
                          {t.graded > 0 ? formatPnL(t.pnl) : '-'}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={6}>
          <Card sx={{ height: '100%' }}>
            <CardContent sx={{ py: 1.5, px: 2 }}>
              <Typography variant="subtitle1" fontWeight={600} gutterBottom sx={{ fontSize: 13 }}>Bet Type Breakdown</Typography>
              <TableContainer>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell sx={{ fontWeight: 600, fontSize: 11, py: 0.5 }}>Type</TableCell>
                      <TableCell align="center" sx={{ fontWeight: 600, fontSize: 11, py: 0.5 }}>Bets</TableCell>
                      <TableCell align="center" sx={{ fontWeight: 600, fontSize: 11, py: 0.5 }}>Wins</TableCell>
                      <TableCell align="center" sx={{ fontWeight: 600, fontSize: 11, py: 0.5 }}>Win Rate</TableCell>
                      <TableCell align="right" sx={{ fontWeight: 600, fontSize: 11, py: 0.5 }}>P/L</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {typePerf.map(t => (
                      <TableRow key={t.type}>
                        <TableCell sx={{ py: 0.5, fontSize: 11, fontWeight: 600 }}>{t.type}</TableCell>
                        <TableCell align="center" sx={{ py: 0.5, fontSize: 11 }}>{t.bets}</TableCell>
                        <TableCell align="center" sx={{ py: 0.5, fontSize: 11 }}>{t.wins}</TableCell>
                        <TableCell align="center" sx={{ py: 0.5, fontSize: 11, color: t.winRate >= 55 ? 'success.main' : t.winRate > 0 ? 'warning.main' : 'text.secondary' }}>
                          {t.winRate > 0 ? `${t.winRate}%` : '-'}
                        </TableCell>
                        <TableCell align="right" sx={{ py: 0.5, fontSize: 11, fontWeight: 600, color: t.pnl >= 0 ? 'success.main' : 'error.main' }}>
                          {t.pnl !== 0 ? formatPnL(t.pnl) : '-'}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* ─── Equity Curve ────────────────────────────────────────── */}
      {equityCurve.length > 0 && (
        <Card sx={{ mb: 2 }}>
          <CardContent sx={{ py: 1.5, px: 2 }}>
            <Typography variant="subtitle1" fontWeight={600} gutterBottom sx={{ fontSize: 13 }}>Bankroll Growth</Typography>
            <Box sx={{ height: 200 }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={[{ date: 'Start', value: 10000 }, ...equityCurve]} margin={{ top: 10, right: 10, bottom: 0, left: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.15} />
                  <XAxis dataKey="date" stroke="#666" fontSize={10} />
                  <YAxis stroke="#666" fontSize={10} tickFormatter={(v: number) => `$${(v / 1000).toFixed(1)}k`} domain={['dataMin - 200', 'dataMax + 200']} />
                  <Tooltip contentStyle={{ backgroundColor: isDark ? '#1e1e1e' : '#fff', border: '1px solid #333', fontSize: 11 }} formatter={(value: number) => [`$${value.toLocaleString()}`, 'Bankroll']} />
                  <ReferenceLine y={10000} stroke="#666" strokeDasharray="3 3" />
                  <Line type="monotone" dataKey="value" stroke={fStats.totalPnl >= 0 ? '#4caf50' : '#ef5350'} strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </Box>
          </CardContent>
        </Card>
      )}

      {/* ─── Detailed Records Table ──────────────────────────────── */}
      <Card>
        <CardContent sx={{ pb: 0.5, pt: 1.5, px: 2 }}>
          <Typography variant="subtitle1" fontWeight={600} sx={{ fontSize: 13 }}>Detailed Records ({sorted.length})</Typography>
        </CardContent>
        {loading && <LinearProgress />}
        <TableContainer>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell sx={hdr}><TableSortLabel active={sortField === 'sport'} direction={sortOrder} onClick={() => handleSort('sport')}>Sport</TableSortLabel></TableCell>
                <TableCell sx={hdr}>Date</TableCell>
                <TableCell sx={hdr}>Time (PST)</TableCell>
                <TableCell sx={hdr}>Team</TableCell>
                <TableCell sx={hdr}><TableSortLabel active={sortField === 'bet_type'} direction={sortOrder} onClick={() => handleSort('bet_type')}>Type</TableSortLabel></TableCell>
                <TableCell sx={hdr}>Pick</TableCell>
                <TableCell sx={hdr} align="center">Line</TableCell>
                <TableCell sx={hdr} align="center">Odds</TableCell>
                <TableCell sx={hdr} align="center"><TableSortLabel active={sortField === 'probability'} direction={sortOrder} onClick={() => handleSort('probability')}>%</TableSortLabel></TableCell>
                <TableCell sx={hdr} align="center"><TableSortLabel active={sortField === 'edge'} direction={sortOrder} onClick={() => handleSort('edge')}>Edge</TableSortLabel></TableCell>
                <TableCell sx={hdr}>Tier</TableCell>
                <TableCell sx={hdr}>W/L</TableCell>
                <TableCell sx={hdr} align="right"><TableSortLabel active={sortField === 'profit_loss'} direction={sortOrder} onClick={() => handleSort('profit_loss')}>P/L</TableSortLabel></TableCell>
                <TableCell sx={hdr} align="right"><TableSortLabel active={sortField === 'clv'} direction={sortOrder} onClick={() => handleSort('clv')}>CLV</TableSortLabel></TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {paginatedBets.map((bet) => {
                const { date: betDate, time: betTime } = formatGameTime(bet.game_time);
                const edgeVal = (bet.edge || 0) * 100;
                const isPending = bet.result === 'pending';
                return (
                  <React.Fragment key={bet.id}>
                    <TableRow sx={{ opacity: isPending ? 0.75 : 1 }}>
                      <TableCell rowSpan={2} sx={{ py: 0.75, fontSize: 11, verticalAlign: 'middle', fontWeight: 600, borderBottom: 1, borderColor: 'divider' }}>{bet.sport}</TableCell>
                      <TableCell rowSpan={2} sx={{ py: 0.75, fontSize: 11, verticalAlign: 'middle', borderBottom: 1, borderColor: 'divider' }}>{betDate}</TableCell>
                      <TableCell rowSpan={2} sx={{ py: 0.75, fontSize: 11, verticalAlign: 'middle', borderBottom: 1, borderColor: 'divider' }}>{betTime}</TableCell>
                      <TableCell sx={{ py: 0.75, fontSize: 11, fontWeight: bet.pick_team === 'away' ? 700 : 400, borderBottom: 0, bgcolor: bet.pick_team === 'away' ? (isDark ? 'rgba(46, 125, 50, 0.12)' : 'rgba(46, 125, 50, 0.08)') : undefined }}>{bet.away_team || '-'}</TableCell>
                      <TableCell rowSpan={2} sx={{ py: 0.75, fontSize: 11, verticalAlign: 'middle', borderBottom: 1, borderColor: 'divider' }}>{formatBetType(bet.bet_type)}</TableCell>
                      <TableCell rowSpan={2} sx={{ py: 0.75, fontSize: 11, verticalAlign: 'middle', fontWeight: 600, color: 'success.main', borderBottom: 1, borderColor: 'divider' }}>{bet.predicted_side}</TableCell>
                      <TableCell rowSpan={2} align="center" sx={{ py: 0.75, fontSize: 11, fontFamily: 'monospace', verticalAlign: 'middle', borderBottom: 1, borderColor: 'divider' }}>
                        {bet.line != null ? (bet.bet_type === 'total' ? bet.line.toFixed(1) : (bet.line > 0 ? `+${bet.line.toFixed(1)}` : bet.line.toFixed(1))) : '-'}
                      </TableCell>
                      <TableCell rowSpan={2} align="center" sx={{ py: 0.75, fontSize: 11, fontFamily: 'monospace', verticalAlign: 'middle', borderBottom: 1, borderColor: 'divider' }}>
                        {bet.odds != null ? (bet.odds > 0 ? `+${bet.odds}` : bet.odds) : '-'}
                      </TableCell>
                      <TableCell rowSpan={2} align="center" sx={{ py: 0.75, fontSize: 11, verticalAlign: 'middle', borderBottom: 1, borderColor: 'divider' }}>{(bet.probability * 100).toFixed(1)}%</TableCell>
                      <TableCell rowSpan={2} align="center" sx={{ py: 0.75, fontSize: 11, verticalAlign: 'middle', borderBottom: 1, borderColor: 'divider', color: edgeVal >= 3 ? 'success.main' : edgeVal >= 1 ? 'warning.main' : 'text.secondary' }}>
                        {edgeVal >= 0 ? '+' : ''}{edgeVal.toFixed(1)}%
                      </TableCell>
                      <TableCell rowSpan={2} sx={{ py: 0.75, fontSize: 11, verticalAlign: 'middle', borderBottom: 1, borderColor: 'divider' }}><TierBadge tier={bet.signal_tier || 'D'} /></TableCell>
                      <TableCell rowSpan={2} sx={{ py: 0.75, fontSize: 11, verticalAlign: 'middle', borderBottom: 1, borderColor: 'divider' }}>{getStatusChip(bet.result)}</TableCell>
                      <TableCell rowSpan={2} align="right" sx={{ py: 0.75, fontSize: 11, verticalAlign: 'middle', borderBottom: 1, borderColor: 'divider', color: bet.profit_loss != null ? (bet.profit_loss > 0 ? 'success.main' : bet.profit_loss < 0 ? 'error.main' : 'inherit') : 'text.secondary', fontWeight: 600 }}>
                        {bet.profit_loss != null ? formatPnL(bet.profit_loss) : '-'}
                      </TableCell>
                      <TableCell rowSpan={2} align="right" sx={{ py: 0.75, fontSize: 11, verticalAlign: 'middle', borderBottom: 1, borderColor: 'divider', color: bet.clv != null ? (bet.clv > 0 ? 'success.main' : bet.clv < 0 ? 'error.main' : 'inherit') : 'text.secondary' }}>
                        {bet.clv != null ? `${bet.clv > 0 ? '+' : ''}${(bet.clv * 100).toFixed(1)}%` : '-'}
                      </TableCell>
                    </TableRow>
                    <TableRow sx={{ opacity: isPending ? 0.75 : 1 }}>
                      <TableCell sx={{ py: 0.75, fontSize: 11, fontWeight: bet.pick_team === 'home' ? 700 : 400, borderBottom: 1, borderColor: 'divider', bgcolor: bet.pick_team === 'home' ? (isDark ? 'rgba(46, 125, 50, 0.12)' : 'rgba(46, 125, 50, 0.08)') : undefined }}>{bet.home_team || '-'}</TableCell>
                    </TableRow>
                  </React.Fragment>
                );
              })}
              {paginatedBets.length === 0 && !loading && (
                <TableRow><TableCell colSpan={14} align="center" sx={{ py: 4 }}>
                  <Typography color="text.secondary">No predictions match the current filters.</Typography>
                </TableCell></TableRow>
              )}
            </TableBody>
          </Table>
        </TableContainer>
        <TablePagination
          component="div"
          count={sorted.length}
          page={page}
          onPageChange={(_, p) => setPage(p)}
          rowsPerPage={rowsPerPage}
          onRowsPerPageChange={(e) => { setRowsPerPage(parseInt(e.target.value)); setPage(0); }}
          rowsPerPageOptions={[25, 50, 100, { value: -1, label: 'All' }]}
          labelRowsPerPage="Records per page:"
        />
      </Card>
    </Box>
  );
};

export default Analytics;