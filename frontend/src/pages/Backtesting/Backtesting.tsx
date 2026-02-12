// src/pages/Backtesting/Backtesting.tsx - Client-side backtesting using real prediction data
import React, { useState, useEffect, useCallback, useMemo } from 'react';
import {
  Box, Card, CardContent, Typography, Grid, TextField, Button,
  FormControl, InputLabel, Select, MenuItem, Table, TableBody,
  TableCell, TableContainer, TableHead, TableRow, Chip, LinearProgress,
  useTheme, TablePagination, Alert
} from '@mui/material';
import { PlayArrow, CheckCircle, Cancel, Schedule, Remove } from '@mui/icons-material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, BarChart, Bar } from 'recharts';
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

interface BacktestResult {
  totalBets: number;
  graded: number;
  pending: number;
  wins: number;
  losses: number;
  pushes: number;
  winRate: number;
  roi: number;
  totalPnl: number;
  finalBankroll: number;
  maxDrawdown: number;
  avgEdge: number;
  avgClv: number;
  equityCurve: { date: string; value: number }[];
  byTier: { tier: string; bets: number; graded: number; wins: number; winRate: number; roi: number; pnl: number }[];
  bySport: { sport: string; bets: number; graded: number; wins: number; winRate: number; roi: number; pnl: number }[];
  byType: { type: string; bets: number; graded: number; wins: number; winRate: number; roi: number; pnl: number }[];
  filteredBets: Bet[];
}

// ─── Helpers ─────────────────────────────────────────────────────────
const TIER_ORDER = ['A', 'B', 'C', 'D'];
const TIER_THRESHOLDS: Record<string, string[]> = {
  'A': ['A'],
  'B': ['A', 'B'],
  'C': ['A', 'B', 'C'],
  'D': ['A', 'B', 'C', 'D'],
};

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

const formatPnL = (v: number) => `${v >= 0 ? '+' : ''}$${Math.abs(v).toFixed(2)}`;
const formatBetType = (bt: string) => bt === 'spread' ? 'Spread' : bt === 'total' ? 'Total' : bt === 'moneyline' ? 'ML' : bt;

const formatGameTime = (gt: string | null) => {
  if (!gt) return { date: '-', time: '-' };
  const d = new Date(gt);
  return {
    date: d.toLocaleDateString('en-US', { month: 'numeric', day: 'numeric', year: 'numeric', timeZone: 'America/Los_Angeles' }),
    time: d.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', timeZone: 'America/Los_Angeles' }),
  };
};

// ─── Component ───────────────────────────────────────────────────────
const Backtesting: React.FC = () => {
  const theme = useTheme();
  const isDark = theme.palette.mode === 'dark';

  // All bets from API (fetched once)
  const [allBets, setAllBets] = useState<Bet[]>([]);
  const [dataLoading, setDataLoading] = useState(true);

  // Config
  const [sport, setSport] = useState('all');
  const [betType, setBetType] = useState('all');
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [initialBankroll, setInitialBankroll] = useState(10000);
  const [betAmount, setBetAmount] = useState(100);
  const [minTier, setMinTier] = useState('B');

  // Results
  const [result, setResult] = useState<BacktestResult | null>(null);
  const [running, setRunning] = useState(false);

  // Bet log pagination
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(25);

  // ─── Load all bets once ────────────────────────────────────────
  useEffect(() => {
    (async () => {
      setDataLoading(true);
      try {
        const data = await api.getBettingSummary({ tiers: 'A,B,C,D', stake: 100, initial_bankroll: 10000 });
        if (data.bets) {
          setAllBets(data.bets);
          // Set smart date defaults from actual data
          const times = data.bets
            .map((b: Bet) => b.game_time ? new Date(b.game_time).getTime() : 0)
            .filter((t: number) => t > 0);
          if (times.length > 0) {
            const minT = new Date(Math.min(...times));
            const maxT = new Date(Math.max(...times));
            setStartDate(minT.toISOString().slice(0, 10));
            setEndDate(maxT.toISOString().slice(0, 10));
          }
        }
      } catch (err) {
        console.error('Failed to load prediction data:', err);
      }
      setDataLoading(false);
    })();
  }, []);

  // ─── Available options from data ───────────────────────────────
  const availableSports = useMemo(() =>
    Array.from(new Set(allBets.map(b => b.sport).filter(Boolean))).sort() as string[],
    [allBets]
  );

  const availableTypes = useMemo(() =>
    Array.from(new Set(allBets.map(b => b.bet_type).filter(Boolean))).sort() as string[],
    [allBets]
  );

  // ─── Run Backtest (client-side) ────────────────────────────────
  const runBacktest = useCallback(() => {
    setRunning(true);
    setPage(0);

    // Small timeout so UI shows "Running..."
    setTimeout(() => {
      const allowedTiers = TIER_THRESHOLDS[minTier] || TIER_ORDER;
      const stakeScale = betAmount / 100; // API returns P/L based on $100 stake

      // 1. Filter bets by config
      const filtered = allBets.filter(b => {
        if (sport !== 'all' && b.sport !== sport) return false;
        if (betType !== 'all' && b.bet_type !== betType) return false;
        if (!allowedTiers.includes(b.signal_tier || 'D')) return false;
        if (b.game_time && startDate) {
          const gt = new Date(b.game_time).toISOString().slice(0, 10);
          if (gt < startDate) return false;
        }
        if (b.game_time && endDate) {
          const gt = new Date(b.game_time).toISOString().slice(0, 10);
          if (gt > endDate) return false;
        }
        return true;
      });

      // 2. Compute results from graded bets
      const graded = filtered.filter(b => b.result !== 'pending');
      const wins = graded.filter(b => b.result === 'win').length;
      const losses = graded.filter(b => b.result === 'loss').length;
      const pushes = graded.filter(b => b.result === 'push').length;
      const decided = wins + losses;

      // Scale P/L by bet amount
      const totalPnl = graded.reduce((s, b) => s + ((b.profit_loss || 0) * stakeScale), 0);
      const winRate = decided > 0 ? (wins / decided * 100) : 0;
      const roi = graded.length > 0 ? (totalPnl / (graded.length * betAmount) * 100) : 0;

      // Avg edge & CLV
      const avgEdge = filtered.length > 0
        ? filtered.reduce((s, b) => s + (b.edge || 0), 0) / filtered.length * 100 : 0;
      const clvBets = graded.filter(b => b.clv != null);
      const avgClv = clvBets.length > 0
        ? clvBets.reduce((s, b) => s + (b.clv || 0), 0) / clvBets.length * 100 : 0;

      // 3. Equity curve (chronological)
      const gradedSorted = [...graded].sort((a, b) => {
        const aT = a.game_time ? new Date(a.game_time).getTime() : 0;
        const bT = b.game_time ? new Date(b.game_time).getTime() : 0;
        return aT - bT;
      });

      let runningBalance = initialBankroll;
      let peak = initialBankroll;
      let maxDrawdown = 0;
      const equityCurve: { date: string; value: number }[] = [{ date: 'Start', value: initialBankroll }];

      gradedSorted.forEach(b => {
        const pnl = (b.profit_loss || 0) * stakeScale;
        runningBalance += pnl;
        if (runningBalance > peak) peak = runningBalance;
        const dd = peak > 0 ? ((peak - runningBalance) / peak * 100) : 0;
        if (dd > maxDrawdown) maxDrawdown = dd;

        const dt = b.game_time ? new Date(b.game_time) : null;
        const label = dt ? dt.toLocaleDateString('en-US', { month: 'short', day: 'numeric', timeZone: 'America/Los_Angeles' }) : '?';
        equityCurve.push({ date: label, value: Math.round(runningBalance * 100) / 100 });
      });

      const finalBankroll = initialBankroll + totalPnl;

      // 4. Breakdowns
      const buildBreakdown = (key: 'signal_tier' | 'sport' | 'bet_type') => {
        const groups = Array.from(new Set(filtered.map(b => (b as any)[key]).filter(Boolean))) as string[];
        return groups.map(g => {
          const gb = filtered.filter(b => (b as any)[key] === g);
          const gg = gb.filter(b => b.result !== 'pending');
          const gw = gg.filter(b => b.result === 'win').length;
          const gd = gg.filter(b => b.result === 'win' || b.result === 'loss').length;
          const gpnl = gg.reduce((s, b) => s + ((b.profit_loss || 0) * stakeScale), 0);
          const groi = gg.length > 0 ? (gpnl / (gg.length * betAmount) * 100) : 0;
          return {
            [key === 'signal_tier' ? 'tier' : key === 'bet_type' ? 'type' : 'sport']: key === 'bet_type' ? formatBetType(g) : g,
            bets: gb.length,
            graded: gg.length,
            wins: gw,
            winRate: gd > 0 ? Math.round(gw / gd * 100) : 0,
            roi: Math.round(groi * 10) / 10,
            pnl: Math.round(gpnl * 100) / 100,
          };
        });
      };

      const byTier = buildBreakdown('signal_tier').sort((a, b) => TIER_ORDER.indexOf((a as any).tier) - TIER_ORDER.indexOf((b as any).tier));
      const bySport = buildBreakdown('sport').sort((a, b) => b.bets - a.bets);
      const byType = buildBreakdown('bet_type');

      setResult({
        totalBets: filtered.length,
        graded: graded.length,
        pending: filtered.length - graded.length,
        wins, losses, pushes, winRate,
        roi: Math.round(roi * 10) / 10,
        totalPnl: Math.round(totalPnl * 100) / 100,
        finalBankroll: Math.round(finalBankroll * 100) / 100,
        maxDrawdown: Math.round(maxDrawdown * 10) / 10,
        avgEdge: Math.round(avgEdge * 10) / 10,
        avgClv: Math.round(avgClv * 100) / 100,
        equityCurve,
        byTier: byTier as any,
        bySport: bySport as any,
        byType: byType as any,
        filteredBets: filtered,
      });

      setRunning(false);
    }, 100);
  }, [allBets, sport, betType, startDate, endDate, initialBankroll, betAmount, minTier]);

  // ─── Bet log sort (graded first) ──────────────────────────────
  const sortedBetLog = useMemo(() => {
    if (!result) return [];
    return [...result.filteredBets].sort((a, b) => {
      const aPending = a.result === 'pending';
      const bPending = b.result === 'pending';
      if (aPending !== bPending) return aPending ? 1 : -1;
      const aT = a.game_time ? new Date(a.game_time).getTime() : 0;
      const bT = b.game_time ? new Date(b.game_time).getTime() : 0;
      return aPending ? aT - bT : bT - aT;
    });
  }, [result]);

  const paginatedBets = sortedBetLog.slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage);
  const stakeScale = betAmount / 100;

  const getStatusChip = (status: string) => {
    switch (status) {
      case 'win': return <Chip icon={<CheckCircle />} label="W" size="small" color="success" />;
      case 'loss': return <Chip icon={<Cancel />} label="L" size="small" color="error" />;
      case 'push': return <Chip icon={<Remove />} label="P" size="small" />;
      default: return <Chip icon={<Schedule />} label="-" size="small" color="primary" />;
    }
  };

  const hdr = { fontWeight: 600, fontSize: 11, py: 0.75, bgcolor: isDark ? 'grey.900' : 'grey.100', color: isDark ? 'grey.100' : 'grey.800', whiteSpace: 'nowrap', borderBottom: 1, borderColor: 'divider' };
  const csx = { height: '100%' };
  const ccsx = { textAlign: 'center', py: 1.5, '&:last-child': { pb: 1.5 }, height: '100%', display: 'flex', flexDirection: 'column', justifyContent: 'center' };
  const tblHdr = { fontWeight: 600, fontSize: 11, py: 0.5 };

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
        <Typography variant="h5" fontWeight={700} sx={{ fontSize: 20 }}>Backtesting</Typography>
        {allBets.length > 0 && (
          <Typography variant="caption" color="text.secondary" sx={{ fontSize: 11 }}>
            {allBets.length} predictions loaded • {allBets.filter(b => b.result !== 'pending').length} graded
          </Typography>
        )}
      </Box>

      <Grid container spacing={2}>
        {/* ─── Configuration Panel ──────────────────────────────── */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent sx={{ py: 2, px: 2 }}>
              <Typography variant="subtitle1" fontWeight={600} sx={{ fontSize: 13, mb: 1.5 }}>Configuration</Typography>

              {dataLoading && <LinearProgress sx={{ mb: 1.5 }} />}

              <Grid container spacing={1.5}>
                <Grid item xs={12}>
                  <FormControl fullWidth size="small">
                    <InputLabel sx={{ fontSize: 11 }}>Sport</InputLabel>
                    <Select value={sport} label="Sport" onChange={(e) => setSport(e.target.value)} sx={{ fontSize: 11 }}>
                      <MenuItem value="all" sx={{ fontSize: 11 }}>All Sports</MenuItem>
                      {availableSports.map(s => <MenuItem key={s} value={s} sx={{ fontSize: 11 }}>{s}</MenuItem>)}
                    </Select>
                  </FormControl>
                </Grid>
                <Grid item xs={12}>
                  <FormControl fullWidth size="small">
                    <InputLabel sx={{ fontSize: 11 }}>Bet Type</InputLabel>
                    <Select value={betType} label="Bet Type" onChange={(e) => setBetType(e.target.value)} sx={{ fontSize: 11 }}>
                      <MenuItem value="all" sx={{ fontSize: 11 }}>All Types</MenuItem>
                      {availableTypes.map(t => <MenuItem key={t} value={t} sx={{ fontSize: 11 }}>{formatBetType(t)}</MenuItem>)}
                    </Select>
                  </FormControl>
                </Grid>
                <Grid item xs={6}>
                  <TextField fullWidth size="small" label="Start Date" type="date" value={startDate}
                    onChange={(e) => setStartDate(e.target.value)}
                    InputLabelProps={{ shrink: true, sx: { fontSize: 11 } }}
                    inputProps={{ style: { fontSize: 11 } }} />
                </Grid>
                <Grid item xs={6}>
                  <TextField fullWidth size="small" label="End Date" type="date" value={endDate}
                    onChange={(e) => setEndDate(e.target.value)}
                    InputLabelProps={{ shrink: true, sx: { fontSize: 11 } }}
                    inputProps={{ style: { fontSize: 11 } }} />
                </Grid>
                <Grid item xs={6}>
                  <TextField fullWidth size="small" label="Initial Bankroll" type="number" value={initialBankroll}
                    onChange={(e) => setInitialBankroll(Math.max(100, Number(e.target.value) || 10000))}
                    InputLabelProps={{ sx: { fontSize: 11 } }} inputProps={{ style: { fontSize: 11 }, min: 100 }} />
                </Grid>
                <Grid item xs={6}>
                  <TextField fullWidth size="small" label="Bet Amount" type="number" value={betAmount}
                    onChange={(e) => setBetAmount(Math.max(1, Number(e.target.value) || 100))}
                    InputLabelProps={{ sx: { fontSize: 11 } }} inputProps={{ style: { fontSize: 11 }, min: 1 }} />
                </Grid>
                <Grid item xs={12}>
                  <FormControl fullWidth size="small">
                    <InputLabel sx={{ fontSize: 11 }}>Min Tier</InputLabel>
                    <Select value={minTier} label="Min Tier" onChange={(e) => setMinTier(e.target.value)} sx={{ fontSize: 11 }}>
                      <MenuItem value="A" sx={{ fontSize: 11 }}>Tier A Only (65%+)</MenuItem>
                      <MenuItem value="B" sx={{ fontSize: 11 }}>Tier B+ (60%+)</MenuItem>
                      <MenuItem value="C" sx={{ fontSize: 11 }}>Tier C+ (55%+)</MenuItem>
                      <MenuItem value="D" sx={{ fontSize: 11 }}>All Tiers</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
                <Grid item xs={12}>
                  <Button fullWidth variant="contained" size="small" startIcon={<PlayArrow sx={{ fontSize: 16 }} />}
                    onClick={runBacktest} disabled={dataLoading || running || allBets.length === 0} sx={{ fontSize: 12 }}>
                    {running ? 'Running...' : dataLoading ? 'Loading Data...' : 'Run Backtest'}
                  </Button>
                </Grid>
              </Grid>

              {allBets.length > 0 && !result && (
                <Alert severity="info" sx={{ mt: 1.5, fontSize: 11, py: 0.5 }}>
                  Configure parameters and click "Run Backtest" to simulate performance using your {allBets.length} real predictions.
                </Alert>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* ─── Results Panel ────────────────────────────────────── */}
        <Grid item xs={12} md={8}>
          {running && <LinearProgress sx={{ mb: 2 }} />}

          {result ? (
            <>
              {/* Stat Cards */}
              <Grid container spacing={1.5} mb={1.5} sx={{ alignItems: 'stretch' }}>
                <Grid item xs={6} sm={4} md={3}>
                  <Card sx={csx}><CardContent sx={ccsx}>
                    <Typography sx={{ fontSize: 10, color: 'text.secondary' }}>Total Bets</Typography>
                    <Typography sx={{ fontSize: 18, fontWeight: 700 }}>{result.totalBets}</Typography>
                    <Typography sx={{ fontSize: 9, color: 'text.secondary' }}>{result.graded} graded • {result.pending} pending</Typography>
                  </CardContent></Card>
                </Grid>
                <Grid item xs={6} sm={4} md={3}>
                  <Card sx={csx}><CardContent sx={ccsx}>
                    <Typography sx={{ fontSize: 10, color: 'text.secondary' }}>Win Rate</Typography>
                    <Typography sx={{ fontSize: 18, fontWeight: 700, color: result.winRate >= 55 ? 'success.main' : result.winRate > 0 ? 'warning.main' : 'text.primary' }}>
                      {result.graded > 0 ? `${result.winRate.toFixed(1)}%` : '-'}
                    </Typography>
                    <Typography sx={{ fontSize: 9, color: 'text.secondary' }}>{result.wins}W-{result.losses}L-{result.pushes}P</Typography>
                  </CardContent></Card>
                </Grid>
                <Grid item xs={6} sm={4} md={3}>
                  <Card sx={csx}><CardContent sx={ccsx}>
                    <Typography sx={{ fontSize: 10, color: 'text.secondary' }}>ROI</Typography>
                    <Typography sx={{ fontSize: 18, fontWeight: 700, color: result.roi >= 0 ? 'success.main' : 'error.main' }}>
                      {result.graded > 0 ? `${result.roi >= 0 ? '+' : ''}${result.roi}%` : '-'}
                    </Typography>
                  </CardContent></Card>
                </Grid>
                <Grid item xs={6} sm={4} md={3}>
                  <Card sx={csx}><CardContent sx={ccsx}>
                    <Typography sx={{ fontSize: 10, color: 'text.secondary' }}>Final Bankroll</Typography>
                    <Typography sx={{ fontSize: 18, fontWeight: 700, color: result.finalBankroll >= initialBankroll ? 'success.main' : 'error.main' }}>
                      ${result.finalBankroll.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                    </Typography>
                  </CardContent></Card>
                </Grid>
                <Grid item xs={6} sm={4} md={3}>
                  <Card sx={csx}><CardContent sx={ccsx}>
                    <Typography sx={{ fontSize: 10, color: 'text.secondary' }}>P/L</Typography>
                    <Typography sx={{ fontSize: 18, fontWeight: 700, color: result.totalPnl >= 0 ? 'success.main' : 'error.main' }}>
                      {formatPnL(result.totalPnl)}
                    </Typography>
                  </CardContent></Card>
                </Grid>
                <Grid item xs={6} sm={4} md={3}>
                  <Card sx={csx}><CardContent sx={ccsx}>
                    <Typography sx={{ fontSize: 10, color: 'text.secondary' }}>Max Drawdown</Typography>
                    <Typography sx={{ fontSize: 18, fontWeight: 700, color: result.maxDrawdown > 5 ? 'error.main' : 'warning.main' }}>
                      {result.maxDrawdown.toFixed(1)}%
                    </Typography>
                  </CardContent></Card>
                </Grid>
                <Grid item xs={6} sm={4} md={3}>
                  <Card sx={csx}><CardContent sx={ccsx}>
                    <Typography sx={{ fontSize: 10, color: 'text.secondary' }}>Avg Edge</Typography>
                    <Typography sx={{ fontSize: 18, fontWeight: 700, color: result.avgEdge > 0 ? 'success.main' : 'text.primary' }}>
                      {result.avgEdge >= 0 ? '+' : ''}{result.avgEdge}%
                    </Typography>
                  </CardContent></Card>
                </Grid>
                <Grid item xs={6} sm={4} md={3}>
                  <Card sx={csx}><CardContent sx={ccsx}>
                    <Typography sx={{ fontSize: 10, color: 'text.secondary' }}>Avg CLV</Typography>
                    <Typography sx={{ fontSize: 18, fontWeight: 700, color: result.avgClv > 0 ? 'success.main' : result.avgClv < 0 ? 'error.main' : 'text.primary' }}>
                      {result.avgClv !== 0 ? `${result.avgClv >= 0 ? '+' : ''}${result.avgClv}%` : '-'}
                    </Typography>
                  </CardContent></Card>
                </Grid>
              </Grid>

              {/* Equity Curve */}
              <Card sx={{ mb: 1.5 }}>
                <CardContent sx={{ py: 1.5, px: 2 }}>
                  <Typography variant="subtitle1" fontWeight={600} sx={{ fontSize: 13, mb: 1 }}>Equity Curve</Typography>
                  <ResponsiveContainer width="100%" height={250}>
                    <LineChart data={result.equityCurve} margin={{ top: 10, right: 10, bottom: 0, left: 10 }}>
                      <CartesianGrid strokeDasharray="3 3" opacity={0.15} />
                      <XAxis dataKey="date" stroke="#666" fontSize={10} />
                      <YAxis stroke="#666" fontSize={10} tickFormatter={(v: number) => `$${(v / 1000).toFixed(1)}k`} domain={['dataMin - 200', 'dataMax + 200']} />
                      <Tooltip contentStyle={{ backgroundColor: isDark ? '#1e1e1e' : '#fff', border: '1px solid #333', fontSize: 11 }} formatter={(value: number) => [`$${value.toLocaleString()}`, 'Bankroll']} />
                      <ReferenceLine y={initialBankroll} stroke="#666" strokeDasharray="3 3" label={{ value: 'Initial', position: 'right', fontSize: 10, fill: '#888' }} />
                      <Line type="monotone" dataKey="value" stroke={result.totalPnl >= 0 ? '#4caf50' : '#ef5350'} strokeWidth={2} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              {/* Breakdown Tables */}
              <Grid container spacing={1.5} mb={1.5}>
                {/* By Tier */}
                <Grid item xs={12} md={6}>
                  <Card sx={{ height: '100%' }}>
                    <CardContent sx={{ py: 1.5, px: 2 }}>
                      <Typography variant="subtitle1" fontWeight={600} sx={{ fontSize: 13, mb: 0.5 }}>Performance by Tier</Typography>
                      <TableContainer>
                        <Table size="small">
                          <TableHead>
                            <TableRow>
                              <TableCell sx={tblHdr}>Tier</TableCell>
                              <TableCell align="center" sx={tblHdr}>Bets</TableCell>
                              <TableCell align="center" sx={tblHdr}>Graded</TableCell>
                              <TableCell align="center" sx={tblHdr}>Win Rate</TableCell>
                              <TableCell align="center" sx={tblHdr}>ROI</TableCell>
                              <TableCell align="right" sx={tblHdr}>P/L</TableCell>
                            </TableRow>
                          </TableHead>
                          <TableBody>
                            {result.byTier.map((t: any) => (
                              <TableRow key={t.tier}>
                                <TableCell sx={{ py: 0.5, fontSize: 11 }}><TierBadge tier={t.tier} /></TableCell>
                                <TableCell align="center" sx={{ py: 0.5, fontSize: 11 }}>{t.bets}</TableCell>
                                <TableCell align="center" sx={{ py: 0.5, fontSize: 11 }}>{t.graded}</TableCell>
                                <TableCell align="center" sx={{ py: 0.5, fontSize: 11, color: t.winRate >= 55 ? 'success.main' : t.graded > 0 ? 'warning.main' : 'text.secondary' }}>
                                  {t.graded > 0 ? `${t.winRate}%` : '-'}
                                </TableCell>
                                <TableCell align="center" sx={{ py: 0.5, fontSize: 11, color: t.roi >= 0 ? 'success.main' : 'error.main' }}>
                                  {t.graded > 0 ? `${t.roi >= 0 ? '+' : ''}${t.roi}%` : '-'}
                                </TableCell>
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

                {/* By Sport */}
                <Grid item xs={12} md={6}>
                  <Card sx={{ height: '100%' }}>
                    <CardContent sx={{ py: 1.5, px: 2 }}>
                      <Typography variant="subtitle1" fontWeight={600} sx={{ fontSize: 13, mb: 0.5 }}>Performance by Sport</Typography>
                      <TableContainer>
                        <Table size="small">
                          <TableHead>
                            <TableRow>
                              <TableCell sx={tblHdr}>Sport</TableCell>
                              <TableCell align="center" sx={tblHdr}>Bets</TableCell>
                              <TableCell align="center" sx={tblHdr}>Graded</TableCell>
                              <TableCell align="center" sx={tblHdr}>Win Rate</TableCell>
                              <TableCell align="center" sx={tblHdr}>ROI</TableCell>
                              <TableCell align="right" sx={tblHdr}>P/L</TableCell>
                            </TableRow>
                          </TableHead>
                          <TableBody>
                            {result.bySport.map((s: any) => (
                              <TableRow key={s.sport}>
                                <TableCell sx={{ py: 0.5, fontSize: 11, fontWeight: 600 }}>{s.sport}</TableCell>
                                <TableCell align="center" sx={{ py: 0.5, fontSize: 11 }}>{s.bets}</TableCell>
                                <TableCell align="center" sx={{ py: 0.5, fontSize: 11 }}>{s.graded}</TableCell>
                                <TableCell align="center" sx={{ py: 0.5, fontSize: 11, color: s.winRate >= 55 ? 'success.main' : s.graded > 0 ? 'warning.main' : 'text.secondary' }}>
                                  {s.graded > 0 ? `${s.winRate}%` : '-'}
                                </TableCell>
                                <TableCell align="center" sx={{ py: 0.5, fontSize: 11, color: s.roi >= 0 ? 'success.main' : 'error.main' }}>
                                  {s.graded > 0 ? `${s.roi >= 0 ? '+' : ''}${s.roi}%` : '-'}
                                </TableCell>
                                <TableCell align="right" sx={{ py: 0.5, fontSize: 11, fontWeight: 600, color: s.pnl >= 0 ? 'success.main' : 'error.main' }}>
                                  {s.graded > 0 ? formatPnL(s.pnl) : '-'}
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

              {/* By Bet Type */}
              <Card sx={{ mb: 1.5 }}>
                <CardContent sx={{ py: 1.5, px: 2 }}>
                  <Typography variant="subtitle1" fontWeight={600} sx={{ fontSize: 13, mb: 0.5 }}>Performance by Bet Type</Typography>
                  <TableContainer>
                    <Table size="small">
                      <TableHead>
                        <TableRow>
                          <TableCell sx={tblHdr}>Type</TableCell>
                          <TableCell align="center" sx={tblHdr}>Bets</TableCell>
                          <TableCell align="center" sx={tblHdr}>Graded</TableCell>
                          <TableCell align="center" sx={tblHdr}>Win Rate</TableCell>
                          <TableCell align="center" sx={tblHdr}>ROI</TableCell>
                          <TableCell align="right" sx={tblHdr}>P/L</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {result.byType.map((t: any) => (
                          <TableRow key={t.type}>
                            <TableCell sx={{ py: 0.5, fontSize: 11, fontWeight: 600 }}>{t.type}</TableCell>
                            <TableCell align="center" sx={{ py: 0.5, fontSize: 11 }}>{t.bets}</TableCell>
                            <TableCell align="center" sx={{ py: 0.5, fontSize: 11 }}>{t.graded}</TableCell>
                            <TableCell align="center" sx={{ py: 0.5, fontSize: 11, color: t.winRate >= 55 ? 'success.main' : t.graded > 0 ? 'warning.main' : 'text.secondary' }}>
                              {t.graded > 0 ? `${t.winRate}%` : '-'}
                            </TableCell>
                            <TableCell align="center" sx={{ py: 0.5, fontSize: 11, color: t.roi >= 0 ? 'success.main' : 'error.main' }}>
                              {t.graded > 0 ? `${t.roi >= 0 ? '+' : ''}${t.roi}%` : '-'}
                            </TableCell>
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

              {/* Detailed Bet Log */}
              <Card>
                <CardContent sx={{ pb: 0.5, pt: 1.5, px: 2 }}>
                  <Typography variant="subtitle1" fontWeight={600} sx={{ fontSize: 13 }}>Bet Log ({sortedBetLog.length})</Typography>
                </CardContent>
                <TableContainer>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell sx={hdr}>Sport</TableCell>
                        <TableCell sx={hdr}>Date</TableCell>
                        <TableCell sx={hdr}>Time (PST)</TableCell>
                        <TableCell sx={hdr}>Team</TableCell>
                        <TableCell sx={hdr}>Type</TableCell>
                        <TableCell sx={hdr}>Pick</TableCell>
                        <TableCell sx={hdr} align="center">Odds</TableCell>
                        <TableCell sx={hdr} align="center">%</TableCell>
                        <TableCell sx={hdr} align="center">Edge</TableCell>
                        <TableCell sx={hdr}>Tier</TableCell>
                        <TableCell sx={hdr}>W/L</TableCell>
                        <TableCell sx={hdr} align="right">P/L</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {paginatedBets.map((bet) => {
                        const { date: betDate, time: betTime } = formatGameTime(bet.game_time);
                        const edgeVal = (bet.edge || 0) * 100;
                        const isPending = bet.result === 'pending';
                        const scaledPnl = bet.profit_loss != null ? bet.profit_loss * stakeScale : null;
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
                                {bet.odds != null ? (bet.odds > 0 ? `+${bet.odds}` : bet.odds) : '-'}
                              </TableCell>
                              <TableCell rowSpan={2} align="center" sx={{ py: 0.75, fontSize: 11, verticalAlign: 'middle', borderBottom: 1, borderColor: 'divider' }}>{(bet.probability * 100).toFixed(1)}%</TableCell>
                              <TableCell rowSpan={2} align="center" sx={{ py: 0.75, fontSize: 11, verticalAlign: 'middle', borderBottom: 1, borderColor: 'divider', color: edgeVal >= 3 ? 'success.main' : edgeVal >= 1 ? 'warning.main' : 'text.secondary' }}>
                                {edgeVal >= 0 ? '+' : ''}{edgeVal.toFixed(1)}%
                              </TableCell>
                              <TableCell rowSpan={2} sx={{ py: 0.75, fontSize: 11, verticalAlign: 'middle', borderBottom: 1, borderColor: 'divider' }}><TierBadge tier={bet.signal_tier || 'D'} /></TableCell>
                              <TableCell rowSpan={2} sx={{ py: 0.75, fontSize: 11, verticalAlign: 'middle', borderBottom: 1, borderColor: 'divider' }}>{getStatusChip(bet.result)}</TableCell>
                              <TableCell rowSpan={2} align="right" sx={{ py: 0.75, fontSize: 11, verticalAlign: 'middle', borderBottom: 1, borderColor: 'divider', color: scaledPnl != null ? (scaledPnl > 0 ? 'success.main' : scaledPnl < 0 ? 'error.main' : 'inherit') : 'text.secondary', fontWeight: 600 }}>
                                {scaledPnl != null ? formatPnL(scaledPnl) : '-'}
                              </TableCell>
                            </TableRow>
                            <TableRow sx={{ opacity: isPending ? 0.75 : 1 }}>
                              <TableCell sx={{ py: 0.75, fontSize: 11, fontWeight: bet.pick_team === 'home' ? 700 : 400, borderBottom: 1, borderColor: 'divider', bgcolor: bet.pick_team === 'home' ? (isDark ? 'rgba(46, 125, 50, 0.12)' : 'rgba(46, 125, 50, 0.08)') : undefined }}>{bet.home_team || '-'}</TableCell>
                            </TableRow>
                          </React.Fragment>
                        );
                      })}
                      {paginatedBets.length === 0 && (
                        <TableRow><TableCell colSpan={12} align="center" sx={{ py: 4 }}>
                          <Typography color="text.secondary" fontSize={12}>No bets match the current configuration.</Typography>
                        </TableCell></TableRow>
                      )}
                    </TableBody>
                  </Table>
                </TableContainer>
                <TablePagination
                  component="div"
                  count={sortedBetLog.length}
                  page={page}
                  onPageChange={(_, p) => setPage(p)}
                  rowsPerPage={rowsPerPage}
                  onRowsPerPageChange={(e) => { setRowsPerPage(parseInt(e.target.value)); setPage(0); }}
                  rowsPerPageOptions={[25, 50, 100, { value: -1, label: 'All' }]}
                  labelRowsPerPage="Bets per page:"
                />
              </Card>
            </>
          ) : (
            <Card>
              <CardContent sx={{ textAlign: 'center', py: 6 }}>
                <Typography color="text.secondary" sx={{ fontSize: 13, mb: 1 }}>
                  {dataLoading ? 'Loading prediction data...' : 'Configure parameters and click "Run Backtest"'}
                </Typography>
                <Typography color="text.secondary" sx={{ fontSize: 11 }}>
                  {dataLoading ? '' : `Simulates flat-stake betting using your ${allBets.length} real predictions.`}
                </Typography>
              </CardContent>
            </Card>
          )}
        </Grid>
      </Grid>
    </Box>
  );
};

export default Backtesting;