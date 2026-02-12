// src/pages/Betting/Betting.tsx - Connected to real prediction data
import React, { useState, useEffect, useCallback } from 'react';
import {
  Box, Card, CardContent, Typography, Grid, Table, TableBody, TableCell,
  TableContainer, TableHead, TableRow, Chip, Button, TextField, Switch,
  FormControlLabel, Alert, LinearProgress, useTheme, TablePagination,
  Tabs, Tab
} from '@mui/material';
import { Refresh, Save, CheckCircle, Cancel, Schedule, Remove } from '@mui/icons-material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import { api } from '../../api/client';
import { formatCurrency, formatPercent } from '../../utils';

interface BetFromAPI {
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

interface StatsFromAPI {
  initial_bankroll: number;
  current_bankroll: number;
  total_bets: number;
  graded_bets: number;
  pending_bets: number;
  wins: number;
  losses: number;
  pushes: number;
  win_rate: number;
  roi: number;
  avg_edge: number;
  avg_clv: number;
  total_pnl: number;
}

interface BettingConfig {
  flat_amount: number;
  initial_bankroll: number;
  bet_tier_a: boolean;
  bet_tier_b: boolean;
  bet_tier_c: boolean;
  bet_tier_d: boolean;
  auto_bet: boolean;
}

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

const Betting: React.FC = () => {
  const theme = useTheme();
  const isDark = theme.palette.mode === 'dark';

  const [config, setConfig] = useState<BettingConfig>(() => {
    try {
      const saved = localStorage.getItem('royaley_betting_config');
      if (saved) return JSON.parse(saved);
    } catch { /* ignore */ }
    return { flat_amount: 100, initial_bankroll: 10000, bet_tier_a: true, bet_tier_b: true, bet_tier_c: false, bet_tier_d: false, auto_bet: false };
  });

  const [stats, setStats] = useState<StatsFromAPI>({
    initial_bankroll: 10000, current_bankroll: 10000, total_bets: 0,
    graded_bets: 0, pending_bets: 0, wins: 0, losses: 0, pushes: 0,
    win_rate: 0, roi: 0, avg_edge: 0, avg_clv: 0, total_pnl: 0,
  });
  const [bets, setBets] = useState<BetFromAPI[]>([]);
  const [equityCurve, setEquityCurve] = useState<{ date: string; value: number }[]>([]);
  const [loading, setLoading] = useState(false);
  const [saved, setSaved] = useState(false);
  const [betTab, setBetTab] = useState(0);
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(20);

  const getActiveTiers = useCallback(() => {
    const tiers: string[] = [];
    if (config.bet_tier_a) tiers.push('A');
    if (config.bet_tier_b) tiers.push('B');
    if (config.bet_tier_c) tiers.push('C');
    if (config.bet_tier_d) tiers.push('D');
    return tiers.join(',') || 'A,B';
  }, [config]);

  const loadData = useCallback(async (showLoading = true) => {
    if (showLoading) setLoading(true);
    try {
      const data = await api.getBettingSummary({
        tiers: getActiveTiers(),
        stake: config.flat_amount,
        initial_bankroll: config.initial_bankroll,
      });
      if (data.stats) setStats(data.stats);
      if (data.bets) setBets(data.bets);
      if (data.equity_curve) setEquityCurve(data.equity_curve);
    } catch (err) {
      console.error('Load betting data error:', err);
    }
    if (showLoading) setLoading(false);
  }, [config.flat_amount, config.initial_bankroll, getActiveTiers]);

  useEffect(() => { loadData(); }, [loadData]);
  useEffect(() => { const iv = setInterval(() => loadData(false), 60000); return () => clearInterval(iv); }, [loadData]);

  const saveConfig = () => {
    localStorage.setItem('royaley_betting_config', JSON.stringify(config));
    setSaved(true);
    setTimeout(() => setSaved(false), 3000);
    loadData();
  };

  const filteredBets = betTab === 0 ? bets
    : betTab === 1 ? bets.filter(b => b.result === 'pending')
    : bets.filter(b => b.result !== 'pending');

  const sortedBets = [...filteredBets].sort((a, b) => {
    const aPending = a.result === 'pending';
    const bPending = b.result === 'pending';
    // Graded bets first, then pending
    if (aPending !== bPending) return aPending ? 1 : -1;
    const aTime = a.game_time ? new Date(a.game_time).getTime() : 0;
    const bTime = b.game_time ? new Date(b.game_time).getTime() : 0;
    // Graded: most recent first. Pending: soonest first.
    return aPending ? aTime - bTime : bTime - aTime;
  });

  const paginatedBets = sortedBets.slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage);

  const getStatusChip = (status: string) => {
    switch (status) {
      case 'win': return <Chip icon={<CheckCircle />} label="W" size="small" color="success" />;
      case 'loss': return <Chip icon={<Cancel />} label="L" size="small" color="error" />;
      case 'push': return <Chip icon={<Remove />} label="P" size="small" />;
      default: return <Chip icon={<Schedule />} label="-" size="small" color="primary" />;
    }
  };

  const formatBetType = (bt: string) => {
    if (bt === 'spread') return 'Spread';
    if (bt === 'total') return 'Total';
    if (bt === 'moneyline') return 'ML';
    return bt;
  };

  const formatLine = (line: number | null, bt: string) => {
    if (line == null) return '-';
    if (bt === 'total') return line.toFixed(1);
    return line > 0 ? `+${line.toFixed(1)}` : line.toFixed(1);
  };

  const formatOdds = (odds: number | null) => {
    if (odds == null) return '-';
    return odds > 0 ? `+${odds}` : `${odds}`;
  };

  const formatGameTime = (gt: string | null) => {
    if (!gt) return { date: '-', time: '-' };
    const d = new Date(gt);
    return {
      date: d.toLocaleDateString('en-US', { month: 'numeric', day: 'numeric', year: 'numeric', timeZone: 'America/Los_Angeles' }),
      time: d.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', timeZone: 'America/Los_Angeles' }),
    };
  };

  const hdr = { fontWeight: 600, fontSize: 11, py: 0.75, bgcolor: isDark ? 'grey.900' : 'grey.100', color: isDark ? 'grey.100' : 'grey.800', whiteSpace: 'nowrap', borderBottom: 1, borderColor: 'divider' };

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
        <Typography variant="h5" fontWeight={700} sx={{ fontSize: 20 }}>Betting</Typography>
        <Button variant="outlined" size="small" startIcon={<Refresh />} onClick={() => loadData()} sx={{ fontSize: 11 }}>Refresh</Button>
      </Box>

      {/* Stats Cards */}
      <Grid container spacing={2} mb={2}>
        <Grid item xs={6} sm={4} md={2}>
          <Card><CardContent sx={{ textAlign: 'center', py: 1.5, '&:last-child': { pb: 1.5 } }}>
            <Typography sx={{ fontSize: 10, color: 'text.secondary' }}>Bankroll</Typography>
            <Typography sx={{ fontSize: 16, fontWeight: 700, color: 'primary.main' }}>{formatCurrency(stats.current_bankroll)}</Typography>
          </CardContent></Card>
        </Grid>
        <Grid item xs={6} sm={4} md={2}>
          <Card><CardContent sx={{ textAlign: 'center', py: 1.5, '&:last-child': { pb: 1.5 } }}>
            <Typography sx={{ fontSize: 10, color: 'text.secondary' }}>Total Bets</Typography>
            <Typography sx={{ fontSize: 16, fontWeight: 700 }}>{stats.total_bets}</Typography>
            {stats.pending_bets > 0 && <Typography sx={{ fontSize: 9, color: 'text.secondary' }}>{stats.graded_bets} graded • {stats.pending_bets} pending</Typography>}
          </CardContent></Card>
        </Grid>
        <Grid item xs={6} sm={4} md={2}>
          <Card><CardContent sx={{ textAlign: 'center', py: 1.5, '&:last-child': { pb: 1.5 } }}>
            <Typography sx={{ fontSize: 10, color: 'text.secondary' }}>Win Rate</Typography>
            <Typography sx={{ fontSize: 16, fontWeight: 700, color: 'success.main' }}>{stats.graded_bets > 0 ? `${stats.win_rate}%` : '-'}</Typography>
            {stats.graded_bets > 0 && <Typography sx={{ fontSize: 9, color: 'text.secondary' }}>{stats.wins}W-{stats.losses}L-{stats.pushes}P</Typography>}
          </CardContent></Card>
        </Grid>
        <Grid item xs={6} sm={4} md={2}>
          <Card><CardContent sx={{ textAlign: 'center', py: 1.5, '&:last-child': { pb: 1.5 } }}>
            <Typography sx={{ fontSize: 10, color: 'text.secondary' }}>ROI</Typography>
            <Typography sx={{ fontSize: 16, fontWeight: 700, color: stats.roi >= 0 ? 'success.main' : 'error.main' }}>{stats.roi >= 0 ? '+' : ''}{stats.roi}%</Typography>
          </CardContent></Card>
        </Grid>
        <Grid item xs={6} sm={4} md={2}>
          <Card><CardContent sx={{ textAlign: 'center', py: 1.5, '&:last-child': { pb: 1.5 } }}>
            <Typography sx={{ fontSize: 10, color: 'text.secondary' }}>Avg CLV</Typography>
            <Typography sx={{ fontSize: 16, fontWeight: 700, color: stats.avg_clv >= 0 ? 'success.main' : 'error.main' }}>{stats.avg_clv >= 0 ? '+' : ''}{stats.avg_clv}%</Typography>
          </CardContent></Card>
        </Grid>
        <Grid item xs={6} sm={4} md={2}>
          <Card><CardContent sx={{ textAlign: 'center', py: 1.5, '&:last-child': { pb: 1.5 } }}>
            <Typography sx={{ fontSize: 10, color: 'text.secondary' }}>P/L</Typography>
            <Typography sx={{ fontSize: 16, fontWeight: 700, color: stats.total_pnl >= 0 ? 'success.main' : 'error.main' }}>{stats.total_pnl >= 0 ? '+' : ''}{formatCurrency(stats.total_pnl)}</Typography>
          </CardContent></Card>
        </Grid>
      </Grid>

      <Grid container spacing={2}>
        {/* Flat Betting Configuration */}
        <Grid item xs={12} md={5}>
          <Card sx={{ mb: 2 }}>
            <CardContent>
              <Typography variant="subtitle1" fontWeight={600} sx={{ fontSize: 13, mb: 1.5 }}>Flat Betting Configuration</Typography>
              <Alert severity="info" sx={{ mb: 2, fontSize: 11, py: 0.5 }}>
                System uses flat betting: same amount for every bet. All records saved for self-reinforcement training.
              </Alert>
              <Grid container spacing={2} sx={{ mb: 1.5 }}>
                <Grid item xs={6}>
                  <TextField fullWidth size="small" label="Bet Amount ($)" type="number" value={config.flat_amount}
                    onChange={(e) => setConfig({ ...config, flat_amount: Math.max(1, parseInt(e.target.value) || 100) })}
                    inputProps={{ min: 1 }} sx={{ '& input': { fontSize: 13 } }} />
                </Grid>
                <Grid item xs={6}>
                  <TextField fullWidth size="small" label="Initial Bankroll ($)" type="number" value={config.initial_bankroll}
                    onChange={(e) => setConfig({ ...config, initial_bankroll: Math.max(100, parseInt(e.target.value) || 10000) })}
                    inputProps={{ min: 100 }} sx={{ '& input': { fontSize: 13 } }} />
                </Grid>
              </Grid>
              <Typography variant="caption" sx={{ fontSize: 11, mb: 1, display: 'block' }}>Bet on Tiers:</Typography>
              <Box display="flex" gap={1} mb={1.5} flexWrap="wrap">
                <FormControlLabel control={<Switch size="small" checked={config.bet_tier_a} onChange={(e) => setConfig({ ...config, bet_tier_a: e.target.checked })} color="success" />} label={<Typography sx={{ fontSize: 11 }}>Tier A (65%+)</Typography>} />
                <FormControlLabel control={<Switch size="small" checked={config.bet_tier_b} onChange={(e) => setConfig({ ...config, bet_tier_b: e.target.checked })} color="primary" />} label={<Typography sx={{ fontSize: 11 }}>Tier B (60-65%)</Typography>} />
                <FormControlLabel control={<Switch size="small" checked={config.bet_tier_c} onChange={(e) => setConfig({ ...config, bet_tier_c: e.target.checked })} color="warning" />} label={<Typography sx={{ fontSize: 11 }}>Tier C (55-60%)</Typography>} />
                <FormControlLabel control={<Switch size="small" checked={config.bet_tier_d} onChange={(e) => setConfig({ ...config, bet_tier_d: e.target.checked })} />} label={<Typography sx={{ fontSize: 11 }}>Tier D (&lt;55%)</Typography>} />
              </Box>
              <FormControlLabel control={<Switch size="small" checked={config.auto_bet} onChange={(e) => setConfig({ ...config, auto_bet: e.target.checked })} />} label={<Typography sx={{ fontSize: 11 }}>Auto-place bets (simulation)</Typography>} sx={{ mb: 1.5 }} />
              <Button fullWidth variant="contained" startIcon={<Save />} onClick={saveConfig} sx={{ fontSize: 12 }}>
                {saved ? '✓ Saved!' : 'Save Configuration'}
              </Button>
            </CardContent>
          </Card>

          {/* Self-Reinforcement Training */}
          <Card>
            <CardContent>
              <Typography variant="subtitle1" fontWeight={600} sx={{ fontSize: 13, mb: 0.5 }}>Self-Reinforcement Training</Typography>
              <Typography variant="caption" color="text.secondary" sx={{ fontSize: 11, display: 'block', mb: 1 }}>Every bet is saved with full feature data for model improvement:</Typography>
              <Box sx={{ fontSize: 11, color: 'text.secondary', mb: 1.5 }}>
                <Typography sx={{ fontSize: 11 }}>• Opening & closing lines</Typography>
                <Typography sx={{ fontSize: 11 }}>• All prediction features</Typography>
                <Typography sx={{ fontSize: 11 }}>• Actual result (W/L/P)</Typography>
                <Typography sx={{ fontSize: 11 }}>• CLV achieved</Typography>
              </Box>
              <Alert severity="success" sx={{ fontSize: 11, py: 0.5 }}>Model retrains weekly using bet history.</Alert>
            </CardContent>
          </Card>
        </Grid>

        {/* Bankroll Growth Chart */}
        <Grid item xs={12} md={7}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography variant="subtitle1" fontWeight={600} sx={{ fontSize: 13, mb: 1 }}>Bankroll Growth</Typography>
              <Box sx={{ height: 340 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={equityCurve} margin={{ top: 10, right: 10, bottom: 0, left: 10 }}>
                    <CartesianGrid strokeDasharray="3 3" opacity={0.15} />
                    <XAxis dataKey="date" stroke="#666" fontSize={10} />
                    <YAxis stroke="#666" fontSize={10} tickFormatter={(v: number) => `$${(v / 1000).toFixed(1)}k`} domain={['dataMin - 200', 'dataMax + 200']} />
                    <Tooltip
                      contentStyle={{ backgroundColor: isDark ? '#1e1e1e' : '#fff', border: '1px solid #333', fontSize: 11 }}
                      formatter={(value: number) => [`$${value.toLocaleString()}`, 'Bankroll']}
                    />
                    <ReferenceLine y={stats.initial_bankroll} stroke="#666" strokeDasharray="3 3" label={{ value: 'Initial', position: 'right', fontSize: 10, fill: '#888' }} />
                    <Line type="monotone" dataKey="value" stroke={stats.total_pnl >= 0 ? '#4caf50' : '#ef5350'} strokeWidth={2} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Bets Table */}
        <Grid item xs={12}>
          <Card>
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', px: 2, pt: 1.5, pb: 0 }}>
              <Tabs value={betTab} onChange={(_, v) => { setBetTab(v); setPage(0); }} sx={{ minHeight: 36 }}>
                <Tab label={`All (${bets.length})`} sx={{ fontSize: 12, minHeight: 36, py: 0.5 }} />
                <Tab label={`Pending (${stats.pending_bets})`} sx={{ fontSize: 12, minHeight: 36, py: 0.5 }} />
                <Tab label={`Graded (${stats.graded_bets})`} sx={{ fontSize: 12, minHeight: 36, py: 0.5 }} />
              </Tabs>
              <Typography variant="caption" color="text.secondary" sx={{ fontSize: 11 }}>
                Avg Edge: <span style={{ color: stats.avg_edge >= 0 ? '#4caf50' : '#ef5350', fontWeight: 600 }}>{stats.avg_edge >= 0 ? '+' : ''}{stats.avg_edge}%</span>
              </Typography>
            </Box>
            {loading && <LinearProgress />}
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
                    <TableCell sx={hdr} align="center">Line</TableCell>
                    <TableCell sx={hdr} align="center">Odds</TableCell>
                    <TableCell sx={hdr} align="center">Stake</TableCell>
                    <TableCell sx={hdr} align="center">%</TableCell>
                    <TableCell sx={hdr} align="center">Edge</TableCell>
                    <TableCell sx={hdr}>Tier</TableCell>
                    <TableCell sx={hdr}>W/L</TableCell>
                    <TableCell sx={hdr} align="right">P/L</TableCell>
                    <TableCell sx={hdr} align="right">CLV</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {paginatedBets.map((bet) => {
                    const { date: betDate, time: betTime } = formatGameTime(bet.game_time);
                    const edgeVal = (bet.edge || 0) * 100;
                    const isNegEdge = edgeVal < 0;
                    const isPending = bet.result === 'pending';

                    return (
                      <React.Fragment key={bet.id}>
                        <TableRow sx={{ opacity: isPending ? 0.85 : 1 }}>
                          <TableCell rowSpan={2} sx={{ py: 0.75, fontSize: 11, verticalAlign: 'middle', fontWeight: 600, borderBottom: 1, borderColor: 'divider' }}>{bet.sport}</TableCell>
                          <TableCell rowSpan={2} sx={{ py: 0.75, fontSize: 11, verticalAlign: 'middle', borderBottom: 1, borderColor: 'divider' }}>{betDate}</TableCell>
                          <TableCell rowSpan={2} sx={{ py: 0.75, fontSize: 11, verticalAlign: 'middle', borderBottom: 1, borderColor: 'divider' }}>{betTime}</TableCell>
                          <TableCell sx={{ py: 0.75, fontSize: 11, fontWeight: bet.pick_team === 'away' ? 700 : 400, borderBottom: 0, bgcolor: bet.pick_team === 'away' ? (isDark ? 'rgba(46, 125, 50, 0.12)' : 'rgba(46, 125, 50, 0.08)') : undefined }}>{bet.away_team || '-'}</TableCell>
                          <TableCell rowSpan={2} sx={{ py: 0.75, fontSize: 11, verticalAlign: 'middle', borderBottom: 1, borderColor: 'divider' }}>{formatBetType(bet.bet_type)}</TableCell>
                          <TableCell rowSpan={2} sx={{ py: 0.75, fontSize: 11, verticalAlign: 'middle', fontWeight: 600, color: isNegEdge ? 'text.secondary' : 'success.main', borderBottom: 1, borderColor: 'divider' }}>{bet.predicted_side}</TableCell>
                          <TableCell rowSpan={2} align="center" sx={{ py: 0.75, fontSize: 11, fontFamily: 'monospace', verticalAlign: 'middle', borderBottom: 1, borderColor: 'divider' }}>{formatLine(bet.line, bet.bet_type)}</TableCell>
                          <TableCell rowSpan={2} align="center" sx={{ py: 0.75, fontSize: 11, fontFamily: 'monospace', verticalAlign: 'middle', borderBottom: 1, borderColor: 'divider' }}>{formatOdds(bet.odds)}</TableCell>
                          <TableCell rowSpan={2} align="center" sx={{ py: 0.75, fontSize: 11, verticalAlign: 'middle', borderBottom: 1, borderColor: 'divider' }}>${config.flat_amount}</TableCell>
                          <TableCell rowSpan={2} align="center" sx={{ py: 0.75, fontSize: 11, verticalAlign: 'middle', borderBottom: 1, borderColor: 'divider' }}>{(bet.probability * 100).toFixed(1)}%</TableCell>
                          <TableCell rowSpan={2} align="center" sx={{ py: 0.75, fontSize: 11, verticalAlign: 'middle', borderBottom: 1, borderColor: 'divider', color: isNegEdge ? 'error.main' : edgeVal >= 3 ? 'success.main' : edgeVal >= 1 ? 'warning.main' : 'text.secondary' }}>
                            {edgeVal >= 0 ? '+' : ''}{edgeVal.toFixed(1)}%
                          </TableCell>
                          <TableCell rowSpan={2} sx={{ py: 0.75, fontSize: 11, verticalAlign: 'middle', borderBottom: 1, borderColor: 'divider' }}><TierBadge tier={bet.signal_tier || 'D'} /></TableCell>
                          <TableCell rowSpan={2} sx={{ py: 0.75, fontSize: 11, verticalAlign: 'middle', borderBottom: 1, borderColor: 'divider' }}>{getStatusChip(bet.result)}</TableCell>
                          <TableCell rowSpan={2} align="right" sx={{ py: 0.75, fontSize: 11, verticalAlign: 'middle', borderBottom: 1, borderColor: 'divider', color: bet.profit_loss != null ? (bet.profit_loss > 0 ? 'success.main' : bet.profit_loss < 0 ? 'error.main' : 'inherit') : 'text.secondary', fontWeight: 600 }}>
                            {bet.profit_loss != null ? `${bet.profit_loss > 0 ? '+' : ''}${formatCurrency(bet.profit_loss)}` : '-'}
                          </TableCell>
                          <TableCell rowSpan={2} align="right" sx={{ py: 0.75, fontSize: 11, verticalAlign: 'middle', borderBottom: 1, borderColor: 'divider', color: bet.clv != null ? (bet.clv > 0 ? 'success.main' : bet.clv < 0 ? 'error.main' : 'inherit') : 'text.secondary' }}>
                            {bet.clv != null ? `${bet.clv > 0 ? '+' : ''}${(bet.clv * 100).toFixed(1)}%` : '-'}
                          </TableCell>
                        </TableRow>
                        <TableRow sx={{ opacity: isPending ? 0.85 : 1 }}>
                          <TableCell sx={{ py: 0.75, fontSize: 11, fontWeight: bet.pick_team === 'home' ? 700 : 400, borderBottom: 1, borderColor: 'divider', bgcolor: bet.pick_team === 'home' ? (isDark ? 'rgba(46, 125, 50, 0.12)' : 'rgba(46, 125, 50, 0.08)') : undefined }}>{bet.home_team || '-'}</TableCell>
                        </TableRow>
                      </React.Fragment>
                    );
                  })}
                  {paginatedBets.length === 0 && !loading && (
                    <TableRow><TableCell colSpan={15} align="center" sx={{ py: 4 }}><Typography color="text.secondary">No bets found. Enable tiers in configuration and save.</Typography></TableCell></TableRow>
                  )}
                </TableBody>
              </Table>
            </TableContainer>
            <TablePagination
              component="div"
              count={sortedBets.length}
              page={page}
              onPageChange={(_, p) => setPage(p)}
              rowsPerPage={rowsPerPage}
              onRowsPerPageChange={(e) => { setRowsPerPage(parseInt(e.target.value)); setPage(0); }}
              rowsPerPageOptions={[20, 50, 100, { value: -1, label: 'All' }]}
              labelRowsPerPage="Bets per page:"
            />
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Betting;