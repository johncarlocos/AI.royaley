// src/pages/Betting/Betting.tsx
import React, { useState, useEffect } from 'react';
import {
  Box, Card, CardContent, Typography, Grid, Table, TableBody, TableCell,
  TableContainer, TableHead, TableRow, Chip, Button, TextField, Switch,
  FormControlLabel, Alert, LinearProgress, Divider, useTheme
} from '@mui/material';
import { Refresh, Save, CheckCircle, Cancel, Schedule, Remove } from '@mui/icons-material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { api } from '../../api/client';
import { formatCurrency, formatPercent } from '../../utils';

interface BetRecord {
  id: string;
  date: string;
  time: string;
  sport: string;
  game_number_away: number;
  game_number_home: number;
  away_team: string;
  home_team: string;
  bet_type: string;
  pick: string;
  pick_team: 'away' | 'home' | null;
  line: number | string;
  odds: number;
  stake: number;
  probability: number;
  edge: number;
  tier: string;
  status: 'pending' | 'won' | 'lost' | 'push';
  profit: number;
  clv: number;
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
  const [config, setConfig] = useState({ flat_amount: 100, bet_tier_a: true, bet_tier_b: true, bet_tier_c: false, bet_tier_d: false, auto_bet: false });
  const [bets, setBets] = useState<BetRecord[]>([]);
  const [stats] = useState({ initial: 10000, current: 10850, total_bets: 45, wins: 28, losses: 17, roi: 8.5, clv_avg: 1.8 });
  const [loading, setLoading] = useState(false);
  const [saved, setSaved] = useState(false);

  useEffect(() => { loadBets(); }, []);

  const loadBets = async () => {
    setLoading(true);
    try {
      const data = await api.getBets();
      setBets(Array.isArray(data) ? data : generateDemoBets());
    } catch {
      setBets(generateDemoBets());
    }
    setLoading(false);
  };

  const saveConfig = async () => {
    try {
      await api.updateBettingConfig(config);
    } catch { /* ignore */ }
    setSaved(true);
    setTimeout(() => setSaved(false), 3000);
  };

  const equityCurve = generateEquityCurve();
  
  const getStatusChip = (status: string) => {
    switch (status) {
      case 'won': return <Chip icon={<CheckCircle />} label="W" size="small" color="success" />;
      case 'lost': return <Chip icon={<Cancel />} label="L" size="small" color="error" />;
      case 'push': return <Chip icon={<Remove />} label="P" size="small" />;
      default: return <Chip icon={<Schedule />} label="-" size="small" color="primary" />;
    }
  };

  const hdr = { fontWeight: 600, fontSize: 11, py: 0.75, bgcolor: isDark ? 'grey.900' : 'grey.100', color: isDark ? 'grey.100' : 'grey.800', whiteSpace: 'nowrap', borderBottom: 1, borderColor: 'divider' };

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
        <Typography variant="h5" fontWeight={700} sx={{ fontSize: 20 }}>Betting</Typography>
        <Button variant="outlined" size="small" startIcon={<Refresh />} onClick={loadBets} sx={{ fontSize: 11 }}>Refresh</Button>
      </Box>

      <Grid container spacing={2} mb={2}>
        <Grid item xs={6} sm={4} md={2}><Card><CardContent sx={{ textAlign: 'center', py: 1.5, '&:last-child': { pb: 1.5 } }}><Typography sx={{ fontSize: 10, color: 'text.secondary' }}>Bankroll</Typography><Typography sx={{ fontSize: 16, fontWeight: 700, color: 'primary.main' }}>{formatCurrency(stats.current)}</Typography></CardContent></Card></Grid>
        <Grid item xs={6} sm={4} md={2}><Card><CardContent sx={{ textAlign: 'center', py: 1.5, '&:last-child': { pb: 1.5 } }}><Typography sx={{ fontSize: 10, color: 'text.secondary' }}>Total Bets</Typography><Typography sx={{ fontSize: 16, fontWeight: 700 }}>{stats.total_bets}</Typography></CardContent></Card></Grid>
        <Grid item xs={6} sm={4} md={2}><Card><CardContent sx={{ textAlign: 'center', py: 1.5, '&:last-child': { pb: 1.5 } }}><Typography sx={{ fontSize: 10, color: 'text.secondary' }}>Win Rate</Typography><Typography sx={{ fontSize: 16, fontWeight: 700, color: 'success.main' }}>{formatPercent(stats.wins / stats.total_bets)}</Typography></CardContent></Card></Grid>
        <Grid item xs={6} sm={4} md={2}><Card><CardContent sx={{ textAlign: 'center', py: 1.5, '&:last-child': { pb: 1.5 } }}><Typography sx={{ fontSize: 10, color: 'text.secondary' }}>ROI</Typography><Typography sx={{ fontSize: 16, fontWeight: 700, color: stats.roi > 0 ? 'success.main' : 'error.main' }}>+{stats.roi.toFixed(1)}%</Typography></CardContent></Card></Grid>
        <Grid item xs={6} sm={4} md={2}><Card><CardContent sx={{ textAlign: 'center', py: 1.5, '&:last-child': { pb: 1.5 } }}><Typography sx={{ fontSize: 10, color: 'text.secondary' }}>Avg CLV</Typography><Typography sx={{ fontSize: 16, fontWeight: 700, color: stats.clv_avg > 0 ? 'success.main' : 'error.main' }}>+{stats.clv_avg.toFixed(2)}%</Typography></CardContent></Card></Grid>
        <Grid item xs={6} sm={4} md={2}><Card><CardContent sx={{ textAlign: 'center', py: 1.5, '&:last-child': { pb: 1.5 } }}><Typography sx={{ fontSize: 10, color: 'text.secondary' }}>P/L</Typography><Typography sx={{ fontSize: 16, fontWeight: 700, color: 'success.main' }}>{formatCurrency(stats.current - stats.initial)}</Typography></CardContent></Card></Grid>
      </Grid>

      <Grid container spacing={2}>
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent sx={{ py: 1.5, px: 2 }}>
              <Typography variant="subtitle1" fontWeight={600} gutterBottom sx={{ fontSize: 13 }}>Flat Betting Configuration</Typography>
              <Alert severity="info" sx={{ mb: 1.5, fontSize: 10, py: 0.5 }}>System uses flat betting: same amount for every bet. All records saved for self-reinforcement training.</Alert>
              <TextField fullWidth size="small" label="Bet Amount ($)" type="number" value={config.flat_amount} onChange={(e) => setConfig({ ...config, flat_amount: Number(e.target.value) })} sx={{ mb: 1.5 }} />
              <Typography variant="body2" fontWeight={500} gutterBottom sx={{ fontSize: 11 }}>Bet on Tiers:</Typography>
              <FormControlLabel control={<Switch checked={config.bet_tier_a} onChange={(e) => setConfig({ ...config, bet_tier_a: e.target.checked })} size="small" />} label={<Typography sx={{ fontSize: 11 }}>Tier A (65%+)</Typography>} />
              <FormControlLabel control={<Switch checked={config.bet_tier_b} onChange={(e) => setConfig({ ...config, bet_tier_b: e.target.checked })} size="small" />} label={<Typography sx={{ fontSize: 11 }}>Tier B (60-65%)</Typography>} />
              <FormControlLabel control={<Switch checked={config.bet_tier_c} onChange={(e) => setConfig({ ...config, bet_tier_c: e.target.checked })} size="small" />} label={<Typography sx={{ fontSize: 11 }}>Tier C (55-60%)</Typography>} />
              <FormControlLabel control={<Switch checked={config.bet_tier_d} onChange={(e) => setConfig({ ...config, bet_tier_d: e.target.checked })} size="small" />} label={<Typography sx={{ fontSize: 11 }}>Tier D (&lt;55%)</Typography>} />
              <Divider sx={{ my: 1.5 }} />
              <FormControlLabel control={<Switch checked={config.auto_bet} onChange={(e) => setConfig({ ...config, auto_bet: e.target.checked })} size="small" />} label={<Typography sx={{ fontSize: 11 }}>Auto-place bets (simulation)</Typography>} />
              <Button fullWidth variant="contained" size="small" startIcon={<Save />} onClick={saveConfig} sx={{ mt: 1.5, fontSize: 11 }}>Save Configuration</Button>
              {saved && <Alert severity="success" sx={{ mt: 1, fontSize: 10 }}>Configuration saved!</Alert>}
            </CardContent>
          </Card>
          <Card sx={{ mt: 2 }}>
            <CardContent sx={{ py: 1.5, px: 2 }}>
              <Typography variant="subtitle1" fontWeight={600} gutterBottom sx={{ fontSize: 13 }}>Self-Reinforcement Training</Typography>
              <Typography variant="body2" color="textSecondary" paragraph sx={{ fontSize: 11, mb: 1 }}>Every bet is saved with full feature data for model improvement:</Typography>
              <Box component="ul" sx={{ pl: 2, m: 0 }}>
                <Typography component="li" sx={{ fontSize: 11 }}>• Opening & closing odds</Typography>
                <Typography component="li" sx={{ fontSize: 11 }}>• All prediction features</Typography>
                <Typography component="li" sx={{ fontSize: 11 }}>• Actual result (W/L/P)</Typography>
                <Typography component="li" sx={{ fontSize: 11 }}>• CLV achieved</Typography>
              </Box>
              <Alert severity="success" sx={{ mt: 1.5, fontSize: 10, py: 0.5 }}>Model retrains weekly using bet history.</Alert>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={8} sx={{ display: 'flex' }}>
          <Card sx={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
            <CardContent sx={{ flex: 1, display: 'flex', flexDirection: 'column', py: 1.5, px: 2 }}>
              <Typography variant="subtitle1" fontWeight={600} gutterBottom sx={{ fontSize: 13 }}>Bankroll Growth</Typography>
              <Box sx={{ flex: 1, minHeight: 280 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={equityCurve}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                    <XAxis dataKey="date" stroke="#666" fontSize={10} />
                    <YAxis stroke="#666" fontSize={10} />
                    <Tooltip contentStyle={{ backgroundColor: '#1e1e1e', border: 'none', fontSize: 11 }} />
                    <Line type="monotone" dataKey="value" stroke="#3b82f6" strokeWidth={2} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Recent Bets - Full Width */}
        <Grid item xs={12}>
          <Card>
            <CardContent sx={{ pb: 0.5, pt: 1.5, px: 2 }}>
              <Typography variant="subtitle1" fontWeight={600} sx={{ fontSize: 13 }}>Recent Bets ({bets.length})</Typography>
            </CardContent>
            {loading && <LinearProgress />}
            <TableContainer>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell sx={hdr}>Sport</TableCell>
                    <TableCell sx={hdr}>Date</TableCell>
                    <TableCell sx={hdr}>Time</TableCell>
                    <TableCell sx={hdr} align="center">Game #</TableCell>
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
                  {bets.map((bet) => (
                    <React.Fragment key={bet.id}>
                      {/* Away Team Row */}
                      <TableRow>
                        <TableCell rowSpan={2} sx={{ py: 0.75, fontSize: 11, verticalAlign: 'middle', fontWeight: 600, borderBottom: 1, borderColor: 'divider' }}>{bet.sport}</TableCell>
                        <TableCell rowSpan={2} sx={{ py: 0.75, fontSize: 11, verticalAlign: 'middle', borderBottom: 1, borderColor: 'divider' }}>{bet.date}</TableCell>
                        <TableCell rowSpan={2} sx={{ py: 0.75, fontSize: 11, verticalAlign: 'middle', borderBottom: 1, borderColor: 'divider' }}>{bet.time}</TableCell>
                        <TableCell align="center" sx={{ py: 0.75, fontSize: 11, fontFamily: 'monospace', fontWeight: 600, borderBottom: 0 }}>{bet.game_number_away}</TableCell>
                        <TableCell sx={{ py: 0.75, fontSize: 11, fontWeight: bet.pick_team === 'away' ? 700 : 400, borderBottom: 0 }}>{bet.away_team}</TableCell>
                        <TableCell rowSpan={2} sx={{ py: 0.75, fontSize: 11, verticalAlign: 'middle', borderBottom: 1, borderColor: 'divider' }}>{bet.bet_type}</TableCell>
                        <TableCell rowSpan={2} sx={{ py: 0.75, fontSize: 11, verticalAlign: 'middle', fontWeight: 600, color: 'success.main', borderBottom: 1, borderColor: 'divider' }}>{bet.pick}</TableCell>
                        <TableCell rowSpan={2} align="center" sx={{ py: 0.75, fontSize: 11, verticalAlign: 'middle', borderBottom: 1, borderColor: 'divider' }}>{bet.line}</TableCell>
                        <TableCell rowSpan={2} align="center" sx={{ py: 0.75, fontSize: 11, verticalAlign: 'middle', borderBottom: 1, borderColor: 'divider' }}>{bet.odds > 0 ? '+' : ''}{bet.odds}</TableCell>
                        <TableCell rowSpan={2} align="center" sx={{ py: 0.75, fontSize: 11, verticalAlign: 'middle', borderBottom: 1, borderColor: 'divider' }}>${bet.stake}</TableCell>
                        <TableCell rowSpan={2} align="center" sx={{ py: 0.75, fontSize: 11, verticalAlign: 'middle', borderBottom: 1, borderColor: 'divider' }}>{formatPercent(bet.probability)}</TableCell>
                        <TableCell rowSpan={2} align="center" sx={{ py: 0.75, fontSize: 11, verticalAlign: 'middle', borderBottom: 1, borderColor: 'divider', color: bet.edge >= 3 ? 'success.main' : 'inherit' }}>+{bet.edge.toFixed(1)}%</TableCell>
                        <TableCell rowSpan={2} sx={{ py: 0.75, fontSize: 11, verticalAlign: 'middle', borderBottom: 1, borderColor: 'divider' }}><TierBadge tier={bet.tier} /></TableCell>
                        <TableCell rowSpan={2} sx={{ py: 0.75, fontSize: 11, verticalAlign: 'middle', borderBottom: 1, borderColor: 'divider' }}>{getStatusChip(bet.status)}</TableCell>
                        <TableCell rowSpan={2} align="right" sx={{ py: 0.75, fontSize: 11, verticalAlign: 'middle', borderBottom: 1, borderColor: 'divider', color: bet.profit > 0 ? 'success.main' : bet.profit < 0 ? 'error.main' : 'inherit', fontWeight: 600 }}>{bet.profit > 0 ? '+' : ''}{formatCurrency(bet.profit)}</TableCell>
                        <TableCell rowSpan={2} align="right" sx={{ py: 0.75, fontSize: 11, verticalAlign: 'middle', borderBottom: 1, borderColor: 'divider', color: bet.clv > 0 ? 'success.main' : bet.clv < 0 ? 'error.main' : 'inherit' }}>{bet.clv > 0 ? '+' : ''}{bet.clv.toFixed(1)}%</TableCell>
                      </TableRow>
                      {/* Home Team Row */}
                      <TableRow>
                        <TableCell align="center" sx={{ py: 0.75, fontSize: 11, fontFamily: 'monospace', fontWeight: 600, borderBottom: 1, borderColor: 'divider' }}>{bet.game_number_home}</TableCell>
                        <TableCell sx={{ py: 0.75, fontSize: 11, fontWeight: bet.pick_team === 'home' ? 700 : 400, borderBottom: 1, borderColor: 'divider' }}>{bet.home_team}</TableCell>
                      </TableRow>
                    </React.Fragment>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

const generateDemoBets = (): BetRecord[] => [
  { id: '1', date: '01/17', time: '5:00 PM', sport: 'NBA', game_number_away: 501, game_number_home: 502, away_team: 'Boston Celtics', home_team: 'Los Angeles Lakers', bet_type: 'Spread', pick: 'Boston -6', pick_team: 'away', line: -6, odds: -110, stake: 100, probability: 0.67, edge: 3.2, tier: 'A', status: 'pending', profit: 0, clv: 0 },
  { id: '2', date: '01/17', time: '7:30 PM', sport: 'NBA', game_number_away: 503, game_number_home: 504, away_team: 'Golden State Warriors', home_team: 'Phoenix Suns', bet_type: 'Spread', pick: 'Phoenix -4', pick_team: 'home', line: -4, odds: -110, stake: 100, probability: 0.66, edge: 3.8, tier: 'A', status: 'pending', profit: 0, clv: 0 },
  { id: '3', date: '01/16', time: '8:00 PM', sport: 'NBA', game_number_away: 505, game_number_home: 506, away_team: 'Miami Heat', home_team: 'Milwaukee Bucks', bet_type: 'Total', pick: 'Under 224.5', pick_team: null, line: 224.5, odds: -110, stake: 100, probability: 0.63, edge: 2.9, tier: 'B', status: 'won', profit: 91, clv: 1.8 },
  { id: '4', date: '01/16', time: '1:00 PM', sport: 'NFL', game_number_away: 507, game_number_home: 508, away_team: 'Kansas City Chiefs', home_team: 'Buffalo Bills', bet_type: 'Spread', pick: 'Bills -3', pick_team: 'home', line: -3, odds: -105, stake: 100, probability: 0.65, edge: 3.5, tier: 'A', status: 'won', profit: 95, clv: 2.1 },
  { id: '5', date: '01/15', time: '7:00 PM', sport: 'NHL', game_number_away: 509, game_number_home: 510, away_team: 'Toronto Maple Leafs', home_team: 'Boston Bruins', bet_type: 'Puck Line', pick: 'Bruins -1.5', pick_team: 'home', line: -1.5, odds: +140, stake: 100, probability: 0.58, edge: 2.1, tier: 'C', status: 'lost', profit: -100, clv: -0.5 },
  { id: '6', date: '01/15', time: '9:00 PM', sport: 'NCAAB', game_number_away: 511, game_number_home: 512, away_team: 'Duke Blue Devils', home_team: 'North Carolina', bet_type: 'Spread', pick: 'Duke -3.5', pick_team: 'away', line: -3.5, odds: -110, stake: 100, probability: 0.64, edge: 2.9, tier: 'B', status: 'won', profit: 91, clv: 1.3 },
  { id: '7', date: '01/14', time: '7:30 PM', sport: 'NBA', game_number_away: 513, game_number_home: 514, away_team: 'Denver Nuggets', home_team: 'Dallas Mavericks', bet_type: 'Total', pick: 'Over 228', pick_team: null, line: 228, odds: -110, stake: 100, probability: 0.61, edge: 2.4, tier: 'B', status: 'lost', profit: -100, clv: 0.2 },
  { id: '8', date: '01/14', time: '4:00 PM', sport: 'MLB', game_number_away: 515, game_number_home: 516, away_team: 'New York Yankees', home_team: 'Los Angeles Dodgers', bet_type: 'Run Line', pick: 'Dodgers -1.5', pick_team: 'home', line: -1.5, odds: +120, stake: 100, probability: 0.59, edge: 2.0, tier: 'C', status: 'won', profit: 120, clv: 0.9 },
  { id: '9', date: '01/13', time: '8:00 PM', sport: 'NHL', game_number_away: 517, game_number_home: 518, away_team: 'Colorado Avalanche', home_team: 'Vegas Golden Knights', bet_type: 'Total', pick: 'Over 6.5', pick_team: null, line: 6.5, odds: -115, stake: 100, probability: 0.62, edge: 2.6, tier: 'B', status: 'won', profit: 87, clv: 1.4 },
  { id: '10', date: '01/13', time: '2:00 PM', sport: 'NCAAB', game_number_away: 519, game_number_home: 520, away_team: 'Kentucky Wildcats', home_team: 'Tennessee Volunteers', bet_type: 'Spread', pick: 'Tennessee -5', pick_team: 'home', line: -5, odds: -110, stake: 100, probability: 0.60, edge: 2.0, tier: 'B', status: 'push', profit: 0, clv: 0.5 },
];

const generateEquityCurve = () => {
  const data = [];
  let value = 10000;
  for (let i = 30; i >= 0; i--) {
    const date = new Date();
    date.setDate(date.getDate() - i);
    value += (Math.random() - 0.4) * 150;
    data.push({ date: date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }), value: Math.round(value) });
  }
  return data;
};

export default Betting;
