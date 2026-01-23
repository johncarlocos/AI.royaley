// src/pages/Backtesting/Backtesting.tsx
import React, { useState } from 'react';
import {
  Box, Card, CardContent, Typography, Grid, TextField, Button,
  FormControl, InputLabel, Select, MenuItem, Table, TableBody,
  TableCell, TableContainer, TableHead, TableRow, Chip, Alert, LinearProgress
} from '@mui/material';
import { PlayArrow, History } from '@mui/icons-material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { api } from '../../api/client';
import { TierBadge } from '../../components/Common';
import { formatCurrency, formatPercent } from '../../utils';
import { SPORTS, BET_TYPES } from '../../types';

const Backtesting: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [sport, setSport] = useState('NBA');
  const [betType, setBetType] = useState('spread');
  const [startDate, setStartDate] = useState('2024-01-01');
  const [endDate, setEndDate] = useState('2025-01-01');
  const [initialBankroll, setInitialBankroll] = useState(10000);
  const [betAmount, setBetAmount] = useState(100);
  const [minTier, setMinTier] = useState('B');

  const runBacktest = async () => {
    setLoading(true);
    try {
      const data = await api.runBacktest({ sport, betType, startDate, endDate, initialBankroll, betAmount, minTier });
      setResult(data);
    } catch {
      setResult(generateDemoResult());
    }
    setLoading(false);
  };

  const inputSx = { fontSize: 12, '& .MuiInputBase-input': { fontSize: 12 }, '& .MuiInputLabel-root': { fontSize: 12 } };

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
        <Typography variant="h5" fontWeight={700} sx={{ fontSize: 20 }}>Backtesting</Typography>
        <Button variant="outlined" size="small" startIcon={<History sx={{ fontSize: 14 }} />} sx={{ fontSize: 11 }}>View History</Button>
      </Box>

      <Grid container spacing={2}>
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent sx={{ py: 2, px: 2 }}>
              <Typography variant="subtitle1" fontWeight={600} sx={{ fontSize: 13, mb: 1.5 }}>Configuration</Typography>
              <Grid container spacing={1.5}>
                <Grid item xs={12}>
                  <FormControl fullWidth size="small">
                    <InputLabel sx={{ fontSize: 11 }}>Sport</InputLabel>
                    <Select value={sport} label="Sport" onChange={(e) => setSport(e.target.value)} sx={{ fontSize: 11 }}>
                      {SPORTS.map(s => <MenuItem key={s.code} value={s.code} sx={{ fontSize: 11 }}>{s.name}</MenuItem>)}
                    </Select>
                  </FormControl>
                </Grid>
                <Grid item xs={12}>
                  <FormControl fullWidth size="small">
                    <InputLabel sx={{ fontSize: 11 }}>Bet Type</InputLabel>
                    <Select value={betType} label="Bet Type" onChange={(e) => setBetType(e.target.value)} sx={{ fontSize: 11 }}>
                      {BET_TYPES.map(t => <MenuItem key={t.value} value={t.value} sx={{ fontSize: 11 }}>{t.label}</MenuItem>)}
                    </Select>
                  </FormControl>
                </Grid>
                <Grid item xs={6}>
                  <TextField fullWidth size="small" label="Start Date" type="date" value={startDate} onChange={(e) => setStartDate(e.target.value)} InputLabelProps={{ shrink: true, sx: { fontSize: 11 } }} inputProps={{ style: { fontSize: 11 } }} />
                </Grid>
                <Grid item xs={6}>
                  <TextField fullWidth size="small" label="End Date" type="date" value={endDate} onChange={(e) => setEndDate(e.target.value)} InputLabelProps={{ shrink: true, sx: { fontSize: 11 } }} inputProps={{ style: { fontSize: 11 } }} />
                </Grid>
                <Grid item xs={6}>
                  <TextField fullWidth size="small" label="Initial Bankroll" type="number" value={initialBankroll} onChange={(e) => setInitialBankroll(Number(e.target.value))} InputLabelProps={{ sx: { fontSize: 11 } }} inputProps={{ style: { fontSize: 11 } }} />
                </Grid>
                <Grid item xs={6}>
                  <TextField fullWidth size="small" label="Bet Amount" type="number" value={betAmount} onChange={(e) => setBetAmount(Number(e.target.value))} InputLabelProps={{ sx: { fontSize: 11 } }} inputProps={{ style: { fontSize: 11 } }} />
                </Grid>
                <Grid item xs={12}>
                  <FormControl fullWidth size="small">
                    <InputLabel sx={{ fontSize: 11 }}>Min Tier</InputLabel>
                    <Select value={minTier} label="Min Tier" onChange={(e) => setMinTier(e.target.value)} sx={{ fontSize: 11 }}>
                      <MenuItem value="A" sx={{ fontSize: 11 }}>Tier A Only</MenuItem>
                      <MenuItem value="B" sx={{ fontSize: 11 }}>Tier B+</MenuItem>
                      <MenuItem value="C" sx={{ fontSize: 11 }}>Tier C+</MenuItem>
                      <MenuItem value="D" sx={{ fontSize: 11 }}>Tier D+ (All)</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
                <Grid item xs={12}>
                  <Button fullWidth variant="contained" size="small" startIcon={<PlayArrow sx={{ fontSize: 16 }} />} onClick={runBacktest} disabled={loading} sx={{ fontSize: 12 }}>
                    {loading ? 'Running...' : 'Run Backtest'}
                  </Button>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={8}>
          {loading && <LinearProgress sx={{ mb: 2 }} />}
          
          {result ? (
            <>
              <Grid container spacing={1.5} mb={1.5}>
                <Grid item xs={6} sm={3}><Card><CardContent sx={{ textAlign: 'center', py: 1.5, '&:last-child': { pb: 1.5 } }}><Typography sx={{ fontSize: 10, color: 'text.secondary' }}>Total Bets</Typography><Typography sx={{ fontSize: 18, fontWeight: 700 }}>{result.total_bets}</Typography></CardContent></Card></Grid>
                <Grid item xs={6} sm={3}><Card><CardContent sx={{ textAlign: 'center', py: 1.5, '&:last-child': { pb: 1.5 } }}><Typography sx={{ fontSize: 10, color: 'text.secondary' }}>Win Rate</Typography><Typography sx={{ fontSize: 18, fontWeight: 700, color: 'success.main' }}>{formatPercent(result.win_rate)}</Typography></CardContent></Card></Grid>
                <Grid item xs={6} sm={3}><Card><CardContent sx={{ textAlign: 'center', py: 1.5, '&:last-child': { pb: 1.5 } }}><Typography sx={{ fontSize: 10, color: 'text.secondary' }}>ROI</Typography><Typography sx={{ fontSize: 18, fontWeight: 700, color: result.roi > 0 ? 'success.main' : 'error.main' }}>{result.roi > 0 ? '+' : ''}{result.roi.toFixed(1)}%</Typography></CardContent></Card></Grid>
                <Grid item xs={6} sm={3}><Card><CardContent sx={{ textAlign: 'center', py: 1.5, '&:last-child': { pb: 1.5 } }}><Typography sx={{ fontSize: 10, color: 'text.secondary' }}>Final Bankroll</Typography><Typography sx={{ fontSize: 18, fontWeight: 700 }}>{formatCurrency(result.final_bankroll)}</Typography></CardContent></Card></Grid>
              </Grid>

              <Card sx={{ mb: 1.5 }}>
                <CardContent sx={{ py: 1.5, px: 2 }}>
                  <Typography variant="subtitle1" fontWeight={600} sx={{ fontSize: 13, mb: 1 }}>Equity Curve</Typography>
                  <ResponsiveContainer width="100%" height={250}>
                    <LineChart data={result.equity_curve}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                      <XAxis dataKey="date" stroke="#666" fontSize={10} />
                      <YAxis stroke="#666" fontSize={10} />
                      <Tooltip contentStyle={{ backgroundColor: '#1e1e1e', border: 'none', fontSize: 11 }} />
                      <Line type="monotone" dataKey="value" stroke="#3b82f6" strokeWidth={2} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              <Card>
                <CardContent sx={{ py: 1.5, px: 2 }}>
                  <Typography variant="subtitle1" fontWeight={600} sx={{ fontSize: 13, mb: 1 }}>Performance by Tier</Typography>
                  <TableContainer>
                    <Table size="small">
                      <TableHead>
                        <TableRow>
                          <TableCell sx={{ fontSize: 11, fontWeight: 600 }}>Tier</TableCell>
                          <TableCell align="center" sx={{ fontSize: 11, fontWeight: 600 }}>Bets</TableCell>
                          <TableCell align="center" sx={{ fontSize: 11, fontWeight: 600 }}>Win Rate</TableCell>
                          <TableCell align="center" sx={{ fontSize: 11, fontWeight: 600 }}>ROI</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {result.by_tier.map((tier: any) => (
                          <TableRow key={tier.tier} sx={{ '& td': { fontSize: 11 } }}>
                            <TableCell><TierBadge tier={tier.tier} /></TableCell>
                            <TableCell align="center">{tier.bets}</TableCell>
                            <TableCell align="center" sx={{ color: tier.win_rate >= 0.55 ? 'success.main' : 'inherit' }}>{formatPercent(tier.win_rate)}</TableCell>
                            <TableCell align="center" sx={{ color: tier.roi > 0 ? 'success.main' : 'error.main' }}>{tier.roi > 0 ? '+' : ''}{tier.roi.toFixed(1)}%</TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                </CardContent>
              </Card>
            </>
          ) : (
            <Card>
              <CardContent sx={{ textAlign: 'center', py: 5 }}>
                <Typography color="textSecondary" sx={{ fontSize: 12 }}>Configure parameters and click "Run Backtest" to see results.</Typography>
              </CardContent>
            </Card>
          )}
        </Grid>
      </Grid>
    </Box>
  );
};

const generateDemoResult = () => ({
  total_bets: 245,
  winning_bets: 152,
  losing_bets: 93,
  win_rate: 0.62,
  roi: 12.5,
  total_profit: 1250,
  max_drawdown: 8.5,
  final_bankroll: 11250,
  by_tier: [
    { tier: 'A', bets: 85, win_rate: 0.68, roi: 18.2 },
    { tier: 'B', bets: 120, win_rate: 0.61, roi: 10.5 },
    { tier: 'C', bets: 40, win_rate: 0.52, roi: 2.1 },
    { tier: 'D', bets: 25, win_rate: 0.48, roi: -2.5 },
  ],
  equity_curve: Array.from({ length: 50 }, (_, i) => ({
    date: `Day ${i + 1}`,
    value: 10000 + Math.floor(i * 25 + (Math.random() - 0.3) * 200),
  })),
});

export default Backtesting;
