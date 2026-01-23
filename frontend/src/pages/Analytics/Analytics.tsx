// src/pages/Analytics/Analytics.tsx
import React, { useState, useEffect } from 'react';
import {
  Box, Card, CardContent, Typography, Grid, Table, TableBody, TableCell,
  TableContainer, TableHead, TableRow, Chip, Select, MenuItem, FormControl,
  InputLabel, Button, TableSortLabel, LinearProgress, useTheme
} from '@mui/material';
import { Refresh, CheckCircle, Cancel, Schedule, Remove } from '@mui/icons-material';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import { api } from '../../api/client';
import { TierBadge } from '../../components/Common';
import { SPORTS } from '../../types';
import { formatPercent } from '../../utils';

interface AnalyticsRecord {
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
  probability: number;
  edge: number;
  clv: number;
  result: 'won' | 'lost' | 'push' | 'pending';
  tier: string;
}

type SortField = 'sport' | 'bet_type' | 'probability' | 'edge' | 'clv';

const Analytics: React.FC = () => {
  const theme = useTheme();
  const isDark = theme.palette.mode === 'dark';
  const [records, setRecords] = useState<AnalyticsRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [sportFilter, setSportFilter] = useState('all');
  const [typeFilter, setTypeFilter] = useState('all');
  const [probFilter, setProbFilter] = useState('all');
  const [edgeFilter, setEdgeFilter] = useState('all');
  const [clvFilter, setClvFilter] = useState('all');
  const [sortField, setSortField] = useState<SortField>('edge');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');

  useEffect(() => { loadAnalytics(); }, []);

  const loadAnalytics = async () => {
    setLoading(true);
    try {
      const data = await api.getAnalytics();
      setRecords(Array.isArray(data) ? data : generateDemoRecords());
    } catch {
      setRecords(generateDemoRecords());
    }
    setLoading(false);
  };

  const filtered = records.filter(r => {
    if (sportFilter !== 'all' && r.sport !== sportFilter) return false;
    if (typeFilter !== 'all' && r.bet_type !== typeFilter) return false;
    if (probFilter === '60+' && r.probability < 0.60) return false;
    if (probFilter === '65+' && r.probability < 0.65) return false;
    if (probFilter === '70+' && r.probability < 0.70) return false;
    if (edgeFilter === '3+' && r.edge < 3) return false;
    if (edgeFilter === '5+' && r.edge < 5) return false;
    if (clvFilter === '1+' && r.clv < 1) return false;
    if (clvFilter === '2+' && r.clv < 2) return false;
    return true;
  });

  const sorted = [...filtered].sort((a, b) => {
    const aVal = a[sortField];
    const bVal = b[sortField];
    if (typeof aVal === 'string' && typeof bVal === 'string') return sortOrder === 'asc' ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
    return sortOrder === 'asc' ? (aVal as number) - (bVal as number) : (bVal as number) - (aVal as number);
  });

  const handleSort = (field: SortField) => {
    if (sortField === field) setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
    else { setSortField(field); setSortOrder('desc'); }
  };

  const stats = {
    total: filtered.length,
    avgEdge: filtered.length > 0 ? filtered.reduce((s, r) => s + r.edge, 0) / filtered.length : 0,
    avgClv: filtered.filter(r => r.result !== 'pending').length > 0 ? filtered.filter(r => r.result !== 'pending').reduce((s, r) => s + r.clv, 0) / filtered.filter(r => r.result !== 'pending').length : 0,
    winRate: filtered.filter(r => r.result !== 'pending').length > 0 ? filtered.filter(r => r.result === 'won').length / filtered.filter(r => r.result !== 'pending').length : 0,
  };

  const sportData = SPORTS.map(s => ({ sport: s.code, count: filtered.filter(r => r.sport === s.code).length, winRate: filtered.filter(r => r.sport === s.code && r.result !== 'pending').length > 0 ? filtered.filter(r => r.sport === s.code && r.result === 'won').length / filtered.filter(r => r.sport === s.code && r.result !== 'pending').length * 100 : 0 })).filter(d => d.count > 0);
  const tierData = ['A', 'B', 'C', 'D'].map(t => ({ name: `Tier ${t}`, value: filtered.filter(r => r.tier === t).length })).filter(d => d.value > 0);
  const COLORS = ['#4caf50', '#2196f3', '#ff9800', '#9e9e9e'];

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
        <Typography variant="h5" fontWeight={700} sx={{ fontSize: 20 }}>Analytics</Typography>
        <Button variant="outlined" size="small" startIcon={<Refresh />} onClick={loadAnalytics} sx={{ fontSize: 11 }}>Refresh</Button>
      </Box>

      <Grid container spacing={2} mb={2}>
        <Grid item xs={6} sm={3}><Card><CardContent sx={{ textAlign: 'center', py: 1.5, '&:last-child': { pb: 1.5 } }}><Typography sx={{ fontSize: 11, color: 'text.secondary' }}>Total Records</Typography><Typography sx={{ fontSize: 20, fontWeight: 700 }}>{stats.total}</Typography></CardContent></Card></Grid>
        <Grid item xs={6} sm={3}><Card><CardContent sx={{ textAlign: 'center', py: 1.5, '&:last-child': { pb: 1.5 } }}><Typography sx={{ fontSize: 11, color: 'text.secondary' }}>Avg Edge</Typography><Typography sx={{ fontSize: 20, fontWeight: 700, color: 'success.main' }}>{stats.avgEdge.toFixed(1)}%</Typography></CardContent></Card></Grid>
        <Grid item xs={6} sm={3}><Card><CardContent sx={{ textAlign: 'center', py: 1.5, '&:last-child': { pb: 1.5 } }}><Typography sx={{ fontSize: 11, color: 'text.secondary' }}>Avg CLV</Typography><Typography sx={{ fontSize: 20, fontWeight: 700, color: stats.avgClv > 0 ? 'success.main' : 'error.main' }}>{stats.avgClv > 0 ? '+' : ''}{stats.avgClv.toFixed(2)}%</Typography></CardContent></Card></Grid>
        <Grid item xs={6} sm={3}><Card><CardContent sx={{ textAlign: 'center', py: 1.5, '&:last-child': { pb: 1.5 } }}><Typography sx={{ fontSize: 11, color: 'text.secondary' }}>Win Rate</Typography><Typography sx={{ fontSize: 20, fontWeight: 700, color: stats.winRate >= 0.55 ? 'success.main' : 'warning.main' }}>{formatPercent(stats.winRate)}</Typography></CardContent></Card></Grid>
      </Grid>

      <Card sx={{ mb: 2 }}>
        <CardContent sx={{ py: 1.5, px: 2 }}>
          <Typography variant="body2" fontWeight={500} gutterBottom sx={{ fontSize: 12 }}>Filter & Sort</Typography>
          <Grid container spacing={2}>
            <Grid item xs={6} sm={4} md={2}><FormControl fullWidth size="small"><InputLabel sx={{ fontSize: 12 }}>Sport</InputLabel><Select value={sportFilter} label="Sport" onChange={(e) => setSportFilter(e.target.value)} sx={{ fontSize: 12 }}><MenuItem value="all" sx={{ fontSize: 12 }}>All Sports</MenuItem>{SPORTS.map(s => <MenuItem key={s.code} value={s.code} sx={{ fontSize: 12 }}>{s.code}</MenuItem>)}</Select></FormControl></Grid>
            <Grid item xs={6} sm={4} md={2}><FormControl fullWidth size="small"><InputLabel sx={{ fontSize: 12 }}>Type</InputLabel><Select value={typeFilter} label="Type" onChange={(e) => setTypeFilter(e.target.value)} sx={{ fontSize: 12 }}><MenuItem value="all" sx={{ fontSize: 12 }}>All Types</MenuItem><MenuItem value="Spread" sx={{ fontSize: 12 }}>Spread</MenuItem><MenuItem value="Total" sx={{ fontSize: 12 }}>Total</MenuItem><MenuItem value="1H Spread" sx={{ fontSize: 12 }}>1H Spread</MenuItem><MenuItem value="1H Total" sx={{ fontSize: 12 }}>1H Total</MenuItem></Select></FormControl></Grid>
            <Grid item xs={6} sm={4} md={2}><FormControl fullWidth size="small"><InputLabel sx={{ fontSize: 12 }}>Probability</InputLabel><Select value={probFilter} label="Probability" onChange={(e) => setProbFilter(e.target.value)} sx={{ fontSize: 12 }}><MenuItem value="all" sx={{ fontSize: 12 }}>All</MenuItem><MenuItem value="60+" sx={{ fontSize: 12 }}>60%+</MenuItem><MenuItem value="65+" sx={{ fontSize: 12 }}>65%+</MenuItem><MenuItem value="70+" sx={{ fontSize: 12 }}>70%+</MenuItem></Select></FormControl></Grid>
            <Grid item xs={6} sm={4} md={2}><FormControl fullWidth size="small"><InputLabel sx={{ fontSize: 12 }}>Edge</InputLabel><Select value={edgeFilter} label="Edge" onChange={(e) => setEdgeFilter(e.target.value)} sx={{ fontSize: 12 }}><MenuItem value="all" sx={{ fontSize: 12 }}>All</MenuItem><MenuItem value="3+" sx={{ fontSize: 12 }}>3%+</MenuItem><MenuItem value="5+" sx={{ fontSize: 12 }}>5%+</MenuItem></Select></FormControl></Grid>
            <Grid item xs={6} sm={4} md={2}><FormControl fullWidth size="small"><InputLabel sx={{ fontSize: 12 }}>CLV</InputLabel><Select value={clvFilter} label="CLV" onChange={(e) => setClvFilter(e.target.value)} sx={{ fontSize: 12 }}><MenuItem value="all" sx={{ fontSize: 12 }}>All</MenuItem><MenuItem value="1+" sx={{ fontSize: 12 }}>+1%+</MenuItem><MenuItem value="2+" sx={{ fontSize: 12 }}>+2%+</MenuItem></Select></FormControl></Grid>
          </Grid>
        </CardContent>
      </Card>

      <Grid container spacing={2} mb={2}>
        <Grid item xs={12} md={8}>
          <Card><CardContent sx={{ py: 1.5, px: 2 }}><Typography variant="subtitle1" fontWeight={600} gutterBottom sx={{ fontSize: 13 }}>Performance by Sport</Typography><ResponsiveContainer width="100%" height={220}><BarChart data={sportData}><CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" /><XAxis dataKey="sport" stroke="#666" fontSize={10} /><YAxis stroke="#666" fontSize={10} /><Tooltip contentStyle={{ backgroundColor: '#1e1e1e', border: 'none', fontSize: 11 }} /><Bar dataKey="winRate" fill="#3b82f6" name="Win Rate %" radius={[4, 4, 0, 0]} /></BarChart></ResponsiveContainer></CardContent></Card>
        </Grid>
        <Grid item xs={12} md={4}>
          <Card><CardContent sx={{ py: 1.5, px: 2 }}><Typography variant="subtitle1" fontWeight={600} gutterBottom sx={{ fontSize: 13 }}>Distribution by Tier</Typography><ResponsiveContainer width="100%" height={220}><PieChart><Pie data={tierData} dataKey="value" nameKey="name" cx="50%" cy="50%" innerRadius={40} outerRadius={70} label={{ fontSize: 10 }}>{tierData.map((_, index) => <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />)}</Pie><Tooltip contentStyle={{ backgroundColor: '#1e1e1e', border: 'none', fontSize: 11 }} /></PieChart></ResponsiveContainer></CardContent></Card>
        </Grid>
      </Grid>

      <Card>
        <CardContent sx={{ pb: 0.5, pt: 1.5, px: 2 }}><Typography variant="subtitle1" fontWeight={600} sx={{ fontSize: 13 }}>Detailed Records ({sorted.length})</Typography></CardContent>
        {loading && <LinearProgress />}
        <TableContainer>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell sx={hdr}><TableSortLabel active={sortField === 'sport'} direction={sortOrder} onClick={() => handleSort('sport')}>Sport</TableSortLabel></TableCell>
                <TableCell sx={hdr}>Date</TableCell>
                <TableCell sx={hdr}>Time</TableCell>
                <TableCell sx={hdr} align="center">Game #</TableCell>
                <TableCell sx={hdr}>Team</TableCell>
                <TableCell sx={hdr}><TableSortLabel active={sortField === 'bet_type'} direction={sortOrder} onClick={() => handleSort('bet_type')}>Type</TableSortLabel></TableCell>
                <TableCell sx={hdr}>Pick</TableCell>
                <TableCell sx={hdr} align="center"><TableSortLabel active={sortField === 'probability'} direction={sortOrder} onClick={() => handleSort('probability')}>%</TableSortLabel></TableCell>
                <TableCell sx={hdr} align="center"><TableSortLabel active={sortField === 'edge'} direction={sortOrder} onClick={() => handleSort('edge')}>Edge</TableSortLabel></TableCell>
                <TableCell sx={hdr} align="center"><TableSortLabel active={sortField === 'clv'} direction={sortOrder} onClick={() => handleSort('clv')}>CLV</TableSortLabel></TableCell>
                <TableCell sx={hdr}>Tier</TableCell>
                <TableCell sx={hdr}>Result</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {sorted.slice(0, 50).map((r) => (
                <React.Fragment key={r.id}>
                  {/* Away Team Row */}
                  <TableRow>
                    <TableCell rowSpan={2} sx={{ py: 0.75, fontSize: 11, verticalAlign: 'middle', fontWeight: 600, borderBottom: 1, borderColor: 'divider' }}>{r.sport}</TableCell>
                    <TableCell rowSpan={2} sx={{ py: 0.75, fontSize: 11, verticalAlign: 'middle', borderBottom: 1, borderColor: 'divider' }}>{r.date}</TableCell>
                    <TableCell rowSpan={2} sx={{ py: 0.75, fontSize: 11, verticalAlign: 'middle', borderBottom: 1, borderColor: 'divider' }}>{r.time}</TableCell>
                    <TableCell align="center" sx={{ py: 0.75, fontSize: 11, fontFamily: 'monospace', fontWeight: 600, borderBottom: 0 }}>{r.game_number_away}</TableCell>
                    <TableCell sx={{ py: 0.75, fontSize: 11, fontWeight: r.pick_team === 'away' ? 700 : 400, borderBottom: 0 }}>{r.away_team}</TableCell>
                    <TableCell rowSpan={2} sx={{ py: 0.75, fontSize: 11, verticalAlign: 'middle', borderBottom: 1, borderColor: 'divider' }}>{r.bet_type}</TableCell>
                    <TableCell rowSpan={2} sx={{ py: 0.75, fontSize: 11, verticalAlign: 'middle', fontWeight: 600, color: 'success.main', borderBottom: 1, borderColor: 'divider' }}>{r.pick}</TableCell>
                    <TableCell rowSpan={2} align="center" sx={{ py: 0.75, fontSize: 11, verticalAlign: 'middle', borderBottom: 1, borderColor: 'divider' }}>{formatPercent(r.probability)}</TableCell>
                    <TableCell rowSpan={2} align="center" sx={{ py: 0.75, fontSize: 11, verticalAlign: 'middle', borderBottom: 1, borderColor: 'divider', color: r.edge >= 3 ? 'success.main' : 'inherit' }}>+{r.edge.toFixed(1)}%</TableCell>
                    <TableCell rowSpan={2} align="center" sx={{ py: 0.75, fontSize: 11, verticalAlign: 'middle', borderBottom: 1, borderColor: 'divider', color: r.clv > 0 ? 'success.main' : r.clv < 0 ? 'error.main' : 'inherit' }}>{r.clv > 0 ? '+' : ''}{r.clv.toFixed(1)}%</TableCell>
                    <TableCell rowSpan={2} sx={{ py: 0.75, fontSize: 11, verticalAlign: 'middle', borderBottom: 1, borderColor: 'divider' }}><TierBadge tier={r.tier} /></TableCell>
                    <TableCell rowSpan={2} sx={{ py: 0.75, fontSize: 11, verticalAlign: 'middle', borderBottom: 1, borderColor: 'divider' }}>{getStatusChip(r.result)}</TableCell>
                  </TableRow>
                  {/* Home Team Row */}
                  <TableRow>
                    <TableCell align="center" sx={{ py: 0.75, fontSize: 11, fontFamily: 'monospace', fontWeight: 600, borderBottom: 1, borderColor: 'divider' }}>{r.game_number_home}</TableCell>
                    <TableCell sx={{ py: 0.75, fontSize: 11, fontWeight: r.pick_team === 'home' ? 700 : 400, borderBottom: 1, borderColor: 'divider' }}>{r.home_team}</TableCell>
                  </TableRow>
                </React.Fragment>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </Card>
    </Box>
  );
};

const generateDemoRecords = (): AnalyticsRecord[] => {
  const games = [
    { sport: 'NBA', away: 'Boston Celtics', home: 'Los Angeles Lakers', gn_a: 501, gn_h: 502 },
    { sport: 'NBA', away: 'Golden State Warriors', home: 'Phoenix Suns', gn_a: 503, gn_h: 504 },
    { sport: 'NBA', away: 'Miami Heat', home: 'Milwaukee Bucks', gn_a: 505, gn_h: 506 },
    { sport: 'NBA', away: 'Denver Nuggets', home: 'Dallas Mavericks', gn_a: 507, gn_h: 508 },
    { sport: 'NFL', away: 'Kansas City Chiefs', home: 'Buffalo Bills', gn_a: 509, gn_h: 510 },
    { sport: 'NFL', away: 'Philadelphia Eagles', home: 'San Francisco 49ers', gn_a: 511, gn_h: 512 },
    { sport: 'NHL', away: 'Toronto Maple Leafs', home: 'Boston Bruins', gn_a: 513, gn_h: 514 },
    { sport: 'NHL', away: 'Colorado Avalanche', home: 'Vegas Golden Knights', gn_a: 515, gn_h: 516 },
    { sport: 'NCAAB', away: 'Duke Blue Devils', home: 'North Carolina', gn_a: 517, gn_h: 518 },
    { sport: 'NCAAB', away: 'Kentucky Wildcats', home: 'Tennessee Volunteers', gn_a: 519, gn_h: 520 },
    { sport: 'MLB', away: 'New York Yankees', home: 'Los Angeles Dodgers', gn_a: 521, gn_h: 522 },
    { sport: 'MLB', away: 'Houston Astros', home: 'Atlanta Braves', gn_a: 523, gn_h: 524 },
    { sport: 'NCAAF', away: 'Alabama Crimson Tide', home: 'Georgia Bulldogs', gn_a: 525, gn_h: 526 },
    { sport: 'WNBA', away: 'Las Vegas Aces', home: 'New York Liberty', gn_a: 527, gn_h: 528 },
    { sport: 'CFL', away: 'Toronto Argonauts', home: 'BC Lions', gn_a: 529, gn_h: 530 },
  ];
  const types = ['Spread', 'Total', '1H Spread', '1H Total'];
  const tiers = ['A', 'A', 'B', 'B', 'B', 'C', 'C', 'D'];
  const results: ('won' | 'lost' | 'push' | 'pending')[] = ['won', 'won', 'won', 'won', 'lost', 'lost', 'pending', 'push'];
  const dates = ['01/17', '01/16', '01/15', '01/14', '01/13', '01/12', '01/11', '01/10'];
  const times = ['5:00 PM', '7:30 PM', '8:00 PM', '1:00 PM', '4:00 PM', '9:00 PM'];
  
  return Array.from({ length: 50 }, (_, i) => {
    const game = games[i % games.length];
    const tier = tiers[Math.floor(Math.random() * tiers.length)];
    const pickTeam = Math.random() > 0.5 ? 'away' : 'home';
    const type = types[Math.floor(Math.random() * types.length)];
    const isTotal = type.includes('Total');
    
    return {
      id: String(i),
      date: dates[Math.floor(Math.random() * dates.length)],
      time: times[Math.floor(Math.random() * times.length)],
      sport: game.sport,
      game_number_away: game.gn_a,
      game_number_home: game.gn_h,
      away_team: game.away,
      home_team: game.home,
      bet_type: type,
      pick: isTotal ? `${Math.random() > 0.5 ? 'Over' : 'Under'} ${Math.floor(Math.random() * 50 + 200)}` : `${pickTeam === 'away' ? game.away.split(' ').pop() : game.home.split(' ').pop()} ${Math.random() > 0.5 ? '-' : '+'}${(Math.random() * 6 + 1).toFixed(1)}`,
      pick_team: isTotal ? null : (pickTeam as 'away' | 'home'),
      probability: tier === 'A' ? 0.65 + Math.random() * 0.08 : tier === 'B' ? 0.60 + Math.random() * 0.05 : tier === 'C' ? 0.55 + Math.random() * 0.05 : 0.50 + Math.random() * 0.05,
      edge: tier === 'A' ? 3 + Math.random() * 3 : tier === 'B' ? 2 + Math.random() * 2 : 1 + Math.random() * 2,
      clv: (Math.random() - 0.3) * 4,
      result: results[Math.floor(Math.random() * results.length)],
      tier,
    };
  });
};

export default Analytics;
