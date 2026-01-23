// src/pages/PlayerProps/PlayerProps.tsx
import React, { useState, useEffect, useMemo } from 'react';
import {
  Box, Card, CardContent, Typography, Grid, Table, TableBody, TableCell,
  TableContainer, TableHead, TableRow, Chip, Select, MenuItem, FormControl,
  InputLabel, Button, Alert, LinearProgress, useTheme, Tabs, Tab, IconButton,
  Dialog, DialogTitle, DialogContent, DialogActions, TablePagination
} from '@mui/material';
import { Refresh, Psychology, CheckCircle, Cancel, Schedule, Remove, ExpandMore } from '@mui/icons-material';
import { api } from '../../api/client';
import { TierBadge } from '../../components/Common';
import { formatPercent } from '../../utils';

interface PlayerProp {
  id: string;
  sport: string;
  date: string;
  time: string;
  player_name: string;
  team: string;
  opponent: string;
  prop_type: string;
  circa_open: number;
  circa_current: number;
  system_open: number;
  system_current: number;
  pick: 'over' | 'under';
  probability: number;
  edge: number;
  tier: string;
  season_avg: number;
  last_5_avg: number;
  last_10_avg: number;
  home_away: 'home' | 'away';
  matchup_rating: string;
  status: 'pending' | 'won' | 'lost' | 'push';
  actual?: number;
  reason: string;
}

const PlayerProps: React.FC = () => {
  const theme = useTheme();
  const isDark = theme.palette.mode === 'dark';
  
  const [props, setProps] = useState<PlayerProp[]>([]);
  const [loading, setLoading] = useState(true);
  const [sportFilter, setSportFilter] = useState('all');
  const [tierFilter, setTierFilter] = useState('all');
  const [tab, setTab] = useState(0);
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(50);
  const [reasonDialog, setReasonDialog] = useState<{ open: boolean; prop: PlayerProp | null }>({ open: false, prop: null });

  useEffect(() => { loadProps(); }, []);

  const loadProps = async () => {
    setLoading(true);
    try {
      const data = await api.getPlayerProps({ sport: sportFilter });
      setProps(Array.isArray(data) ? data : generateDemoPlayerProps());
    } catch {
      setProps(generateDemoPlayerProps());
    }
    setLoading(false);
  };

  const filteredProps = useMemo(() => {
    return props.filter(p => {
      const sportMatch = sportFilter === 'all' || p.sport === sportFilter;
      const tierMatch = tierFilter === 'all' || p.tier === tierFilter;
      const tabMatch = tab === 0 || (tab === 1 && p.status === 'pending') || (tab === 2 && p.status !== 'pending');
      return sportMatch && tierMatch && tabMatch;
    });
  }, [props, sportFilter, tierFilter, tab]);

  const totalCount = filteredProps.length;
  const paginated = useMemo(() => {
    if (rowsPerPage === -1) return filteredProps;
    return filteredProps.slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage);
  }, [filteredProps, page, rowsPerPage]);

  const stats = {
    total: props.length,
    tierA: props.filter(p => p.tier === 'A').length,
    avgEdge: props.length > 0 ? props.reduce((s, p) => s + p.edge, 0) / props.length : 0,
    winRate: props.filter(p => p.status !== 'pending').length > 0 ? props.filter(p => p.status === 'won').length / props.filter(p => p.status !== 'pending').length : 0,
  };

  const getStatusChip = (status: string) => {
    switch (status) {
      case 'won': return <Chip icon={<CheckCircle />} label="" size="small" color="success" sx={{ '& .MuiChip-icon': { fontSize: 16, ml: 0.5, mr: -0.5 }, minWidth: 28, height: 22 }} />;
      case 'lost': return <Chip icon={<Cancel />} label="" size="small" color="error" sx={{ '& .MuiChip-icon': { fontSize: 16, ml: 0.5, mr: -0.5 }, minWidth: 28, height: 22 }} />;
      case 'push': return <Chip icon={<Remove />} label="" size="small" sx={{ '& .MuiChip-icon': { fontSize: 16, ml: 0.5, mr: -0.5 }, minWidth: 28, height: 22 }} />;
      default: return <Chip icon={<Schedule />} label="" size="small" sx={{ '& .MuiChip-icon': { fontSize: 16, ml: 0.5, mr: -0.5 }, minWidth: 28, height: 22 }} />;
    }
  };

  const hdr = { fontWeight: 600, fontSize: 11, py: 0.75, bgcolor: isDark ? 'grey.900' : 'grey.100', color: isDark ? 'grey.100' : 'grey.800', whiteSpace: 'nowrap', borderBottom: 1, borderColor: 'divider' };

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={1.5}>
        <Box display="flex" alignItems="center" gap={1}>
          <Psychology sx={{ fontSize: 24, color: 'primary.main' }} />
          <Typography variant="h5" fontWeight={700} sx={{ fontSize: 20 }}>Player Props</Typography>
        </Box>
        <Button variant="contained" size="small" startIcon={<Refresh sx={{ fontSize: 14 }} />} onClick={loadProps} sx={{ fontSize: 11 }}>Refresh</Button>
      </Box>

      <Alert severity="info" icon={<Psychology sx={{ fontSize: 18 }} />} sx={{ mb: 1.5, py: 0.5, '& .MuiAlert-message': { fontSize: 11 } }}>
        <strong>Player Props Predictions:</strong> ML models predict Over/Under on player stats using season averages, recent form, matchup, and rest days.
      </Alert>

      <Grid container spacing={1.5} mb={1.5}>
        <Grid item xs={6} sm={3}><Card><CardContent sx={{ textAlign: 'center', py: 1.5, '&:last-child': { pb: 1.5 } }}><Typography sx={{ fontSize: 11, color: 'text.secondary' }}>Total Props</Typography><Typography sx={{ fontSize: 20, fontWeight: 700 }}>{stats.total}</Typography></CardContent></Card></Grid>
        <Grid item xs={6} sm={3}><Card><CardContent sx={{ textAlign: 'center', py: 1.5, '&:last-child': { pb: 1.5 } }}><Typography sx={{ fontSize: 11, color: 'text.secondary' }}>Tier A Props</Typography><Typography sx={{ fontSize: 20, fontWeight: 700, color: 'success.main' }}>{stats.tierA}</Typography></CardContent></Card></Grid>
        <Grid item xs={6} sm={3}><Card><CardContent sx={{ textAlign: 'center', py: 1.5, '&:last-child': { pb: 1.5 } }}><Typography sx={{ fontSize: 11, color: 'text.secondary' }}>Avg Edge</Typography><Typography sx={{ fontSize: 20, fontWeight: 700, color: 'info.main' }}>+{stats.avgEdge.toFixed(1)}%</Typography></CardContent></Card></Grid>
        <Grid item xs={6} sm={3}><Card><CardContent sx={{ textAlign: 'center', py: 1.5, '&:last-child': { pb: 1.5 } }}><Typography sx={{ fontSize: 11, color: 'text.secondary' }}>Win Rate</Typography><Typography sx={{ fontSize: 20, fontWeight: 700, color: stats.winRate >= 0.55 ? 'success.main' : 'warning.main' }}>{formatPercent(stats.winRate)}</Typography></CardContent></Card></Grid>
      </Grid>

      {/* Tabs + Filters Row - Same format as Predictions */}
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', borderBottom: 1, borderColor: 'divider', mb: 1.5 }}>
        <Tabs value={tab} onChange={(_, v) => { setTab(v); setPage(0); }} sx={{ minHeight: 40 }}>
          <Tab label={`All (${props.length})`} sx={{ fontSize: 12, minHeight: 40, py: 0.5 }} />
          <Tab label="Pending" sx={{ fontSize: 12, minHeight: 40, py: 0.5 }} />
          <Tab label="Graded" sx={{ fontSize: 12, minHeight: 40, py: 0.5 }} />
        </Tabs>
        <Box display="flex" alignItems="center" gap={1.5}>
          <FormControl size="small" sx={{ minWidth: 100 }}>
            <InputLabel sx={{ fontSize: 12 }}>Sport</InputLabel>
            <Select value={sportFilter} label="Sport" onChange={(e) => { setSportFilter(e.target.value); setPage(0); }} sx={{ fontSize: 12, height: 34 }}>
              <MenuItem value="all" sx={{ fontSize: 12 }}>All Sports</MenuItem>
              <MenuItem value="NBA" sx={{ fontSize: 12 }}>NBA</MenuItem>
              <MenuItem value="NFL" sx={{ fontSize: 12 }}>NFL</MenuItem>
              <MenuItem value="MLB" sx={{ fontSize: 12 }}>MLB</MenuItem>
              <MenuItem value="NHL" sx={{ fontSize: 12 }}>NHL</MenuItem>
              <MenuItem value="NCAAB" sx={{ fontSize: 12 }}>NCAAB</MenuItem>
            </Select>
          </FormControl>
          <FormControl size="small" sx={{ minWidth: 100 }}>
            <InputLabel sx={{ fontSize: 12 }}>Tier</InputLabel>
            <Select value={tierFilter} label="Tier" onChange={(e) => { setTierFilter(e.target.value); setPage(0); }} sx={{ fontSize: 12, height: 34 }}>
              <MenuItem value="all" sx={{ fontSize: 12 }}>All Tiers</MenuItem>
              <MenuItem value="A" sx={{ fontSize: 12 }}>Tier A</MenuItem>
              <MenuItem value="B" sx={{ fontSize: 12 }}>Tier B</MenuItem>
              <MenuItem value="C" sx={{ fontSize: 12 }}>Tier C</MenuItem>
              <MenuItem value="D" sx={{ fontSize: 12 }}>Tier D</MenuItem>
            </Select>
          </FormControl>
          <Typography variant="caption" color="text.secondary" sx={{ fontSize: 11 }}>{totalCount} rows • {props.length} props</Typography>
        </Box>
      </Box>

      <Card>
        {loading && <LinearProgress />}
        <TableContainer>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell sx={hdr}>Sport</TableCell>
                <TableCell sx={hdr}>Date</TableCell>
                <TableCell sx={hdr}>Time</TableCell>
                <TableCell sx={hdr}>Player</TableCell>
                <TableCell sx={hdr}>Team</TableCell>
                <TableCell sx={hdr}>Opp</TableCell>
                <TableCell sx={hdr}>H/A</TableCell>
                <TableCell sx={hdr}>Prop</TableCell>
                <TableCell sx={hdr} align="center">Circa O</TableCell>
                <TableCell sx={hdr} align="center">Circa.</TableCell>
                <TableCell sx={hdr} align="center">System O</TableCell>
                <TableCell sx={hdr} align="center">System.</TableCell>
                <TableCell sx={hdr} align="center">Pick</TableCell>
                <TableCell sx={hdr} align="center">%</TableCell>
                <TableCell sx={hdr} align="center">Edge</TableCell>
                <TableCell sx={hdr}>Tier</TableCell>
                <TableCell sx={hdr} align="center">Avg</TableCell>
                <TableCell sx={hdr} align="center">L5</TableCell>
                <TableCell sx={hdr} align="center">L10</TableCell>
                <TableCell sx={hdr}>Match</TableCell>
                <TableCell sx={hdr}>W/L</TableCell>
                <TableCell sx={hdr} align="center">Act</TableCell>
                <TableCell sx={hdr}></TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {paginated.map((prop) => (
                <TableRow key={prop.id} sx={{ '& td': { py: 0.75, fontSize: 11, borderBottom: 1, borderColor: 'divider' } }}>
                  <TableCell sx={{ fontWeight: 600 }}>{prop.sport}</TableCell>
                  <TableCell>{prop.date}</TableCell>
                  <TableCell>{prop.time}</TableCell>
                  <TableCell sx={{ fontWeight: 600 }}>{prop.player_name}</TableCell>
                  <TableCell>{prop.team}</TableCell>
                  <TableCell>{prop.opponent}</TableCell>
                  <TableCell><Chip label={prop.home_away === 'home' ? 'H' : 'A'} size="small" color={prop.home_away === 'home' ? 'success' : 'default'} sx={{ fontSize: 9, height: 18, minWidth: 20 }} /></TableCell>
                  <TableCell sx={{ textTransform: 'capitalize' }}>{prop.prop_type.replace('_', ' ')}</TableCell>
                  <TableCell align="center">{prop.circa_open}</TableCell>
                  <TableCell align="center" sx={{ fontWeight: 600 }}>{prop.circa_current}</TableCell>
                  <TableCell align="center" sx={{ color: 'info.main' }}>{prop.system_open}</TableCell>
                  <TableCell align="center" sx={{ color: 'info.main', fontWeight: 600 }}>{prop.system_current}</TableCell>
                  <TableCell align="center"><Chip label={prop.pick.toUpperCase()} size="small" color={prop.pick === 'over' ? 'success' : 'error'} sx={{ fontSize: 9, fontWeight: 600, height: 18, minWidth: 40 }} /></TableCell>
                  <TableCell align="center">{formatPercent(prop.probability)}</TableCell>
                  <TableCell align="center" sx={{ color: prop.edge >= 3 ? 'success.main' : prop.edge >= 1.5 ? 'warning.main' : 'text.secondary' }}>+{prop.edge.toFixed(1)}%</TableCell>
                  <TableCell><TierBadge tier={prop.tier} /></TableCell>
                  <TableCell align="center">{prop.season_avg.toFixed(1)}</TableCell>
                  <TableCell align="center" sx={{ color: prop.last_5_avg > prop.season_avg ? 'success.main' : prop.last_5_avg < prop.season_avg ? 'error.main' : 'inherit' }}>{prop.last_5_avg.toFixed(1)}</TableCell>
                  <TableCell align="center" sx={{ color: prop.last_10_avg > prop.season_avg ? 'success.main' : prop.last_10_avg < prop.season_avg ? 'error.main' : 'inherit' }}>{prop.last_10_avg.toFixed(1)}</TableCell>
                  <TableCell><Chip label={prop.matchup_rating.substring(0, 4)} size="small" color={prop.matchup_rating === 'Elite' ? 'success' : prop.matchup_rating === 'Good' ? 'primary' : prop.matchup_rating === 'Neutral' ? 'default' : 'error'} sx={{ fontSize: 9, height: 18 }} /></TableCell>
                  <TableCell>{getStatusChip(prop.status)}</TableCell>
                  <TableCell align="center" sx={{ fontWeight: prop.actual !== undefined ? 600 : 400 }}>{prop.actual !== undefined ? prop.actual : '-'}</TableCell>
                  <TableCell><IconButton size="small" onClick={() => setReasonDialog({ open: true, prop })}><ExpandMore sx={{ fontSize: 16 }} /></IconButton></TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
        <TablePagination
          component="div"
          count={totalCount}
          page={page}
          onPageChange={(_, p) => setPage(p)}
          rowsPerPage={rowsPerPage}
          onRowsPerPageChange={(e) => { setRowsPerPage(parseInt(e.target.value)); setPage(0); }}
          rowsPerPageOptions={[50, 100, 200, 500, { value: -1, label: 'All' }]}
          labelRowsPerPage="Rows per page:"
        />
      </Card>

      <Dialog open={reasonDialog.open} onClose={() => setReasonDialog({ open: false, prop: null })} maxWidth="sm" fullWidth>
        <DialogTitle sx={{ fontSize: 14, fontWeight: 600 }}>{reasonDialog.prop?.player_name} - {reasonDialog.prop?.prop_type.replace('_', ' ')} {reasonDialog.prop?.pick.toUpperCase()}</DialogTitle>
        <DialogContent>
          <Typography variant="body2" sx={{ fontSize: 12, mb: 2 }}>{reasonDialog.prop?.reason}</Typography>
          <Grid container spacing={1}>
            <Grid item xs={4}><Typography sx={{ fontSize: 10, color: 'text.secondary' }}>Season Avg</Typography><Typography sx={{ fontSize: 12, fontWeight: 600 }}>{reasonDialog.prop?.season_avg.toFixed(1)}</Typography></Grid>
            <Grid item xs={4}><Typography sx={{ fontSize: 10, color: 'text.secondary' }}>Last 5 Avg</Typography><Typography sx={{ fontSize: 12, fontWeight: 600 }}>{reasonDialog.prop?.last_5_avg.toFixed(1)}</Typography></Grid>
            <Grid item xs={4}><Typography sx={{ fontSize: 10, color: 'text.secondary' }}>Last 10 Avg</Typography><Typography sx={{ fontSize: 12, fontWeight: 600 }}>{reasonDialog.prop?.last_10_avg.toFixed(1)}</Typography></Grid>
            <Grid item xs={3}><Typography sx={{ fontSize: 10, color: 'text.secondary' }}>Circa O</Typography><Typography sx={{ fontSize: 12, fontWeight: 600 }}>{reasonDialog.prop?.circa_open}</Typography></Grid>
            <Grid item xs={3}><Typography sx={{ fontSize: 10, color: 'text.secondary' }}>Circa.</Typography><Typography sx={{ fontSize: 12, fontWeight: 600 }}>{reasonDialog.prop?.circa_current}</Typography></Grid>
            <Grid item xs={3}><Typography sx={{ fontSize: 10, color: 'text.secondary' }}>System O</Typography><Typography sx={{ fontSize: 12, fontWeight: 600, color: 'info.main' }}>{reasonDialog.prop?.system_open}</Typography></Grid>
            <Grid item xs={3}><Typography sx={{ fontSize: 10, color: 'text.secondary' }}>System.</Typography><Typography sx={{ fontSize: 12, fontWeight: 600, color: 'info.main' }}>{reasonDialog.prop?.system_current}</Typography></Grid>
          </Grid>
        </DialogContent>
        <DialogActions><Button onClick={() => setReasonDialog({ open: false, prop: null })} size="small">Close</Button></DialogActions>
      </Dialog>
    </Box>
  );
};

const generateDemoPlayerProps = (): PlayerProp[] => {
  return [
    { id: '1', sport: 'NBA', date: '1/17', time: '5:00 PM', player_name: 'LeBron James', team: 'LAL', opponent: 'BOS', prop_type: 'points', circa_open: 25.5, circa_current: 26.5, system_open: 26.0, system_current: 27.5, pick: 'over', probability: 0.68, edge: 4.2, tier: 'A', season_avg: 25.8, last_5_avg: 27.4, last_10_avg: 26.5, home_away: 'home', matchup_rating: 'Good', status: 'pending', reason: 'Strong recent form with 27.4 PPG in last 5 games. Boston allows 4th most points to SFs.' },
    { id: '2', sport: 'NBA', date: '1/17', time: '5:00 PM', player_name: 'Stephen Curry', team: 'GSW', opponent: 'PHX', prop_type: 'threes', circa_open: 4.5, circa_current: 4.5, system_open: 4.5, system_current: 5.0, pick: 'over', probability: 0.62, edge: 2.8, tier: 'B', season_avg: 4.8, last_5_avg: 5.2, last_10_avg: 4.9, home_away: 'away', matchup_rating: 'Elite', status: 'pending', reason: 'Phoenix ranks 28th in 3PT defense. Curry averaging 5.2 3PM in last 5 games.' },
    { id: '3', sport: 'NBA', date: '1/17', time: '7:30 PM', player_name: 'Nikola Jokic', team: 'DEN', opponent: 'DAL', prop_type: 'assists', circa_open: 9.5, circa_current: 9.5, system_open: 9.0, system_current: 8.5, pick: 'under', probability: 0.58, edge: 1.9, tier: 'C', season_avg: 9.2, last_5_avg: 8.6, last_10_avg: 8.9, home_away: 'home', matchup_rating: 'Neutral', status: 'pending', reason: 'Dallas forces low assist totals. Jokic trending down with 8.6 APG in last 5.' },
    { id: '4', sport: 'NBA', date: '1/17', time: '7:30 PM', player_name: 'Anthony Davis', team: 'LAL', opponent: 'BOS', prop_type: 'rebounds', circa_open: 11.5, circa_current: 11.5, system_open: 11.0, system_current: 11.0, pick: 'under', probability: 0.53, edge: 0.8, tier: 'D', season_avg: 11.2, last_5_avg: 10.8, last_10_avg: 11.0, home_away: 'home', matchup_rating: 'Tough', status: 'pending', reason: 'Boston strong on boards. AD averaging slightly under line recently. Low confidence play.' },
    { id: '5', sport: 'NBA', date: '1/16', time: '8:00 PM', player_name: 'Jayson Tatum', team: 'BOS', opponent: 'LAL', prop_type: 'rebounds', circa_open: 8.5, circa_current: 8.5, system_open: 9.0, system_current: 9.5, pick: 'over', probability: 0.65, edge: 3.5, tier: 'A', season_avg: 8.8, last_5_avg: 9.2, last_10_avg: 9.0, home_away: 'away', matchup_rating: 'Good', status: 'won', actual: 11, reason: 'LAL weak on defensive boards. Tatum crashing glass more recently.' },
    { id: '6', sport: 'NBA', date: '1/16', time: '7:30 PM', player_name: 'Luka Doncic', team: 'DAL', opponent: 'DEN', prop_type: 'pra', circa_open: 55.5, circa_current: 56.5, system_open: 56.0, system_current: 57.5, pick: 'over', probability: 0.61, edge: 2.4, tier: 'B', season_avg: 56.4, last_5_avg: 57.8, last_10_avg: 56.8, home_away: 'away', matchup_rating: 'Tough', status: 'lost', actual: 52, reason: 'High usage rate game expected. Luka averaging 57.8 PRA in last 5.' },
    { id: '7', sport: 'NBA', date: '1/16', time: '7:30 PM', player_name: 'Devin Booker', team: 'PHX', opponent: 'GSW', prop_type: 'points', circa_open: 26.5, circa_current: 27.0, system_open: 26.5, system_current: 27.0, pick: 'over', probability: 0.54, edge: 1.0, tier: 'D', season_avg: 26.2, last_5_avg: 26.8, last_10_avg: 26.5, home_away: 'home', matchup_rating: 'Neutral', status: 'push', reason: 'Even matchup. Booker consistent but no strong edge. Borderline play.' },
    { id: '8', sport: 'NFL', date: '1/19', time: '1:00 PM', player_name: 'Patrick Mahomes', team: 'KC', opponent: 'BUF', prop_type: 'passing_yards', circa_open: 285.5, circa_current: 290.5, system_open: 295.0, system_current: 300.5, pick: 'over', probability: 0.65, edge: 3.5, tier: 'A', season_avg: 295.2, last_5_avg: 305.8, last_10_avg: 300.4, home_away: 'away', matchup_rating: 'Tough', status: 'pending', reason: 'Playoff atmosphere elevates Mahomes. Buffalo allows chunk plays.' },
    { id: '9', sport: 'NFL', date: '1/19', time: '4:30 PM', player_name: 'Josh Allen', team: 'BUF', opponent: 'KC', prop_type: 'rushing_yards', circa_open: 35.5, circa_current: 38.5, system_open: 40.0, system_current: 42.5, pick: 'over', probability: 0.62, edge: 2.8, tier: 'B', season_avg: 38.4, last_5_avg: 42.6, last_10_avg: 40.5, home_away: 'home', matchup_rating: 'Good', status: 'pending', reason: 'KC defense vulnerable to mobile QBs. Allen running more recently.' },
    { id: '10', sport: 'NFL', date: '1/19', time: '4:30 PM', player_name: 'Travis Kelce', team: 'KC', opponent: 'BUF', prop_type: 'receiving_yards', circa_open: 62.5, circa_current: 65.5, system_open: 63.0, system_current: 64.0, pick: 'under', probability: 0.52, edge: 0.5, tier: 'D', season_avg: 64.2, last_5_avg: 58.4, last_10_avg: 61.2, home_away: 'away', matchup_rating: 'Tough', status: 'pending', reason: 'Buffalo strong vs TEs. Kelce trending down. Very low confidence.' },
    { id: '11', sport: 'NHL', date: '1/17', time: '7:00 PM', player_name: 'Connor McDavid', team: 'EDM', opponent: 'COL', prop_type: 'points', circa_open: 1.5, circa_current: 1.5, system_open: 1.5, system_current: 2.0, pick: 'over', probability: 0.65, edge: 3.5, tier: 'A', season_avg: 1.7, last_5_avg: 2.0, last_10_avg: 1.8, home_away: 'home', matchup_rating: 'Good', status: 'pending', reason: 'Elite matchup. McDavid on 5-game point streak.' },
    { id: '12', sport: 'NHL', date: '1/17', time: '7:00 PM', player_name: 'Nathan MacKinnon', team: 'COL', opponent: 'EDM', prop_type: 'points', circa_open: 1.5, circa_current: 1.5, system_open: 1.5, system_current: 1.5, pick: 'over', probability: 0.55, edge: 1.2, tier: 'C', season_avg: 1.5, last_5_avg: 1.4, last_10_avg: 1.5, home_away: 'away', matchup_rating: 'Neutral', status: 'pending', reason: 'Road game vs tough opponent. Slight edge on points.' },
  ];
};

export default PlayerProps;
