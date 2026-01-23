// src/pages/GameProps/GameProps.tsx
import React, { useState, useEffect, useMemo } from 'react';
import {
  Box, Card, CardContent, Typography, Grid, Table, TableBody, TableCell,
  TableContainer, TableHead, TableRow, Chip, Select, MenuItem, FormControl,
  InputLabel, Button, Alert, LinearProgress, useTheme, Tabs, Tab, IconButton,
  Dialog, DialogTitle, DialogContent, DialogActions, TablePagination
} from '@mui/material';
import { Refresh, CheckCircle, Cancel, Schedule, Remove, ExpandMore, SportsScore, Casino } from '@mui/icons-material';
import { TierBadge } from '../../components/Common';
import { formatPercent } from '../../utils';

interface GameProp {
  id: string;
  sport: string;
  date: string;
  time: string;
  game_number: number;
  away_team: string;
  home_team: string;
  prop_type: string;
  prop_label: string;
  team: string;
  circa_open: number | string;
  circa_current: number | string;
  system_open: number | string;
  system_current: number | string;
  pick: string;
  pick_direction: 'over' | 'under' | 'away' | 'home';
  probability: number;
  edge: number;
  tier: string;
  status: 'pending' | 'won' | 'lost' | 'push';
  actual?: number | string;
  reason: string;
}

const GameProps: React.FC = () => {
  const theme = useTheme();
  const isDark = theme.palette.mode === 'dark';
  
  const [gameProps, setGameProps] = useState<GameProp[]>([]);
  const [loading, setLoading] = useState(true);
  const [sportFilter, setSportFilter] = useState('all');
  const [tierFilter, setTierFilter] = useState('all');
  const [tab, setTab] = useState(0);
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(50);
  const [reasonDialog, setReasonDialog] = useState<{ open: boolean; prop: GameProp | null }>({ open: false, prop: null });

  useEffect(() => { loadGameProps(); }, []);

  const loadGameProps = async () => {
    setLoading(true);
    try {
      setGameProps(generateDemoGameProps());
    } catch {
      setGameProps(generateDemoGameProps());
    }
    setLoading(false);
  };

  const filteredGameProps = useMemo(() => {
    return gameProps.filter(p => {
      const sportMatch = sportFilter === 'all' || p.sport === sportFilter;
      const tierMatch = tierFilter === 'all' || p.tier === tierFilter;
      const tabMatch = tab === 0 || (tab === 1 && p.status === 'pending') || (tab === 2 && p.status !== 'pending');
      return sportMatch && tierMatch && tabMatch;
    });
  }, [gameProps, sportFilter, tierFilter, tab]);

  const totalCount = filteredGameProps.length;
  const paginated = useMemo(() => {
    if (rowsPerPage === -1) return filteredGameProps;
    return filteredGameProps.slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage);
  }, [filteredGameProps, page, rowsPerPage]);

  const stats = {
    total: gameProps.length,
    tierA: gameProps.filter(p => p.tier === 'A').length,
    avgEdge: gameProps.length > 0 ? gameProps.reduce((s, p) => s + p.edge, 0) / gameProps.length : 0,
    winRate: gameProps.filter(p => p.status !== 'pending').length > 0 ? gameProps.filter(p => p.status === 'won').length / gameProps.filter(p => p.status !== 'pending').length : 0,
  };

  const getStatusChip = (status: string) => {
    switch (status) {
      case 'won': return <Chip icon={<CheckCircle />} label="" size="small" color="success" sx={{ '& .MuiChip-icon': { fontSize: 16, ml: 0.5, mr: -0.5 }, minWidth: 28, height: 22 }} />;
      case 'lost': return <Chip icon={<Cancel />} label="" size="small" color="error" sx={{ '& .MuiChip-icon': { fontSize: 16, ml: 0.5, mr: -0.5 }, minWidth: 28, height: 22 }} />;
      case 'push': return <Chip icon={<Remove />} label="" size="small" sx={{ '& .MuiChip-icon': { fontSize: 16, ml: 0.5, mr: -0.5 }, minWidth: 28, height: 22 }} />;
      default: return <Chip icon={<Schedule />} label="" size="small" sx={{ '& .MuiChip-icon': { fontSize: 16, ml: 0.5, mr: -0.5 }, minWidth: 28, height: 22 }} />;
    }
  };

  const formatLine = (value: number | string | undefined): string => {
    if (value === undefined || value === null || value === '') return '-';
    const num = typeof value === 'string' ? parseFloat(value) : value;
    if (isNaN(num)) return String(value);
    return num > 0 ? `+${num}` : `${num}`;
  };

  const hdr = { fontWeight: 600, fontSize: 11, py: 0.75, bgcolor: isDark ? 'grey.900' : 'grey.100', color: isDark ? 'grey.100' : 'grey.800', whiteSpace: 'nowrap', borderBottom: 1, borderColor: 'divider' };

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={1.5}>
        <Box display="flex" alignItems="center" gap={1}>
          <SportsScore sx={{ fontSize: 24, color: 'secondary.main' }} />
          <Typography variant="h5" fontWeight={700} sx={{ fontSize: 20 }}>Game Props</Typography>
        </Box>
        <Button variant="contained" color="secondary" size="small" startIcon={<Refresh sx={{ fontSize: 14 }} />} onClick={loadGameProps} sx={{ fontSize: 11 }}>Refresh</Button>
      </Box>

      <Alert severity="success" icon={<Casino sx={{ fontSize: 18 }} />} sx={{ mb: 1.5, py: 0.5, '& .MuiAlert-message': { fontSize: 11 } }}>
        <strong>Game Props Predictions:</strong> Team totals, 1st half spreads/totals, 1st quarter lines, and alternate lines.
      </Alert>

      <Grid container spacing={1.5} mb={1.5}>
        <Grid item xs={6} sm={3}><Card><CardContent sx={{ textAlign: 'center', py: 1.5, '&:last-child': { pb: 1.5 } }}><Typography sx={{ fontSize: 11, color: 'text.secondary' }}>Total Game Props</Typography><Typography sx={{ fontSize: 20, fontWeight: 700 }}>{stats.total}</Typography></CardContent></Card></Grid>
        <Grid item xs={6} sm={3}><Card><CardContent sx={{ textAlign: 'center', py: 1.5, '&:last-child': { pb: 1.5 } }}><Typography sx={{ fontSize: 11, color: 'text.secondary' }}>Tier A Props</Typography><Typography sx={{ fontSize: 20, fontWeight: 700, color: 'success.main' }}>{stats.tierA}</Typography></CardContent></Card></Grid>
        <Grid item xs={6} sm={3}><Card><CardContent sx={{ textAlign: 'center', py: 1.5, '&:last-child': { pb: 1.5 } }}><Typography sx={{ fontSize: 11, color: 'text.secondary' }}>Avg Edge</Typography><Typography sx={{ fontSize: 20, fontWeight: 700, color: 'info.main' }}>+{stats.avgEdge.toFixed(1)}%</Typography></CardContent></Card></Grid>
        <Grid item xs={6} sm={3}><Card><CardContent sx={{ textAlign: 'center', py: 1.5, '&:last-child': { pb: 1.5 } }}><Typography sx={{ fontSize: 11, color: 'text.secondary' }}>Win Rate</Typography><Typography sx={{ fontSize: 20, fontWeight: 700, color: stats.winRate >= 0.55 ? 'success.main' : 'warning.main' }}>{formatPercent(stats.winRate)}</Typography></CardContent></Card></Grid>
      </Grid>

      {/* Tabs + Filters Row - Same format as Predictions */}
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', borderBottom: 1, borderColor: 'divider', mb: 1.5 }}>
        <Tabs value={tab} onChange={(_, v) => { setTab(v); setPage(0); }} sx={{ minHeight: 40 }}>
          <Tab label={`All (${gameProps.length})`} sx={{ fontSize: 12, minHeight: 40, py: 0.5 }} />
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
              <MenuItem value="NCAAF" sx={{ fontSize: 12 }}>NCAAF</MenuItem>
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
          <Typography variant="caption" color="text.secondary" sx={{ fontSize: 11 }}>{totalCount} rows • {gameProps.length} props</Typography>
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
                <TableCell sx={hdr} align="center">Game#</TableCell>
                <TableCell sx={hdr}>Away</TableCell>
                <TableCell sx={hdr}>Home</TableCell>
                <TableCell sx={hdr}>Type</TableCell>
                <TableCell sx={hdr}>Team</TableCell>
                <TableCell sx={hdr} align="center">Circa O</TableCell>
                <TableCell sx={hdr} align="center">Circa.</TableCell>
                <TableCell sx={hdr} align="center">System O</TableCell>
                <TableCell sx={hdr} align="center">System.</TableCell>
                <TableCell sx={hdr}>Pick</TableCell>
                <TableCell sx={hdr} align="center">%</TableCell>
                <TableCell sx={hdr} align="center">Edge</TableCell>
                <TableCell sx={hdr}>Tier</TableCell>
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
                  <TableCell align="center" sx={{ fontFamily: 'monospace', fontWeight: 600 }}>{prop.game_number}</TableCell>
                  <TableCell>{prop.away_team}</TableCell>
                  <TableCell>{prop.home_team}</TableCell>
                  <TableCell><Chip label={prop.prop_label} size="small" color={prop.prop_type.includes('spread') ? 'primary' : 'secondary'} sx={{ fontSize: 9, height: 18 }} /></TableCell>
                  <TableCell sx={{ fontWeight: prop.team ? 600 : 400 }}>{prop.team || '-'}</TableCell>
                  <TableCell align="center">{formatLine(prop.circa_open)}</TableCell>
                  <TableCell align="center" sx={{ fontWeight: 600 }}>{formatLine(prop.circa_current)}</TableCell>
                  <TableCell align="center" sx={{ color: 'info.main' }}>{formatLine(prop.system_open)}</TableCell>
                  <TableCell align="center" sx={{ color: 'info.main', fontWeight: 600 }}>{formatLine(prop.system_current)}</TableCell>
                  <TableCell><Typography sx={{ fontSize: 11, fontWeight: 600, color: 'success.main' }}>{prop.pick}</Typography></TableCell>
                  <TableCell align="center">{formatPercent(prop.probability)}</TableCell>
                  <TableCell align="center" sx={{ color: prop.edge >= 3 ? 'success.main' : prop.edge >= 1.5 ? 'warning.main' : 'text.secondary' }}>+{prop.edge.toFixed(1)}%</TableCell>
                  <TableCell><TierBadge tier={prop.tier} /></TableCell>
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
        <DialogTitle sx={{ fontSize: 14, fontWeight: 600 }}>{reasonDialog.prop?.away_team} @ {reasonDialog.prop?.home_team} - {reasonDialog.prop?.prop_label}</DialogTitle>
        <DialogContent>
          <Box sx={{ mb: 2, p: 1.5, bgcolor: 'action.hover', borderRadius: 1 }}>
            <Typography sx={{ fontSize: 14, fontWeight: 700, color: 'success.main' }}>{reasonDialog.prop?.pick}</Typography>
          </Box>
          <Typography variant="body2" sx={{ fontSize: 12, mb: 2 }}>{reasonDialog.prop?.reason}</Typography>
          <Grid container spacing={1}>
            <Grid item xs={3}><Typography sx={{ fontSize: 10, color: 'text.secondary' }}>Circa O</Typography><Typography sx={{ fontSize: 12, fontWeight: 600 }}>{formatLine(reasonDialog.prop?.circa_open)}</Typography></Grid>
            <Grid item xs={3}><Typography sx={{ fontSize: 10, color: 'text.secondary' }}>Circa.</Typography><Typography sx={{ fontSize: 12, fontWeight: 600 }}>{formatLine(reasonDialog.prop?.circa_current)}</Typography></Grid>
            <Grid item xs={3}><Typography sx={{ fontSize: 10, color: 'text.secondary' }}>System O</Typography><Typography sx={{ fontSize: 12, fontWeight: 600, color: 'info.main' }}>{formatLine(reasonDialog.prop?.system_open)}</Typography></Grid>
            <Grid item xs={3}><Typography sx={{ fontSize: 10, color: 'text.secondary' }}>System.</Typography><Typography sx={{ fontSize: 12, fontWeight: 600, color: 'info.main' }}>{formatLine(reasonDialog.prop?.system_current)}</Typography></Grid>
            <Grid item xs={4}><Typography sx={{ fontSize: 10, color: 'text.secondary' }}>Probability</Typography><Typography sx={{ fontSize: 12, fontWeight: 600 }}>{formatPercent(reasonDialog.prop?.probability || 0)}</Typography></Grid>
            <Grid item xs={4}><Typography sx={{ fontSize: 10, color: 'text.secondary' }}>Edge</Typography><Typography sx={{ fontSize: 12, fontWeight: 600, color: 'success.main' }}>+{reasonDialog.prop?.edge.toFixed(1)}%</Typography></Grid>
            <Grid item xs={4}><Typography sx={{ fontSize: 10, color: 'text.secondary' }}>Tier</Typography><Typography sx={{ fontSize: 12 }}><TierBadge tier={reasonDialog.prop?.tier || 'D'} /></Typography></Grid>
          </Grid>
        </DialogContent>
        <DialogActions><Button onClick={() => setReasonDialog({ open: false, prop: null })} size="small">Close</Button></DialogActions>
      </Dialog>
    </Box>
  );
};

const generateDemoGameProps = (): GameProp[] => {
  return [
    { id: 'gp1', sport: 'NBA', date: '1/17', time: '5:00 PM', game_number: 501, away_team: 'Celtics', home_team: 'Lakers', prop_type: 'team_total', prop_label: 'TT', team: 'Lakers', circa_open: 112.5, circa_current: 113.5, system_open: 114.0, system_current: 115.5, pick: 'Over 113.5', pick_direction: 'over', probability: 0.66, edge: 3.8, tier: 'A', status: 'pending', reason: 'Lakers averaging 118 PPG at home. Celtics allow 112 on road.' },
    { id: 'gp2', sport: 'NBA', date: '1/17', time: '5:00 PM', game_number: 501, away_team: 'Celtics', home_team: 'Lakers', prop_type: 'team_total', prop_label: 'TT', team: 'Celtics', circa_open: 115.5, circa_current: 116.5, system_open: 114.0, system_current: 113.5, pick: 'Under 116.5', pick_direction: 'under', probability: 0.62, edge: 2.5, tier: 'B', status: 'pending', reason: 'Lakers improved defense at home. Celtics struggle in back-to-back road games.' },
    { id: 'gp3', sport: 'NBA', date: '1/17', time: '7:30 PM', game_number: 503, away_team: 'Warriors', home_team: 'Suns', prop_type: '1h_spread', prop_label: '1H Spd', team: 'Warriors', circa_open: 2.5, circa_current: 2.0, system_open: 1.5, system_current: 1.0, pick: 'Warriors +2', pick_direction: 'away', probability: 0.64, edge: 3.2, tier: 'A', status: 'pending', reason: 'Warriors excellent 1H team, outscoring opponents by 3.2 PPG in first halves.' },
    { id: 'gp4', sport: 'NBA', date: '1/17', time: '7:30 PM', game_number: 503, away_team: 'Warriors', home_team: 'Suns', prop_type: '1h_total', prop_label: '1H Tot', team: '', circa_open: 115.5, circa_current: 116.0, system_open: 116.0, system_current: 116.5, pick: 'Over 116', pick_direction: 'over', probability: 0.54, edge: 0.8, tier: 'D', status: 'pending', reason: 'Close to line. Slight lean on over but low confidence.' },
    { id: 'gp5', sport: 'NBA', date: '1/17', time: '5:00 PM', game_number: 501, away_team: 'Celtics', home_team: 'Lakers', prop_type: '1h_total', prop_label: '1H Tot', team: '', circa_open: 115.5, circa_current: 116.0, system_open: 117.0, system_current: 118.5, pick: 'Over 116', pick_direction: 'over', probability: 0.65, edge: 3.4, tier: 'A', status: 'pending', reason: 'Both teams score heavily in first halves. Combined 1H average of 118.2.' },
    { id: 'gp6', sport: 'NBA', date: '1/17', time: '7:30 PM', game_number: 503, away_team: 'Warriors', home_team: 'Suns', prop_type: '1q_total', prop_label: '1Q Tot', team: '', circa_open: 58.5, circa_current: 59.0, system_open: 60.0, system_current: 61.5, pick: 'Over 59', pick_direction: 'over', probability: 0.64, edge: 3.1, tier: 'B', status: 'pending', reason: 'High-paced matchup. Both teams start fast.' },
    { id: 'gp7', sport: 'NBA', date: '1/17', time: '5:00 PM', game_number: 501, away_team: 'Celtics', home_team: 'Lakers', prop_type: 'alt_spread', prop_label: 'Alt Spd', team: 'Celtics', circa_open: -8.5, circa_current: -9.5, system_open: -10.0, system_current: -11.5, pick: 'Celtics -9.5', pick_direction: 'away', probability: 0.58, edge: 2.1, tier: 'C', status: 'pending', reason: 'Celtics dominating Lakers recently. 4 of last 5 wins by 10+.' },
    { id: 'gp8', sport: 'NBA', date: '1/17', time: '7:30 PM', game_number: 503, away_team: 'Warriors', home_team: 'Suns', prop_type: 'alt_total', prop_label: 'Alt Tot', team: '', circa_open: 235.5, circa_current: 236.5, system_open: 236.0, system_current: 236.0, pick: 'Under 236.5', pick_direction: 'under', probability: 0.51, edge: 0.3, tier: 'D', status: 'pending', reason: 'Very close to system projection. No real edge.' },
    { id: 'gp9', sport: 'NFL', date: '1/19', time: '1:00 PM', game_number: 505, away_team: 'Chiefs', home_team: 'Bills', prop_type: '1h_spread', prop_label: '1H Spd', team: 'Chiefs', circa_open: 1.5, circa_current: 1.0, system_open: 0.5, system_current: -0.5, pick: 'Chiefs +1', pick_direction: 'away', probability: 0.63, edge: 2.9, tier: 'B', status: 'pending', reason: 'Chiefs historically strong in playoff first halves. Mahomes 8-2 ATS in 1H playoff games.' },
    { id: 'gp10', sport: 'NFL', date: '1/19', time: '1:00 PM', game_number: 505, away_team: 'Chiefs', home_team: 'Bills', prop_type: '1h_total', prop_label: '1H Tot', team: '', circa_open: 24.5, circa_current: 24.0, system_open: 22.5, system_current: 22.0, pick: 'Under 24', pick_direction: 'under', probability: 0.61, edge: 2.3, tier: 'B', status: 'pending', reason: 'Playoff games typically start slow. Cold weather factor.' },
    { id: 'gp11', sport: 'NFL', date: '1/19', time: '1:00 PM', game_number: 505, away_team: 'Chiefs', home_team: 'Bills', prop_type: 'alt_spread', prop_label: 'Alt Spd', team: 'Bills', circa_open: -6.5, circa_current: -7.0, system_open: -6.0, system_current: -5.5, pick: 'Bills -7', pick_direction: 'home', probability: 0.52, edge: 0.6, tier: 'D', status: 'pending', reason: 'Chiefs keep games close. Low confidence alt spread.' },
    { id: 'gp12', sport: 'NHL', date: '1/17', time: '7:00 PM', game_number: 507, away_team: 'Leafs', home_team: 'Bruins', prop_type: 'team_total', prop_label: 'TT', team: 'Bruins', circa_open: 3.5, circa_current: 3.5, system_open: 3.5, system_current: 4.0, pick: 'Over 3.5', pick_direction: 'over', probability: 0.61, edge: 2.4, tier: 'B', status: 'pending', reason: 'Bruins averaging 3.8 goals at home. Power play clicking.' },
    { id: 'gp13', sport: 'NHL', date: '1/17', time: '7:00 PM', game_number: 507, away_team: 'Leafs', home_team: 'Bruins', prop_type: '1q_total', prop_label: '1P Tot', team: '', circa_open: 1.5, circa_current: 1.5, system_open: 2.0, system_current: 2.5, pick: 'Over 1.5', pick_direction: 'over', probability: 0.58, edge: 1.8, tier: 'C', status: 'pending', reason: 'Both teams score early. Combined 1P goals: 2.4 average.' },
    { id: 'gp14', sport: 'NBA', date: '1/16', time: '8:00 PM', game_number: 497, away_team: 'Heat', home_team: 'Bucks', prop_type: 'team_total', prop_label: 'TT', team: 'Bucks', circa_open: 118.5, circa_current: 119.5, system_open: 120.0, system_current: 121.5, pick: 'Over 119.5', pick_direction: 'over', probability: 0.65, edge: 3.2, tier: 'A', status: 'won', actual: 124, reason: 'Bucks averaging 122 at home. Heat defense depleted.' },
    { id: 'gp15', sport: 'NFL', date: '1/12', time: '4:30 PM', game_number: 495, away_team: 'Cowboys', home_team: 'Packers', prop_type: '1h_spread', prop_label: '1H Spd', team: 'Packers', circa_open: -3.5, circa_current: -3.0, system_open: -2.5, system_current: -2.0, pick: 'Packers -3', pick_direction: 'home', probability: 0.62, edge: 2.6, tier: 'B', status: 'lost', actual: 'Cowboys +3', reason: 'Packers strong at home. Cowboys poor 1H road team.' },
    { id: 'gp16', sport: 'NBA', date: '1/16', time: '7:30 PM', game_number: 499, away_team: 'Mavs', home_team: 'Nuggets', prop_type: 'alt_total', prop_label: 'Alt Tot', team: '', circa_open: 228.5, circa_current: 229.0, system_open: 229.0, system_current: 229.5, pick: 'Over 229', pick_direction: 'over', probability: 0.53, edge: 0.7, tier: 'D', status: 'push', reason: 'Borderline play. Landed exactly on total.' },
  ];
};

export default GameProps;
