// src/pages/PlayerProps/PlayerProps.tsx
import React, { useState, useEffect, useMemo } from 'react';
import {
  Box, Card, CardContent, Typography, Grid, Table, TableBody, TableCell,
  TableContainer, TableHead, TableRow, Chip, Select, MenuItem, FormControl,
  InputLabel, Button, Alert, LinearProgress, useTheme, Tabs, Tab, IconButton,
  Dialog, DialogTitle, DialogContent, DialogActions, TablePagination
} from '@mui/material';
import { Refresh, Psychology, CheckCircle, Cancel, Schedule, Remove, ExpandMore, SportsBasketball } from '@mui/icons-material';
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
      const data = await api.getPlayerProps({ sport: sportFilter !== 'all' ? sportFilter : undefined });
      setProps(Array.isArray(data) ? data : []);
    } catch {
      setProps([]);
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
    winRate: props.filter(p => p.status !== 'pending').length > 0 ? props.filter(p => p.status === 'won' || p.status === 'win' as any).length / props.filter(p => p.status !== 'pending').length : 0,
  };

  const getStatusChip = (status: string) => {
    switch (status) {
      case 'won': case 'win': return <Chip icon={<CheckCircle />} label="" size="small" color="success" sx={{ '& .MuiChip-icon': { fontSize: 16, ml: 0.5, mr: -0.5 }, minWidth: 28, height: 22 }} />;
      case 'lost': case 'loss': return <Chip icon={<Cancel />} label="" size="small" color="error" sx={{ '& .MuiChip-icon': { fontSize: 16, ml: 0.5, mr: -0.5 }, minWidth: 28, height: 22 }} />;
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
        {' '}Tier A (58%+) = Strong Edge · Tier B (55-58%) = Clear +EV · Tier C (52-55%) = Modest Edge · Tier D (&lt;52%) = Track Only
      </Alert>

      <Grid container spacing={1.5} mb={1.5}>
        <Grid item xs={6} sm={3}><Card><CardContent sx={{ textAlign: 'center', py: 1.5, '&:last-child': { pb: 1.5 } }}><Typography sx={{ fontSize: 11, color: 'text.secondary' }}>Total Props</Typography><Typography sx={{ fontSize: 20, fontWeight: 700 }}>{stats.total}</Typography></CardContent></Card></Grid>
        <Grid item xs={6} sm={3}><Card><CardContent sx={{ textAlign: 'center', py: 1.5, '&:last-child': { pb: 1.5 } }}><Typography sx={{ fontSize: 11, color: 'text.secondary' }}>Tier A Props</Typography><Typography sx={{ fontSize: 20, fontWeight: 700, color: 'success.main' }}>{stats.tierA}</Typography></CardContent></Card></Grid>
        <Grid item xs={6} sm={3}><Card><CardContent sx={{ textAlign: 'center', py: 1.5, '&:last-child': { pb: 1.5 } }}><Typography sx={{ fontSize: 11, color: 'text.secondary' }}>Avg Edge</Typography><Typography sx={{ fontSize: 20, fontWeight: 700, color: 'info.main' }}>+{stats.avgEdge.toFixed(1)}%</Typography></CardContent></Card></Grid>
        <Grid item xs={6} sm={3}><Card><CardContent sx={{ textAlign: 'center', py: 1.5, '&:last-child': { pb: 1.5 } }}><Typography sx={{ fontSize: 11, color: 'text.secondary' }}>Win Rate</Typography><Typography sx={{ fontSize: 20, fontWeight: 700, color: props.length === 0 ? 'text.secondary' : stats.winRate >= 0.55 ? 'success.main' : 'warning.main' }}>{props.length > 0 ? formatPercent(stats.winRate) : '-'}</Typography></CardContent></Card></Grid>
      </Grid>

      {/* Tabs + Filters Row */}
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
              <MenuItem value="WNBA" sx={{ fontSize: 12 }}>WNBA</MenuItem>
            </Select>
          </FormControl>
          <FormControl size="small" sx={{ minWidth: 100 }}>
            <InputLabel sx={{ fontSize: 12 }}>Tier</InputLabel>
            <Select value={tierFilter} label="Tier" onChange={(e) => { setTierFilter(e.target.value); setPage(0); }} sx={{ fontSize: 12, height: 34 }}>
              <MenuItem value="all" sx={{ fontSize: 12 }}>All Tiers</MenuItem>
              <MenuItem value="A" sx={{ fontSize: 12 }}>Tier A (58%+)</MenuItem>
              <MenuItem value="B" sx={{ fontSize: 12 }}>Tier B (55-58%)</MenuItem>
              <MenuItem value="C" sx={{ fontSize: 12 }}>Tier C (52-55%)</MenuItem>
              <MenuItem value="D" sx={{ fontSize: 12 }}>Tier D (&lt;52%)</MenuItem>
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
                <TableCell sx={hdr} align="center">Line</TableCell>
                <TableCell sx={hdr} align="center">Predicted</TableCell>
                <TableCell sx={hdr} align="center">Pick</TableCell>
                <TableCell sx={hdr} align="center">%</TableCell>
                <TableCell sx={hdr} align="center">Edge</TableCell>
                <TableCell sx={hdr}>Tier</TableCell>
                <TableCell sx={hdr} align="center">Avg</TableCell>
                <TableCell sx={hdr} align="center">L5</TableCell>
                <TableCell sx={hdr} align="center">L10</TableCell>
                <TableCell sx={hdr}>W/L</TableCell>
                <TableCell sx={hdr} align="center">Act</TableCell>
                <TableCell sx={hdr}></TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {!loading && paginated.length === 0 && (
                <TableRow>
                  <TableCell colSpan={20} sx={{ textAlign: 'center', py: 6 }}>
                    <SportsBasketball sx={{ fontSize: 48, color: 'text.disabled', mb: 1 }} />
                    <Typography color="text.secondary" sx={{ fontSize: 14, fontWeight: 600, mb: 0.5 }}>No Player Props Yet</Typography>
                    <Typography color="text.disabled" sx={{ fontSize: 12 }}>
                      Player props predictions will appear here once the pipeline is configured with player stats data.
                    </Typography>
                  </TableCell>
                </TableRow>
              )}
              {paginated.map((prop) => (
                <TableRow key={prop.id} sx={{ '& td': { py: 0.75, fontSize: 11, borderBottom: 1, borderColor: 'divider' } }}>
                  <TableCell sx={{ fontWeight: 600 }}>{prop.sport}</TableCell>
                  <TableCell>{prop.date}</TableCell>
                  <TableCell>{prop.time}</TableCell>
                  <TableCell sx={{ fontWeight: 600 }}>{prop.player_name}</TableCell>
                  <TableCell>{prop.team}</TableCell>
                  <TableCell>{prop.opponent}</TableCell>
                  <TableCell><Chip label={prop.home_away === 'home' ? 'H' : 'A'} size="small" color={prop.home_away === 'home' ? 'success' : 'default'} sx={{ fontSize: 9, height: 18, minWidth: 20 }} /></TableCell>
                  <TableCell sx={{ textTransform: 'capitalize' }}>{prop.prop_type.replace(/_/g, ' ')}</TableCell>
                  <TableCell align="center">{prop.circa_current}</TableCell>
                  <TableCell align="center" sx={{ color: 'info.main', fontWeight: 600 }}>{prop.system_current}</TableCell>
                  <TableCell align="center"><Chip label={prop.pick.toUpperCase()} size="small" color={prop.pick === 'over' ? 'success' : 'error'} sx={{ fontSize: 9, fontWeight: 600, height: 18, minWidth: 40 }} /></TableCell>
                  <TableCell align="center">{formatPercent(prop.probability)}</TableCell>
                  <TableCell align="center" sx={{ color: prop.edge >= 3 ? 'success.main' : prop.edge >= 1.5 ? 'warning.main' : 'text.secondary' }}>+{prop.edge.toFixed(1)}%</TableCell>
                  <TableCell><TierBadge tier={prop.tier} /></TableCell>
                  <TableCell align="center">{prop.season_avg > 0 ? prop.season_avg.toFixed(1) : '-'}</TableCell>
                  <TableCell align="center" sx={{ color: prop.last_5_avg > prop.season_avg ? 'success.main' : prop.last_5_avg < prop.season_avg ? 'error.main' : 'inherit' }}>{prop.last_5_avg > 0 ? prop.last_5_avg.toFixed(1) : '-'}</TableCell>
                  <TableCell align="center" sx={{ color: prop.last_10_avg > prop.season_avg ? 'success.main' : prop.last_10_avg < prop.season_avg ? 'error.main' : 'inherit' }}>{prop.last_10_avg > 0 ? prop.last_10_avg.toFixed(1) : '-'}</TableCell>
                  <TableCell>{getStatusChip(prop.status)}</TableCell>
                  <TableCell align="center" sx={{ fontWeight: prop.actual != null ? 600 : 400 }}>{prop.actual != null ? prop.actual : '-'}</TableCell>
                  <TableCell><IconButton size="small" onClick={() => setReasonDialog({ open: true, prop })}><ExpandMore sx={{ fontSize: 16 }} /></IconButton></TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
        {paginated.length > 0 && (
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
        )}
      </Card>

      <Dialog open={reasonDialog.open} onClose={() => setReasonDialog({ open: false, prop: null })} maxWidth="sm" fullWidth>
        <DialogTitle sx={{ fontSize: 14, fontWeight: 600 }}>{reasonDialog.prop?.player_name} - {reasonDialog.prop?.prop_type.replace(/_/g, ' ')} {reasonDialog.prop?.pick.toUpperCase()}</DialogTitle>
        <DialogContent>
          {reasonDialog.prop?.reason && (
            <Typography variant="body2" sx={{ fontSize: 12, mb: 2 }}>{reasonDialog.prop.reason}</Typography>
          )}
          <Grid container spacing={1}>
            <Grid item xs={4}><Typography sx={{ fontSize: 10, color: 'text.secondary' }}>Season Avg</Typography><Typography sx={{ fontSize: 12, fontWeight: 600 }}>{reasonDialog.prop?.season_avg ? reasonDialog.prop.season_avg.toFixed(1) : '-'}</Typography></Grid>
            <Grid item xs={4}><Typography sx={{ fontSize: 10, color: 'text.secondary' }}>Last 5 Avg</Typography><Typography sx={{ fontSize: 12, fontWeight: 600 }}>{reasonDialog.prop?.last_5_avg ? reasonDialog.prop.last_5_avg.toFixed(1) : '-'}</Typography></Grid>
            <Grid item xs={4}><Typography sx={{ fontSize: 10, color: 'text.secondary' }}>Last 10 Avg</Typography><Typography sx={{ fontSize: 12, fontWeight: 600 }}>{reasonDialog.prop?.last_10_avg ? reasonDialog.prop.last_10_avg.toFixed(1) : '-'}</Typography></Grid>
            <Grid item xs={4}><Typography sx={{ fontSize: 10, color: 'text.secondary' }}>Line</Typography><Typography sx={{ fontSize: 12, fontWeight: 600 }}>{reasonDialog.prop?.circa_current}</Typography></Grid>
            <Grid item xs={4}><Typography sx={{ fontSize: 10, color: 'text.secondary' }}>Predicted</Typography><Typography sx={{ fontSize: 12, fontWeight: 600, color: 'info.main' }}>{reasonDialog.prop?.system_current}</Typography></Grid>
            <Grid item xs={4}><Typography sx={{ fontSize: 10, color: 'text.secondary' }}>Edge</Typography><Typography sx={{ fontSize: 12, fontWeight: 600, color: 'success.main' }}>+{reasonDialog.prop?.edge.toFixed(1)}%</Typography></Grid>
          </Grid>
        </DialogContent>
        <DialogActions><Button onClick={() => setReasonDialog({ open: false, prop: null })} size="small">Close</Button></DialogActions>
      </Dialog>
    </Box>
  );
};

export default PlayerProps;