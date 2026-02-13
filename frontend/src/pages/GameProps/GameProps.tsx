// src/pages/GameProps/GameProps.tsx
import React, { useState, useEffect, useMemo } from 'react';
import {
  Box, Card, CardContent, Typography, Grid, Table, TableBody, TableCell,
  TableContainer, TableHead, TableRow, Chip, Select, MenuItem, FormControl,
  InputLabel, Button, Alert, LinearProgress, useTheme, Tabs, Tab, IconButton,
  Dialog, DialogTitle, DialogContent, DialogActions
} from '@mui/material';
import {
  Refresh, CheckCircle, Cancel, Schedule, ExpandMore,
  Person, Casino, FlashOn, AccessTime, SportsScore
} from '@mui/icons-material';
import { formatPercent } from '../../utils';
import axios from 'axios';

const BASE_URL = import.meta.env.VITE_API_URL || '/api/v1';

interface GameProp {
  id: string;
  sport: string;
  gameDate: string;
  gameTime: string;
  teams: string;
  player: string;
  playerPosition: string;
  playerTeamSide: string;
  propType: string;
  propLabel: string;
  propColor: string;
  line: number;
  oddsOver: string;
  oddsUnder: string;
  pick: 'OVER' | 'UNDER';
  projection: number;
  probability: number;
  edge: number;
  tier: string;
  average: number;
  lastSeason: string;
  lastSeasonTrend: 'up' | 'down' | 'flat';
  matchTier: string;
  status: 'pending' | 'won' | 'lost' | 'push';
  actual?: number;
  category: 'player_stats' | 'scoring_props' | 'game_events';
}

const GameProps: React.FC = () => {
  const theme = useTheme();
  const isDark = theme.palette.mode === 'dark';

  const [gameProps, setGameProps] = useState<GameProp[]>([]);
  const [loading, setLoading] = useState(true);
  const [sportFilter, setSportFilter] = useState('all');
  const [tierFilter, setTierFilter] = useState('all');
  const [propTypeFilter, setPropTypeFilter] = useState('all');
  const [categoryTab, setCategoryTab] = useState(0);
  const [statusTab, setStatusTab] = useState(0);
  const [detailDialog, setDetailDialog] = useState<{ open: boolean; prop: GameProp | null }>({ open: false, prop: null });

  useEffect(() => { loadGameProps(); }, []);

  const loadGameProps = async () => {
    setLoading(true);
    try {
      const { data } = await axios.get(`${BASE_URL}/public/game-props`, {
        params: sportFilter !== 'all' ? { sport: sportFilter } : undefined
      });
      setGameProps(Array.isArray(data) ? data : []);
    } catch {
      setGameProps([]);
    }
    setLoading(false);
  };

  const categoryLabels = ['player_stats', 'scoring_props', 'game_events'];

  const filteredGameProps = useMemo(() => {
    return gameProps.filter(p => {
      const sportMatch = sportFilter === 'all' || p.sport === sportFilter;
      const tierMatch = tierFilter === 'all' || p.tier === tierFilter;
      const propTypeMatch = propTypeFilter === 'all' || p.propType === propTypeFilter;
      const categoryMatch = p.category === categoryLabels[categoryTab];
      const statusMatch = statusTab === 0 ||
        (statusTab === 1 && p.status === 'pending') ||
        (statusTab === 2 && p.status !== 'pending');
      return sportMatch && tierMatch && propTypeMatch && categoryMatch && statusMatch;
    });
  }, [gameProps, sportFilter, tierFilter, propTypeFilter, categoryTab, statusTab]);

  const categoryCounts = useMemo(() => ({
    player_stats: gameProps.filter(p => p.category === 'player_stats').length,
    scoring_props: gameProps.filter(p => p.category === 'scoring_props').length,
    game_events: gameProps.filter(p => p.category === 'game_events').length,
  }), [gameProps]);

  const stats = useMemo(() => {
    const filtered = gameProps.filter(p => p.category === categoryLabels[categoryTab]);
    const graded = filtered.filter(p => p.status !== 'pending');
    const wins = graded.filter(p => p.status === 'won' || p.status === 'win' as any).length;
    const winRate = graded.length > 0 ? (wins / graded.length) * 100 : 0;
    const units = graded.reduce((sum, p) => {
      if (p.status === 'won' || p.status === 'win' as any) return sum + 1;
      if (p.status === 'lost' || p.status === 'loss' as any) return sum - 1.1;
      return sum;
    }, 0);

    return {
      total: filtered.length,
      pending: filtered.filter(p => p.status === 'pending').length,
      tierA: filtered.filter(p => p.tier === 'A').length,
      avgEdge: filtered.length > 0 ? filtered.reduce((s, p) => s + p.edge, 0) / filtered.length : 0,
      winRate,
      units,
    };
  }, [gameProps, categoryTab]);

  const getPropChip = (label: string, color: string) => {
    const colorMap: Record<string, any> = {
      'blue': { bg: '#3b82f6', text: 'white' },
      'green': { bg: '#22c55e', text: 'white' },
      'orange': { bg: '#f97316', text: 'white' },
      'purple': { bg: '#a855f7', text: 'white' },
      'red': { bg: '#ef4444', text: 'white' },
      'teal': { bg: '#14b8a6', text: 'white' },
      'pink': { bg: '#ec4899', text: 'white' },
      'yellow': { bg: '#eab308', text: 'black' },
    };
    const colors = colorMap[color] || colorMap['blue'];
    return (
      <Chip
        label={label}
        size="small"
        sx={{
          bgcolor: colors.bg,
          color: colors.text,
          fontSize: 10,
          fontWeight: 600,
          height: 22,
          minWidth: 55,
          '& .MuiChip-label': { px: 1 }
        }}
      />
    );
  };

  const getPickChip = (pick: 'OVER' | 'UNDER') => (
    <Chip
      label={pick}
      size="small"
      sx={{
        bgcolor: pick === 'OVER' ? '#22c55e' : '#f97316',
        color: 'white',
        fontSize: 10,
        fontWeight: 700,
        height: 22,
        minWidth: 50,
      }}
    />
  );

  const getTierBadge = (tier: string) => {
    const colors: Record<string, { bg: string; text: string }> = {
      'A': { bg: '#22c55e', text: 'white' },
      'B': { bg: '#3b82f6', text: 'white' },
      'C': { bg: '#f97316', text: 'white' },
      'D': { bg: '#6b7280', text: 'white' },
    };
    const color = colors[tier] || colors['D'];
    return (
      <Box sx={{
        width: 24,
        height: 24,
        borderRadius: '50%',
        bgcolor: color.bg,
        color: color.text,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        fontSize: 11,
        fontWeight: 700,
      }}>
        {tier}
      </Box>
    );
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'won': case 'win': return <CheckCircle sx={{ color: '#22c55e', fontSize: 18 }} />;
      case 'lost': case 'loss': return <Cancel sx={{ color: '#ef4444', fontSize: 18 }} />;
      case 'push': return <Schedule sx={{ color: '#eab308', fontSize: 18 }} />;
      default: return <AccessTime sx={{ color: '#6b7280', fontSize: 18 }} />;
    }
  };

  const getLSTrend = (value: string, trend: 'up' | 'down' | 'flat') => {
    const color = trend === 'up' ? '#22c55e' : trend === 'down' ? '#ef4444' : '#6b7280';
    const icon = trend === 'up' ? '↗' : trend === 'down' ? '↘' : '';
    return (
      <Typography sx={{ color, fontSize: 12, fontWeight: 500 }}>
        {value}{icon}
      </Typography>
    );
  };

  const hdr = { fontSize: 11, fontWeight: 600, color: 'text.secondary', py: 1.5, borderBottom: 2, borderColor: 'divider' };

  return (
    <Box>
      {/* Header */}
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 3 }}>
        <Typography variant="h5" sx={{ fontWeight: 700, display: 'flex', alignItems: 'center', gap: 1 }}>
          <Casino sx={{ color: 'primary.main' }} />
          Game Props
        </Typography>
        <Button variant="contained" startIcon={<Refresh />} onClick={loadGameProps} size="small">
          Refresh
        </Button>
      </Box>

      {/* Description */}
      <Alert severity="info" sx={{ mb: 3 }} icon={<FlashOn />}>
        <Typography variant="body2">
          <strong>Game Props Predictions:</strong> Player stat props (passing yards, rushing, receiving, rebounds, assists), scoring props (anytime TD, first TD scorer), and game event props (overtime, safety, first to score).
          {' '}Tier A (58%+) = Strong Edge · Tier B (55-58%) = Clear +EV · Tier C (52-55%) = Modest Edge · Tier D (&lt;52%) = Track Only
        </Typography>
      </Alert>

      {/* Stats Cards */}
      <Grid container spacing={2} sx={{ mb: 3 }}>
        {[
          { label: 'Total Props', value: stats.total, color: 'text.primary' },
          { label: 'Pending', value: stats.pending, color: 'primary.main' },
          { label: 'Tier A Picks', value: stats.tierA, color: 'success.main' },
          { label: 'Avg Edge', value: stats.total > 0 ? `+${stats.avgEdge.toFixed(1)}%` : '-', color: 'success.main' },
          { label: 'Win Rate', value: stats.total > 0 ? `${stats.winRate.toFixed(1)}%` : '-', color: 'text.primary' },
          { label: 'Units +/-', value: stats.total > 0 ? (stats.units >= 0 ? `+${stats.units.toFixed(1)}u` : `${stats.units.toFixed(1)}u`) : '-', color: stats.units >= 0 ? 'success.main' : 'error.main' },
        ].map((stat) => (
          <Grid item xs={6} sm={4} md={2} key={stat.label}>
            <Card sx={{ textAlign: 'center', py: 2, bgcolor: isDark ? 'grey.900' : 'grey.50' }}>
              <Typography variant="caption" color="text.secondary">{stat.label}</Typography>
              <Typography variant="h5" sx={{ fontWeight: 700, color: stat.color }}>{stat.value}</Typography>
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* Category Tabs */}
      <Card sx={{ mb: 2 }}>
        <Tabs value={categoryTab} onChange={(_, v) => setCategoryTab(v)} sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tab
            icon={<Person sx={{ fontSize: 18 }} />}
            iconPosition="start"
            label={<>PLAYER STATS <Chip label={categoryCounts.player_stats} size="small" sx={{ ml: 1, height: 20, fontSize: 11 }} /></>}
          />
          <Tab
            icon={<Casino sx={{ fontSize: 18 }} />}
            iconPosition="start"
            label={<>SCORING PROPS <Chip label={categoryCounts.scoring_props} size="small" sx={{ ml: 1, height: 20, fontSize: 11 }} /></>}
          />
          <Tab
            icon={<FlashOn sx={{ fontSize: 18 }} />}
            iconPosition="start"
            label={<>GAME EVENTS <Chip label={categoryCounts.game_events} size="small" sx={{ ml: 1, height: 20, fontSize: 11 }} /></>}
          />
        </Tabs>
      </Card>

      {/* Main Content */}
      <Card>
        <CardContent sx={{ p: 2 }}>
          {/* Filters Row */}
          <Box sx={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: 2, mb: 2 }}>
            <Box sx={{ display: 'flex', gap: 1 }}>
              {['ALL', 'PENDING', 'GRADED'].map((label, idx) => (
                <Chip
                  key={label}
                  label={idx === 0 ? `ALL (${filteredGameProps.length})` : label}
                  onClick={() => setStatusTab(idx)}
                  variant={statusTab === idx ? 'filled' : 'outlined'}
                  color={statusTab === idx ? 'primary' : 'default'}
                  size="small"
                  sx={{ fontWeight: 600 }}
                />
              ))}
            </Box>

            <FormControl size="small" sx={{ minWidth: 120 }}>
              <InputLabel sx={{ fontSize: 12 }}>Sport</InputLabel>
              <Select value={sportFilter} onChange={(e) => setSportFilter(e.target.value)} label="Sport" sx={{ fontSize: 12 }}>
                <MenuItem value="all">All Sports</MenuItem>
                <MenuItem value="NFL">NFL</MenuItem>
                <MenuItem value="NBA">NBA</MenuItem>
                <MenuItem value="NHL">NHL</MenuItem>
                <MenuItem value="MLB">MLB</MenuItem>
                <MenuItem value="NCAAB">NCAAB</MenuItem>
                <MenuItem value="WNBA">WNBA</MenuItem>
              </Select>
            </FormControl>

            <FormControl size="small" sx={{ minWidth: 80 }}>
              <InputLabel sx={{ fontSize: 12 }}>Tier</InputLabel>
              <Select value={tierFilter} onChange={(e) => setTierFilter(e.target.value)} label="Tier" sx={{ fontSize: 12 }}>
                <MenuItem value="all">All</MenuItem>
                <MenuItem value="A">A (58%+)</MenuItem>
                <MenuItem value="B">B (55-58%)</MenuItem>
                <MenuItem value="C">C (52-55%)</MenuItem>
                <MenuItem value="D">D (&lt;52%)</MenuItem>
              </Select>
            </FormControl>

            <FormControl size="small" sx={{ minWidth: 120 }}>
              <InputLabel sx={{ fontSize: 12 }}>Prop Type</InputLabel>
              <Select value={propTypeFilter} onChange={(e) => setPropTypeFilter(e.target.value)} label="Prop Type" sx={{ fontSize: 12 }}>
                <MenuItem value="all">All Props</MenuItem>
                <MenuItem value="pass_yds">Pass Yds</MenuItem>
                <MenuItem value="rush_yds">Rush Yds</MenuItem>
                <MenuItem value="rec_yds">Rec Yds</MenuItem>
                <MenuItem value="points">Points</MenuItem>
                <MenuItem value="rebounds">Rebounds</MenuItem>
                <MenuItem value="assists">Assists</MenuItem>
                <MenuItem value="threes">3PM</MenuItem>
                <MenuItem value="sog">SOG</MenuItem>
              </Select>
            </FormControl>

            <Box sx={{ flex: 1 }} />
            <Typography variant="body2" color="text.secondary">{filteredGameProps.length} props</Typography>
          </Box>
        </CardContent>

        {/* Table */}
        {loading && <LinearProgress />}
        <TableContainer>
          <Table size="small" sx={{ minWidth: 1400 }}>
            <TableHead>
              <TableRow sx={{ bgcolor: isDark ? 'grey.900' : 'grey.50' }}>
                <TableCell sx={hdr}>Sport</TableCell>
                <TableCell sx={hdr}>Game</TableCell>
                <TableCell sx={hdr}>Player</TableCell>
                <TableCell sx={hdr}>Prop</TableCell>
                <TableCell sx={{ ...hdr, textAlign: 'center' }}>Line</TableCell>
                <TableCell sx={{ ...hdr, textAlign: 'center' }}>O/U</TableCell>
                <TableCell sx={{ ...hdr, textAlign: 'center' }}>Pick</TableCell>
                <TableCell sx={{ ...hdr, textAlign: 'center' }}>Proj</TableCell>
                <TableCell sx={{ ...hdr, textAlign: 'center' }}>Prob</TableCell>
                <TableCell sx={{ ...hdr, textAlign: 'center' }}>Edge</TableCell>
                <TableCell sx={{ ...hdr, textAlign: 'center' }}>Tier</TableCell>
                <TableCell sx={{ ...hdr, textAlign: 'center' }}>Avg</TableCell>
                <TableCell sx={{ ...hdr, textAlign: 'center' }}>LS</TableCell>
                <TableCell sx={{ ...hdr, textAlign: 'center' }}>Match</TableCell>
                <TableCell sx={{ ...hdr, textAlign: 'center' }}>W/L</TableCell>
                <TableCell sx={{ ...hdr, textAlign: 'center' }}>Act</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {!loading && filteredGameProps.length === 0 && (
                <TableRow>
                  <TableCell colSpan={16} sx={{ textAlign: 'center', py: 8 }}>
                    <SportsScore sx={{ fontSize: 48, color: 'text.disabled', mb: 1 }} />
                    <Typography color="text.secondary" sx={{ fontSize: 14, fontWeight: 600, mb: 0.5 }}>No Game Props Yet</Typography>
                    <Typography color="text.disabled" sx={{ fontSize: 12 }}>
                      Game props predictions will appear here once the pipeline is configured with player stats and game event data.
                    </Typography>
                  </TableCell>
                </TableRow>
              )}
              {filteredGameProps.map((prop) => (
                <TableRow
                  key={prop.id}
                  sx={{
                    '&:hover': { bgcolor: isDark ? 'grey.800' : 'grey.50' },
                    '& td': { py: 1.5, fontSize: 12, borderBottom: 1, borderColor: 'divider' }
                  }}
                >
                  <TableCell>
                    <Typography sx={{ fontWeight: 600, fontSize: 12 }}>{prop.sport}</Typography>
                  </TableCell>
                  <TableCell>
                    <Typography sx={{ fontSize: 11, fontWeight: 500 }}>{prop.gameDate} {prop.gameTime}</Typography>
                    <Typography sx={{ fontSize: 11, color: 'text.secondary' }}>{prop.teams}</Typography>
                  </TableCell>
                  <TableCell>
                    <Typography sx={{ fontWeight: 600, fontSize: 12 }}>{prop.player}</Typography>
                    <Typography sx={{ fontSize: 10, color: 'text.secondary' }}>{prop.playerPosition} • {prop.playerTeamSide}</Typography>
                  </TableCell>
                  <TableCell>{getPropChip(prop.propLabel, prop.propColor)}</TableCell>
                  <TableCell align="center">
                    <Typography sx={{ fontWeight: 600, fontSize: 13 }}>{prop.line}</Typography>
                  </TableCell>
                  <TableCell align="center">
                    <Typography sx={{ fontSize: 10, color: 'text.secondary' }}>O {prop.oddsOver}</Typography>
                    <Typography sx={{ fontSize: 10, color: 'text.secondary' }}>U {prop.oddsUnder}</Typography>
                  </TableCell>
                  <TableCell align="center">{getPickChip(prop.pick)}</TableCell>
                  <TableCell align="center">
                    <Typography sx={{ fontWeight: 600, fontSize: 12, color: 'primary.main' }}>{prop.projection}</Typography>
                  </TableCell>
                  <TableCell align="center">
                    <Typography sx={{ fontSize: 12 }}>{(prop.probability * 100).toFixed(1)}%</Typography>
                  </TableCell>
                  <TableCell align="center">
                    <Typography sx={{ fontSize: 12, fontWeight: 600, color: 'success.main' }}>+{prop.edge.toFixed(1)}%</Typography>
                  </TableCell>
                  <TableCell align="center">{getTierBadge(prop.tier)}</TableCell>
                  <TableCell align="center">
                    <Typography sx={{ fontSize: 12 }}>{prop.average}</Typography>
                  </TableCell>
                  <TableCell align="center">{getLSTrend(prop.lastSeason, prop.lastSeasonTrend)}</TableCell>
                  <TableCell align="center">{getTierBadge(prop.matchTier)}</TableCell>
                  <TableCell align="center">{getStatusIcon(prop.status)}</TableCell>
                  <TableCell align="center">
                    {prop.actual != null ? (
                      <Typography sx={{ fontWeight: 600, fontSize: 12 }}>{prop.actual}</Typography>
                    ) : (
                      <IconButton size="small" onClick={() => setDetailDialog({ open: true, prop })}>
                        <ExpandMore sx={{ fontSize: 16 }} />
                      </IconButton>
                    )}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </Card>

      {/* Detail Dialog */}
      <Dialog open={detailDialog.open} onClose={() => setDetailDialog({ open: false, prop: null })} maxWidth="sm" fullWidth>
        <DialogTitle sx={{ fontSize: 14, fontWeight: 600 }}>
          {detailDialog.prop?.player} - {detailDialog.prop?.propLabel}
        </DialogTitle>
        <DialogContent>
          <Grid container spacing={2} sx={{ mt: 1 }}>
            <Grid item xs={6}>
              <Typography variant="caption" color="text.secondary">Game</Typography>
              <Typography variant="body2" fontWeight={600}>{detailDialog.prop?.teams}</Typography>
            </Grid>
            <Grid item xs={6}>
              <Typography variant="caption" color="text.secondary">Time</Typography>
              <Typography variant="body2" fontWeight={600}>{detailDialog.prop?.gameDate} {detailDialog.prop?.gameTime}</Typography>
            </Grid>
            <Grid item xs={4}>
              <Typography variant="caption" color="text.secondary">Line</Typography>
              <Typography variant="body2" fontWeight={600}>{detailDialog.prop?.line}</Typography>
            </Grid>
            <Grid item xs={4}>
              <Typography variant="caption" color="text.secondary">Projection</Typography>
              <Typography variant="body2" fontWeight={600} color="primary.main">{detailDialog.prop?.projection}</Typography>
            </Grid>
            <Grid item xs={4}>
              <Typography variant="caption" color="text.secondary">Edge</Typography>
              <Typography variant="body2" fontWeight={600} color="success.main">+{detailDialog.prop?.edge.toFixed(1)}%</Typography>
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDetailDialog({ open: false, prop: null })} size="small">Close</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default GameProps;