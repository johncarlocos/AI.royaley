// src/pages/GameProps/GameProps.tsx
import React, { useState, useEffect, useMemo } from 'react';
import {
  Box, Card, CardContent, Typography, Grid, Table, TableBody, TableCell,
  TableContainer, TableHead, TableRow, Chip, Select, MenuItem, FormControl,
  InputLabel, Button, Alert, LinearProgress, useTheme, Tabs, Tab, IconButton,
  Dialog, DialogTitle, DialogContent, DialogActions, Tooltip
} from '@mui/material';
import { 
  Refresh, CheckCircle, Cancel, Schedule, ExpandMore, 
  Person, Casino, FlashOn, TrendingUp, TrendingDown, AccessTime
} from '@mui/icons-material';
import { TierBadge } from '../../components/Common';
import { formatPercent } from '../../utils';

interface GameProp {
  id: string;
  sport: string;
  sportIcon: string;
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
      setGameProps(generateDemoGameProps());
    } catch {
      setGameProps(generateDemoGameProps());
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
    const wins = graded.filter(p => p.status === 'won').length;
    const winRate = graded.length > 0 ? (wins / graded.length) * 100 : 0;
    const units = graded.reduce((sum, p) => {
      if (p.status === 'won') return sum + 1;
      if (p.status === 'lost') return sum - 1.1;
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
      case 'won': return <CheckCircle sx={{ color: '#22c55e', fontSize: 18 }} />;
      case 'lost': return <Cancel sx={{ color: '#ef4444', fontSize: 18 }} />;
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

  if (loading) {
    return (
      <Box sx={{ p: 3 }}>
        <LinearProgress />
        <Typography sx={{ mt: 2, textAlign: 'center' }}>Loading game props...</Typography>
      </Box>
    );
  }

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
        </Typography>
      </Alert>

      {/* Stats Cards */}
      <Grid container spacing={2} sx={{ mb: 3 }}>
        {[
          { label: 'Total Props', value: stats.total, color: 'text.primary' },
          { label: 'Pending', value: stats.pending, color: 'primary.main' },
          { label: 'Tier A Picks', value: stats.tierA, color: 'success.main' },
          { label: 'Avg Edge', value: `+${stats.avgEdge.toFixed(1)}%`, color: 'success.main' },
          { label: 'Win Rate', value: `${stats.winRate.toFixed(1)}%`, color: 'text.primary' },
          { label: 'Units +/-', value: stats.units >= 0 ? `+${stats.units.toFixed(1)}u` : `${stats.units.toFixed(1)}u`, color: stats.units >= 0 ? 'success.main' : 'error.main' },
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
            {/* Status Tabs */}
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

            {/* Dropdowns */}
            <FormControl size="small" sx={{ minWidth: 120 }}>
              <InputLabel sx={{ fontSize: 12 }}>Sport</InputLabel>
              <Select value={sportFilter} onChange={(e) => setSportFilter(e.target.value)} label="Sport" sx={{ fontSize: 12 }}>
                <MenuItem value="all">All Sports</MenuItem>
                <MenuItem value="NFL">NFL</MenuItem>
                <MenuItem value="NBA">NBA</MenuItem>
                <MenuItem value="NHL">NHL</MenuItem>
                <MenuItem value="MLB">MLB</MenuItem>
              </Select>
            </FormControl>

            <FormControl size="small" sx={{ minWidth: 80 }}>
              <InputLabel sx={{ fontSize: 12 }}>Tier</InputLabel>
              <Select value={tierFilter} onChange={(e) => setTierFilter(e.target.value)} label="Tier" sx={{ fontSize: 12 }}>
                <MenuItem value="all">All</MenuItem>
                <MenuItem value="A">A</MenuItem>
                <MenuItem value="B">B</MenuItem>
                <MenuItem value="C">C</MenuItem>
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
              </Select>
            </FormControl>

            <Box sx={{ flex: 1 }} />
            <Typography variant="body2" color="text.secondary">{filteredGameProps.length} props</Typography>
          </Box>
        </CardContent>

        {/* Table */}
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
              {filteredGameProps.map((prop) => (
                <TableRow 
                  key={prop.id} 
                  sx={{ 
                    '&:hover': { bgcolor: isDark ? 'grey.800' : 'grey.50' },
                    '& td': { py: 1.5, fontSize: 12, borderBottom: 1, borderColor: 'divider' } 
                  }}
                >
                  {/* Sport */}
                  <TableCell>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Typography sx={{ fontSize: 16 }}>{prop.sportIcon}</Typography>
                      <Typography sx={{ fontWeight: 600, fontSize: 12 }}>{prop.sport}</Typography>
                    </Box>
                  </TableCell>
                  
                  {/* Game */}
                  <TableCell>
                    <Typography sx={{ fontSize: 11, fontWeight: 500 }}>{prop.gameDate} {prop.gameTime}</Typography>
                    <Typography sx={{ fontSize: 11, color: 'text.secondary' }}>{prop.teams}</Typography>
                  </TableCell>
                  
                  {/* Player */}
                  <TableCell>
                    <Typography sx={{ fontWeight: 600, fontSize: 12 }}>{prop.player}</Typography>
                    <Typography sx={{ fontSize: 10, color: 'text.secondary' }}>{prop.playerPosition} • {prop.playerTeamSide}</Typography>
                  </TableCell>
                  
                  {/* Prop */}
                  <TableCell>{getPropChip(prop.propLabel, prop.propColor)}</TableCell>
                  
                  {/* Line */}
                  <TableCell align="center">
                    <Typography sx={{ fontWeight: 600, fontSize: 13 }}>{prop.line}</Typography>
                  </TableCell>
                  
                  {/* O/U */}
                  <TableCell align="center">
                    <Typography sx={{ fontSize: 10, color: 'text.secondary' }}>O {prop.oddsOver}</Typography>
                    <Typography sx={{ fontSize: 10, color: 'text.secondary' }}>U {prop.oddsUnder}</Typography>
                  </TableCell>
                  
                  {/* Pick */}
                  <TableCell align="center">{getPickChip(prop.pick)}</TableCell>
                  
                  {/* Proj */}
                  <TableCell align="center">
                    <Typography sx={{ fontWeight: 600, fontSize: 12, color: 'primary.main' }}>{prop.projection}</Typography>
                  </TableCell>
                  
                  {/* Prob */}
                  <TableCell align="center">
                    <Typography sx={{ fontSize: 12 }}>{(prop.probability * 100).toFixed(1)}%</Typography>
                  </TableCell>
                  
                  {/* Edge */}
                  <TableCell align="center">
                    <Typography sx={{ fontSize: 12, fontWeight: 600, color: 'success.main' }}>+{prop.edge.toFixed(1)}%</Typography>
                  </TableCell>
                  
                  {/* Tier */}
                  <TableCell align="center">{getTierBadge(prop.tier)}</TableCell>
                  
                  {/* Avg */}
                  <TableCell align="center">
                    <Typography sx={{ fontSize: 12 }}>{prop.average}</Typography>
                  </TableCell>
                  
                  {/* LS */}
                  <TableCell align="center">{getLSTrend(prop.lastSeason, prop.lastSeasonTrend)}</TableCell>
                  
                  {/* Match */}
                  <TableCell align="center">{getTierBadge(prop.matchTier)}</TableCell>
                  
                  {/* W/L */}
                  <TableCell align="center">{getStatusIcon(prop.status)}</TableCell>
                  
                  {/* Act */}
                  <TableCell align="center">
                    {prop.actual !== undefined ? (
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

const generateDemoGameProps = (): GameProp[] => {
  return [
    // Player Stats - NFL
    { id: 'ps1', sport: 'NFL', sportIcon: '🏈', gameDate: '1/19', gameTime: '3:00 PM', teams: 'KC vs BUF', player: 'Patrick Mahomes', playerPosition: 'QB', playerTeamSide: 'AWAY', propType: 'pass_yds', propLabel: 'Pass Yds', propColor: 'blue', line: 285.5, oddsOver: '-115', oddsUnder: '-105', pick: 'OVER', projection: 302.5, probability: 0.65, edge: 4.2, tier: 'B', average: 292.4, lastSeason: '305.8', lastSeasonTrend: 'flat', matchTier: 'B', status: 'pending', category: 'player_stats' },
    { id: 'ps2', sport: 'NFL', sportIcon: '🏈', gameDate: '1/19', gameTime: '3:00 PM', teams: 'BUF vs KC', player: 'Josh Allen', playerPosition: 'QB', playerTeamSide: 'HOME', propType: 'rush_yds', propLabel: 'Rush Yds', propColor: 'green', line: 38.5, oddsOver: '-110', oddsUnder: '-110', pick: 'OVER', projection: 45.2, probability: 0.62, edge: 3.5, tier: 'A', average: 42.6, lastSeason: '48.2', lastSeasonTrend: 'up', matchTier: 'A', status: 'pending', category: 'player_stats' },
    { id: 'ps3', sport: 'NFL', sportIcon: '🏈', gameDate: '1/19', gameTime: '3:00 PM', teams: 'KC vs BUF', player: 'Travis Kelce', playerPosition: 'TE', playerTeamSide: 'AWAY', propType: 'rec_yds', propLabel: 'Rec Yds', propColor: 'orange', line: 65.5, oddsOver: '-108', oddsUnder: '-112', pick: 'UNDER', projection: 58.4, probability: 0.58, edge: 2.1, tier: 'B', average: 62.8, lastSeason: '55.2', lastSeasonTrend: 'down', matchTier: 'C', status: 'pending', category: 'player_stats' },
    { id: 'ps4', sport: 'NFL', sportIcon: '🏈', gameDate: '1/19', gameTime: '3:00 PM', teams: 'BUF vs KC', player: 'Stefon Diggs', playerPosition: 'WR', playerTeamSide: 'HOME', propType: 'rec', propLabel: 'Rec', propColor: 'red', line: 6.5, oddsOver: '-120', oddsUnder: '+100', pick: 'OVER', projection: 7.8, probability: 0.64, edge: 3.8, tier: 'A', average: 7.2, lastSeason: '8.4', lastSeasonTrend: 'up', matchTier: 'A', status: 'pending', category: 'player_stats' },
    
    // Player Stats - NBA
    { id: 'ps5', sport: 'NBA', sportIcon: '🏀', gameDate: '1/20', gameTime: '7:30 PM', teams: 'LAL vs BOS', player: 'LeBron James', playerPosition: 'SF', playerTeamSide: 'HOME', propType: 'points', propLabel: 'Points', propColor: 'purple', line: 26.5, oddsOver: '-112', oddsUnder: '-108', pick: 'OVER', projection: 28.4, probability: 0.66, edge: 4.5, tier: 'A', average: 25.8, lastSeason: '28.2', lastSeasonTrend: 'up', matchTier: 'A', status: 'pending', category: 'player_stats' },
    { id: 'ps6', sport: 'NBA', sportIcon: '🏀', gameDate: '1/20', gameTime: '7:30 PM', teams: 'LAL vs BOS', player: 'Anthony Davis', playerPosition: 'PF', playerTeamSide: 'HOME', propType: 'rebounds', propLabel: 'Reb', propColor: 'teal', line: 11.5, oddsOver: '-105', oddsUnder: '-115', pick: 'OVER', projection: 12.8, probability: 0.61, edge: 2.8, tier: 'B', average: 12.2, lastSeason: '13.4', lastSeasonTrend: 'up', matchTier: 'B', status: 'pending', category: 'player_stats' },
    { id: 'ps7', sport: 'NBA', sportIcon: '🏀', gameDate: '1/20', gameTime: '10:00 PM', teams: 'GSW vs PHX', player: 'Stephen Curry', playerPosition: 'PG', playerTeamSide: 'AWAY', propType: 'threes', propLabel: '3PM', propColor: 'yellow', line: 4.5, oddsOver: '-105', oddsUnder: '-115', pick: 'OVER', projection: 5.2, probability: 0.63, edge: 3.2, tier: 'A', average: 4.8, lastSeason: '5.4', lastSeasonTrend: 'up', matchTier: 'A', status: 'pending', category: 'player_stats' },
    { id: 'ps8', sport: 'NBA', sportIcon: '🏀', gameDate: '1/20', gameTime: '7:30 PM', teams: 'BOS vs LAL', player: 'Jayson Tatum', playerPosition: 'SF', playerTeamSide: 'AWAY', propType: 'assists', propLabel: 'Ast', propColor: 'pink', line: 4.5, oddsOver: '-108', oddsUnder: '-112', pick: 'UNDER', projection: 3.8, probability: 0.58, edge: 1.9, tier: 'C', average: 4.6, lastSeason: '3.6', lastSeasonTrend: 'down', matchTier: 'C', status: 'pending', category: 'player_stats' },
    
    // Player Stats - NHL
    { id: 'ps9', sport: 'NHL', sportIcon: '🏒', gameDate: '1/20', gameTime: '7:00 PM', teams: 'EDM vs COL', player: 'Connor McDavid', playerPosition: 'C', playerTeamSide: 'HOME', propType: 'points', propLabel: 'Points', propColor: 'purple', line: 1.5, oddsOver: '-130', oddsUnder: '+110', pick: 'OVER', projection: 2.1, probability: 0.68, edge: 5.2, tier: 'A', average: 1.7, lastSeason: '2.2', lastSeasonTrend: 'up', matchTier: 'A', status: 'pending', category: 'player_stats' },
    { id: 'ps10', sport: 'NHL', sportIcon: '🏒', gameDate: '1/20', gameTime: '7:00 PM', teams: 'COL vs EDM', player: 'Nathan MacKinnon', playerPosition: 'C', playerTeamSide: 'AWAY', propType: 'sog', propLabel: 'SOG', propColor: 'blue', line: 4.5, oddsOver: '-115', oddsUnder: '-105', pick: 'OVER', projection: 5.4, probability: 0.62, edge: 3.1, tier: 'B', average: 4.8, lastSeason: '5.2', lastSeasonTrend: 'up', matchTier: 'B', status: 'pending', category: 'player_stats' },
    
    // Player Stats - Graded
    { id: 'ps11', sport: 'NBA', sportIcon: '🏀', gameDate: '1/18', gameTime: '8:00 PM', teams: 'MIL vs MIA', player: 'Giannis Antetokounmpo', playerPosition: 'PF', playerTeamSide: 'HOME', propType: 'pra', propLabel: 'PRA', propColor: 'teal', line: 55.5, oddsOver: '-110', oddsUnder: '-110', pick: 'OVER', projection: 58.4, probability: 0.64, edge: 3.5, tier: 'A', average: 56.8, lastSeason: '59.2', lastSeasonTrend: 'flat', matchTier: 'A', status: 'won', actual: 62, category: 'player_stats' },
    { id: 'ps12', sport: 'NFL', sportIcon: '🏈', gameDate: '1/14', gameTime: '4:30 PM', teams: 'DAL vs GB', player: 'Lamar Jackson', playerPosition: 'QB', playerTeamSide: 'HOME', propType: 'pass_yds', propLabel: 'Pass Yds', propColor: 'blue', line: 245.5, oddsOver: '-110', oddsUnder: '-110', pick: 'OVER', projection: 268.2, probability: 0.61, edge: 2.8, tier: 'B', average: 252.4, lastSeason: '262.8', lastSeasonTrend: 'up', matchTier: 'B', status: 'lost', actual: 218, category: 'player_stats' },
    
    // Scoring Props
    { id: 'sp1', sport: 'NFL', sportIcon: '🏈', gameDate: '1/19', gameTime: '3:00 PM', teams: 'KC vs BUF', player: 'Isiah Pacheco', playerPosition: 'RB', playerTeamSide: 'AWAY', propType: 'atd', propLabel: 'Any TD', propColor: 'green', line: 0.5, oddsOver: '-125', oddsUnder: '+105', pick: 'OVER', projection: 0.8, probability: 0.58, edge: 2.4, tier: 'B', average: 0.6, lastSeason: '0.7', lastSeasonTrend: 'up', matchTier: 'B', status: 'pending', category: 'scoring_props' },
    { id: 'sp2', sport: 'NFL', sportIcon: '🏈', gameDate: '1/19', gameTime: '3:00 PM', teams: 'KC vs BUF', player: 'Travis Kelce', playerPosition: 'TE', playerTeamSide: 'AWAY', propType: 'ftd', propLabel: 'First TD', propColor: 'red', line: 0.5, oddsOver: '+850', oddsUnder: '-1200', pick: 'OVER', projection: 0.15, probability: 0.12, edge: 3.8, tier: 'A', average: 0.08, lastSeason: '0.1', lastSeasonTrend: 'up', matchTier: 'A', status: 'pending', category: 'scoring_props' },
    { id: 'sp3', sport: 'NBA', sportIcon: '🏀', gameDate: '1/20', gameTime: '7:30 PM', teams: 'LAL vs BOS', player: 'LeBron James', playerPosition: 'SF', playerTeamSide: 'HOME', propType: 'dd', propLabel: 'Double-Dbl', propColor: 'purple', line: 0.5, oddsOver: '-180', oddsUnder: '+150', pick: 'OVER', projection: 0.75, probability: 0.72, edge: 4.1, tier: 'A', average: 0.68, lastSeason: '0.72', lastSeasonTrend: 'up', matchTier: 'A', status: 'pending', category: 'scoring_props' },
    { id: 'sp4', sport: 'NHL', sportIcon: '🏒', gameDate: '1/20', gameTime: '7:00 PM', teams: 'EDM vs COL', player: 'Connor McDavid', playerPosition: 'C', playerTeamSide: 'HOME', propType: 'goal', propLabel: 'Goal', propColor: 'orange', line: 0.5, oddsOver: '-105', oddsUnder: '-115', pick: 'OVER', projection: 0.62, probability: 0.58, edge: 2.9, tier: 'B', average: 0.52, lastSeason: '0.58', lastSeasonTrend: 'up', matchTier: 'B', status: 'pending', category: 'scoring_props' },
    { id: 'sp5', sport: 'NFL', sportIcon: '🏈', gameDate: '1/19', gameTime: '3:00 PM', teams: 'BUF vs KC', player: 'James Cook', playerPosition: 'RB', playerTeamSide: 'HOME', propType: 'atd', propLabel: 'Any TD', propColor: 'green', line: 0.5, oddsOver: '-140', oddsUnder: '+115', pick: 'OVER', projection: 0.72, probability: 0.62, edge: 3.1, tier: 'B', average: 0.58, lastSeason: '0.65', lastSeasonTrend: 'up', matchTier: 'B', status: 'pending', category: 'scoring_props' },
    { id: 'sp6', sport: 'NBA', sportIcon: '🏀', gameDate: '1/20', gameTime: '10:00 PM', teams: 'GSW vs PHX', player: 'Kevin Durant', playerPosition: 'SF', playerTeamSide: 'HOME', propType: 'dd', propLabel: 'Double-Dbl', propColor: 'purple', line: 0.5, oddsOver: '+125', oddsUnder: '-150', pick: 'UNDER', projection: 0.38, probability: 0.55, edge: 1.8, tier: 'C', average: 0.42, lastSeason: '0.35', lastSeasonTrend: 'down', matchTier: 'C', status: 'pending', category: 'scoring_props' },
    { id: 'sp7', sport: 'NFL', sportIcon: '🏈', gameDate: '1/14', gameTime: '1:00 PM', teams: 'PHI vs TB', player: 'Jalen Hurts', playerPosition: 'QB', playerTeamSide: 'AWAY', propType: 'atd', propLabel: 'Any TD', propColor: 'green', line: 0.5, oddsOver: '-165', oddsUnder: '+140', pick: 'OVER', projection: 0.82, probability: 0.68, edge: 4.2, tier: 'A', average: 0.72, lastSeason: '0.78', lastSeasonTrend: 'up', matchTier: 'A', status: 'won', actual: 1, category: 'scoring_props' },
    { id: 'sp8', sport: 'NBA', sportIcon: '🏀', gameDate: '1/17', gameTime: '8:00 PM', teams: 'DEN vs MIN', player: 'Nikola Jokic', playerPosition: 'C', playerTeamSide: 'AWAY', propType: 'td', propLabel: 'Triple-Dbl', propColor: 'teal', line: 0.5, oddsOver: '+220', oddsUnder: '-280', pick: 'OVER', projection: 0.35, probability: 0.32, edge: 5.8, tier: 'A', average: 0.28, lastSeason: '0.32', lastSeasonTrend: 'up', matchTier: 'A', status: 'won', actual: 1, category: 'scoring_props' },
    
    // Game Events
    { id: 'ge1', sport: 'NFL', sportIcon: '🏈', gameDate: '1/19', gameTime: '3:00 PM', teams: 'KC vs BUF', player: 'Game', playerPosition: '-', playerTeamSide: '-', propType: 'ot', propLabel: 'Overtime', propColor: 'pink', line: 0.5, oddsOver: '+650', oddsUnder: '-950', pick: 'UNDER', projection: 0.08, probability: 0.92, edge: 2.1, tier: 'C', average: 0.05, lastSeason: '0.06', lastSeasonTrend: 'flat', matchTier: 'C', status: 'pending', category: 'game_events' },
    { id: 'ge2', sport: 'NFL', sportIcon: '🏈', gameDate: '1/19', gameTime: '3:00 PM', teams: 'KC vs BUF', player: 'Game', playerPosition: '-', playerTeamSide: '-', propType: 'safety', propLabel: 'Safety', propColor: 'red', line: 0.5, oddsOver: '+1200', oddsUnder: '-2000', pick: 'UNDER', projection: 0.04, probability: 0.96, edge: 1.5, tier: 'D', average: 0.02, lastSeason: '0.03', lastSeasonTrend: 'flat', matchTier: 'D', status: 'pending', category: 'game_events' },
    { id: 'ge3', sport: 'NFL', sportIcon: '🏈', gameDate: '1/19', gameTime: '3:00 PM', teams: 'KC vs BUF', player: 'KC', playerPosition: '-', playerTeamSide: 'AWAY', propType: 'fts', propLabel: 'First Score', propColor: 'green', line: 0.5, oddsOver: '-105', oddsUnder: '-115', pick: 'OVER', projection: 0.52, probability: 0.54, edge: 1.8, tier: 'C', average: 0.48, lastSeason: '0.51', lastSeasonTrend: 'up', matchTier: 'C', status: 'pending', category: 'game_events' },
    { id: 'ge4', sport: 'NBA', sportIcon: '🏀', gameDate: '1/20', gameTime: '7:30 PM', teams: 'LAL vs BOS', player: 'BOS', playerPosition: '-', playerTeamSide: 'AWAY', propType: 'fts', propLabel: 'First Score', propColor: 'green', line: 0.5, oddsOver: '-125', oddsUnder: '+105', pick: 'OVER', projection: 0.58, probability: 0.56, edge: 2.2, tier: 'B', average: 0.54, lastSeason: '0.56', lastSeasonTrend: 'flat', matchTier: 'B', status: 'pending', category: 'game_events' },
    { id: 'ge5', sport: 'NHL', sportIcon: '🏒', gameDate: '1/20', gameTime: '7:00 PM', teams: 'EDM vs COL', player: 'Game', playerPosition: '-', playerTeamSide: '-', propType: 'ot', propLabel: 'Overtime', propColor: 'pink', line: 0.5, oddsOver: '+280', oddsUnder: '-360', pick: 'UNDER', projection: 0.22, probability: 0.78, edge: 2.8, tier: 'B', average: 0.25, lastSeason: '0.24', lastSeasonTrend: 'flat', matchTier: 'B', status: 'pending', category: 'game_events' },
    { id: 'ge6', sport: 'NFL', sportIcon: '🏈', gameDate: '1/19', gameTime: '3:00 PM', teams: 'KC vs BUF', player: 'Game', playerPosition: '-', playerTeamSide: '-', propType: 'fgm', propLabel: '50+ FG', propColor: 'blue', line: 0.5, oddsOver: '+175', oddsUnder: '-220', pick: 'UNDER', projection: 0.32, probability: 0.68, edge: 2.4, tier: 'B', average: 0.28, lastSeason: '0.30', lastSeasonTrend: 'flat', matchTier: 'B', status: 'pending', category: 'game_events' },
    { id: 'ge7', sport: 'NFL', sportIcon: '🏈', gameDate: '1/14', gameTime: '4:30 PM', teams: 'DAL vs GB', player: 'Game', playerPosition: '-', playerTeamSide: '-', propType: 'ot', propLabel: 'Overtime', propColor: 'pink', line: 0.5, oddsOver: '+600', oddsUnder: '-900', pick: 'UNDER', projection: 0.10, probability: 0.90, edge: 1.8, tier: 'C', average: 0.08, lastSeason: '0.09', lastSeasonTrend: 'flat', matchTier: 'C', status: 'won', actual: 0, category: 'game_events' },
    { id: 'ge8', sport: 'NBA', sportIcon: '🏀', gameDate: '1/17', gameTime: '8:00 PM', teams: 'DEN vs MIN', player: 'DEN', playerPosition: '-', playerTeamSide: 'AWAY', propType: 'fts', propLabel: 'First Score', propColor: 'green', line: 0.5, oddsOver: '+105', oddsUnder: '-125', pick: 'OVER', projection: 0.48, probability: 0.52, edge: 3.2, tier: 'B', average: 0.46, lastSeason: '0.48', lastSeasonTrend: 'flat', matchTier: 'B', status: 'lost', actual: 0, category: 'game_events' },
  ];
};

export default GameProps;
