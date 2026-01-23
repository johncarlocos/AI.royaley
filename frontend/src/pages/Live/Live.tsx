// src/pages/Live/Live.tsx
import React, { useState, useEffect } from 'react';
import {
  Box, Typography, Select, MenuItem, FormControl, InputLabel, Button,
  LinearProgress, Table, TableBody, TableCell, TableContainer, TableHead,
  TableRow, Paper, Chip, useTheme, alpha
} from '@mui/material';
import { PlayCircle, Refresh, FiberManualRecord } from '@mui/icons-material';
import { api } from '../../api/client';
import { SPORTS } from '../../types';

interface LiveGame {
  id: string;
  sport: string;
  date: string;
  time: string;
  gameNumber: number;
  homeTeam: string;
  awayTeam: string;
  homeRecord: string;
  awayRecord: string;
  homeScore: number | null;
  awayScore: number | null;
  period: string;
  status: 'live' | 'upcoming' | 'halftime' | 'final';
  spread: { away: string; home: string };
  total: string;
  prediction: {
    pick: string;
    type: string;
    probability: number;
    edge: number;
    tier: string;
  } | null;
}

// Tier badge component
const TierBadge: React.FC<{ tier: string }> = ({ tier }) => {
  const colors: Record<string, string> = { A: '#4caf50', B: '#2196f3', C: '#ff9800', D: '#9e9e9e' };
  return (
    <Chip
      label={`Tier ${tier}`}
      size="small"
      sx={{
        bgcolor: colors[tier] || colors.D,
        color: 'white',
        fontWeight: 600,
        fontSize: '0.7rem',
        height: 22,
      }}
    />
  );
};

// Status indicator
const StatusBadge: React.FC<{ status: string; period?: string }> = ({ status, period }) => {
  if (status === 'live') {
    return (
      <Box display="flex" alignItems="center" gap={0.5}>
        <FiberManualRecord sx={{ fontSize: 10, color: 'error.main', animation: 'pulse 1.5s infinite' }} />
        <Typography variant="caption" color="error.main" fontWeight={600}>
          {period || 'LIVE'}
        </Typography>
      </Box>
    );
  }
  if (status === 'halftime') {
    return <Chip label="HALF" size="small" sx={{ bgcolor: 'warning.main', color: 'white', height: 20, fontSize: '0.65rem' }} />;
  }
  if (status === 'final') {
    return <Chip label="FINAL" size="small" sx={{ bgcolor: 'grey.600', color: 'white', height: 20, fontSize: '0.65rem' }} />;
  }
  return <Typography variant="caption" color="text.secondary">{period}</Typography>;
};

const Live: React.FC = () => {
  const theme = useTheme();
  const [games, setGames] = useState<LiveGame[]>([]);
  const [loading, setLoading] = useState(true);
  const [sportFilter, setSportFilter] = useState('all');
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());

  const isDark = theme.palette.mode === 'dark';
  const headerBg = isDark ? '#1a2332' : '#f5f5f5';
  const grayColBg = isDark ? alpha('#64748b', 0.15) : alpha('#94a3b8', 0.15);
  const rowHoverBg = isDark ? alpha('#3b82f6', 0.08) : alpha('#3b82f6', 0.04);

  useEffect(() => {
    loadGames();
    const interval = setInterval(() => {
      loadGames();
    }, 30000); // Refresh every 30 seconds for live
    return () => clearInterval(interval);
  }, [sportFilter]);

  const loadGames = async () => {
    setLoading(true);
    try {
      const data = await api.getGames({ sport: sportFilter !== 'all' ? sportFilter : undefined });
      setGames(Array.isArray(data) ? data : generateDemoGames());
    } catch {
      setGames(generateDemoGames());
    }
    setLoading(false);
    setLastUpdate(new Date());
  };

  const filtered = sportFilter === 'all' ? games : games.filter(g => g.sport === sportFilter);
  const liveCount = filtered.filter(g => g.status === 'live' || g.status === 'halftime').length;

  // Common cell style
  const cellSx = { py: 1, px: 1.5, fontSize: '0.8rem', borderRight: `1px solid ${isDark ? '#2d3748' : '#e2e8f0'}` };
  const grayCellSx = { ...cellSx, bgcolor: grayColBg };
  const headerCellSx = { py: 1, px: 1.5, fontWeight: 600, fontSize: '0.75rem', borderRight: `1px solid ${isDark ? '#2d3748' : '#e2e8f0'}`, bgcolor: headerBg };

  return (
    <Box>
      {/* Header */}
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
        <Box display="flex" alignItems="center" gap={2}>
          <Typography variant="h5" fontWeight={700} sx={{ fontSize: 20 }}>
            <PlayCircle sx={{ verticalAlign: 'middle', mr: 1, color: 'error.main' }} />
            LIVE
          </Typography>
          {liveCount > 0 && (
            <Chip
              icon={<FiberManualRecord sx={{ fontSize: '10px !important' }} />}
              label={`${liveCount} Live`}
              color="error"
              size="small"
              sx={{ animation: 'pulse 2s infinite' }}
            />
          )}
          <Typography variant="caption" color="text.secondary">
            Updated: {lastUpdate.toLocaleTimeString()}
          </Typography>
        </Box>
        <Box display="flex" gap={2}>
          <FormControl size="small" sx={{ minWidth: 120 }}>
            <InputLabel>Sport</InputLabel>
            <Select value={sportFilter} label="Sport" onChange={(e) => setSportFilter(e.target.value)}>
              <MenuItem value="all">All Sports</MenuItem>
              {SPORTS.map(s => <MenuItem key={s.code} value={s.code}>{s.code}</MenuItem>)}
            </Select>
          </FormControl>
          <Button variant="outlined" startIcon={<Refresh />} onClick={loadGames} size="small">
            Refresh
          </Button>
        </Box>
      </Box>

      {loading && <LinearProgress sx={{ mb: 2 }} />}

      {/* Live Games Table */}
      <Paper variant="outlined" sx={{ overflow: 'hidden' }}>
        <TableContainer sx={{ maxHeight: 'calc(100vh - 200px)' }}>
          <Table size="small" stickyHeader>
            <TableHead>
              <TableRow>
                <TableCell sx={{ ...headerCellSx, minWidth: 60 }}>Sport</TableCell>
                <TableCell sx={{ ...headerCellSx, minWidth: 70 }}>Time</TableCell>
                <TableCell sx={{ ...headerCellSx, minWidth: 50, textAlign: 'center' }}>Game #</TableCell>
                <TableCell sx={{ ...headerCellSx, minWidth: 140 }}>Team</TableCell>
                <TableCell sx={{ ...headerCellSx, minWidth: 50, textAlign: 'center' }}>Record</TableCell>
                <TableCell sx={{ ...headerCellSx, minWidth: 50, textAlign: 'center' }}>Score</TableCell>
                <TableCell sx={{ ...headerCellSx, minWidth: 70, textAlign: 'center' }}>Status</TableCell>
                <TableCell sx={{ ...headerCellSx, minWidth: 60, textAlign: 'center' }}>Spread</TableCell>
                <TableCell sx={{ ...headerCellSx, minWidth: 60, textAlign: 'center' }}>Total</TableCell>
                <TableCell sx={{ ...headerCellSx, minWidth: 100 }}>Pick</TableCell>
                <TableCell sx={{ ...headerCellSx, minWidth: 50, textAlign: 'center' }}>%</TableCell>
                <TableCell sx={{ ...headerCellSx, minWidth: 50, textAlign: 'center' }}>Edge</TableCell>
                <TableCell sx={{ ...headerCellSx, minWidth: 70, textAlign: 'center', borderRight: 'none' }}>Tier</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {filtered.map((game, gameIdx) => (
                <React.Fragment key={game.id}>
                  {/* Away Team Row */}
                  <TableRow
                    sx={{
                      bgcolor: game.status === 'live' ? alpha('#ef4444', 0.08) : 'transparent',
                      '&:hover': { bgcolor: rowHoverBg },
                      borderTop: gameIdx > 0 ? `2px solid ${isDark ? '#4a5568' : '#cbd5e1'}` : undefined,
                    }}
                  >
                    <TableCell rowSpan={2} sx={{ ...grayCellSx, verticalAlign: 'middle', fontWeight: 600 }}>
                      {game.sport}
                    </TableCell>
                    <TableCell rowSpan={2} sx={{ ...grayCellSx, verticalAlign: 'middle' }}>
                      {game.time}
                    </TableCell>
                    <TableCell sx={{ ...cellSx, textAlign: 'center', bgcolor: isDark ? '#2a3441' : '#f0f4f8' }}>
                      {game.gameNumber}
                    </TableCell>
                    <TableCell sx={{ ...cellSx, fontWeight: game.prediction?.pick.includes(game.awayTeam) ? 700 : 400 }}>
                      {game.awayTeam}
                    </TableCell>
                    <TableCell sx={{ ...cellSx, textAlign: 'center', color: 'text.secondary' }}>
                      {game.awayRecord}
                    </TableCell>
                    <TableCell sx={{ ...cellSx, textAlign: 'center', fontWeight: 700, fontSize: '0.9rem' }}>
                      {game.homeScore !== null ? game.awayScore : '-'}
                    </TableCell>
                    <TableCell rowSpan={2} sx={{ ...cellSx, verticalAlign: 'middle', textAlign: 'center' }}>
                      <StatusBadge status={game.status} period={game.period} />
                    </TableCell>
                    <TableCell sx={{ ...cellSx, textAlign: 'center' }}>
                      {game.spread.away}
                    </TableCell>
                    <TableCell rowSpan={2} sx={{ ...cellSx, textAlign: 'center', verticalAlign: 'middle' }}>
                      {game.total}
                    </TableCell>
                    <TableCell rowSpan={2} sx={{ ...cellSx, verticalAlign: 'middle' }}>
                      {game.prediction ? (
                        <Box>
                          <Typography sx={{ fontSize: 11, fontWeight: 600, color: 'success.main', lineHeight: 1.3 }}>
                            {game.prediction.pick}
                          </Typography>
                          <Typography sx={{ fontSize: 10, color: 'text.secondary' }}>
                            {game.prediction.type}
                          </Typography>
                        </Box>
                      ) : (
                        <Typography sx={{ fontSize: 10, color: 'text.disabled' }}>-</Typography>
                      )}
                    </TableCell>
                    <TableCell rowSpan={2} sx={{ ...cellSx, textAlign: 'center', verticalAlign: 'middle', fontWeight: 600 }}>
                      {game.prediction ? `${(game.prediction.probability * 100).toFixed(0)}%` : '-'}
                    </TableCell>
                    <TableCell rowSpan={2} sx={{ ...cellSx, textAlign: 'center', verticalAlign: 'middle', color: game.prediction && game.prediction.edge > 0 ? 'success.main' : 'text.secondary' }}>
                      {game.prediction ? `+${game.prediction.edge.toFixed(1)}%` : '-'}
                    </TableCell>
                    <TableCell rowSpan={2} sx={{ ...cellSx, textAlign: 'center', verticalAlign: 'middle', borderRight: 'none' }}>
                      {game.prediction ? <TierBadge tier={game.prediction.tier} /> : '-'}
                    </TableCell>
                  </TableRow>

                  {/* Home Team Row */}
                  <TableRow
                    sx={{
                      bgcolor: game.status === 'live' ? alpha('#ef4444', 0.08) : 'transparent',
                      '&:hover': { bgcolor: rowHoverBg },
                    }}
                  >
                    <TableCell sx={{ ...cellSx, textAlign: 'center', bgcolor: isDark ? '#2a3441' : '#f0f4f8' }}>
                      {game.gameNumber + 1}
                    </TableCell>
                    <TableCell sx={{ ...cellSx, fontWeight: game.prediction?.pick.includes(game.homeTeam) ? 700 : 400 }}>
                      {game.homeTeam}
                    </TableCell>
                    <TableCell sx={{ ...cellSx, textAlign: 'center', color: 'text.secondary' }}>
                      {game.homeRecord}
                    </TableCell>
                    <TableCell sx={{ ...cellSx, textAlign: 'center', fontWeight: 700, fontSize: '0.9rem' }}>
                      {game.homeScore !== null ? game.homeScore : '-'}
                    </TableCell>
                    <TableCell sx={{ ...cellSx, textAlign: 'center' }}>
                      {game.spread.home}
                    </TableCell>
                  </TableRow>
                </React.Fragment>
              ))}
            </TableBody>
          </Table>
        </TableContainer>

        {filtered.length === 0 && !loading && (
          <Box sx={{ textAlign: 'center', py: 6 }}>
            <Typography color="text.secondary">No games found.</Typography>
          </Box>
        )}
      </Paper>

      {/* Summary Stats */}
      <Box display="flex" gap={3} mt={2}>
        <Typography variant="body2" color="text.secondary">
          <strong>{filtered.filter(g => g.status === 'live').length}</strong> Live
        </Typography>
        <Typography variant="body2" color="text.secondary">
          <strong>{filtered.filter(g => g.status === 'halftime').length}</strong> Halftime
        </Typography>
        <Typography variant="body2" color="text.secondary">
          <strong>{filtered.filter(g => g.status === 'upcoming').length}</strong> Upcoming
        </Typography>
        <Typography variant="body2" color="text.secondary">
          <strong>{filtered.filter(g => g.status === 'final').length}</strong> Final
        </Typography>
        <Typography variant="body2" color="text.secondary">
          <strong>{filtered.filter(g => g.prediction !== null).length}</strong> With Predictions
        </Typography>
      </Box>

      {/* CSS for pulse animation */}
      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }
      `}</style>
    </Box>
  );
};

// Generate demo live games
const generateDemoGames = (): LiveGame[] => [
  {
    id: '1', sport: 'NBA', date: '1/19/2026', time: '7:30 PM', gameNumber: 601,
    awayTeam: 'Boston Celtics', homeTeam: 'Los Angeles Lakers',
    awayRecord: '32-12', homeRecord: '25-18',
    awayScore: 58, homeScore: 52,
    period: 'Q2 4:32', status: 'live',
    spread: { away: '+4.5', home: '-4.5' }, total: 'O/U 224.5',
    prediction: { pick: 'Celtics +4.5', type: 'Spread', probability: 0.67, edge: 3.2, tier: 'A' }
  },
  {
    id: '2', sport: 'NBA', date: '1/19/2026', time: '8:00 PM', gameNumber: 603,
    awayTeam: 'Phoenix Suns', homeTeam: 'Golden State Warriors',
    awayRecord: '26-19', homeRecord: '28-17',
    awayScore: 45, homeScore: 48,
    period: 'Q2 8:15', status: 'live',
    spread: { away: '+3', home: '-3' }, total: 'O/U 231',
    prediction: { pick: 'Under 231', type: 'Total', probability: 0.62, edge: 2.1, tier: 'B' }
  },
  {
    id: '3', sport: 'NFL', date: '1/19/2026', time: '6:30 PM', gameNumber: 605,
    awayTeam: 'Buffalo Bills', homeTeam: 'Kansas City Chiefs',
    awayRecord: '14-4', homeRecord: '15-3',
    awayScore: 14, homeScore: 17,
    period: 'Halftime', status: 'halftime',
    spread: { away: '+2.5', home: '-2.5' }, total: 'O/U 48.5',
    prediction: { pick: 'Chiefs -2.5', type: 'Spread', probability: 0.65, edge: 2.8, tier: 'A' }
  },
  {
    id: '4', sport: 'NHL', date: '1/19/2026', time: '7:00 PM', gameNumber: 607,
    awayTeam: 'Toronto Maple Leafs', homeTeam: 'Boston Bruins',
    awayRecord: '29-14-4', homeRecord: '31-11-5',
    awayScore: 2, homeScore: 3,
    period: 'P2 12:45', status: 'live',
    spread: { away: '+1.5', home: '-1.5' }, total: 'O/U 6.5',
    prediction: { pick: 'Over 6.5', type: 'Total', probability: 0.58, edge: 1.5, tier: 'C' }
  },
  {
    id: '5', sport: 'NCAAB', date: '1/19/2026', time: '9:00 PM', gameNumber: 609,
    awayTeam: 'Duke Blue Devils', homeTeam: 'UNC Tar Heels',
    awayRecord: '16-3', homeRecord: '15-4',
    awayScore: null, homeScore: null,
    period: '9:00 PM', status: 'upcoming',
    spread: { away: '-2.5', home: '+2.5' }, total: 'O/U 152.5',
    prediction: { pick: 'Duke -2.5', type: 'Spread', probability: 0.64, edge: 2.5, tier: 'B' }
  },
  {
    id: '6', sport: 'NBA', date: '1/19/2026', time: '10:30 PM', gameNumber: 611,
    awayTeam: 'Denver Nuggets', homeTeam: 'LA Clippers',
    awayRecord: '30-14', homeRecord: '27-17',
    awayScore: null, homeScore: null,
    period: '10:30 PM', status: 'upcoming',
    spread: { away: '-1.5', home: '+1.5' }, total: 'O/U 222',
    prediction: { pick: 'Nuggets -1.5', type: 'Spread', probability: 0.61, edge: 1.9, tier: 'B' }
  },
  {
    id: '7', sport: 'NHL', date: '1/19/2026', time: '4:00 PM', gameNumber: 613,
    awayTeam: 'Colorado Avalanche', homeTeam: 'Vegas Golden Knights',
    awayRecord: '30-13-3', homeRecord: '32-10-4',
    awayScore: 4, homeScore: 5,
    period: 'Final', status: 'final',
    spread: { away: '+1.5', home: '-1.5' }, total: 'O/U 6.5',
    prediction: { pick: 'Over 6.5', type: 'Total', probability: 0.66, edge: 3.0, tier: 'A' }
  },
  {
    id: '8', sport: 'NBA', date: '1/19/2026', time: '5:00 PM', gameNumber: 615,
    awayTeam: 'Milwaukee Bucks', homeTeam: 'Miami Heat',
    awayRecord: '27-17', homeRecord: '24-20',
    awayScore: 112, homeScore: 105,
    period: 'Final', status: 'final',
    spread: { away: '-4', home: '+4' }, total: 'O/U 218.5',
    prediction: { pick: 'Bucks -4', type: 'Spread', probability: 0.63, edge: 2.2, tier: 'B' }
  },
  {
    id: '9', sport: 'MLB', date: '1/19/2026', time: '1:05 PM', gameNumber: 617,
    awayTeam: 'NY Yankees', homeTeam: 'LA Dodgers',
    awayRecord: '58-34', homeRecord: '62-30',
    awayScore: 4, homeScore: 6,
    period: 'Final', status: 'final',
    spread: { away: '+1.5', home: '-1.5' }, total: 'O/U 8.5',
    prediction: { pick: 'Dodgers -1.5', type: 'Run Line', probability: 0.59, edge: 1.8, tier: 'C' }
  },
  {
    id: '10', sport: 'ATP', date: '1/19/2026', time: '11:00 AM', gameNumber: 619,
    awayTeam: 'Novak Djokovic', homeTeam: 'Carlos Alcaraz',
    awayRecord: '45-8', homeRecord: '52-6',
    awayScore: null, homeScore: null,
    period: '11:00 AM', status: 'upcoming',
    spread: { away: '+1.5', home: '-1.5' }, total: 'O/U 3.5 sets',
    prediction: { pick: 'Alcaraz -1.5', type: 'Set Spread', probability: 0.57, edge: 1.2, tier: 'C' }
  },
];

export default Live;
