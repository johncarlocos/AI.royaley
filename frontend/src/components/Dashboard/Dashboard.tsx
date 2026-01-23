// src/components/Dashboard/Dashboard.tsx - Dashboard (Performance moved to Predictions)
import React, { useState, useEffect } from 'react';
import {
  Box, Card, CardContent, Typography, Grid, Chip, useTheme, LinearProgress
} from '@mui/material';
import {
  TrendingUp, SportsBasketball, SportsFootball, SportsHockey, SportsBaseball,
  SportsTennis, CheckCircle, Schedule, EmojiEvents, Warning
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';

interface TodayPrediction {
  sport: string;
  game: string;
  pick: string;
  tier: string;
  time: string;
}

const Dashboard: React.FC = () => {
  const theme = useTheme();
  const isDark = theme.palette.mode === 'dark';
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setTimeout(() => setLoading(false), 300);
  }, []);

  // Today's predictions
  const todayPredictions: TodayPrediction[] = [
    { sport: 'NBA', game: 'Celtics vs Lakers', pick: 'Boston -6', tier: 'A', time: '5:00 PM' },
    { sport: 'NBA', game: 'Warriors vs Suns', pick: 'Phoenix -4', tier: 'A', time: '7:30 PM' },
    { sport: 'NFL', game: 'Chiefs vs Bills', pick: 'Bills -3.5', tier: 'A', time: '1:00 PM' },
    { sport: 'NHL', game: 'Maple Leafs vs Bruins', pick: 'Under 6', tier: 'B', time: '7:00 PM' },
  ];

  // Quick stats
  const quickStats = [
    { label: 'Today\'s Picks', value: '24', color: 'primary' },
    { label: 'Tier A Picks', value: '6', color: 'success' },
    { label: 'Pending', value: '18', color: 'warning' },
    { label: 'Graded Today', value: '12', color: 'info' },
  ];

  // Best performers
  const bestPerformers = [
    { label: 'Tier A Predictions', value: '68.5% Win Rate', color: 'success' },
    { label: 'CFL Games', value: '75.0% Win Rate', color: 'success' },
    { label: 'NFL Full Game', value: '+9.8% ROI', color: 'success' },
    { label: '2H Spreads', value: '60.3% Win Rate', color: 'success' },
  ];

  // Areas to monitor
  const areasToMonitor = [
    { label: 'Tier D Predictions', value: '51.1% Win Rate', color: 'warning' },
    { label: '2H Totals', value: '55.6% Win Rate', color: 'warning' },
    { label: 'Tier D ROI', value: '-1.8% ROI', color: 'error' },
    { label: 'Tier D CLV', value: '-0.2% CLV', color: 'error' },
  ];

  // Recent activity
  const recentActivity = [
    { icon: <CheckCircle sx={{ fontSize: 18, color: 'success.main' }} />, text: 'Bills -3 WON (+2.1 units)', time: '2h ago' },
    { icon: <CheckCircle sx={{ fontSize: 18, color: 'success.main' }} />, text: 'Lakers Under 224.5 WON (+1.8 units)', time: '3h ago' },
    { icon: <Warning sx={{ fontSize: 18, color: 'error.main' }} />, text: 'Celtics -5.5 LOST (-1.0 units)', time: '4h ago' },
    { icon: <Schedule sx={{ fontSize: 18, color: 'info.main' }} />, text: 'New Tier A: Warriors +3', time: '5h ago' },
    { icon: <CheckCircle sx={{ fontSize: 18, color: 'success.main' }} />, text: 'Bruins ML WON (+1.5 units)', time: '6h ago' },
  ];

  const getSportIcon = (sport: string) => {
    switch (sport) {
      case 'NBA': case 'NCAAB': case 'WNBA': return <SportsBasketball sx={{ fontSize: 20 }} />;
      case 'NFL': case 'NCAAF': case 'CFL': return <SportsFootball sx={{ fontSize: 20 }} />;
      case 'NHL': return <SportsHockey sx={{ fontSize: 20 }} />;
      case 'MLB': return <SportsBaseball sx={{ fontSize: 20 }} />;
      case 'ATP': case 'WTA': return <SportsTennis sx={{ fontSize: 20 }} />;
      default: return <TrendingUp sx={{ fontSize: 20 }} />;
    }
  };

  const getTierColor = (tier: string) => {
    switch (tier) {
      case 'A': return 'success';
      case 'B': return 'primary';
      case 'C': return 'warning';
      default: return 'default';
    }
  };

  if (loading) return <LinearProgress />;

  return (
    <Box>
      {/* Header */}
      <Box mb={2.5}>
        <Typography variant="h5" fontWeight={700} sx={{ fontSize: 20 }}>Dashboard</Typography>
        <Typography variant="body2" color="text.secondary" sx={{ fontSize: 13 }}>
          Quick overview • Full performance stats available in Predictions page
        </Typography>
      </Box>

      {/* Quick Stats */}
      <Grid container spacing={2} mb={2.5}>
        {quickStats.map((stat, idx) => (
          <Grid item xs={6} sm={3} key={idx}>
            <Card sx={{ textAlign: 'center', cursor: 'pointer', '&:hover': { bgcolor: isDark ? 'grey.800' : 'grey.50' } }} onClick={() => navigate('/predictions')}>
              <CardContent sx={{ py: 2, '&:last-child': { pb: 2 } }}>
                <Typography sx={{ fontSize: 13, color: 'text.secondary', mb: 0.5 }}>{stat.label}</Typography>
                <Typography sx={{ fontSize: 28, fontWeight: 700, color: `${stat.color}.main` }}>{stat.value}</Typography>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      <Grid container spacing={2.5}>
        {/* Today's Top Picks */}
        <Grid item xs={12} md={6}>
          <Card sx={{ height: '100%' }}>
            <CardContent sx={{ py: 2, px: 2, '&:last-child': { pb: 2 } }}>
              <Box display="flex" alignItems="center" justifyContent="space-between" mb={1.5}>
                <Box display="flex" alignItems="center" gap={1}>
                  <EmojiEvents sx={{ fontSize: 22, color: 'warning.main' }} />
                  <Typography variant="subtitle1" fontWeight={600} sx={{ fontSize: 16 }}>Today's Top Picks</Typography>
                </Box>
                <Chip label="View All" size="small" sx={{ fontSize: 11, height: 24, cursor: 'pointer' }} onClick={() => navigate('/predictions')} />
              </Box>
              {todayPredictions.map((pred, idx) => (
                <Box key={idx} sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', py: 1.25, borderBottom: idx < todayPredictions.length - 1 ? 1 : 0, borderColor: 'divider' }}>
                  <Box display="flex" alignItems="center" gap={1.5}>
                    <Box sx={{ color: 'text.secondary' }}>{getSportIcon(pred.sport)}</Box>
                    <Box>
                      <Typography sx={{ fontSize: 14, fontWeight: 600 }}>{pred.pick}</Typography>
                      <Typography sx={{ fontSize: 12, color: 'text.secondary' }}>{pred.game}</Typography>
                    </Box>
                  </Box>
                  <Box display="flex" alignItems="center" gap={1.5}>
                    <Typography sx={{ fontSize: 12, color: 'text.secondary' }}>{pred.time}</Typography>
                    <Chip label={pred.tier} size="small" color={getTierColor(pred.tier) as any} sx={{ fontSize: 11, height: 22, minWidth: 28, fontWeight: 700 }} />
                  </Box>
                </Box>
              ))}
            </CardContent>
          </Card>
        </Grid>

        {/* Recent Activity */}
        <Grid item xs={12} md={6}>
          <Card sx={{ height: '100%' }}>
            <CardContent sx={{ py: 2, px: 2, '&:last-child': { pb: 2 } }}>
              <Box display="flex" alignItems="center" gap={1} mb={1.5}>
                <Schedule sx={{ fontSize: 22, color: 'info.main' }} />
                <Typography variant="subtitle1" fontWeight={600} sx={{ fontSize: 16 }}>Recent Activity</Typography>
              </Box>
              {recentActivity.map((activity, idx) => (
                <Box key={idx} sx={{ display: 'flex', alignItems: 'center', gap: 1.5, py: 1 }}>
                  {activity.icon}
                  <Typography sx={{ fontSize: 13, flex: 1 }}>{activity.text}</Typography>
                  <Typography sx={{ fontSize: 11, color: 'text.secondary' }}>{activity.time}</Typography>
                </Box>
              ))}
            </CardContent>
          </Card>
        </Grid>

        {/* Best Performers */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent sx={{ py: 2, px: 2, '&:last-child': { pb: 2 } }}>
              <Box display="flex" alignItems="center" gap={1} mb={1.5}>
                <Typography sx={{ fontSize: 20 }}>🔥</Typography>
                <Typography variant="subtitle1" fontWeight={600} sx={{ fontSize: 16 }}>Best Performers</Typography>
              </Box>
              {bestPerformers.map((item, idx) => (
                <Box key={idx} sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', py: 0.75 }}>
                  <Typography sx={{ fontSize: 13 }}>{item.label}</Typography>
                  <Chip label={item.value} size="small" color={item.color as any} sx={{ fontSize: 11, height: 22 }} />
                </Box>
              ))}
            </CardContent>
          </Card>
        </Grid>

        {/* Areas to Monitor */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent sx={{ py: 2, px: 2, '&:last-child': { pb: 2 } }}>
              <Box display="flex" alignItems="center" gap={1} mb={1.5}>
                <Typography sx={{ fontSize: 20 }}>⚠️</Typography>
                <Typography variant="subtitle1" fontWeight={600} sx={{ fontSize: 16 }}>Areas to Monitor</Typography>
              </Box>
              {areasToMonitor.map((item, idx) => (
                <Box key={idx} sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', py: 0.75 }}>
                  <Typography sx={{ fontSize: 13 }}>{item.label}</Typography>
                  <Chip label={item.value} size="small" color={item.color as any} sx={{ fontSize: 11, height: 22 }} />
                </Box>
              ))}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Dashboard;
