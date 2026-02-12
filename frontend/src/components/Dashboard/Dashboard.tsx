// src/components/Dashboard/Dashboard.tsx - Dashboard (Performance moved to Predictions)
import React, { useState, useEffect, useCallback } from 'react';
import {
  Box, Card, CardContent, Typography, Grid, Chip, useTheme, LinearProgress
} from '@mui/material';
import {
  TrendingUp, SportsBasketball, SportsFootball, SportsHockey, SportsBaseball,
  SportsTennis, CheckCircle, Schedule, EmojiEvents, Warning
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import { api } from '../../api/client';

interface TodayPrediction {
  sport: string;
  game: string;
  pick: string;
  tier: string;
  time: string;
}

const REFRESH_MS = 60000; // 1-minute auto-refresh

const Dashboard: React.FC = () => {
  const theme = useTheme();
  const isDark = theme.palette.mode === 'dark';
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);

  // Live data state (replaces hardcoded arrays)
  const [todayPredictions, setTodayPredictions] = useState<TodayPrediction[]>([]);
  const [quickStats, setQuickStats] = useState([
    { label: 'Today\'s Picks', value: '0', color: 'primary' },
    { label: 'Tier A Picks', value: '0', color: 'success' },
    { label: 'Pending', value: '0', color: 'warning' },
    { label: 'Graded Today', value: '0', color: 'info' },
  ]);
  const [bestPerformers, setBestPerformers] = useState<Array<{ label: string; value: string; color: string }>>([]);
  const [areasToMonitor, setAreasToMonitor] = useState<Array<{ label: string; value: string; color: string }>>([]);
  const [recentActivity, setRecentActivity] = useState<Array<{ icon: React.ReactNode; text: string; time: string }>>([]);

  const getActivityIcon = (iconType: string) => {
    switch (iconType) {
      case 'win': return <CheckCircle sx={{ fontSize: 18, color: 'success.main' }} />;
      case 'loss': return <Warning sx={{ fontSize: 18, color: 'error.main' }} />;
      default: return <Schedule sx={{ fontSize: 18, color: 'info.main' }} />;
    }
  };

  const fetchDashboard = useCallback(async (showLoading = false) => {
    if (showLoading) setLoading(true);
    try {
      const stats = await api.getDashboardStats();

      // Quick stats
      setQuickStats([
        { label: 'Today\'s Picks', value: String(stats.total_predictions || 0), color: 'primary' },
        { label: 'Tier A Picks', value: String(stats.tier_a_count || 0), color: 'success' },
        { label: 'Pending', value: String(stats.pending_count || 0), color: 'warning' },
        { label: 'Graded Today', value: String(stats.graded_today || 0), color: 'info' },
      ]);

      // Today's predictions (top picks)
      setTodayPredictions(
        (stats.top_picks || []).map((p: any) => ({
          sport: p.sport || '',
          game: p.game || '',
          pick: p.pick || '',
          tier: p.tier || 'D',
          time: p.time ? new Date(p.time).toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', timeZone: 'America/Los_Angeles' }) : '',
        }))
      );

      // Best performers
      setBestPerformers(
        (stats.best_performers || []).map((p: any) => ({
          label: p.label || '',
          value: p.value || '',
          color: p.color || 'success',
        }))
      );

      // Areas to monitor
      setAreasToMonitor(
        (stats.areas_to_monitor || []).map((p: any) => ({
          label: p.label || '',
          value: p.value || '',
          color: p.color || 'warning',
        }))
      );

      // Recent activity
      setRecentActivity(
        (stats.recent_activity || []).map((a: any) => ({
          icon: getActivityIcon(a.icon || 'pending'),
          text: a.text || '',
          time: a.time || '',
        }))
      );
    } catch (err) {
      console.error('Dashboard fetch error:', err);
    }
    if (showLoading) setLoading(false);
  }, []);

  // Initial load
  useEffect(() => {
    fetchDashboard(true);
  }, [fetchDashboard]);

  // Auto-refresh every 60 seconds
  useEffect(() => {
    const interval = setInterval(() => fetchDashboard(false), REFRESH_MS);
    return () => clearInterval(interval);
  }, [fetchDashboard]);

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
              {todayPredictions.length > 0 ? todayPredictions.map((pred, idx) => (
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
              )) : (
                <Box sx={{ textAlign: 'center', py: 4, color: 'text.secondary' }}>
                  <Schedule sx={{ fontSize: 40, mb: 1, opacity: 0.4 }} />
                  <Typography sx={{ fontSize: 14 }}>No predictions yet</Typography>
                  <Typography sx={{ fontSize: 12, mt: 0.5 }}>Predictions will appear once generated</Typography>
                </Box>
              )}
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
              {recentActivity.length > 0 ? recentActivity.map((activity, idx) => (
                <Box key={idx} sx={{ display: 'flex', alignItems: 'center', gap: 1.5, py: 1 }}>
                  {activity.icon}
                  <Typography sx={{ fontSize: 13, flex: 1 }}>{activity.text}</Typography>
                  <Typography sx={{ fontSize: 11, color: 'text.secondary' }}>{activity.time}</Typography>
                </Box>
              )) : (
                <Box sx={{ textAlign: 'center', py: 4, color: 'text.secondary' }}>
                  <Typography sx={{ fontSize: 14 }}>No graded predictions yet</Typography>
                </Box>
              )}
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
              {bestPerformers.length > 0 ? bestPerformers.map((item, idx) => (
                <Box key={idx} sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', py: 0.75 }}>
                  <Typography sx={{ fontSize: 13 }}>{item.label}</Typography>
                  <Chip label={item.value} size="small" color={item.color as any} sx={{ fontSize: 11, height: 22 }} />
                </Box>
              )) : (
                <Box sx={{ textAlign: 'center', py: 2, color: 'text.secondary' }}>
                  <Typography sx={{ fontSize: 13 }}>No graded data yet</Typography>
                </Box>
              )}
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
              {areasToMonitor.length > 0 ? areasToMonitor.map((item, idx) => (
                <Box key={idx} sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', py: 0.75 }}>
                  <Typography sx={{ fontSize: 13 }}>{item.label}</Typography>
                  <Chip label={item.value} size="small" color={item.color as any} sx={{ fontSize: 11, height: 22 }} />
                </Box>
              )) : (
                <Box sx={{ textAlign: 'center', py: 2, color: 'text.secondary' }}>
                  <Typography sx={{ fontSize: 13 }}>No data to monitor yet</Typography>
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Dashboard;