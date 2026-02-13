// src/components/Dashboard/Dashboard.tsx - Full Dashboard with all KPIs
import React, { useState, useEffect, useCallback } from 'react';
import {
  Box, Card, Typography, Grid, Chip, useTheme, LinearProgress, Tooltip, Button
} from '@mui/material';
import {
  TrendingUp, TrendingDown, SportsBasketball, SportsFootball, SportsHockey,
  SportsBaseball, SportsTennis, CheckCircle, Schedule, EmojiEvents, Warning,
  Refresh, Casino, Psychology, Speed, SportsScore, Error as ErrorIcon
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import { api } from '../../api/client';
import { useSettingsStore } from '../../store';
import { formatTime, getTimezoneAbbr } from '../../utils/formatters';

const REFRESH_MS = 60000;

interface DashboardData {
  total_predictions: number;
  tier_a_count: number;
  pending_count: number;
  graded_today: number;
  graded_total: number;
  wins: number;
  losses: number;
  pushes: number;
  win_rate: number;
  roi: number;
  total_pnl: number;
  avg_clv: number;
  top_picks: any[];
  recent_activity: any[];
  best_performers: any[];
  areas_to_monitor: any[];
  sport_breakdown: any[];
  tier_breakdown: any[];
  bet_type_breakdown: any[];
  upcoming_games_count: number;
  active_models_count: number;
}

const Dashboard: React.FC = () => {
  const theme = useTheme();
  const isDark = theme.palette.mode === 'dark';
  const navigate = useNavigate();
  const { timezone, timeFormat } = useSettingsStore();
  const [loading, setLoading] = useState(true);
  const [data, setData] = useState<DashboardData | null>(null);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());

  const fetchDashboard = useCallback(async (showLoading = false) => {
    if (showLoading) setLoading(true);
    try {
      const stats = await api.getDashboardStats();
      setData(stats);
      setLastUpdate(new Date());
    } catch (err) {
      console.error('Dashboard fetch error:', err);
    }
    if (showLoading) setLoading(false);
  }, []);

  useEffect(() => { fetchDashboard(true); }, [fetchDashboard]);
  useEffect(() => {
    const interval = setInterval(() => fetchDashboard(false), REFRESH_MS);
    return () => clearInterval(interval);
  }, [fetchDashboard]);

  const getSportIcon = (sport: string) => {
    switch (sport) {
      case 'NBA': case 'NCAAB': case 'WNBA': return <SportsBasketball sx={{ fontSize: 18 }} />;
      case 'NFL': case 'NCAAF': case 'CFL': return <SportsFootball sx={{ fontSize: 18 }} />;
      case 'NHL': return <SportsHockey sx={{ fontSize: 18 }} />;
      case 'MLB': return <SportsBaseball sx={{ fontSize: 18 }} />;
      case 'ATP': case 'WTA': return <SportsTennis sx={{ fontSize: 18 }} />;
      default: return <SportsScore sx={{ fontSize: 18 }} />;
    }
  };

  const getTierColor = (tier: string): 'success' | 'primary' | 'warning' | 'default' => {
    switch (tier) { case 'A': return 'success'; case 'B': return 'primary'; case 'C': return 'warning'; default: return 'default'; }
  };

  const formatPnl = (v: number) => (v >= 0 ? `+$${v.toFixed(0)}` : `-$${Math.abs(v).toFixed(0)}`);
  const pnlColor = (v: number) => v > 0 ? 'success.main' : v < 0 ? 'error.main' : 'text.secondary';

  const formatGameTime = (iso: string) => {
    if (!iso) return '';
    try {
      const d = new Date(iso);
      return formatTime(d, timezone, timeFormat);
    } catch { return ''; }
  };

  if (loading && !data) return <LinearProgress />;
  const d = data || {} as DashboardData;

  // Compute max for sport bar widths
  const maxSportCount = Math.max(...(d.sport_breakdown || []).map((s: any) => s.count), 1);

  // Card component for consistent styling
  const StatCard = ({ label, value, sub, color, icon, onClick }: {
    label: string; value: string | number; sub?: string; color: string;
    icon?: React.ReactNode; onClick?: () => void;
  }) => (
    <Card sx={{
      cursor: onClick ? 'pointer' : 'default', height: '100%',
      '&:hover': onClick ? { bgcolor: isDark ? 'rgba(255,255,255,0.04)' : 'grey.50' } : {},
    }} onClick={onClick}>
      <Box sx={{ py: 1.5, px: 2 }}>
        <Box display="flex" alignItems="center" justifyContent="space-between">
          <Typography sx={{ fontSize: 11, color: 'text.secondary', textTransform: 'uppercase', letterSpacing: 0.5, fontWeight: 500 }}>
            {label}
          </Typography>
          {icon && <Box sx={{ color: 'text.secondary', opacity: 0.5 }}>{icon}</Box>}
        </Box>
        <Typography sx={{ fontSize: 24, fontWeight: 700, color, lineHeight: 1.3, mt: 0.25 }}>
          {value}
        </Typography>
        {sub && <Typography sx={{ fontSize: 11, color: 'text.secondary', mt: 0.25 }}>{sub}</Typography>}
      </Box>
    </Card>
  );

  return (
    <Box>
      {/* Header */}
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
        <Box>
          <Typography variant="h5" fontWeight={700} sx={{ fontSize: 20 }}>Dashboard</Typography>
          <Typography variant="body2" color="text.secondary" sx={{ fontSize: 12 }}>
            Updated {formatTime(lastUpdate, timezone, timeFormat)} {getTimezoneAbbr(timezone)}
          </Typography>
        </Box>
        <Button variant="outlined" size="small" startIcon={<Refresh sx={{ fontSize: 14 }} />}
          onClick={() => fetchDashboard(true)} sx={{ fontSize: 12 }}>Refresh</Button>
      </Box>

      {/* Row 1: 8 KPI Stat Cards */}
      <Grid container spacing={1.5} mb={2}>
        <Grid item xs={6} sm={3} md={1.5}>
          <StatCard label="Today's Picks" value={d.total_predictions || 0} color="primary.main"
            icon={<Casino sx={{ fontSize: 16 }} />} onClick={() => navigate('/predictions')} />
        </Grid>
        <Grid item xs={6} sm={3} md={1.5}>
          <StatCard label="Tier A" value={d.tier_a_count || 0} color="success.main"
            sub={d.total_predictions ? `${((d.tier_a_count || 0) / d.total_predictions * 100).toFixed(0)}% of total` : ''}
            icon={<EmojiEvents sx={{ fontSize: 16 }} />} onClick={() => navigate('/predictions')} />
        </Grid>
        <Grid item xs={6} sm={3} md={1.5}>
          <StatCard label="Win Rate" value={d.win_rate ? `${d.win_rate}%` : '\u2014'}
            color={d.win_rate >= 55 ? 'success.main' : d.win_rate >= 50 ? 'warning.main' : d.win_rate > 0 ? 'error.main' : 'text.secondary'}
            sub={d.graded_total > 0 ? `${d.wins}W - ${d.losses}L` : 'No graded yet'}
            icon={<TrendingUp sx={{ fontSize: 16 }} />} />
        </Grid>
        <Grid item xs={6} sm={3} md={1.5}>
          <StatCard label="ROI" value={d.roi ? `${d.roi > 0 ? '+' : ''}${d.roi}%` : '\u2014'}
            color={d.roi > 0 ? 'success.main' : d.roi < 0 ? 'error.main' : 'text.secondary'}
            icon={<TrendingUp sx={{ fontSize: 16 }} />} onClick={() => navigate('/analytics')} />
        </Grid>
        <Grid item xs={6} sm={3} md={1.5}>
          <StatCard label="P&L" value={d.graded_total > 0 ? formatPnl(d.total_pnl || 0) : '\u2014'}
            color={pnlColor(d.total_pnl || 0)}
            icon={d.total_pnl >= 0 ? <TrendingUp sx={{ fontSize: 16 }} /> : <TrendingDown sx={{ fontSize: 16 }} />}
            onClick={() => navigate('/betting')} />
        </Grid>
        <Grid item xs={6} sm={3} md={1.5}>
          <StatCard label="Avg CLV" value={d.avg_clv ? `${d.avg_clv > 0 ? '+' : ''}${d.avg_clv}\u00A2` : '\u2014'}
            color={d.avg_clv > 0 ? 'success.main' : d.avg_clv < 0 ? 'error.main' : 'text.secondary'}
            sub="Closing line value" />
        </Grid>
        <Grid item xs={6} sm={3} md={1.5}>
          <StatCard label="Pending" value={d.pending_count || 0} color="warning.main"
            sub={d.graded_today > 0 ? `${d.graded_today} graded today` : ''}
            icon={<Schedule sx={{ fontSize: 16 }} />} />
        </Grid>
        <Grid item xs={6} sm={3} md={1.5}>
          <StatCard label="Models" value={d.active_models_count || 0} color="info.main"
            sub={`${d.upcoming_games_count || 0} upcoming games`}
            icon={<Psychology sx={{ fontSize: 16 }} />} onClick={() => navigate('/models')} />
        </Grid>
      </Grid>

      {/* Row 2: Top Picks + Sport Breakdown */}
      <Grid container spacing={2} mb={2}>
        {/* Today's Top Picks */}
        <Grid item xs={12} md={7}>
          <Card sx={{ height: '100%' }}>
            <Box sx={{ px: 2, py: 1.25, borderBottom: 1, borderColor: 'divider', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <Box display="flex" alignItems="center" gap={1}>
                <EmojiEvents sx={{ fontSize: 20, color: 'warning.main' }} />
                <Typography sx={{ fontSize: 15, fontWeight: 600 }}>Today's Top Picks</Typography>
                {d.top_picks?.length > 0 && (
                  <Chip label={d.top_picks.length} size="small" sx={{ fontSize: 10, height: 18 }} />
                )}
              </Box>
              <Chip label="View All" size="small" variant="outlined"
                sx={{ fontSize: 11, height: 24, cursor: 'pointer' }} onClick={() => navigate('/predictions')} />
            </Box>
            <Box sx={{ p: 0 }}>
              {(d.top_picks || []).length > 0 ? d.top_picks.map((pred: any, idx: number) => (
                <Box key={idx} sx={{
                  display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                  px: 2, py: 1.25,
                  borderBottom: idx < d.top_picks.length - 1 ? 1 : 0, borderColor: 'divider',
                  '&:hover': { bgcolor: isDark ? 'rgba(255,255,255,0.02)' : 'rgba(0,0,0,0.01)' },
                }}>
                  <Box display="flex" alignItems="center" gap={1.5} flex={1} minWidth={0}>
                    <Box sx={{ color: 'text.secondary' }}>{getSportIcon(pred.sport)}</Box>
                    <Box flex={1} minWidth={0}>
                      <Typography sx={{ fontSize: 14, fontWeight: 600, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                        {pred.pick}
                      </Typography>
                      <Typography sx={{ fontSize: 11, color: 'text.secondary', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                        {pred.game}
                      </Typography>
                    </Box>
                  </Box>
                  <Box display="flex" alignItems="center" gap={1} ml={1}>
                    <Tooltip title={`${(pred.probability * 100).toFixed(1)}% probability`} arrow>
                      <Box sx={{ width: 55, display: 'flex', alignItems: 'center', gap: 0.5 }}>
                        <LinearProgress variant="determinate" value={pred.probability * 100}
                          color={pred.probability >= 0.65 ? 'success' : pred.probability >= 0.55 ? 'primary' : 'warning'}
                          sx={{ flex: 1, height: 4, borderRadius: 2 }} />
                        <Typography sx={{ fontSize: 10, color: 'text.secondary', minWidth: 28, textAlign: 'right' }}>
                          {(pred.probability * 100).toFixed(0)}%
                        </Typography>
                      </Box>
                    </Tooltip>
                    {pred.edge > 0 && (
                      <Tooltip title={`${pred.edge.toFixed(1)}% edge over market`} arrow>
                        <Typography sx={{ fontSize: 10, color: 'success.main', fontWeight: 600, minWidth: 30, textAlign: 'right' }}>
                          +{pred.edge.toFixed(1)}
                        </Typography>
                      </Tooltip>
                    )}
                    <Typography sx={{ fontSize: 11, color: 'text.secondary', minWidth: 55, textAlign: 'right' }}>
                      {formatGameTime(pred.time)}
                    </Typography>
                    <Chip label={pred.tier} size="small" color={getTierColor(pred.tier)}
                      sx={{ fontSize: 10, height: 20, minWidth: 24, fontWeight: 700, '& .MuiChip-label': { px: 0.5 } }} />
                  </Box>
                </Box>
              )) : (
                <Box sx={{ textAlign: 'center', py: 5, color: 'text.secondary' }}>
                  <Schedule sx={{ fontSize: 36, mb: 1, opacity: 0.3 }} />
                  <Typography sx={{ fontSize: 13 }}>No upcoming predictions</Typography>
                  <Typography sx={{ fontSize: 11, mt: 0.5 }}>Picks appear once models generate predictions</Typography>
                </Box>
              )}
            </Box>
          </Card>
        </Grid>

        {/* Right Column: Sport Breakdown + Distribution */}
        <Grid item xs={12} md={5}>
          <Box display="flex" flexDirection="column" gap={2} height="100%">
            {/* Sport Breakdown */}
            <Card sx={{ flex: 1 }}>
              <Box sx={{ px: 2, py: 1.25, borderBottom: 1, borderColor: 'divider' }}>
                <Typography sx={{ fontSize: 15, fontWeight: 600 }}>Picks by Sport</Typography>
              </Box>
              <Box sx={{ px: 2, py: 1.5 }}>
                {(d.sport_breakdown || []).length > 0 ? d.sport_breakdown.map((s: any, idx: number) => (
                  <Box key={idx} sx={{ display: 'flex', alignItems: 'center', gap: 1.5, mb: idx < d.sport_breakdown.length - 1 ? 1 : 0 }}>
                    <Box sx={{ color: 'text.secondary', width: 20, display: 'flex', justifyContent: 'center' }}>
                      {getSportIcon(s.sport)}
                    </Box>
                    <Typography sx={{ fontSize: 12, fontWeight: 500, width: 50 }}>{s.sport}</Typography>
                    <Box flex={1}>
                      <LinearProgress variant="determinate" value={(s.count / maxSportCount) * 100}
                        sx={{ height: 8, borderRadius: 4, bgcolor: isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)',
                          '& .MuiLinearProgress-bar': { borderRadius: 4 } }} />
                    </Box>
                    <Typography sx={{ fontSize: 12, fontWeight: 600, minWidth: 28, textAlign: 'right' }}>{s.count}</Typography>
                    {s.tier_a > 0 && (
                      <Chip label={`${s.tier_a}A`} size="small" color="success"
                        sx={{ fontSize: 9, height: 16, '& .MuiChip-label': { px: 0.4 } }} />
                    )}
                  </Box>
                )) : (
                  <Typography sx={{ fontSize: 12, color: 'text.secondary', textAlign: 'center', py: 2 }}>
                    No sport data yet
                  </Typography>
                )}
              </Box>
            </Card>

            {/* Tier + Bet Type Distribution */}
            <Card>
              <Box sx={{ px: 2, py: 1.25, borderBottom: 1, borderColor: 'divider' }}>
                <Typography sx={{ fontSize: 15, fontWeight: 600 }}>Distribution</Typography>
              </Box>
              <Box sx={{ px: 2, py: 1.5 }}>
                <Grid container spacing={2}>
                  <Grid item xs={6}>
                    <Typography sx={{ fontSize: 11, color: 'text.secondary', mb: 0.75, fontWeight: 500 }}>BY TIER</Typography>
                    {(d.tier_breakdown || []).length > 0 ? d.tier_breakdown.map((t: any, idx: number) => (
                      <Box key={idx} display="flex" alignItems="center" justifyContent="space-between" mb={0.5}>
                        <Chip label={`Tier ${t.tier}`} size="small" color={getTierColor(t.tier)}
                          sx={{ fontSize: 10, height: 20, fontWeight: 600 }} />
                        <Typography sx={{ fontSize: 12, fontWeight: 600 }}>{t.count}</Typography>
                      </Box>
                    )) : <Typography sx={{ fontSize: 11, color: 'text.secondary' }}>{'\u2014'}</Typography>}
                  </Grid>
                  <Grid item xs={6}>
                    <Typography sx={{ fontSize: 11, color: 'text.secondary', mb: 0.75, fontWeight: 500 }}>BY TYPE</Typography>
                    {(d.bet_type_breakdown || []).length > 0 ? d.bet_type_breakdown.map((b: any, idx: number) => (
                      <Box key={idx} display="flex" alignItems="center" justifyContent="space-between" mb={0.5}>
                        <Typography sx={{ fontSize: 12, color: 'text.secondary' }}>
                          {b.type === 'moneyline' ? 'ML' : b.type === 'spread' ? 'Spread' : b.type === 'total' ? 'Total' : b.type}
                        </Typography>
                        <Typography sx={{ fontSize: 12, fontWeight: 600 }}>{b.count}</Typography>
                      </Box>
                    )) : <Typography sx={{ fontSize: 11, color: 'text.secondary' }}>{'\u2014'}</Typography>}
                  </Grid>
                </Grid>
              </Box>
            </Card>
          </Box>
        </Grid>
      </Grid>

      {/* Row 3: Recent Activity + Performance */}
      <Grid container spacing={2} mb={2}>
        <Grid item xs={12} md={7}>
          <Card sx={{ height: '100%' }}>
            <Box sx={{ px: 2, py: 1.25, borderBottom: 1, borderColor: 'divider', display: 'flex', alignItems: 'center', gap: 1 }}>
              <Schedule sx={{ fontSize: 20, color: 'info.main' }} />
              <Typography sx={{ fontSize: 15, fontWeight: 600 }}>Recent Activity</Typography>
              {d.graded_total > 0 && (
                <Chip label={`${d.graded_total} graded`} size="small" sx={{ fontSize: 10, height: 18, ml: 'auto' }} />
              )}
            </Box>
            <Box sx={{ px: 0, maxHeight: 320, overflow: 'auto' }}>
              {(d.recent_activity || []).length > 0 ? d.recent_activity.map((a: any, idx: number) => (
                <Box key={idx} sx={{
                  display: 'flex', alignItems: 'center', gap: 1.5,
                  px: 2, py: 1,
                  borderBottom: idx < d.recent_activity.length - 1 ? 1 : 0, borderColor: 'divider',
                  '&:hover': { bgcolor: isDark ? 'rgba(255,255,255,0.02)' : 'rgba(0,0,0,0.01)' },
                }}>
                  {a.result === 'win' ? (
                    <CheckCircle sx={{ fontSize: 18, color: 'success.main' }} />
                  ) : a.result === 'loss' ? (
                    <ErrorIcon sx={{ fontSize: 18, color: 'error.main' }} />
                  ) : (
                    <Schedule sx={{ fontSize: 18, color: 'warning.main' }} />
                  )}
                  <Typography sx={{ fontSize: 13, flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {a.text}
                  </Typography>
                  <Typography sx={{ fontSize: 11, fontWeight: 600, color: pnlColor(a.pnl || 0), minWidth: 45, textAlign: 'right' }}>
                    {a.pnl ? formatPnl(a.pnl) : ''}
                  </Typography>
                  <Chip label={a.tier} size="small" color={getTierColor(a.tier)}
                    sx={{ fontSize: 9, height: 18, minWidth: 22, '& .MuiChip-label': { px: 0.4 } }} />
                  <Typography sx={{ fontSize: 10, color: 'text.secondary', minWidth: 45, textAlign: 'right' }}>
                    {a.time}
                  </Typography>
                </Box>
              )) : (
                <Box sx={{ textAlign: 'center', py: 5, color: 'text.secondary' }}>
                  <Typography sx={{ fontSize: 13 }}>No graded predictions yet</Typography>
                  <Typography sx={{ fontSize: 11, mt: 0.5 }}>Results appear here once games are graded</Typography>
                </Box>
              )}
            </Box>
          </Card>
        </Grid>

        <Grid item xs={12} md={5}>
          <Box display="flex" flexDirection="column" gap={2} height="100%">
            {/* Best Performers */}
            <Card sx={{ flex: 1 }}>
              <Box sx={{ px: 2, py: 1.25, borderBottom: 1, borderColor: 'divider', display: 'flex', alignItems: 'center', gap: 1 }}>
                <Typography sx={{ fontSize: 18 }}>{'\uD83D\uDD25'}</Typography>
                <Typography sx={{ fontSize: 15, fontWeight: 600 }}>Best Performers</Typography>
              </Box>
              <Box sx={{ px: 2, py: 1.5 }}>
                {(d.best_performers || []).length > 0 ? d.best_performers.map((item: any, idx: number) => (
                  <Box key={idx} sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 0.75 }}>
                    <Typography sx={{ fontSize: 13 }}>{item.label}</Typography>
                    <Chip label={item.value} size="small" color={item.color as any}
                      sx={{ fontSize: 11, height: 22, fontWeight: 600 }} />
                  </Box>
                )) : (
                  <Typography sx={{ fontSize: 12, color: 'text.secondary', textAlign: 'center', py: 2 }}>
                    {d.graded_total > 0 ? 'Need 3+ graded per category' : 'No graded data yet'}
                  </Typography>
                )}
              </Box>
            </Card>

            {/* Areas to Monitor */}
            <Card sx={{ flex: 1 }}>
              <Box sx={{ px: 2, py: 1.25, borderBottom: 1, borderColor: 'divider', display: 'flex', alignItems: 'center', gap: 1 }}>
                <Warning sx={{ fontSize: 20, color: 'warning.main' }} />
                <Typography sx={{ fontSize: 15, fontWeight: 600 }}>Areas to Monitor</Typography>
              </Box>
              <Box sx={{ px: 2, py: 1.5 }}>
                {(d.areas_to_monitor || []).length > 0 ? d.areas_to_monitor.map((item: any, idx: number) => (
                  <Box key={idx} sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 0.75 }}>
                    <Typography sx={{ fontSize: 13 }}>{item.label}</Typography>
                    <Chip label={item.value} size="small" color={item.color as any}
                      sx={{ fontSize: 11, height: 22, fontWeight: 600 }} />
                  </Box>
                )) : (
                  <Typography sx={{ fontSize: 12, color: 'text.secondary', textAlign: 'center', py: 2 }}>
                    {d.graded_total > 0 ? 'All categories performing well' : 'No data to monitor yet'}
                  </Typography>
                )}
              </Box>
            </Card>
          </Box>
        </Grid>
      </Grid>

      {/* Row 4: Quick Status Bar */}
      <Card>
        <Box sx={{ px: 2.5, py: 1.25, display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 2 }}>
          <Box display="flex" alignItems="center" gap={3}>
            <Box display="flex" alignItems="center" gap={0.75}>
              <Speed sx={{ fontSize: 16, color: 'success.main' }} />
              <Typography sx={{ fontSize: 12, color: 'text.secondary' }}>
                <strong>{d.active_models_count || 0}</strong> active models
              </Typography>
            </Box>
            <Box display="flex" alignItems="center" gap={0.75}>
              <SportsScore sx={{ fontSize: 16, color: 'info.main' }} />
              <Typography sx={{ fontSize: 12, color: 'text.secondary' }}>
                <strong>{d.upcoming_games_count || 0}</strong> upcoming games
              </Typography>
            </Box>
            <Box display="flex" alignItems="center" gap={0.75}>
              <Casino sx={{ fontSize: 16, color: 'primary.main' }} />
              <Typography sx={{ fontSize: 12, color: 'text.secondary' }}>
                <strong>{d.total_predictions || 0}</strong> total predictions
              </Typography>
            </Box>
            {d.graded_total > 0 && (
              <Box display="flex" alignItems="center" gap={0.75}>
                <CheckCircle sx={{ fontSize: 16, color: 'success.main' }} />
                <Typography sx={{ fontSize: 12, color: 'text.secondary' }}>
                  <strong>{d.graded_total}</strong> graded ({d.wins}W-{d.losses}L{d.pushes > 0 ? `-${d.pushes}P` : ''})
                </Typography>
              </Box>
            )}
          </Box>
          <Typography sx={{ fontSize: 11, color: 'text.secondary' }}>
            Auto-refresh every 60s
          </Typography>
        </Box>
      </Card>
    </Box>
  );
};

export default Dashboard;