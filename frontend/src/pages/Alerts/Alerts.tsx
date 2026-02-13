// src/pages/Alerts/Alerts.tsx - System Health Dashboard (Real Data)
import React, { useState, useEffect, useCallback } from 'react';
import {
  Box, Card, Typography, Grid, Chip, Button, LinearProgress, useTheme, Tooltip
} from '@mui/material';
import {
  CheckCircle, Warning, Error as ErrorIcon, Info, Refresh, Storage, Memory, Speed,
  Cloud, Psychology, Api, DataObject, MonitorHeart, Dns, Security, Timer, TrendingUp, Circle
} from '@mui/icons-material';
import { api } from '../../api/client';
import { useAlertStore } from '../../store';

interface HealthComponent {
  name: string;
  icon: string;
  status: 'good' | 'warning' | 'error';
  value: string;
  details?: string;
}

interface AlertItem {
  id: string;
  type: 'error' | 'warning' | 'info' | 'success';
  message: string;
  timestamp: string;
}

interface SystemHealthResponse {
  health_score: number;
  components: HealthComponent[];
  alerts: AlertItem[];
  quick_stats: {
    uptime: string;
    cpu_percent: number;
    cpu_cores: number;
    memory_percent: number;
    memory_used_gb: number;
    memory_total_gb: number;
    disk_percent: number;
    disk_used_gb: number;
    disk_total_gb: number;
  };
  counts: {
    good: number;
    warning: number;
    error: number;
    total: number;
  };
  updated_at: string;
}

const Alerts: React.FC = () => {
  const theme = useTheme();
  const isDark = theme.palette.mode === 'dark';
  const [data, setData] = useState<SystemHealthResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());
  const { markAllRead } = useAlertStore();

  const loadHealth = useCallback(async () => {
    setLoading(true);
    try {
      const resp = await api.getSystemHealth();
      setData(resp);
      if (resp.updated_at) setLastUpdate(new Date(resp.updated_at));
    } catch {
      // Keep existing data on error
    }
    setLoading(false);
  }, []);

  useEffect(() => {
    loadHealth();
    const interval = setInterval(loadHealth, 30000);
    return () => clearInterval(interval);
  }, [loadHealth]);

  const healthScore = data?.health_score ?? 0;
  const components = data?.components ?? [];
  const alerts = data?.alerts ?? [];
  const counts = data?.counts ?? { good: 0, warning: 0, error: 0, total: 0 };
  const qs = data?.quick_stats;

  const getIcon = (iconName: string) => {
    const iconMap: Record<string, React.ReactNode> = {
      database: <Storage sx={{ fontSize: 18 }} />,
      redis: <Memory sx={{ fontSize: 18 }} />,
      speed: <Speed sx={{ fontSize: 18 }} />,
      cloud: <Cloud sx={{ fontSize: 18 }} />,
      model: <Psychology sx={{ fontSize: 18 }} />,
      api: <Api sx={{ fontSize: 18 }} />,
      disk: <DataObject sx={{ fontSize: 18 }} />,
      server: <Dns sx={{ fontSize: 18 }} />,
      security: <Security sx={{ fontSize: 18 }} />,
      timer: <Timer sx={{ fontSize: 18 }} />,
      trend: <TrendingUp sx={{ fontSize: 18 }} />,
    };
    return iconMap[iconName] || <MonitorHeart sx={{ fontSize: 18 }} />;
  };

  const getAlertIcon = (type: string) => {
    switch (type) {
      case 'error': return <ErrorIcon sx={{ fontSize: 18, color: 'error.main' }} />;
      case 'warning': return <Warning sx={{ fontSize: 18, color: 'warning.main' }} />;
      case 'success': return <CheckCircle sx={{ fontSize: 18, color: 'success.main' }} />;
      default: return <Info sx={{ fontSize: 18, color: 'info.main' }} />;
    }
  };

  const quickStats = qs ? [
    { label: 'Uptime', value: qs.uptime, sub: 'since restart', color: 'success.main' },
    { label: 'CPU', value: `${qs.cpu_percent}%`, sub: `${qs.cpu_cores} cores`, color: qs.cpu_percent >= 80 ? 'warning.main' : qs.cpu_percent >= 90 ? 'error.main' : 'success.main' },
    { label: 'Memory', value: `${qs.memory_percent}%`, sub: `${qs.memory_used_gb}/${qs.memory_total_gb} GB`, color: qs.memory_percent >= 80 ? 'warning.main' : qs.memory_percent >= 90 ? 'error.main' : 'text.primary' },
    { label: 'Disk', value: `${qs.disk_percent}%`, sub: `${qs.disk_used_gb}/${qs.disk_total_gb} GB`, color: qs.disk_percent >= 80 ? 'warning.main' : qs.disk_percent >= 90 ? 'error.main' : 'text.primary' },
    { label: 'Components', value: `${counts.total}`, sub: `${counts.good} healthy`, color: counts.error > 0 ? 'error.main' : counts.warning > 0 ? 'warning.main' : 'success.main' },
    { label: 'Health', value: `${healthScore}%`, sub: 'overall', color: healthScore >= 80 ? 'success.main' : healthScore >= 60 ? 'warning.main' : 'error.main' },
  ] : [];

  return (
    <Box>
      {/* Header */}
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
        <Box display="flex" alignItems="center" gap={2}>
          <Typography variant="h5" fontWeight={700} sx={{ fontSize: 20 }}>System Health</Typography>
          <Typography variant="caption" color="text.secondary">
            Updated: {lastUpdate.toLocaleTimeString()}
          </Typography>
        </Box>
        <Box display="flex" gap={1.5}>
          <Button variant="outlined" size="small" onClick={markAllRead} sx={{ fontSize: 12, py: 0.75 }}>Mark All Read</Button>
          <Button variant="outlined" size="small" startIcon={<Refresh sx={{ fontSize: 16 }} />} onClick={loadHealth} sx={{ fontSize: 12, py: 0.75 }}>Refresh</Button>
        </Box>
      </Box>

      {loading && !data && <LinearProgress sx={{ mb: 1.5, height: 3 }} />}

      {/* Overall Health Bar */}
      <Card sx={{ mb: 2 }}>
        <Box sx={{ px: 2.5, py: 1.5 }}>
          <Box display="flex" alignItems="center" justifyContent="space-between" mb={0.75}>
            <Box display="flex" alignItems="center" gap={1.5}>
              <Circle sx={{ fontSize: 12, color: healthScore >= 80 ? 'success.main' : healthScore >= 60 ? 'warning.main' : 'error.main' }} />
              <Typography sx={{ fontSize: 15, fontWeight: 600 }}>Overall System Health</Typography>
            </Box>
            <Box display="flex" alignItems="center" gap={2}>
              <Box display="flex" alignItems="center" gap={0.5}>
                <CheckCircle sx={{ fontSize: 14, color: 'success.main' }} />
                <Typography sx={{ fontSize: 12, color: 'success.main' }}>{counts.good}</Typography>
              </Box>
              <Box display="flex" alignItems="center" gap={0.5}>
                <Warning sx={{ fontSize: 14, color: 'warning.main' }} />
                <Typography sx={{ fontSize: 12, color: 'warning.main' }}>{counts.warning}</Typography>
              </Box>
              <Box display="flex" alignItems="center" gap={0.5}>
                <ErrorIcon sx={{ fontSize: 14, color: 'error.main' }} />
                <Typography sx={{ fontSize: 12, color: 'error.main' }}>{counts.error}</Typography>
              </Box>
              <Typography sx={{ fontSize: 18, fontWeight: 700, color: healthScore >= 80 ? 'success.main' : healthScore >= 60 ? 'warning.main' : 'error.main', ml: 1 }}>
                {healthScore}%
              </Typography>
            </Box>
          </Box>
          <LinearProgress
            variant="determinate"
            value={healthScore}
            sx={{
              height: 6,
              borderRadius: 3,
              bgcolor: isDark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.08)',
              '& .MuiLinearProgress-bar': {
                bgcolor: healthScore >= 80 ? 'success.main' : healthScore >= 60 ? 'warning.main' : 'error.main',
                borderRadius: 3
              }
            }}
          />
        </Box>
      </Card>

      {/* Quick Stats Row */}
      {quickStats.length > 0 && (
        <Grid container spacing={1.5} mb={2}>
          {quickStats.map((stat, idx) => (
            <Grid item xs={4} sm={2} key={idx}>
              <Card sx={{ textAlign: 'center', py: 1.5, px: 1 }}>
                <Typography sx={{ fontSize: 12, color: 'text.secondary' }}>{stat.label}</Typography>
                <Typography sx={{ fontSize: 20, fontWeight: 700, color: stat.color }}>{stat.value}</Typography>
                <Typography sx={{ fontSize: 11, color: 'text.secondary' }}>{stat.sub}</Typography>
              </Card>
            </Grid>
          ))}
        </Grid>
      )}

      <Grid container spacing={2}>
        {/* System Components */}
        <Grid item xs={12} md={7}>
          <Card sx={{ height: '100%' }}>
            <Box sx={{ px: 2, py: 1.25, borderBottom: 1, borderColor: 'divider' }}>
              <Typography sx={{ fontSize: 16, fontWeight: 600 }}>System Components ({components.length})</Typography>
            </Box>
            <Box sx={{ p: 1.5 }}>
              {components.length > 0 ? (
                <Grid container spacing={1}>
                  {components.map((ind, idx) => (
                    <Grid item xs={6} sm={4} md={3} key={idx}>
                      <Tooltip title={ind.details || ''} arrow>
                        <Box sx={{
                          p: 1.25,
                          borderRadius: 2,
                          bgcolor: isDark ? 'rgba(255,255,255,0.03)' : 'rgba(0,0,0,0.02)',
                          border: 1,
                          borderColor: ind.status === 'error' ? 'error.main' : ind.status === 'warning' ? 'warning.main' : 'transparent',
                          '&:hover': { bgcolor: isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.04)' }
                        }}>
                          <Box display="flex" alignItems="center" justifyContent="space-between" mb={0.75}>
                            <Box display="flex" alignItems="center" gap={0.75}>
                              <Box sx={{ color: 'text.secondary' }}>{getIcon(ind.icon)}</Box>
                              <Typography sx={{ fontSize: 12, fontWeight: 500, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: 80 }}>{ind.name}</Typography>
                            </Box>
                            {ind.status === 'good' && <CheckCircle sx={{ fontSize: 16, color: 'success.main' }} />}
                            {ind.status === 'warning' && <Warning sx={{ fontSize: 16, color: 'warning.main' }} />}
                            {ind.status === 'error' && <ErrorIcon sx={{ fontSize: 16, color: 'error.main' }} />}
                          </Box>
                          <Chip
                            label={ind.value}
                            size="small"
                            color={ind.status === 'good' ? 'success' : ind.status === 'warning' ? 'warning' : 'error'}
                            sx={{ fontSize: 10, height: 20, width: '100%', '& .MuiChip-label': { px: 0.75 } }}
                          />
                        </Box>
                      </Tooltip>
                    </Grid>
                  ))}
                </Grid>
              ) : (
                <Box sx={{ textAlign: 'center', py: 4 }}>
                  <Typography color="text.secondary">Loading health data...</Typography>
                </Box>
              )}
            </Box>
          </Card>
        </Grid>

        {/* Recent Alerts */}
        <Grid item xs={12} md={5}>
          <Card sx={{ height: '100%' }}>
            <Box sx={{ px: 2, py: 1.25, borderBottom: 1, borderColor: 'divider' }}>
              <Typography sx={{ fontSize: 16, fontWeight: 600 }}>System Events ({alerts.length})</Typography>
            </Box>
            <Box sx={{ p: 1.5, maxHeight: 500, overflow: 'auto' }}>
              {alerts.length > 0 ? alerts.map((alert) => (
                <Box key={alert.id} sx={{
                  display: 'flex',
                  alignItems: 'flex-start',
                  gap: 1.5,
                  py: 1,
                  px: 1.25,
                  mb: 0.75,
                  borderRadius: 1.5,
                  bgcolor: isDark ? 'rgba(255,255,255,0.02)' : 'rgba(0,0,0,0.01)',
                  '&:hover': { bgcolor: isDark ? 'rgba(255,255,255,0.04)' : 'rgba(0,0,0,0.03)' }
                }}>
                  {getAlertIcon(alert.type)}
                  <Box flex={1} minWidth={0}>
                    <Typography sx={{ fontSize: 13, lineHeight: 1.4, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{alert.message}</Typography>
                  </Box>
                  <Typography sx={{ fontSize: 11, color: 'text.secondary', whiteSpace: 'nowrap' }}>{alert.timestamp}</Typography>
                </Box>
              )) : (
                <Box sx={{ textAlign: 'center', py: 4 }}>
                  <Typography color="text.secondary" sx={{ fontSize: 13 }}>No events to display</Typography>
                </Box>
              )}
            </Box>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Alerts;
