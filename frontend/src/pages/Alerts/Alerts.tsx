// src/pages/Alerts/Alerts.tsx - System Health Dashboard (Card-based, No Scroll)
import React, { useState, useEffect } from 'react';
import {
  Box, Card, Typography, Grid, Chip, Button, LinearProgress, useTheme, Tooltip
} from '@mui/material';
import { CheckCircle, Warning, Error, Info, Refresh, Storage, Memory, Speed, Cloud, Psychology, Api, DataObject, MonitorHeart, Dns, Security, Timer, TrendingUp, Circle } from '@mui/icons-material';
import { api } from '../../api/client';
import { useAlertStore } from '../../store';

interface HealthIndicator {
  name: string;
  status: 'good' | 'warning' | 'error';
  value: string;
  details?: string;
  icon: string;
}

interface AlertItem {
  id: string;
  type: 'error' | 'warning' | 'info' | 'success';
  message: string;
  timestamp: string;
}

const Alerts: React.FC = () => {
  const theme = useTheme();
  const isDark = theme.palette.mode === 'dark';
  const [indicators, setIndicators] = useState<HealthIndicator[]>([]);
  const [alerts, setAlerts] = useState<AlertItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [overallHealth] = useState(85);
  const { markAllRead } = useAlertStore();

  useEffect(() => {
    loadHealth();
    const interval = setInterval(loadHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  const loadHealth = async () => {
    setLoading(true);
    try { await api.getHealth(); } catch { /* ignore */ }
    setIndicators(generateDemoIndicators());
    setAlerts(generateDemoAlerts());
    setLoading(false);
  };

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
      case 'error': return <Error sx={{ fontSize: 18, color: 'error.main' }} />;
      case 'warning': return <Warning sx={{ fontSize: 18, color: 'warning.main' }} />;
      case 'success': return <CheckCircle sx={{ fontSize: 18, color: 'success.main' }} />;
      default: return <Info sx={{ fontSize: 18, color: 'info.main' }} />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'good': return 'success.main';
      case 'warning': return 'warning.main';
      case 'error': return 'error.main';
      default: return 'text.secondary';
    }
  };

  const goodCount = indicators.filter(i => i.status === 'good').length;
  const warningCount = indicators.filter(i => i.status === 'warning').length;
  const errorCount = indicators.filter(i => i.status === 'error').length;

  // Quick stats data
  const quickStats = [
    { label: 'Uptime', value: '99.9%', sub: '30 days', color: 'success.main' },
    { label: 'API Calls', value: '12.4K', sub: 'Today', color: 'text.primary' },
    { label: 'Latency', value: '45ms', sub: 'p95: 89ms', color: 'success.main' },
    { label: 'CPU', value: '34%', sub: '24 cores', color: 'text.primary' },
    { label: 'Memory', value: '82%', sub: '420/512 GB', color: 'warning.main' },
    { label: 'GPU', value: '28%', sub: 'RTX 6000', color: 'success.main' },
  ];

  return (
    <Box>
      {/* Header */}
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
        <Typography variant="h5" fontWeight={700} sx={{ fontSize: 20 }}>System Health</Typography>
        <Box display="flex" gap={1.5}>
          <Button variant="outlined" size="small" onClick={markAllRead} sx={{ fontSize: 12, py: 0.75 }}>Mark All Read</Button>
          <Button variant="outlined" size="small" startIcon={<Refresh sx={{ fontSize: 16 }} />} onClick={loadHealth} sx={{ fontSize: 12, py: 0.75 }}>Refresh</Button>
        </Box>
      </Box>

      {loading && <LinearProgress sx={{ mb: 1.5, height: 3 }} />}

      {/* Overall Health Bar - Compact */}
      <Card sx={{ mb: 2 }}>
        <Box sx={{ px: 2.5, py: 1.5 }}>
          <Box display="flex" alignItems="center" justifyContent="space-between" mb={0.75}>
            <Box display="flex" alignItems="center" gap={1.5}>
              <Circle sx={{ fontSize: 12, color: overallHealth >= 80 ? 'success.main' : overallHealth >= 60 ? 'warning.main' : 'error.main' }} />
              <Typography sx={{ fontSize: 15, fontWeight: 600 }}>Overall System Health</Typography>
            </Box>
            <Box display="flex" alignItems="center" gap={2}>
              <Box display="flex" alignItems="center" gap={0.5}>
                <CheckCircle sx={{ fontSize: 14, color: 'success.main' }} />
                <Typography sx={{ fontSize: 12, color: 'success.main' }}>{goodCount}</Typography>
              </Box>
              <Box display="flex" alignItems="center" gap={0.5}>
                <Warning sx={{ fontSize: 14, color: 'warning.main' }} />
                <Typography sx={{ fontSize: 12, color: 'warning.main' }}>{warningCount}</Typography>
              </Box>
              <Box display="flex" alignItems="center" gap={0.5}>
                <Error sx={{ fontSize: 14, color: 'error.main' }} />
                <Typography sx={{ fontSize: 12, color: 'error.main' }}>{errorCount}</Typography>
              </Box>
              <Typography sx={{ fontSize: 18, fontWeight: 700, color: overallHealth >= 80 ? 'success.main' : overallHealth >= 60 ? 'warning.main' : 'error.main', ml: 1 }}>
                {overallHealth}%
              </Typography>
            </Box>
          </Box>
          <LinearProgress 
            variant="determinate" 
            value={overallHealth} 
            sx={{ 
              height: 6, 
              borderRadius: 3, 
              bgcolor: isDark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.08)',
              '& .MuiLinearProgress-bar': { 
                bgcolor: overallHealth >= 80 ? 'success.main' : overallHealth >= 60 ? 'warning.main' : 'error.main',
                borderRadius: 3
              }
            }} 
          />
        </Box>
      </Card>

      {/* Quick Stats Row */}
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

      <Grid container spacing={2}>
        {/* System Components - Card Grid */}
        <Grid item xs={12} md={7}>
          <Card sx={{ height: '100%' }}>
            <Box sx={{ px: 2, py: 1.25, borderBottom: 1, borderColor: 'divider' }}>
              <Typography sx={{ fontSize: 16, fontWeight: 600 }}>System Components</Typography>
            </Box>
            <Box sx={{ p: 1.5 }}>
              <Grid container spacing={1}>
                {indicators.map((ind, idx) => (
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
                          {ind.status === 'error' && <Error sx={{ fontSize: 16, color: 'error.main' }} />}
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
            </Box>
          </Card>
        </Grid>

        {/* Recent Alerts - Compact List */}
        <Grid item xs={12} md={5}>
          <Card sx={{ height: '100%' }}>
            <Box sx={{ px: 2, py: 1.25, borderBottom: 1, borderColor: 'divider' }}>
              <Typography sx={{ fontSize: 16, fontWeight: 600 }}>Recent Alerts ({alerts.length})</Typography>
            </Box>
            <Box sx={{ p: 1.5 }}>
              {alerts.map((alert) => (
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
              ))}
            </Box>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

const generateDemoIndicators = (): HealthIndicator[] => [
  { name: 'PostgreSQL', status: 'good', value: 'Connected', details: 'Latency: 2ms', icon: 'database' },
  { name: 'Redis', status: 'good', value: '94% Hit', details: 'Memory: 2.1GB', icon: 'redis' },
  { name: 'API Server', status: 'good', value: 'Running', details: '4 workers', icon: 'server' },
  { name: 'Latency', status: 'good', value: '45ms', details: 'p95 OK', icon: 'speed' },
  { name: 'OddsAPI', status: 'good', value: '450/500', details: 'Resets 23h', icon: 'cloud' },
  { name: 'Pinnacle', status: 'good', value: '890/1K', details: 'CLV active', icon: 'cloud' },
  { name: 'ESPN', status: 'error', value: 'Limited', details: 'Retry 15m', icon: 'api' },
  { name: 'Accuracy', status: 'good', value: '64.2%', details: '>60% target', icon: 'model' },
  { name: 'CLV Avg', status: 'good', value: '+1.8%', details: 'Positive', icon: 'trend' },
  { name: 'Disk', status: 'warning', value: '78%', details: '1.56/2TB', icon: 'disk' },
  { name: 'Memory', status: 'warning', value: '82%', details: '420/512GB', icon: 'disk' },
  { name: 'Worker', status: 'good', value: 'Active', details: '12 queued', icon: 'timer' },
  { name: 'SSL', status: 'good', value: 'Valid', details: '89 days', icon: 'security' },
  { name: 'GPU', status: 'good', value: 'Ready', details: 'RTX 6000', icon: 'server' },
  { name: 'Scheduler', status: 'good', value: 'Running', details: '5 jobs', icon: 'timer' },
  { name: 'Backup', status: 'good', value: 'OK', details: '4h ago', icon: 'database' },
];

const generateDemoAlerts = (): AlertItem[] => [
  { id: '1', type: 'error', message: 'ESPN API rate limit (429)', timestamp: '2m' },
  { id: '2', type: 'warning', message: 'Memory usage > 80%', timestamp: '15m' },
  { id: '3', type: 'success', message: 'Tier A: Lakers -3.5 @ 67.2%', timestamp: '1h' },
  { id: '4', type: 'success', message: 'Model trained (AUC: 0.68)', timestamp: '2h' },
  { id: '5', type: 'info', message: 'Daily backup done', timestamp: '4h' },
  { id: '6', type: 'success', message: 'Tier B: Celtics +2 @ 62.1%', timestamp: '5h' },
  { id: '7', type: 'info', message: 'Odds refresh complete', timestamp: '6h' },
  { id: '8', type: 'warning', message: 'Disk approaching 80%', timestamp: '8h' },
  { id: '9', type: 'success', message: 'Tier A: Warriors -5 @ 66.8%', timestamp: '10h' },
  { id: '10', type: 'info', message: 'System restart complete', timestamp: '12h' },
];

export default Alerts;
