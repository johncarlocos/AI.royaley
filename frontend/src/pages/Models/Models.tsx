// src/pages/Models/Models.tsx - Real data from /models and /models/training endpoints
import React, { useState, useEffect, useCallback, useMemo } from 'react';
import {
  Box, Card, CardContent, Typography, Grid, Table, TableBody, TableCell,
  TableContainer, TableHead, TableRow, Chip, Button, LinearProgress,
  Alert, IconButton, Collapse, FormControl, InputLabel, Select, MenuItem,
  useTheme, Dialog, DialogTitle, DialogContent, DialogActions, TableSortLabel
} from '@mui/material';
import { PlayArrow, CheckCircle, Error as ErrorIcon, HourglassEmpty, ExpandMore, ExpandLess, Refresh, Science } from '@mui/icons-material';
import { api } from '../../api/client';

// ─── Types ───────────────────────────────────────────────────────────
interface Model {
  id: string | number;
  sport_code: string;
  bet_type: string;
  framework: string;
  version: string;
  status: string;
  accuracy: number | null;
  auc: number | null;
  log_loss: number | null;
  created_at: string;
  promoted_at: string | null;
}

interface TrainingRun {
  id: string | number;
  sport_code: string;
  bet_type: string;
  framework: string;
  status: string;
  started_at: string;
  completed_at: string | null;
  duration_seconds: number | null;
  best_model_id: number | null;
  metrics: Record<string, any> | null;
}

type SortField = 'sport_code' | 'bet_type' | 'framework' | 'accuracy' | 'auc' | 'created_at' | 'status';

// ─── Helpers ─────────────────────────────────────────────────────────
const statusColor = (s: string) => {
  switch (s) {
    case 'production': return 'success';
    case 'ready': return 'info';
    case 'training': case 'running': case 'pending': return 'primary';
    case 'deprecated': return 'default';
    case 'failed': return 'error';
    default: return 'default';
  }
};

const frameworkLabel = (f: string) => {
  const labels: Record<string, string> = {
    'meta_ensemble': 'Meta-Ensemble',
    'h2o': 'H2O AutoML',
    'autogluon': 'AutoGluon',
    'sklearn': 'Sklearn Ensemble',
    'deep_learning': 'Deep Learning',
    'quantum': 'Quantum ML',
  };
  return labels[f] || f;
};

const formatDate = (d: string | null) => {
  if (!d) return '-';
  const dt = new Date(d);
  return dt.toLocaleDateString('en-US', { month: 'numeric', day: 'numeric', year: 'numeric', timeZone: 'America/Los_Angeles' });
};

const formatDuration = (secs: number | null) => {
  if (!secs) return '-';
  if (secs < 60) return `${secs}s`;
  if (secs < 3600) return `${Math.floor(secs / 60)}m ${secs % 60}s`;
  return `${Math.floor(secs / 3600)}h ${Math.floor((secs % 3600) / 60)}m`;
};

// ─── Component ───────────────────────────────────────────────────────
const Models: React.FC = () => {
  const theme = useTheme();
  const isDark = theme.palette.mode === 'dark';

  const [models, setModels] = useState<Model[]>([]);
  const [trainingRuns, setTrainingRuns] = useState<TrainingRun[]>([]);
  const [loading, setLoading] = useState(true);
  const [modelsError, setModelsError] = useState<string | null>(null);
  const [trainingError, setTrainingError] = useState<string | null>(null);

  // Filters
  const [sportFilter, setSportFilter] = useState('all');
  const [statusFilter, setStatusFilter] = useState('all');
  const [frameworkFilter, setFrameworkFilter] = useState('all');

  // Sort
  const [sortField, setSortField] = useState<SortField>('accuracy');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');

  // Training dialog
  const [trainDialogOpen, setTrainDialogOpen] = useState(false);
  const [trainSport, setTrainSport] = useState('NBA');
  const [trainBetType, setTrainBetType] = useState('spread');
  const [trainFramework, setTrainFramework] = useState('meta_ensemble');
  const [trainSubmitting, setTrainSubmitting] = useState(false);
  const [trainResult, setTrainResult] = useState<string | null>(null);

  // Expanded training run
  const [expandedRun, setExpandedRun] = useState<string | number | null>(null);

  // ─── Load Data ──────────────────────────────────────────────────
  const loadData = useCallback(async () => {
    setLoading(true);
    setModelsError(null);
    setTrainingError(null);

    // Load models
    try {
      const data = await api.getModels();
      setModels(Array.isArray(data) ? data : []);
    } catch (err: any) {
      console.error('Models load error:', err);
      setModelsError(err?.response?.status === 401 ? 'Authentication required' : 'Failed to load models');
      setModels([]);
    }

    // Load training runs
    try {
      const data = await api.getTrainingRuns({ limit: 50 });
      setTrainingRuns(Array.isArray(data) ? data : []);
    } catch (err: any) {
      console.error('Training runs load error:', err);
      setTrainingError(err?.response?.status === 401 ? 'Authentication required' : 'Failed to load training runs');
      setTrainingRuns([]);
    }

    setLoading(false);
  }, []);

  useEffect(() => { loadData(); }, [loadData]);

  // Auto-refresh if training is running
  useEffect(() => {
    const hasRunning = trainingRuns.some(r => r.status === 'running' || r.status === 'pending');
    if (!hasRunning) return;
    const interval = setInterval(loadData, 15000);
    return () => clearInterval(interval);
  }, [trainingRuns, loadData]);

  // ─── Derived data ─────────────────────────────────────────────
  const availableSports = useMemo(() =>
    Array.from(new Set(models.map(m => m.sport_code).filter(Boolean))).sort() as string[],
    [models]
  );

  const availableFrameworks = useMemo(() =>
    Array.from(new Set(models.map(m => m.framework).filter(Boolean))).sort() as string[],
    [models]
  );

  const productionModels = useMemo(() => models.filter(m => m.status === 'production'), [models]);
  const readyModels = useMemo(() => models.filter(m => m.status === 'ready'), [models]);

  // Stats
  const stats = useMemo(() => {
    const prod = productionModels;
    const accuracies = prod.filter(m => m.accuracy != null).map(m => m.accuracy as number);
    const aucs = prod.filter(m => m.auc != null).map(m => m.auc as number);
    return {
      total: models.length,
      production: prod.length,
      ready: readyModels.length,
      training: trainingRuns.filter(r => r.status === 'running' || r.status === 'pending').length,
      avgAccuracy: accuracies.length > 0 ? accuracies.reduce((s, v) => s + v, 0) / accuracies.length : 0,
      avgAuc: aucs.length > 0 ? aucs.reduce((s, v) => s + v, 0) / aucs.length : 0,
      sports: new Set(prod.map(m => m.sport_code)).size,
    };
  }, [models, productionModels, readyModels, trainingRuns]);

  // Filtered & sorted models
  const filtered = useMemo(() => models.filter(m => {
    if (sportFilter !== 'all' && m.sport_code !== sportFilter) return false;
    if (statusFilter !== 'all' && m.status !== statusFilter) return false;
    if (frameworkFilter !== 'all' && m.framework !== frameworkFilter) return false;
    return true;
  }), [models, sportFilter, statusFilter, frameworkFilter]);

  const sorted = useMemo(() => [...filtered].sort((a, b) => {
    let aV: string | number, bV: string | number;
    switch (sortField) {
      case 'sport_code': aV = a.sport_code || ''; bV = b.sport_code || ''; break;
      case 'bet_type': aV = a.bet_type || ''; bV = b.bet_type || ''; break;
      case 'framework': aV = a.framework || ''; bV = b.framework || ''; break;
      case 'accuracy': aV = a.accuracy || 0; bV = b.accuracy || 0; break;
      case 'auc': aV = a.auc || 0; bV = b.auc || 0; break;
      case 'created_at': aV = a.created_at || ''; bV = b.created_at || ''; break;
      case 'status': aV = a.status || ''; bV = b.status || ''; break;
      default: aV = a.accuracy || 0; bV = b.accuracy || 0;
    }
    if (typeof aV === 'string') return sortOrder === 'asc' ? aV.localeCompare(bV as string) : (bV as string).localeCompare(aV);
    return sortOrder === 'asc' ? (aV as number) - (bV as number) : (bV as number) - (aV as number);
  }), [filtered, sortField, sortOrder]);

  const handleSort = (field: SortField) => {
    if (sortField === field) setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
    else { setSortField(field); setSortOrder('desc'); }
  };

  // ─── Training ─────────────────────────────────────────────────
  const handleTrain = async () => {
    setTrainSubmitting(true);
    setTrainResult(null);
    try {
      await api.trainModel({ sport_code: trainSport, bet_type: trainBetType, framework: trainFramework });
      setTrainResult('Training started successfully');
      setTimeout(() => { setTrainDialogOpen(false); setTrainResult(null); loadData(); }, 1500);
    } catch (err: any) {
      const msg = err?.response?.data?.detail || 'Failed to start training';
      setTrainResult(`Error: ${msg}`);
    }
    setTrainSubmitting(false);
  };

  // Active/recent training runs
  const activeRuns = useMemo(() => trainingRuns.filter(r => r.status === 'running' || r.status === 'pending'), [trainingRuns]);
  const recentRuns = useMemo(() => trainingRuns.filter(r => r.status !== 'running' && r.status !== 'pending').slice(0, 10), [trainingRuns]);

  const hdr = { fontWeight: 600, fontSize: 11, py: 0.75, bgcolor: isDark ? 'grey.900' : 'grey.100', color: isDark ? 'grey.100' : 'grey.800', whiteSpace: 'nowrap', borderBottom: 1, borderColor: 'divider' };
  const csx = { height: '100%' };
  const ccsx = { textAlign: 'center', py: 1.5, '&:last-child': { pb: 1.5 }, height: '100%', display: 'flex', flexDirection: 'column', justifyContent: 'center' };

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
        <Typography variant="h5" fontWeight={700} sx={{ fontSize: 20 }}>ML Models</Typography>
        <Box display="flex" gap={1}>
          <Button variant="contained" size="small" startIcon={<Science sx={{ fontSize: 14 }} />} onClick={() => setTrainDialogOpen(true)} sx={{ fontSize: 11 }}>
            Train New Model
          </Button>
          <Button variant="outlined" size="small" startIcon={<Refresh sx={{ fontSize: 14 }} />} onClick={loadData} sx={{ fontSize: 11 }}>Refresh</Button>
        </Box>
      </Box>

      {loading && <LinearProgress sx={{ mb: 2 }} />}

      {/* ─── Stat Cards ─────────────────────────────────────────── */}
      <Grid container spacing={2} mb={2} sx={{ alignItems: 'stretch' }}>
        <Grid item xs={6} sm={4} md={2}>
          <Card sx={csx}><CardContent sx={ccsx}>
            <Typography sx={{ fontSize: 10, color: 'text.secondary' }}>Total Models</Typography>
            <Typography sx={{ fontSize: 18, fontWeight: 700 }}>{stats.total}</Typography>
          </CardContent></Card>
        </Grid>
        <Grid item xs={6} sm={4} md={2}>
          <Card sx={csx}><CardContent sx={ccsx}>
            <Typography sx={{ fontSize: 10, color: 'text.secondary' }}>Production</Typography>
            <Typography sx={{ fontSize: 18, fontWeight: 700, color: 'success.main' }}>{stats.production}</Typography>
            <Typography sx={{ fontSize: 9, color: 'text.secondary' }}>{stats.sports} sports</Typography>
          </CardContent></Card>
        </Grid>
        <Grid item xs={6} sm={4} md={2}>
          <Card sx={csx}><CardContent sx={ccsx}>
            <Typography sx={{ fontSize: 10, color: 'text.secondary' }}>Ready</Typography>
            <Typography sx={{ fontSize: 18, fontWeight: 700, color: 'info.main' }}>{stats.ready}</Typography>
          </CardContent></Card>
        </Grid>
        <Grid item xs={6} sm={4} md={2}>
          <Card sx={csx}><CardContent sx={ccsx}>
            <Typography sx={{ fontSize: 10, color: 'text.secondary' }}>Training</Typography>
            <Typography sx={{ fontSize: 18, fontWeight: 700, color: stats.training > 0 ? 'primary.main' : 'text.primary' }}>{stats.training}</Typography>
          </CardContent></Card>
        </Grid>
        <Grid item xs={6} sm={4} md={2}>
          <Card sx={csx}><CardContent sx={ccsx}>
            <Typography sx={{ fontSize: 10, color: 'text.secondary' }}>Avg Accuracy</Typography>
            <Typography sx={{ fontSize: 18, fontWeight: 700, color: stats.avgAccuracy >= 0.60 ? 'success.main' : 'warning.main' }}>
              {stats.avgAccuracy > 0 ? `${(stats.avgAccuracy * 100).toFixed(1)}%` : '-'}
            </Typography>
          </CardContent></Card>
        </Grid>
        <Grid item xs={6} sm={4} md={2}>
          <Card sx={csx}><CardContent sx={ccsx}>
            <Typography sx={{ fontSize: 10, color: 'text.secondary' }}>Avg AUC</Typography>
            <Typography sx={{ fontSize: 18, fontWeight: 700, color: stats.avgAuc >= 0.65 ? 'success.main' : 'warning.main' }}>
              {stats.avgAuc > 0 ? stats.avgAuc.toFixed(3) : '-'}
            </Typography>
          </CardContent></Card>
        </Grid>
      </Grid>

      {/* ─── Error Alerts ───────────────────────────────────────── */}
      {modelsError && <Alert severity="warning" sx={{ mb: 2, fontSize: 12 }}>{modelsError} — model data may be incomplete.</Alert>}
      {trainingError && <Alert severity="info" sx={{ mb: 2, fontSize: 12 }}>{trainingError}</Alert>}

      {/* ─── Active Training Runs ────────────────────────────────── */}
      {activeRuns.length > 0 && (
        <Card sx={{ mb: 2, border: '1px solid', borderColor: 'primary.main' }}>
          <CardContent sx={{ py: 1.5, px: 2 }}>
            <Typography variant="subtitle1" fontWeight={600} sx={{ fontSize: 13, mb: 1 }}>
              Active Training ({activeRuns.length})
            </Typography>
            <Grid container spacing={1.5}>
              {activeRuns.map(run => (
                <Grid item xs={12} sm={6} md={4} key={run.id}>
                  <Card variant="outlined">
                    <CardContent sx={{ py: 1, px: 1.5, '&:last-child': { pb: 1 } }}>
                      <Box display="flex" justifyContent="space-between" alignItems="center">
                        <Typography fontWeight={600} fontSize={12}>{run.sport_code} - {run.bet_type}</Typography>
                        <Chip size="small" label={run.status} color="primary" sx={{ fontSize: 10 }} />
                      </Box>
                      <Typography fontSize={10} color="text.secondary">{frameworkLabel(run.framework)}</Typography>
                      <LinearProgress sx={{ mt: 0.5 }} />
                      <Typography fontSize={10} color="text.secondary" mt={0.5}>
                        Started {formatDate(run.started_at)}
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
              ))}
            </Grid>
          </CardContent>
        </Card>
      )}

      {/* ─── Recent Training Runs ────────────────────────────────── */}
      {recentRuns.length > 0 && (
        <Card sx={{ mb: 2 }}>
          <CardContent sx={{ py: 1.5, px: 2 }}>
            <Typography variant="subtitle1" fontWeight={600} sx={{ fontSize: 13, mb: 0.5 }}>Recent Training Runs</Typography>
            <TableContainer>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell sx={{ fontWeight: 600, fontSize: 11, py: 0.5 }}>Sport</TableCell>
                    <TableCell sx={{ fontWeight: 600, fontSize: 11, py: 0.5 }}>Bet Type</TableCell>
                    <TableCell sx={{ fontWeight: 600, fontSize: 11, py: 0.5 }}>Framework</TableCell>
                    <TableCell sx={{ fontWeight: 600, fontSize: 11, py: 0.5 }}>Status</TableCell>
                    <TableCell sx={{ fontWeight: 600, fontSize: 11, py: 0.5 }}>Started</TableCell>
                    <TableCell sx={{ fontWeight: 600, fontSize: 11, py: 0.5 }}>Duration</TableCell>
                    <TableCell sx={{ fontWeight: 600, fontSize: 11, py: 0.5 }}></TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {recentRuns.map(run => (
                    <React.Fragment key={run.id}>
                      <TableRow hover>
                        <TableCell sx={{ py: 0.5, fontSize: 11, fontWeight: 600 }}>{run.sport_code}</TableCell>
                        <TableCell sx={{ py: 0.5, fontSize: 11, textTransform: 'capitalize' }}>{run.bet_type}</TableCell>
                        <TableCell sx={{ py: 0.5, fontSize: 11 }}>{frameworkLabel(run.framework)}</TableCell>
                        <TableCell sx={{ py: 0.5, fontSize: 11 }}>
                          <Chip size="small" label={run.status} color={statusColor(run.status) as any} sx={{ fontSize: 10 }} />
                        </TableCell>
                        <TableCell sx={{ py: 0.5, fontSize: 11 }}>{formatDate(run.started_at)}</TableCell>
                        <TableCell sx={{ py: 0.5, fontSize: 11 }}>{formatDuration(run.duration_seconds)}</TableCell>
                        <TableCell sx={{ py: 0.5 }}>
                          {run.metrics && (
                            <IconButton size="small" onClick={() => setExpandedRun(expandedRun === run.id ? null : run.id)}>
                              {expandedRun === run.id ? <ExpandLess sx={{ fontSize: 16 }} /> : <ExpandMore sx={{ fontSize: 16 }} />}
                            </IconButton>
                          )}
                        </TableCell>
                      </TableRow>
                      {run.metrics && expandedRun === run.id && (
                        <TableRow>
                          <TableCell colSpan={7} sx={{ py: 1, bgcolor: isDark ? 'grey.900' : 'grey.50' }}>
                            <Typography fontSize={10} fontFamily="monospace" component="pre" sx={{ m: 0 }}>
                              {JSON.stringify(run.metrics, null, 2)}
                            </Typography>
                          </TableCell>
                        </TableRow>
                      )}
                    </React.Fragment>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </CardContent>
        </Card>
      )}

      {/* ─── Filters ────────────────────────────────────────────── */}
      <Card sx={{ mb: 2 }}>
        <CardContent sx={{ py: 1.5, px: 2 }}>
          <Typography variant="body2" fontWeight={500} gutterBottom sx={{ fontSize: 12 }}>Filter Models</Typography>
          <Grid container spacing={2}>
            <Grid item xs={6} sm={4} md>
              <FormControl fullWidth size="small">
                <InputLabel sx={{ fontSize: 12 }}>Sport</InputLabel>
                <Select value={sportFilter} label="Sport" onChange={(e) => setSportFilter(e.target.value)} sx={{ fontSize: 12 }}>
                  <MenuItem value="all" sx={{ fontSize: 12 }}>All Sports</MenuItem>
                  {availableSports.map(s => <MenuItem key={s} value={s} sx={{ fontSize: 12 }}>{s}</MenuItem>)}
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={6} sm={4} md>
              <FormControl fullWidth size="small">
                <InputLabel sx={{ fontSize: 12 }}>Status</InputLabel>
                <Select value={statusFilter} label="Status" onChange={(e) => setStatusFilter(e.target.value)} sx={{ fontSize: 12 }}>
                  <MenuItem value="all" sx={{ fontSize: 12 }}>All</MenuItem>
                  <MenuItem value="production" sx={{ fontSize: 12 }}>Production</MenuItem>
                  <MenuItem value="ready" sx={{ fontSize: 12 }}>Ready</MenuItem>
                  <MenuItem value="training" sx={{ fontSize: 12 }}>Training</MenuItem>
                  <MenuItem value="deprecated" sx={{ fontSize: 12 }}>Deprecated</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={6} sm={4} md>
              <FormControl fullWidth size="small">
                <InputLabel sx={{ fontSize: 12 }}>Framework</InputLabel>
                <Select value={frameworkFilter} label="Framework" onChange={(e) => setFrameworkFilter(e.target.value)} sx={{ fontSize: 12 }}>
                  <MenuItem value="all" sx={{ fontSize: 12 }}>All</MenuItem>
                  {availableFrameworks.map(f => <MenuItem key={f} value={f} sx={{ fontSize: 12 }}>{frameworkLabel(f)}</MenuItem>)}
                </Select>
              </FormControl>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* ─── Models Table ───────────────────────────────────────── */}
      <Card>
        <CardContent sx={{ pb: 0.5, pt: 1.5, px: 2 }}>
          <Typography variant="subtitle1" fontWeight={600} sx={{ fontSize: 13 }}>
            Models ({sorted.length})
          </Typography>
        </CardContent>
        <TableContainer>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell sx={hdr}>
                  <TableSortLabel active={sortField === 'sport_code'} direction={sortOrder} onClick={() => handleSort('sport_code')}>Sport</TableSortLabel>
                </TableCell>
                <TableCell sx={hdr}>
                  <TableSortLabel active={sortField === 'bet_type'} direction={sortOrder} onClick={() => handleSort('bet_type')}>Bet Type</TableSortLabel>
                </TableCell>
                <TableCell sx={hdr}>
                  <TableSortLabel active={sortField === 'framework'} direction={sortOrder} onClick={() => handleSort('framework')}>Framework</TableSortLabel>
                </TableCell>
                <TableCell sx={hdr}>Version</TableCell>
                <TableCell sx={hdr} align="center">
                  <TableSortLabel active={sortField === 'accuracy'} direction={sortOrder} onClick={() => handleSort('accuracy')}>Accuracy</TableSortLabel>
                </TableCell>
                <TableCell sx={hdr} align="center">
                  <TableSortLabel active={sortField === 'auc'} direction={sortOrder} onClick={() => handleSort('auc')}>AUC</TableSortLabel>
                </TableCell>
                <TableCell sx={hdr}>
                  <TableSortLabel active={sortField === 'status'} direction={sortOrder} onClick={() => handleSort('status')}>Status</TableSortLabel>
                </TableCell>
                <TableCell sx={hdr}>
                  <TableSortLabel active={sortField === 'created_at'} direction={sortOrder} onClick={() => handleSort('created_at')}>Trained</TableSortLabel>
                </TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {sorted.map((model) => {
                const acc = model.accuracy != null ? model.accuracy : null;
                const accPct = acc != null ? (acc > 1 ? acc : acc * 100) : null;
                return (
                  <TableRow key={model.id} hover>
                    <TableCell sx={{ py: 0.75, fontSize: 11 }}>
                      <Chip label={model.sport_code} size="small" sx={{ fontWeight: 600, fontSize: 10 }} />
                    </TableCell>
                    <TableCell sx={{ py: 0.75, fontSize: 11, textTransform: 'capitalize' }}>{model.bet_type}</TableCell>
                    <TableCell sx={{ py: 0.75, fontSize: 11 }}>{frameworkLabel(model.framework)}</TableCell>
                    <TableCell sx={{ py: 0.75, fontSize: 11, fontFamily: 'monospace', color: 'text.secondary' }}>{model.version || '-'}</TableCell>
                    <TableCell align="center" sx={{ py: 0.75, fontSize: 11 }}>
                      {accPct != null ? (
                        <Typography variant="body2" sx={{ fontSize: 11, color: accPct >= 65 ? 'success.main' : accPct >= 60 ? 'warning.main' : accPct >= 55 ? 'text.primary' : 'error.main', fontWeight: 600 }}>
                          {accPct.toFixed(1)}%
                        </Typography>
                      ) : '-'}
                    </TableCell>
                    <TableCell align="center" sx={{ py: 0.75, fontSize: 11 }}>
                      {model.auc != null ? (
                        <Typography variant="body2" sx={{ fontSize: 11, fontWeight: 500 }}>{model.auc.toFixed(3)}</Typography>
                      ) : '-'}
                    </TableCell>
                    <TableCell sx={{ py: 0.75, fontSize: 11 }}>
                      <Chip size="small" label={model.status} color={statusColor(model.status) as any} sx={{ fontSize: 10, textTransform: 'capitalize' }} />
                    </TableCell>
                    <TableCell sx={{ py: 0.75, fontSize: 11 }}>{formatDate(model.created_at)}</TableCell>
                  </TableRow>
                );
              })}
              {sorted.length === 0 && !loading && (
                <TableRow>
                  <TableCell colSpan={8} align="center" sx={{ py: 4 }}>
                    <Typography color="text.secondary" fontSize={12}>
                      {models.length === 0 ? 'No models found. Train your first model to get started.' : 'No models match the current filters.'}
                    </Typography>
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </TableContainer>
      </Card>

      {/* ─── Train Dialog ───────────────────────────────────────── */}
      <Dialog open={trainDialogOpen} onClose={() => setTrainDialogOpen(false)} maxWidth="xs" fullWidth>
        <DialogTitle sx={{ fontSize: 14, fontWeight: 600, pb: 1 }}>Train New Model</DialogTitle>
        <DialogContent>
          <Grid container spacing={2} sx={{ mt: 0 }}>
            <Grid item xs={12}>
              <FormControl fullWidth size="small">
                <InputLabel sx={{ fontSize: 12 }}>Sport</InputLabel>
                <Select value={trainSport} label="Sport" onChange={(e) => setTrainSport(e.target.value)} sx={{ fontSize: 12 }}>
                  {['NBA', 'NFL', 'NHL', 'MLB', 'NCAAB', 'NCAAF', 'WNBA', 'CFL', 'ATP', 'WTA'].map(s =>
                    <MenuItem key={s} value={s} sx={{ fontSize: 12 }}>{s}</MenuItem>
                  )}
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12}>
              <FormControl fullWidth size="small">
                <InputLabel sx={{ fontSize: 12 }}>Bet Type</InputLabel>
                <Select value={trainBetType} label="Bet Type" onChange={(e) => setTrainBetType(e.target.value)} sx={{ fontSize: 12 }}>
                  {['spread', 'moneyline', 'total'].map(t =>
                    <MenuItem key={t} value={t} sx={{ fontSize: 12, textTransform: 'capitalize' }}>{t}</MenuItem>
                  )}
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12}>
              <FormControl fullWidth size="small">
                <InputLabel sx={{ fontSize: 12 }}>Framework</InputLabel>
                <Select value={trainFramework} label="Framework" onChange={(e) => setTrainFramework(e.target.value)} sx={{ fontSize: 12 }}>
                  <MenuItem value="meta_ensemble" sx={{ fontSize: 12 }}>Meta-Ensemble (All Frameworks)</MenuItem>
                  <MenuItem value="h2o" sx={{ fontSize: 12 }}>H2O AutoML</MenuItem>
                  <MenuItem value="autogluon" sx={{ fontSize: 12 }}>AutoGluon</MenuItem>
                  <MenuItem value="sklearn" sx={{ fontSize: 12 }}>Sklearn Ensemble</MenuItem>
                  <MenuItem value="deep_learning" sx={{ fontSize: 12 }}>Deep Learning (TF/LSTM)</MenuItem>
                  <MenuItem value="quantum" sx={{ fontSize: 12 }}>Quantum ML</MenuItem>
                </Select>
              </FormControl>
            </Grid>
          </Grid>
          {trainResult && (
            <Alert severity={trainResult.startsWith('Error') ? 'error' : 'success'} sx={{ mt: 2, fontSize: 11 }}>
              {trainResult}
            </Alert>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setTrainDialogOpen(false)} size="small" sx={{ fontSize: 11 }}>Cancel</Button>
          <Button onClick={handleTrain} variant="contained" size="small" disabled={trainSubmitting} startIcon={<PlayArrow sx={{ fontSize: 14 }} />} sx={{ fontSize: 11 }}>
            {trainSubmitting ? 'Starting...' : 'Start Training'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default Models;