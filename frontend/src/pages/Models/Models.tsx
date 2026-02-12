// src/pages/Models/Models.tsx - Real data with promote, cancel, reinforce
import React, { useState, useEffect, useCallback, useMemo } from 'react';
import {
  Box, Card, CardContent, Typography, Grid, Table, TableBody, TableCell,
  TableContainer, TableHead, TableRow, Chip, Button, LinearProgress,
  Alert, IconButton, Collapse, FormControl, InputLabel, Select, MenuItem,
  useTheme, Dialog, DialogTitle, DialogContent, DialogActions, TableSortLabel,
  Tooltip
} from '@mui/material';
import {
  PlayArrow, CheckCircle, Error as ErrorIcon, ExpandMore, ExpandLess,
  Refresh, Science, Star, StarBorder, Cancel, AutoFixHigh
} from '@mui/icons-material';
import { api } from '../../api/client';

// ─── Types ───────────────────────────────────────────────────────────
interface Model {
  id: string;
  sport_code: string;
  bet_type: string;
  framework: string;
  version: string;
  status: string;
  accuracy: number | null;
  raw_accuracy: number | null;
  wfv_accuracy: number | null;
  auc: number | null;
  raw_auc: number | null;
  wfv_auc: number | null;
  log_loss: number | null;
  wfv_roi: number | null;
  wfv_n_folds: number | null;
  created_at: string;
  training_samples: number | null;
}

interface TrainingRun {
  id: string;
  sport_code: string;
  bet_type: string;
  framework: string;
  status: string;
  started_at: string;
  completed_at: string | null;
  duration_seconds: number | null;
  metrics: Record<string, any> | null;
  error_message: string | null;
}

type SortField = 'sport_code' | 'bet_type' | 'framework' | 'accuracy' | 'auc' | 'created_at' | 'status' | 'wfv_accuracy' | 'training_samples';

// ─── Helpers ─────────────────────────────────────────────────────────
const statusColor = (s: string): 'success' | 'info' | 'primary' | 'default' | 'error' | 'warning' => {
  switch (s) {
    case 'production': return 'success';
    case 'ready': return 'info';
    case 'running': case 'pending': return 'primary';
    case 'failed': return 'error';
    default: return 'default';
  }
};

const frameworkLabel = (f: string) => {
  const m: Record<string, string> = {
    'meta_ensemble': 'Meta-Ensemble', 'h2o': 'H2O AutoML', 'autogluon': 'AutoGluon',
    'sklearn': 'Sklearn', 'deep_learning': 'Deep Learning', 'quantum': 'Quantum ML',
  };
  return m[f] || f;
};

const formatDate = (d: string | null) => {
  if (!d) return '-';
  return new Date(d).toLocaleDateString('en-US', { month: 'numeric', day: 'numeric', year: 'numeric', timeZone: 'America/Los_Angeles' });
};

const formatDuration = (secs: number | null) => {
  if (!secs) return '-';
  if (secs < 60) return `${secs}s`;
  if (secs < 3600) return `${Math.floor(secs / 60)}m ${secs % 60}s`;
  return `${Math.floor(secs / 3600)}h ${Math.floor((secs % 3600) / 60)}m`;
};

const fmtAcc = (v: number | null) => {
  if (v == null || v === 0) return null;
  const pct = v > 1 ? v : v * 100;
  return pct;
};

// ─── Component ───────────────────────────────────────────────────────
const Models: React.FC = () => {
  const theme = useTheme();
  const isDark = theme.palette.mode === 'dark';

  const [models, setModels] = useState<Model[]>([]);
  const [trainingRuns, setTrainingRuns] = useState<TrainingRun[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Filters
  const [sportFilter, setSportFilter] = useState('all');
  const [statusFilter, setStatusFilter] = useState('all');
  const [frameworkFilter, setFrameworkFilter] = useState('all');

  // Sort
  const [sortField, setSortField] = useState<SortField>('accuracy');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');

  // Dialogs
  const [reinforceOpen, setReinforceOpen] = useState(false);
  const [reinforceSport, setReinforceSport] = useState('');
  const [reinforceBetType, setReinforceBetType] = useState('');
  const [reinforceFramework, setReinforceFramework] = useState('meta_ensemble');
  const [actionResult, setActionResult] = useState<{ msg: string; ok: boolean } | null>(null);
  const [submitting, setSubmitting] = useState(false);

  // Expanded
  const [expandedRun, setExpandedRun] = useState<string | null>(null);

  // ─── Load Data ──────────────────────────────────────────────────
  const loadData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [modelsData, runsData] = await Promise.allSettled([
        api.getModels(),
        api.getTrainingRuns({ limit: 50 }),
      ]);
      setModels(modelsData.status === 'fulfilled' && Array.isArray(modelsData.value) ? modelsData.value : []);
      setTrainingRuns(runsData.status === 'fulfilled' && Array.isArray(runsData.value) ? runsData.value : []);
      if (modelsData.status === 'rejected') setError('Failed to load models');
    } catch (err) {
      setError('Failed to load data');
    }
    setLoading(false);
  }, []);

  useEffect(() => { loadData(); }, [loadData]);

  // Auto-refresh if training running
  useEffect(() => {
    const hasRunning = trainingRuns.some(r => r.status === 'running' || r.status === 'pending');
    if (!hasRunning) return;
    const interval = setInterval(loadData, 15000);
    return () => clearInterval(interval);
  }, [trainingRuns, loadData]);

  // ─── Derived ───────────────────────────────────────────────────
  const availableSports = useMemo(() => Array.from(new Set(models.map(m => m.sport_code).filter(Boolean))).sort(), [models]);
  const availableFrameworks = useMemo(() => Array.from(new Set(models.map(m => m.framework).filter(Boolean))).sort(), [models]);

  // Unique sport+betType combos for reinforce
  const sportBetCombos = useMemo(() => {
    const seen = new Set<string>();
    return models.filter(m => {
      const k = `${m.sport_code}|${m.bet_type}`;
      if (seen.has(k)) return false;
      seen.add(k);
      return true;
    }).map(m => ({ sport: m.sport_code, betType: m.bet_type }));
  }, [models]);

  const productionModels = useMemo(() => models.filter(m => m.status === 'production'), [models]);

  const stats = useMemo(() => {
    const prod = productionModels;
    const all = models;
    const accs = all.map(m => fmtAcc(m.wfv_accuracy) ?? fmtAcc(m.accuracy)).filter(v => v != null && v > 0 && v < 100) as number[];
    const aucs = all.map(m => m.wfv_auc ?? m.auc).filter(v => v != null && v > 0 && v < 1) as number[];
    return {
      total: all.length,
      production: prod.length,
      ready: all.length - prod.length,
      training: trainingRuns.filter(r => r.status === 'running' || r.status === 'pending').length,
      avgAccuracy: accs.length > 0 ? accs.reduce((s, v) => s + v, 0) / accs.length : 0,
      avgAuc: aucs.length > 0 ? aucs.reduce((s, v) => s + v, 0) / aucs.length : 0,
      sports: new Set(all.map(m => m.sport_code)).size,
    };
  }, [models, productionModels, trainingRuns]);

  // Filtered + sorted
  const filtered = useMemo(() => models.filter(m => {
    if (sportFilter !== 'all' && m.sport_code !== sportFilter) return false;
    if (statusFilter !== 'all' && m.status !== statusFilter) return false;
    if (frameworkFilter !== 'all' && m.framework !== frameworkFilter) return false;
    return true;
  }), [models, sportFilter, statusFilter, frameworkFilter]);

  const sorted = useMemo(() => [...filtered].sort((a, b) => {
    let aV: any, bV: any;
    switch (sortField) {
      case 'sport_code': aV = a.sport_code; bV = b.sport_code; break;
      case 'bet_type': aV = a.bet_type; bV = b.bet_type; break;
      case 'framework': aV = a.framework; bV = b.framework; break;
      case 'accuracy': aV = fmtAcc(a.wfv_accuracy) ?? fmtAcc(a.accuracy) ?? 0; bV = fmtAcc(b.wfv_accuracy) ?? fmtAcc(b.accuracy) ?? 0; break;
      case 'wfv_accuracy': aV = fmtAcc(a.wfv_accuracy) ?? 0; bV = fmtAcc(b.wfv_accuracy) ?? 0; break;
      case 'auc': aV = a.wfv_auc ?? a.auc ?? 0; bV = b.wfv_auc ?? b.auc ?? 0; break;
      case 'created_at': aV = a.created_at || ''; bV = b.created_at || ''; break;
      case 'status': aV = a.status === 'production' ? 0 : 1; bV = b.status === 'production' ? 0 : 1; break;
      case 'training_samples': aV = a.training_samples ?? 0; bV = b.training_samples ?? 0; break;
      default: aV = fmtAcc(a.accuracy) ?? 0; bV = fmtAcc(b.accuracy) ?? 0;
    }
    if (typeof aV === 'string') return sortOrder === 'asc' ? aV.localeCompare(bV) : bV.localeCompare(aV);
    return sortOrder === 'asc' ? aV - bV : bV - aV;
  }), [filtered, sortField, sortOrder]);

  const handleSort = (field: SortField) => {
    if (sortField === field) setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
    else { setSortField(field); setSortOrder('desc'); }
  };

  // ─── Actions ───────────────────────────────────────────────────
  const handlePromote = async (modelId: string) => {
    try {
      const res = await api.promoteModel(modelId);
      setActionResult({ msg: res.message || 'Promoted', ok: true });
      loadData();
    } catch { setActionResult({ msg: 'Failed to promote', ok: false }); }
    setTimeout(() => setActionResult(null), 3000);
  };

  const handleDeprecate = async (modelId: string) => {
    try {
      const res = await api.deprecateModel(modelId);
      setActionResult({ msg: res.message || 'Deprecated', ok: true });
      loadData();
    } catch { setActionResult({ msg: 'Failed to deprecate', ok: false }); }
    setTimeout(() => setActionResult(null), 3000);
  };

  const handleCancelRun = async (runId: string) => {
    try {
      await api.cancelTrainingRun(runId);
      setActionResult({ msg: 'Training run cancelled', ok: true });
      loadData();
    } catch { setActionResult({ msg: 'Failed to cancel', ok: false }); }
    setTimeout(() => setActionResult(null), 3000);
  };

  const handleReinforce = async () => {
    if (!reinforceSport || !reinforceBetType) return;
    setSubmitting(true);
    try {
      const res = await api.reinforceModel({ sport_code: reinforceSport, bet_type: reinforceBetType, framework: reinforceFramework });
      if (res.error) {
        setActionResult({ msg: res.error, ok: false });
      } else {
        setActionResult({ msg: res.message || 'Reinforcement training started', ok: true });
        setReinforceOpen(false);
        loadData();
      }
    } catch (err: any) {
      setActionResult({ msg: err?.response?.data?.error || 'Failed to start reinforcement', ok: false });
    }
    setSubmitting(false);
    setTimeout(() => setActionResult(null), 4000);
  };

  const openReinforce = (sport?: string, betType?: string) => {
    setReinforceSport(sport || sportBetCombos[0]?.sport || 'NBA');
    setReinforceBetType(betType || sportBetCombos[0]?.betType || 'spread');
    setReinforceOpen(true);
  };

  // Active/recent training
  const activeRuns = useMemo(() => trainingRuns.filter(r => r.status === 'running' || r.status === 'pending'), [trainingRuns]);
  const recentRuns = useMemo(() => trainingRuns.filter(r => r.status !== 'running' && r.status !== 'pending').slice(0, 10), [trainingRuns]);

  const hdr = { fontWeight: 600, fontSize: 11, py: 0.75, bgcolor: isDark ? 'grey.900' : 'grey.100', color: isDark ? 'grey.100' : 'grey.800', whiteSpace: 'nowrap', borderBottom: 1, borderColor: 'divider' };
  const csx = { height: '100%' };
  const ccsx = { textAlign: 'center', py: 1.5, '&:last-child': { pb: 1.5 }, height: '100%', display: 'flex', flexDirection: 'column', justifyContent: 'center' };

  const AccuracyCell: React.FC<{ model: Model }> = ({ model }) => {
    const wfv = fmtAcc(model.wfv_accuracy);
    const raw = fmtAcc(model.raw_accuracy);
    const display = wfv ?? raw;
    if (!display) return <Typography fontSize={11}>-</Typography>;

    const isWfv = wfv != null && wfv > 0;
    const color = display >= 65 ? 'success.main' : display >= 58 ? 'warning.main' : display >= 52 ? 'text.primary' : 'error.main';

    return (
      <Tooltip title={
        <Box>
          <Typography fontSize={10}>WFV Accuracy: {wfv != null ? `${wfv.toFixed(1)}%` : 'N/A'}</Typography>
          <Typography fontSize={10}>Raw Accuracy: {raw != null ? `${raw.toFixed(1)}%` : 'N/A'}</Typography>
          {model.wfv_n_folds != null && <Typography fontSize={10}>WFV Folds: {model.wfv_n_folds}</Typography>}
          {model.wfv_roi != null && <Typography fontSize={10}>WFV ROI: {(model.wfv_roi * 100).toFixed(1)}%</Typography>}
          {raw != null && raw > 90 && <Typography fontSize={10} color="warning.main">⚠️ Raw accuracy inflated (data leakage)</Typography>}
        </Box>
      } arrow>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, justifyContent: 'center' }}>
          <Typography sx={{ fontSize: 11, color, fontWeight: 600 }}>{display.toFixed(1)}%</Typography>
          {isWfv && <Chip label="WFV" size="small" sx={{ height: 14, fontSize: 8, bgcolor: 'success.main', color: '#fff' }} />}
        </Box>
      </Tooltip>
    );
  };

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
        <Typography variant="h5" fontWeight={700} sx={{ fontSize: 20 }}>ML Models</Typography>
        <Box display="flex" gap={1}>
          <Button variant="contained" size="small" startIcon={<AutoFixHigh sx={{ fontSize: 14 }} />}
            onClick={() => openReinforce()} sx={{ fontSize: 11, bgcolor: '#ff9800', '&:hover': { bgcolor: '#f57c00' } }}>
            Reinforce Model
          </Button>
          <Button variant="outlined" size="small" startIcon={<Refresh sx={{ fontSize: 14 }} />} onClick={loadData} sx={{ fontSize: 11 }}>Refresh</Button>
        </Box>
      </Box>

      {loading && <LinearProgress sx={{ mb: 2 }} />}
      {error && <Alert severity="warning" sx={{ mb: 1.5, fontSize: 11 }}>{error}</Alert>}
      {actionResult && <Alert severity={actionResult.ok ? 'success' : 'error'} sx={{ mb: 1.5, fontSize: 11 }}>{actionResult.msg}</Alert>}

      {/* ─── Stat Cards ─────────────────────────────────────────── */}
      <Grid container spacing={2} mb={2} sx={{ alignItems: 'stretch' }}>
        {[
          { label: 'Total Models', val: stats.total },
          { label: 'Production', val: stats.production, color: 'success.main', sub: `${stats.sports} sports` },
          { label: 'Ready', val: stats.ready, color: 'info.main' },
          { label: 'Training', val: stats.training, color: stats.training > 0 ? 'primary.main' : undefined },
          { label: 'Avg WFV Accuracy', val: stats.avgAccuracy > 0 ? `${stats.avgAccuracy.toFixed(1)}%` : '-', color: stats.avgAccuracy >= 58 ? 'success.main' : 'warning.main' },
          { label: 'Avg WFV AUC', val: stats.avgAuc > 0 ? stats.avgAuc.toFixed(3) : '-', color: stats.avgAuc >= 0.58 ? 'success.main' : 'warning.main' },
        ].map((c, i) => (
          <Grid item xs={6} sm={4} md={2} key={i}>
            <Card sx={csx}><CardContent sx={ccsx}>
              <Typography sx={{ fontSize: 10, color: 'text.secondary' }}>{c.label}</Typography>
              <Typography sx={{ fontSize: 18, fontWeight: 700, color: c.color || 'text.primary' }}>{c.val}</Typography>
              {c.sub && <Typography sx={{ fontSize: 9, color: 'text.secondary' }}>{c.sub}</Typography>}
            </CardContent></Card>
          </Grid>
        ))}
      </Grid>

      {/* ─── Active Training ─────────────────────────────────────── */}
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
                        <Box display="flex" gap={0.5}>
                          <Chip size="small" label={run.status} color="primary" sx={{ fontSize: 10 }} />
                          <Tooltip title="Cancel this training run"><IconButton size="small" color="error" onClick={() => handleCancelRun(run.id)}><Cancel sx={{ fontSize: 14 }} /></IconButton></Tooltip>
                        </Box>
                      </Box>
                      <Typography fontSize={10} color="text.secondary">{frameworkLabel(run.framework)}</Typography>
                      <LinearProgress sx={{ mt: 0.5 }} />
                      <Typography fontSize={10} color="text.secondary" mt={0.5}>Started {formatDate(run.started_at)}</Typography>
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
                    {['Sport', 'Bet Type', 'Framework', 'Status', 'Started', 'Duration', ''].map(h => (
                      <TableCell key={h} sx={{ fontWeight: 600, fontSize: 11, py: 0.5 }}>{h}</TableCell>
                    ))}
                  </TableRow>
                </TableHead>
                <TableBody>
                  {recentRuns.map(run => (
                    <React.Fragment key={run.id}>
                      <TableRow hover>
                        <TableCell sx={{ py: 0.5, fontSize: 11, fontWeight: 600 }}>{run.sport_code}</TableCell>
                        <TableCell sx={{ py: 0.5, fontSize: 11, textTransform: 'capitalize' }}>{run.bet_type}</TableCell>
                        <TableCell sx={{ py: 0.5, fontSize: 11 }}>{frameworkLabel(run.framework)}</TableCell>
                        <TableCell sx={{ py: 0.5 }}><Chip size="small" label={run.status} color={statusColor(run.status)} sx={{ fontSize: 10 }} /></TableCell>
                        <TableCell sx={{ py: 0.5, fontSize: 11 }}>{formatDate(run.started_at)}</TableCell>
                        <TableCell sx={{ py: 0.5, fontSize: 11 }}>{formatDuration(run.duration_seconds)}</TableCell>
                        <TableCell sx={{ py: 0.5 }}>
                          {(run.metrics || run.error_message) && (
                            <IconButton size="small" onClick={() => setExpandedRun(expandedRun === run.id ? null : run.id)}>
                              {expandedRun === run.id ? <ExpandLess sx={{ fontSize: 16 }} /> : <ExpandMore sx={{ fontSize: 16 }} />}
                            </IconButton>
                          )}
                        </TableCell>
                      </TableRow>
                      {expandedRun === run.id && (
                        <TableRow>
                          <TableCell colSpan={7} sx={{ py: 1, bgcolor: isDark ? 'grey.900' : 'grey.50' }}>
                            {run.error_message && <Alert severity="error" sx={{ mb: 1, fontSize: 10 }}>{run.error_message}</Alert>}
                            {run.metrics && <Typography fontSize={10} fontFamily="monospace" component="pre" sx={{ m: 0, whiteSpace: 'pre-wrap' }}>{JSON.stringify(run.metrics, null, 2)}</Typography>}
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
          <Typography variant="subtitle1" fontWeight={600} sx={{ fontSize: 13 }}>Models ({sorted.length})</Typography>
        </CardContent>
        <TableContainer>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell sx={hdr}><TableSortLabel active={sortField === 'sport_code'} direction={sortOrder} onClick={() => handleSort('sport_code')}>Sport</TableSortLabel></TableCell>
                <TableCell sx={hdr}><TableSortLabel active={sortField === 'bet_type'} direction={sortOrder} onClick={() => handleSort('bet_type')}>Bet Type</TableSortLabel></TableCell>
                <TableCell sx={hdr}><TableSortLabel active={sortField === 'framework'} direction={sortOrder} onClick={() => handleSort('framework')}>Framework</TableSortLabel></TableCell>
                <TableCell sx={hdr}>Version</TableCell>
                <TableCell sx={hdr} align="center"><TableSortLabel active={sortField === 'accuracy'} direction={sortOrder} onClick={() => handleSort('accuracy')}>Accuracy</TableSortLabel></TableCell>
                <TableCell sx={hdr} align="center"><TableSortLabel active={sortField === 'auc'} direction={sortOrder} onClick={() => handleSort('auc')}>AUC</TableSortLabel></TableCell>
                <TableCell sx={hdr} align="center"><TableSortLabel active={sortField === 'training_samples'} direction={sortOrder} onClick={() => handleSort('training_samples')}>Samples</TableSortLabel></TableCell>
                <TableCell sx={hdr}><TableSortLabel active={sortField === 'status'} direction={sortOrder} onClick={() => handleSort('status')}>Status</TableSortLabel></TableCell>
                <TableCell sx={hdr}><TableSortLabel active={sortField === 'created_at'} direction={sortOrder} onClick={() => handleSort('created_at')}>Trained</TableSortLabel></TableCell>
                <TableCell sx={hdr}>Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {sorted.map((model) => {
                const isProd = model.status === 'production';
                return (
                  <TableRow key={model.id} hover sx={{ bgcolor: isProd ? (isDark ? 'rgba(76, 175, 80, 0.08)' : 'rgba(76, 175, 80, 0.04)') : undefined }}>
                    <TableCell sx={{ py: 0.75, fontSize: 11 }}><Chip label={model.sport_code} size="small" sx={{ fontWeight: 600, fontSize: 10 }} /></TableCell>
                    <TableCell sx={{ py: 0.75, fontSize: 11, textTransform: 'capitalize' }}>{model.bet_type}</TableCell>
                    <TableCell sx={{ py: 0.75, fontSize: 11 }}>{frameworkLabel(model.framework)}</TableCell>
                    <TableCell sx={{ py: 0.75, fontSize: 10, fontFamily: 'monospace', color: 'text.secondary' }}>{model.version || '-'}</TableCell>
                    <TableCell align="center" sx={{ py: 0.75 }}><AccuracyCell model={model} /></TableCell>
                    <TableCell align="center" sx={{ py: 0.75, fontSize: 11 }}>
                      {(model.wfv_auc ?? model.auc) != null ? (
                        <Tooltip title={`Raw: ${model.raw_auc?.toFixed(3) || 'N/A'} | WFV: ${model.wfv_auc?.toFixed(3) || 'N/A'}`}>
                          <Typography fontSize={11} fontWeight={500}>{((model.wfv_auc ?? model.auc) as number).toFixed(3)}</Typography>
                        </Tooltip>
                      ) : '-'}
                    </TableCell>
                    <TableCell align="center" sx={{ py: 0.75, fontSize: 11, color: 'text.secondary' }}>
                      {model.training_samples != null ? model.training_samples.toLocaleString() : '-'}
                    </TableCell>
                    <TableCell sx={{ py: 0.75 }}>
                      <Chip size="small" label={isProd ? 'Production' : 'Ready'}
                        color={isProd ? 'success' : 'default'}
                        icon={isProd ? <Star sx={{ fontSize: 12 }} /> : undefined}
                        sx={{ fontSize: 10, textTransform: 'capitalize' }} />
                    </TableCell>
                    <TableCell sx={{ py: 0.75, fontSize: 11 }}>{formatDate(model.created_at)}</TableCell>
                    <TableCell sx={{ py: 0.75 }}>
                      <Box display="flex" gap={0.5}>
                        {!isProd ? (
                          <Tooltip title="Promote to production">
                            <IconButton size="small" color="success" onClick={() => handlePromote(model.id)}><Star sx={{ fontSize: 14 }} /></IconButton>
                          </Tooltip>
                        ) : (
                          <Tooltip title="Remove from production">
                            <IconButton size="small" color="warning" onClick={() => handleDeprecate(model.id)}><StarBorder sx={{ fontSize: 14 }} /></IconButton>
                          </Tooltip>
                        )}
                        <Tooltip title="Reinforce (retrain with latest data)">
                          <IconButton size="small" color="info" onClick={() => openReinforce(model.sport_code, model.bet_type)}>
                            <AutoFixHigh sx={{ fontSize: 14 }} />
                          </IconButton>
                        </Tooltip>
                      </Box>
                    </TableCell>
                  </TableRow>
                );
              })}
              {sorted.length === 0 && !loading && (
                <TableRow><TableCell colSpan={10} align="center" sx={{ py: 4 }}>
                  <Typography color="text.secondary" fontSize={12}>
                    {models.length === 0 ? 'No models found.' : 'No models match filters.'}
                  </Typography>
                </TableCell></TableRow>
              )}
            </TableBody>
          </Table>
        </TableContainer>
      </Card>

      {/* ─── Reinforce Dialog ───────────────────────────────────── */}
      <Dialog open={reinforceOpen} onClose={() => setReinforceOpen(false)} maxWidth="xs" fullWidth>
        <DialogTitle sx={{ fontSize: 14, fontWeight: 600, pb: 1 }}>
          <Box display="flex" alignItems="center" gap={1}>
            <AutoFixHigh sx={{ color: '#ff9800' }} /> Reinforce Model
          </Box>
        </DialogTitle>
        <DialogContent>
          <Typography fontSize={11} color="text.secondary" mb={2}>
            Retrains the model using the latest available data. The new model appears as "Ready" and can be promoted to production if it performs better.
          </Typography>
          <Grid container spacing={2} sx={{ mt: 0 }}>
            <Grid item xs={12}>
              <FormControl fullWidth size="small">
                <InputLabel sx={{ fontSize: 12 }}>Sport</InputLabel>
                <Select value={reinforceSport} label="Sport" onChange={(e) => {
                  setReinforceSport(e.target.value);
                  const combo = sportBetCombos.find(c => c.sport === e.target.value);
                  if (combo) setReinforceBetType(combo.betType);
                }} sx={{ fontSize: 12 }}>
                  {Array.from(new Set(sportBetCombos.map(c => c.sport))).map(s =>
                    <MenuItem key={s} value={s} sx={{ fontSize: 12 }}>{s}</MenuItem>
                  )}
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12}>
              <FormControl fullWidth size="small">
                <InputLabel sx={{ fontSize: 12 }}>Bet Type</InputLabel>
                <Select value={reinforceBetType} label="Bet Type" onChange={(e) => setReinforceBetType(e.target.value)} sx={{ fontSize: 12 }}>
                  {sportBetCombos.filter(c => c.sport === reinforceSport).map(c =>
                    <MenuItem key={c.betType} value={c.betType} sx={{ fontSize: 12, textTransform: 'capitalize' }}>{c.betType}</MenuItem>
                  )}
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12}>
              <FormControl fullWidth size="small">
                <InputLabel sx={{ fontSize: 12 }}>Framework</InputLabel>
                <Select value={reinforceFramework} label="Framework" onChange={(e) => setReinforceFramework(e.target.value)} sx={{ fontSize: 12 }}>
                  <MenuItem value="meta_ensemble" sx={{ fontSize: 12 }}>Meta-Ensemble (All)</MenuItem>
                  <MenuItem value="h2o" sx={{ fontSize: 12 }}>H2O AutoML</MenuItem>
                  <MenuItem value="autogluon" sx={{ fontSize: 12 }}>AutoGluon</MenuItem>
                  <MenuItem value="sklearn" sx={{ fontSize: 12 }}>Sklearn Ensemble</MenuItem>
                </Select>
              </FormControl>
            </Grid>
          </Grid>
          {/* Show current production model stats for comparison */}
          {(() => {
            const current = productionModels.find(m => m.sport_code === reinforceSport && m.bet_type === reinforceBetType);
            if (!current) return null;
            const acc = fmtAcc(current.wfv_accuracy) ?? fmtAcc(current.accuracy);
            return (
              <Alert severity="info" sx={{ mt: 2, fontSize: 10 }}>
                Current production: {frameworkLabel(current.framework)}{acc ? ` • ${acc.toFixed(1)}% acc` : ''}{current.wfv_auc ? ` • ${current.wfv_auc.toFixed(3)} AUC` : ''}
              </Alert>
            );
          })()}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setReinforceOpen(false)} size="small" sx={{ fontSize: 11 }}>Cancel</Button>
          <Button onClick={handleReinforce} variant="contained" size="small" disabled={submitting}
            startIcon={<AutoFixHigh sx={{ fontSize: 14 }} />}
            sx={{ fontSize: 11, bgcolor: '#ff9800', '&:hover': { bgcolor: '#f57c00' } }}>
            {submitting ? 'Starting...' : 'Start Reinforcement'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default Models;