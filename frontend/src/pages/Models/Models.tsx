// src/pages/Models/Models.tsx
import React, { useState, useEffect } from 'react';
import {
  Box, Card, CardContent, Typography, Grid, Table, TableBody, TableCell,
  TableContainer, TableHead, TableRow, Chip, Button, LinearProgress,
  Stepper, Step, StepLabel, StepContent, Alert, IconButton, Collapse
} from '@mui/material';
import { PlayArrow, CheckCircle, Error, HourglassEmpty, ExpandMore, ExpandLess, Refresh } from '@mui/icons-material';
import { api } from '../../api/client';
import { formatPercent } from '../../utils';

interface Model {
  id: string;
  sport: string;
  bet_type: string;
  framework: string;
  accuracy: number;
  auc: number;
  is_production: boolean;
  training_date: string;
}

interface PipelineStep {
  name: string;
  status: 'complete' | 'running' | 'pending' | 'error';
  progress?: number;
}

interface Pipeline {
  sport: string;
  bet_type: string;
  status: 'idle' | 'running' | 'complete' | 'error';
  current_step: number;
  steps: PipelineStep[];
}

const PIPELINE_STEPS = [
  { name: 'Data Validation', description: 'Check data completeness (95%+ required)' },
  { name: 'Feature Engineering', description: 'Generate 60-85 features' },
  { name: 'H2O AutoML', description: 'Train 50+ models automatically' },
  { name: 'AutoGluon', description: 'Multi-layer stack ensemble' },
  { name: 'Sklearn Ensemble', description: 'XGBoost, LightGBM, CatBoost' },
  { name: 'Meta-Ensemble', description: 'Combine all frameworks' },
  { name: 'Calibration', description: 'Isotonic regression for probabilities' },
  { name: 'Deploy', description: 'Promote to production' },
];

const Models: React.FC = () => {
  const [models, setModels] = useState<Model[]>([]);
  const [pipelines, setPipelines] = useState<Pipeline[]>([]);
  const [loading, setLoading] = useState(true);
  const [expandedPipeline, setExpandedPipeline] = useState<string | null>(null);

  useEffect(() => { loadModels(); }, []);

  const loadModels = async () => {
    setLoading(true);
    try {
      const data = await api.getModels();
      setModels(Array.isArray(data) ? data : generateDemoModels());
    } catch {
      setModels(generateDemoModels());
    }
    setPipelines(generateDemoPipelines());
    setLoading(false);
  };

  const startTraining = (sport: string, betType: string) => {
    setPipelines(prev => prev.map(p => 
      p.sport === sport && p.bet_type === betType 
        ? { ...p, status: 'running' as const, current_step: 0, steps: p.steps.map((s, i) => ({ ...s, status: i === 0 ? 'running' as const : 'pending' as const })) }
        : p
    ));
  };

  const getStepIcon = (status: string) => {
    switch (status) {
      case 'complete': return <CheckCircle color="success" />;
      case 'running': return <HourglassEmpty color="primary" />;
      case 'error': return <Error color="error" />;
      default: return <HourglassEmpty color="disabled" />;
    }
  };

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h5" fontWeight={700} sx={{ fontSize: 20 }}>ML Models</Typography>
        <Button variant="outlined" startIcon={<Refresh />} onClick={loadModels}>Refresh</Button>
      </Box>

      <Alert severity="info" sx={{ mb: 3 }}><strong>Sequential Pipeline:</strong> Models must be trained in order. Each step depends on the previous step completing successfully.</Alert>

      <Typography variant="h6" gutterBottom>Training Pipelines</Typography>
      <Grid container spacing={2} mb={3}>
        {pipelines.map((pipeline) => (
          <Grid item xs={12} md={6} key={`${pipeline.sport}-${pipeline.bet_type}`}>
            <Card>
              <CardContent>
                <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
                  <Box>
                    <Typography variant="subtitle1" fontWeight={600}>{pipeline.sport} - {pipeline.bet_type}</Typography>
                    <Chip size="small" label={pipeline.status} color={pipeline.status === 'complete' ? 'success' : pipeline.status === 'running' ? 'primary' : pipeline.status === 'error' ? 'error' : 'default'} />
                  </Box>
                  <Box>
                    <Button size="small" variant="contained" startIcon={<PlayArrow />} onClick={() => startTraining(pipeline.sport, pipeline.bet_type)} disabled={pipeline.status === 'running'}>Train</Button>
                    <IconButton size="small" onClick={() => setExpandedPipeline(expandedPipeline === `${pipeline.sport}-${pipeline.bet_type}` ? null : `${pipeline.sport}-${pipeline.bet_type}`)}>
                      {expandedPipeline === `${pipeline.sport}-${pipeline.bet_type}` ? <ExpandLess /> : <ExpandMore />}
                    </IconButton>
                  </Box>
                </Box>
                {pipeline.status === 'running' && (
                  <Box mb={2}>
                    <LinearProgress variant="determinate" value={(pipeline.current_step / 8) * 100} />
                    <Typography variant="caption" color="textSecondary">Step {pipeline.current_step + 1}/8: {PIPELINE_STEPS[pipeline.current_step]?.name}</Typography>
                  </Box>
                )}
                <Collapse in={expandedPipeline === `${pipeline.sport}-${pipeline.bet_type}`}>
                  <Stepper activeStep={pipeline.current_step} orientation="vertical">
                    {pipeline.steps.map((step, index) => (
                      <Step key={step.name} completed={step.status === 'complete'}>
                        <StepLabel StepIconComponent={() => getStepIcon(step.status)} error={step.status === 'error'}>
                          <Typography variant="body2" fontWeight={step.status === 'running' ? 600 : 400}>{step.name}</Typography>
                        </StepLabel>
                        <StepContent>
                          <Typography variant="caption" color="textSecondary">{PIPELINE_STEPS[index]?.description}</Typography>
                          {step.status === 'running' && step.progress !== undefined && <LinearProgress variant="determinate" value={step.progress} sx={{ mt: 1 }} />}
                        </StepContent>
                      </Step>
                    ))}
                  </Stepper>
                </Collapse>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      <Typography variant="h6" gutterBottom>Production Models</Typography>
      <Card>
        {loading && <LinearProgress />}
        <TableContainer>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Sport</TableCell>
                <TableCell>Bet Type</TableCell>
                <TableCell>Framework</TableCell>
                <TableCell align="center">Accuracy</TableCell>
                <TableCell align="center">AUC</TableCell>
                <TableCell>Status</TableCell>
                <TableCell>Last Trained</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {models.map((model) => (
                <TableRow key={model.id} hover>
                  <TableCell><Chip label={model.sport} size="small" /></TableCell>
                  <TableCell sx={{ textTransform: 'capitalize' }}>{model.bet_type}</TableCell>
                  <TableCell>{model.framework}</TableCell>
                  <TableCell align="center"><Typography variant="body2" sx={{ color: model.accuracy >= 0.65 ? 'success.main' : model.accuracy >= 0.60 ? 'warning.main' : 'error.main' }}>{formatPercent(model.accuracy)}</Typography></TableCell>
                  <TableCell align="center">{model.auc.toFixed(3)}</TableCell>
                  <TableCell><Chip label={model.is_production ? 'Production' : 'Staging'} size="small" color={model.is_production ? 'success' : 'default'} /></TableCell>
                  <TableCell>{model.training_date}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </Card>
    </Box>
  );
};

const generateDemoModels = (): Model[] => [
  // NBA Models
  { id: '1', sport: 'NBA', bet_type: 'Spread', framework: 'Meta-Ensemble', accuracy: 0.672, auc: 0.712, is_production: true, training_date: '2025-01-15' },
  { id: '2', sport: 'NBA', bet_type: 'Total', framework: 'Meta-Ensemble', accuracy: 0.658, auc: 0.698, is_production: true, training_date: '2025-01-15' },
  { id: '3', sport: 'NBA', bet_type: '1H Spread', framework: 'AutoGluon', accuracy: 0.645, auc: 0.682, is_production: true, training_date: '2025-01-15' },
  { id: '4', sport: 'NBA', bet_type: '1H Total', framework: 'AutoGluon', accuracy: 0.638, auc: 0.675, is_production: true, training_date: '2025-01-15' },
  { id: '5', sport: 'NBA', bet_type: '2H Spread', framework: 'H2O AutoML', accuracy: 0.612, auc: 0.658, is_production: true, training_date: '2025-01-15' },
  { id: '6', sport: 'NBA', bet_type: '2H Total', framework: 'H2O AutoML', accuracy: 0.605, auc: 0.648, is_production: true, training_date: '2025-01-15' },
  // NFL Models
  { id: '7', sport: 'NFL', bet_type: 'Spread', framework: 'Meta-Ensemble', accuracy: 0.665, auc: 0.705, is_production: true, training_date: '2025-01-14' },
  { id: '8', sport: 'NFL', bet_type: 'Total', framework: 'Meta-Ensemble', accuracy: 0.652, auc: 0.692, is_production: true, training_date: '2025-01-14' },
  { id: '9', sport: 'NFL', bet_type: '1H Spread', framework: 'AutoGluon', accuracy: 0.642, auc: 0.678, is_production: true, training_date: '2025-01-14' },
  { id: '10', sport: 'NFL', bet_type: '1H Total', framework: 'AutoGluon', accuracy: 0.635, auc: 0.672, is_production: true, training_date: '2025-01-14' },
  // NHL Models
  { id: '11', sport: 'NHL', bet_type: 'Puck Line', framework: 'Meta-Ensemble', accuracy: 0.648, auc: 0.688, is_production: true, training_date: '2025-01-14' },
  { id: '12', sport: 'NHL', bet_type: 'Total', framework: 'Meta-Ensemble', accuracy: 0.655, auc: 0.695, is_production: true, training_date: '2025-01-14' },
  { id: '13', sport: 'NHL', bet_type: 'Moneyline', framework: 'AutoGluon', accuracy: 0.612, auc: 0.658, is_production: true, training_date: '2025-01-14' },
  { id: '14', sport: 'NHL', bet_type: '1P Spread', framework: 'H2O AutoML', accuracy: 0.598, auc: 0.642, is_production: true, training_date: '2025-01-14' },
  // MLB Models
  { id: '15', sport: 'MLB', bet_type: 'Run Line', framework: 'Meta-Ensemble', accuracy: 0.642, auc: 0.682, is_production: true, training_date: '2025-01-13' },
  { id: '16', sport: 'MLB', bet_type: 'Total', framework: 'Meta-Ensemble', accuracy: 0.658, auc: 0.698, is_production: true, training_date: '2025-01-13' },
  { id: '17', sport: 'MLB', bet_type: 'Moneyline', framework: 'AutoGluon', accuracy: 0.618, auc: 0.662, is_production: true, training_date: '2025-01-13' },
  { id: '18', sport: 'MLB', bet_type: 'F5 Total', framework: 'H2O AutoML', accuracy: 0.645, auc: 0.685, is_production: true, training_date: '2025-01-13' },
  // NCAAB Models
  { id: '19', sport: 'NCAAB', bet_type: 'Spread', framework: 'Meta-Ensemble', accuracy: 0.638, auc: 0.678, is_production: true, training_date: '2025-01-13' },
  { id: '20', sport: 'NCAAB', bet_type: 'Total', framework: 'Meta-Ensemble', accuracy: 0.632, auc: 0.672, is_production: true, training_date: '2025-01-13' },
  { id: '21', sport: 'NCAAB', bet_type: '1H Spread', framework: 'AutoGluon', accuracy: 0.625, auc: 0.665, is_production: true, training_date: '2025-01-13' },
  { id: '22', sport: 'NCAAB', bet_type: '1H Total', framework: 'H2O AutoML', accuracy: 0.618, auc: 0.658, is_production: true, training_date: '2025-01-13' },
  // NCAAF Models
  { id: '23', sport: 'NCAAF', bet_type: 'Spread', framework: 'Meta-Ensemble', accuracy: 0.645, auc: 0.685, is_production: true, training_date: '2025-01-12' },
  { id: '24', sport: 'NCAAF', bet_type: 'Total', framework: 'Meta-Ensemble', accuracy: 0.638, auc: 0.678, is_production: true, training_date: '2025-01-12' },
  { id: '25', sport: 'NCAAF', bet_type: '1H Spread', framework: 'AutoGluon', accuracy: 0.628, auc: 0.668, is_production: true, training_date: '2025-01-12' },
  // WNBA Models
  { id: '26', sport: 'WNBA', bet_type: 'Spread', framework: 'Meta-Ensemble', accuracy: 0.635, auc: 0.675, is_production: true, training_date: '2025-01-12' },
  { id: '27', sport: 'WNBA', bet_type: 'Total', framework: 'AutoGluon', accuracy: 0.628, auc: 0.668, is_production: true, training_date: '2025-01-12' },
  { id: '28', sport: 'WNBA', bet_type: 'Moneyline', framework: 'H2O AutoML', accuracy: 0.608, auc: 0.652, is_production: true, training_date: '2025-01-12' },
  // CFL Models
  { id: '29', sport: 'CFL', bet_type: 'Spread', framework: 'Meta-Ensemble', accuracy: 0.652, auc: 0.692, is_production: true, training_date: '2025-01-11' },
  { id: '30', sport: 'CFL', bet_type: 'Total', framework: 'AutoGluon', accuracy: 0.645, auc: 0.685, is_production: true, training_date: '2025-01-11' },
  { id: '31', sport: 'CFL', bet_type: 'Moneyline', framework: 'H2O AutoML', accuracy: 0.622, auc: 0.665, is_production: true, training_date: '2025-01-11' },
  // ATP Tennis Models
  { id: '32', sport: 'ATP', bet_type: 'Match Winner', framework: 'Meta-Ensemble', accuracy: 0.668, auc: 0.708, is_production: true, training_date: '2025-01-11' },
  { id: '33', sport: 'ATP', bet_type: 'Set Spread', framework: 'AutoGluon', accuracy: 0.642, auc: 0.682, is_production: true, training_date: '2025-01-11' },
  { id: '34', sport: 'ATP', bet_type: 'Total Games', framework: 'H2O AutoML', accuracy: 0.635, auc: 0.675, is_production: true, training_date: '2025-01-11' },
  // WTA Tennis Models
  { id: '35', sport: 'WTA', bet_type: 'Match Winner', framework: 'Meta-Ensemble', accuracy: 0.662, auc: 0.702, is_production: true, training_date: '2025-01-10' },
  { id: '36', sport: 'WTA', bet_type: 'Set Spread', framework: 'AutoGluon', accuracy: 0.638, auc: 0.678, is_production: true, training_date: '2025-01-10' },
  { id: '37', sport: 'WTA', bet_type: 'Total Games', framework: 'H2O AutoML', accuracy: 0.628, auc: 0.668, is_production: true, training_date: '2025-01-10' },
  // Staging Models
  { id: '38', sport: 'NBA', bet_type: 'Spread', framework: 'XGBoost', accuracy: 0.655, auc: 0.695, is_production: false, training_date: '2025-01-16' },
  { id: '39', sport: 'NFL', bet_type: 'Total', framework: 'LightGBM', accuracy: 0.648, auc: 0.688, is_production: false, training_date: '2025-01-16' },
  { id: '40', sport: 'NHL', bet_type: 'Moneyline', framework: 'CatBoost', accuracy: 0.625, auc: 0.668, is_production: false, training_date: '2025-01-16' },
];

const generateDemoPipelines = (): Pipeline[] => [
  { sport: 'NBA', bet_type: 'Spread', status: 'complete', current_step: 8, steps: PIPELINE_STEPS.map((s) => ({ name: s.name, status: 'complete' as const })) },
  { sport: 'NBA', bet_type: 'Total', status: 'running', current_step: 4, steps: PIPELINE_STEPS.map((s, i) => ({ name: s.name, status: i < 4 ? 'complete' as const : i === 4 ? 'running' as const : 'pending' as const, progress: i === 4 ? 65 : undefined })) },
  { sport: 'NFL', bet_type: 'Spread', status: 'complete', current_step: 8, steps: PIPELINE_STEPS.map((s) => ({ name: s.name, status: 'complete' as const })) },
  { sport: 'NFL', bet_type: 'Total', status: 'idle', current_step: 0, steps: PIPELINE_STEPS.map((s) => ({ name: s.name, status: 'pending' as const })) },
  { sport: 'NHL', bet_type: 'Puck Line', status: 'complete', current_step: 8, steps: PIPELINE_STEPS.map((s) => ({ name: s.name, status: 'complete' as const })) },
  { sport: 'NHL', bet_type: 'Total', status: 'idle', current_step: 0, steps: PIPELINE_STEPS.map((s) => ({ name: s.name, status: 'pending' as const })) },
  { sport: 'MLB', bet_type: 'Run Line', status: 'complete', current_step: 8, steps: PIPELINE_STEPS.map((s) => ({ name: s.name, status: 'complete' as const })) },
  { sport: 'NCAAB', bet_type: 'Spread', status: 'running', current_step: 6, steps: PIPELINE_STEPS.map((s, i) => ({ name: s.name, status: i < 6 ? 'complete' as const : i === 6 ? 'running' as const : 'pending' as const, progress: i === 6 ? 35 : undefined })) },
  { sport: 'NCAAF', bet_type: 'Spread', status: 'complete', current_step: 8, steps: PIPELINE_STEPS.map((s) => ({ name: s.name, status: 'complete' as const })) },
  { sport: 'ATP', bet_type: 'Match Winner', status: 'complete', current_step: 8, steps: PIPELINE_STEPS.map((s) => ({ name: s.name, status: 'complete' as const })) },
];

export default Models;
