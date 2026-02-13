// src/pages/Settings/Settings.tsx
import React, { useState } from 'react';
import {
  Box, Card, CardContent, Typography, Grid, Tabs, Tab, TextField, Button,
  Switch, FormControlLabel, Select, MenuItem, FormControl, InputLabel,
  Table, TableBody, TableCell, TableContainer, TableHead, TableRow,
  Chip, LinearProgress, Alert, Divider, IconButton, Dialog, DialogTitle,
  DialogContent, DialogActions, List, ListItem, ListItemText, Paper,
  Radio, RadioGroup
} from '@mui/material';
import { Palette, Casino, Notifications, Psychology, Storage, Security, Api, Save, Telegram, Email, Visibility, VisibilityOff, Refresh, CheckCircle, Warning, PlayArrow, Add, Delete, Send, AccountBalance, TrendingUp, Shield, SportsSoccer, Memory, Speed, Tune, CalendarMonth, Timeline, Science, Hub } from '@mui/icons-material';
import { useSettingsStore, useBettingStore, useMLConfigStore } from '../../store';
import { TIMEZONES } from '../../types';

interface TabPanelProps { children?: React.ReactNode; index: number; value: number; }
const TabPanel = ({ children, value, index }: TabPanelProps) => <Box hidden={value !== index} sx={{ py: 2, px: 4 }}>{value === index && children}</Box>;

interface TelegramAccount { id: string; name: string; chatId: string; enabled: boolean; }
interface EmailAccount { id: string; email: string; enabled: boolean; }

const Settings: React.FC = () => {
  const [tab, setTab] = useState(0);
  const { theme, setTheme, oddsFormat, setOddsFormat, timezone, setTimezone, timeFormat, setTimeFormat } = useSettingsStore();
  const betting = useBettingStore();
  const ml = useMLConfigStore();
  const [telegramToken, setTelegramToken] = useState('');
  const [showToken, setShowToken] = useState(false);
  
  // Multiple Telegram/Email accounts
  const [telegramAccounts, setTelegramAccounts] = useState<TelegramAccount[]>([
    { id: '1', name: 'Main Account', chatId: '123456789', enabled: true },
  ]);
  const [emailAccounts, setEmailAccounts] = useState<EmailAccount[]>([
    { id: '1', email: 'main@example.com', enabled: true },
  ]);
  const [telegramDialog, setTelegramDialog] = useState(false);
  const [emailDialog, setEmailDialog] = useState(false);
  const [newTelegram, setNewTelegram] = useState({ name: '', chatId: '' });
  const [newEmail, setNewEmail] = useState('');

  const handleSave = () => alert('Settings saved!');
  const testTelegram = (chatId: string) => alert(`Telegram test sent to: ${chatId}`);
  const testEmail = (email: string) => alert(`Email test sent to: ${email}`);
  
  const addTelegramAccount = () => {
    if (newTelegram.name && newTelegram.chatId) {
      setTelegramAccounts([...telegramAccounts, { id: Date.now().toString(), ...newTelegram, enabled: true }]);
      setNewTelegram({ name: '', chatId: '' });
      setTelegramDialog(false);
    }
  };
  const removeTelegramAccount = (id: string) => setTelegramAccounts(telegramAccounts.filter(t => t.id !== id));
  const toggleTelegramAccount = (id: string) => setTelegramAccounts(telegramAccounts.map(t => t.id === id ? { ...t, enabled: !t.enabled } : t));
  
  const addEmailAccount = () => {
    if (newEmail && newEmail.includes('@')) {
      setEmailAccounts([...emailAccounts, { id: Date.now().toString(), email: newEmail, enabled: true }]);
      setNewEmail('');
      setEmailDialog(false);
    }
  };
  const removeEmailAccount = (id: string) => setEmailAccounts(emailAccounts.filter(e => e.id !== id));
  const toggleEmailAccount = (id: string) => setEmailAccounts(emailAccounts.map(e => e.id === id ? { ...e, enabled: !e.enabled } : e));

  return (
    <Box>
      <Typography variant="h5" fontWeight={700} sx={{ fontSize: 20 }} mb={2}>Settings</Typography>
      <Card>
        <Box sx={{ borderBottom: 1, borderColor: 'divider', px: 2 }}>
          <Tabs value={tab} onChange={(_, v) => setTab(v)} variant="scrollable" scrollButtons="auto">
            <Tab icon={<Palette sx={{ fontSize: 18 }} />} label="Appearance" iconPosition="start" sx={{ fontSize: 13, minHeight: 48 }} />
            <Tab icon={<Casino sx={{ fontSize: 18 }} />} label="Betting" iconPosition="start" sx={{ fontSize: 13, minHeight: 48 }} />
            <Tab icon={<Notifications sx={{ fontSize: 18 }} />} label="Notifications" iconPosition="start" sx={{ fontSize: 13, minHeight: 48 }} />
            <Tab icon={<Psychology sx={{ fontSize: 18 }} />} label="ML Models" iconPosition="start" sx={{ fontSize: 13, minHeight: 48 }} />
            <Tab icon={<Storage sx={{ fontSize: 18 }} />} label="Data Collection" iconPosition="start" sx={{ fontSize: 13, minHeight: 48 }} />
            <Tab icon={<Api sx={{ fontSize: 18 }} />} label="Data Pipeline" iconPosition="start" sx={{ fontSize: 13, minHeight: 48 }} />
            <Tab icon={<Security sx={{ fontSize: 18 }} />} label="Security" iconPosition="start" sx={{ fontSize: 13, minHeight: 48 }} />
          </Tabs>
        </Box>

        <TabPanel value={tab} index={0}>
          <Typography variant="subtitle1" fontWeight={600} sx={{ fontSize: 15, mb: 2 }}>Display Preferences</Typography>
          <Grid container spacing={2.5}>
            <Grid item xs={12} sm={6}><FormControl fullWidth size="small"><InputLabel sx={{ fontSize: 13 }}>Theme</InputLabel><Select value={theme} label="Theme" onChange={(e) => setTheme(e.target.value as 'light' | 'dark')} sx={{ fontSize: 13 }}><MenuItem value="dark" sx={{ fontSize: 13 }}>Dark</MenuItem><MenuItem value="light" sx={{ fontSize: 13 }}>Light</MenuItem></Select></FormControl></Grid>
            <Grid item xs={12} sm={6}><FormControl fullWidth size="small"><InputLabel sx={{ fontSize: 13 }}>Time Format</InputLabel><Select value={timeFormat} label="Time Format" onChange={(e) => setTimeFormat(e.target.value as '12h' | '24h')} sx={{ fontSize: 13 }}><MenuItem value="12h" sx={{ fontSize: 13 }}>12-hour (3:30 PM)</MenuItem><MenuItem value="24h" sx={{ fontSize: 13 }}>24-hour (15:30)</MenuItem></Select></FormControl></Grid>
            <Grid item xs={12} sm={6}><FormControl fullWidth size="small"><InputLabel sx={{ fontSize: 13 }}>Odds Format</InputLabel><Select value={oddsFormat} label="Odds Format" onChange={(e) => setOddsFormat(e.target.value as 'american' | 'decimal' | 'fractional')} sx={{ fontSize: 13 }}><MenuItem value="american" sx={{ fontSize: 13 }}>American (-110)</MenuItem><MenuItem value="decimal" sx={{ fontSize: 13 }}>Decimal (1.91)</MenuItem><MenuItem value="fractional" sx={{ fontSize: 13 }}>Fractional (10/11)</MenuItem></Select></FormControl></Grid>
            <Grid item xs={12} sm={6}><FormControl fullWidth size="small"><InputLabel sx={{ fontSize: 13 }}>Timezone</InputLabel><Select value={timezone} label="Timezone" onChange={(e) => setTimezone(e.target.value)} sx={{ fontSize: 13 }}>{TIMEZONES.map(tz => <MenuItem key={tz.value} value={tz.value} sx={{ fontSize: 13 }}>{tz.label}</MenuItem>)}</Select></FormControl></Grid>
          </Grid>
        </TabPanel>

        <TabPanel value={tab} index={1}>
          {/* ── Section 1: Bankroll ── */}
          <Box sx={{ mb: 3 }}>
            <Box display="flex" alignItems="center" gap={1} mb={1.5}>
              <AccountBalance sx={{ fontSize: 18, color: 'primary.main' }} />
              <Typography variant="subtitle1" fontWeight={600} sx={{ fontSize: 15 }}>Bankroll</Typography>
            </Box>
            <Grid container spacing={2}>
              <Grid item xs={12} sm={6}>
                <TextField fullWidth size="small" label="Initial Bankroll ($)" type="number"
                  value={betting.initialBankroll}
                  onChange={(e) => betting.setBetting({ initialBankroll: Math.max(100, parseInt(e.target.value) || 10000) })}
                  inputProps={{ min: 100, step: 500 }}
                  InputLabelProps={{ sx: { fontSize: 13 } }}
                  sx={{ '& input': { fontSize: 13 } }}
                  helperText="Starting capital for tracking P/L and bankroll growth"
                  FormHelperTextProps={{ sx: { fontSize: 11 } }}
                />
              </Grid>
            </Grid>
          </Box>

          <Divider sx={{ my: 2 }} />

          {/* ── Section 2: Bet Sizing ── */}
          <Box sx={{ mb: 3 }}>
            <Box display="flex" alignItems="center" gap={1} mb={1.5}>
              <TrendingUp sx={{ fontSize: 18, color: 'primary.main' }} />
              <Typography variant="subtitle1" fontWeight={600} sx={{ fontSize: 15 }}>Bet Sizing Strategy</Typography>
            </Box>

            <FormControl component="fieldset" sx={{ mb: 2 }}>
              <RadioGroup value={betting.betSizing} onChange={(e) => betting.setBetting({ betSizing: e.target.value as 'flat' | 'percentage' | 'kelly' })}>
                <FormControlLabel value="flat" control={<Radio size="small" />}
                  label={<Typography sx={{ fontSize: 13 }}>Flat — Same dollar amount for every bet</Typography>} />
                <FormControlLabel value="percentage" control={<Radio size="small" />}
                  label={<Typography sx={{ fontSize: 13 }}>Percentage — Fixed % of current bankroll per bet</Typography>} />
                <FormControlLabel value="kelly" control={<Radio size="small" />}
                  label={<Typography sx={{ fontSize: 13 }}>Kelly Criterion — Optimal sizing based on edge & odds</Typography>} />
              </RadioGroup>
            </FormControl>

            <Grid container spacing={2}>
              {betting.betSizing === 'flat' && (
                <Grid item xs={12} sm={6}>
                  <TextField fullWidth size="small" label="Flat Bet Amount ($)" type="number"
                    value={betting.flatAmount}
                    onChange={(e) => betting.setBetting({ flatAmount: Math.max(1, parseInt(e.target.value) || 100) })}
                    inputProps={{ min: 1, step: 10 }}
                    InputLabelProps={{ sx: { fontSize: 13 } }}
                    sx={{ '& input': { fontSize: 13 } }}
                  />
                </Grid>
              )}
              {betting.betSizing === 'percentage' && (
                <Grid item xs={12} sm={6}>
                  <TextField fullWidth size="small" label="Bankroll % per Bet" type="number"
                    value={betting.percentageAmount}
                    onChange={(e) => betting.setBetting({ percentageAmount: Math.min(25, Math.max(0.5, parseFloat(e.target.value) || 2)) })}
                    inputProps={{ min: 0.5, max: 25, step: 0.5 }}
                    InputLabelProps={{ sx: { fontSize: 13 } }}
                    sx={{ '& input': { fontSize: 13 } }}
                    helperText={`Current stake: $${Math.round(betting.initialBankroll * betting.percentageAmount / 100)}`}
                    FormHelperTextProps={{ sx: { fontSize: 11 } }}
                  />
                </Grid>
              )}
              {betting.betSizing === 'kelly' && (
                <Grid item xs={12} sm={6}>
                  <FormControl fullWidth size="small">
                    <InputLabel sx={{ fontSize: 13 }}>Kelly Fraction</InputLabel>
                    <Select value={betting.kellyFraction} label="Kelly Fraction"
                      onChange={(e) => betting.setBetting({ kellyFraction: e.target.value as 'quarter' | 'half' | 'full' })}
                      sx={{ fontSize: 13 }}>
                      <MenuItem value="quarter" sx={{ fontSize: 13 }}>Quarter Kelly (25%) — Conservative</MenuItem>
                      <MenuItem value="half" sx={{ fontSize: 13 }}>Half Kelly (50%) — Balanced</MenuItem>
                      <MenuItem value="full" sx={{ fontSize: 13 }}>Full Kelly (100%) — Aggressive</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
              )}
            </Grid>
          </Box>

          <Divider sx={{ my: 2 }} />

          {/* ── Section 3: Tier Filters ── */}
          <Box sx={{ mb: 3 }}>
            <Box display="flex" alignItems="center" gap={1} mb={1.5}>
              <Casino sx={{ fontSize: 18, color: 'primary.main' }} />
              <Typography variant="subtitle1" fontWeight={600} sx={{ fontSize: 15 }}>Prediction Tiers to Bet</Typography>
            </Box>
            <Typography variant="body2" color="text.secondary" sx={{ fontSize: 12, mb: 1.5 }}>
              Only predictions matching enabled tiers will be included in your betting portfolio.
            </Typography>
            <Grid container spacing={1}>
              <Grid item xs={6} sm={3}>
                <Paper variant="outlined" sx={{ p: 1.5, textAlign: 'center', borderColor: betting.tierA ? 'success.main' : 'divider', bgcolor: betting.tierA ? 'rgba(76, 175, 80, 0.08)' : 'transparent' }}>
                  <FormControlLabel control={<Switch size="small" checked={betting.tierA} onChange={(e) => betting.setBetting({ tierA: e.target.checked })} color="success" />}
                    label={<Typography sx={{ fontSize: 13, fontWeight: 600 }}>Tier A</Typography>} sx={{ m: 0 }} />
                  <Typography sx={{ fontSize: 11, color: 'text.secondary' }}>58%+ confidence</Typography>
                </Paper>
              </Grid>
              <Grid item xs={6} sm={3}>
                <Paper variant="outlined" sx={{ p: 1.5, textAlign: 'center', borderColor: betting.tierB ? 'info.main' : 'divider', bgcolor: betting.tierB ? 'rgba(33, 150, 243, 0.08)' : 'transparent' }}>
                  <FormControlLabel control={<Switch size="small" checked={betting.tierB} onChange={(e) => betting.setBetting({ tierB: e.target.checked })} color="info" />}
                    label={<Typography sx={{ fontSize: 13, fontWeight: 600 }}>Tier B</Typography>} sx={{ m: 0 }} />
                  <Typography sx={{ fontSize: 11, color: 'text.secondary' }}>55–58% confidence</Typography>
                </Paper>
              </Grid>
              <Grid item xs={6} sm={3}>
                <Paper variant="outlined" sx={{ p: 1.5, textAlign: 'center', borderColor: betting.tierC ? 'warning.main' : 'divider', bgcolor: betting.tierC ? 'rgba(255, 152, 0, 0.08)' : 'transparent' }}>
                  <FormControlLabel control={<Switch size="small" checked={betting.tierC} onChange={(e) => betting.setBetting({ tierC: e.target.checked })} color="warning" />}
                    label={<Typography sx={{ fontSize: 13, fontWeight: 600 }}>Tier C</Typography>} sx={{ m: 0 }} />
                  <Typography sx={{ fontSize: 11, color: 'text.secondary' }}>52–55% confidence</Typography>
                </Paper>
              </Grid>
              <Grid item xs={6} sm={3}>
                <Paper variant="outlined" sx={{ p: 1.5, textAlign: 'center', opacity: 0.5 }}>
                  <FormControlLabel control={<Switch size="small" disabled checked={false} />}
                    label={<Typography sx={{ fontSize: 13, fontWeight: 600 }}>Tier D</Typography>} sx={{ m: 0 }} />
                  <Typography sx={{ fontSize: 11, color: 'text.secondary' }}>&lt;52% — disabled</Typography>
                </Paper>
              </Grid>
            </Grid>
          </Box>

          <Divider sx={{ my: 2 }} />

          {/* ── Section 4: Bet Types ── */}
          <Box sx={{ mb: 3 }}>
            <Box display="flex" alignItems="center" gap={1} mb={1.5}>
              <SportsSoccer sx={{ fontSize: 18, color: 'primary.main' }} />
              <Typography variant="subtitle1" fontWeight={600} sx={{ fontSize: 15 }}>Bet Types</Typography>
            </Box>
            <Typography variant="body2" color="text.secondary" sx={{ fontSize: 12, mb: 1.5 }}>
              Select which bet types to include. Disabling a type excludes those predictions entirely.
            </Typography>
            <Box display="flex" gap={3} flexWrap="wrap">
              <FormControlLabel control={<Switch size="small" checked={betting.betSpread} onChange={(e) => betting.setBetting({ betSpread: e.target.checked })} />}
                label={<Typography sx={{ fontSize: 13 }}>Spread</Typography>} />
              <FormControlLabel control={<Switch size="small" checked={betting.betMoneyline} onChange={(e) => betting.setBetting({ betMoneyline: e.target.checked })} />}
                label={<Typography sx={{ fontSize: 13 }}>Moneyline</Typography>} />
              <FormControlLabel control={<Switch size="small" checked={betting.betTotal} onChange={(e) => betting.setBetting({ betTotal: e.target.checked })} />}
                label={<Typography sx={{ fontSize: 13 }}>Totals (O/U)</Typography>} />
            </Box>
          </Box>

          <Divider sx={{ my: 2 }} />

          {/* ── Section 5: Risk Controls ── */}
          <Box sx={{ mb: 3 }}>
            <Box display="flex" alignItems="center" gap={1} mb={1.5}>
              <Shield sx={{ fontSize: 18, color: 'primary.main' }} />
              <Typography variant="subtitle1" fontWeight={600} sx={{ fontSize: 15 }}>Risk Controls</Typography>
            </Box>
            <Grid container spacing={2}>
              <Grid item xs={12} sm={6}>
                <TextField fullWidth size="small" label="Minimum Edge (%)" type="number"
                  value={betting.minEdge}
                  onChange={(e) => betting.setBetting({ minEdge: Math.max(0, parseFloat(e.target.value) || 1) })}
                  inputProps={{ min: 0, max: 20, step: 0.5 }}
                  InputLabelProps={{ sx: { fontSize: 13 } }}
                  sx={{ '& input': { fontSize: 13 } }}
                  helperText="Skip predictions with edge below this threshold"
                  FormHelperTextProps={{ sx: { fontSize: 11 } }}
                />
              </Grid>
              <Grid item xs={12} sm={6}>
                <TextField fullWidth size="small" label="Max Daily Bets" type="number"
                  value={betting.maxDailyBets}
                  onChange={(e) => betting.setBetting({ maxDailyBets: Math.max(1, parseInt(e.target.value) || 20) })}
                  inputProps={{ min: 1, max: 100 }}
                  InputLabelProps={{ sx: { fontSize: 13 } }}
                  sx={{ '& input': { fontSize: 13 } }}
                  helperText="Cap exposure per day"
                  FormHelperTextProps={{ sx: { fontSize: 11 } }}
                />
              </Grid>
              <Grid item xs={12} sm={6}>
                <TextField fullWidth size="small" label="Min Odds (American)" type="number"
                  value={betting.minOdds}
                  onChange={(e) => betting.setBetting({ minOdds: parseInt(e.target.value) || -500 })}
                  InputLabelProps={{ sx: { fontSize: 13 } }}
                  sx={{ '& input': { fontSize: 13 } }}
                  helperText="e.g. -500 to avoid heavy favorites"
                  FormHelperTextProps={{ sx: { fontSize: 11 } }}
                />
              </Grid>
              <Grid item xs={12} sm={6}>
                <TextField fullWidth size="small" label="Max Odds (American)" type="number"
                  value={betting.maxOdds}
                  onChange={(e) => betting.setBetting({ maxOdds: parseInt(e.target.value) || 500 })}
                  InputLabelProps={{ sx: { fontSize: 13 } }}
                  sx={{ '& input': { fontSize: 13 } }}
                  helperText="e.g. +500 to avoid long shots"
                  FormHelperTextProps={{ sx: { fontSize: 11 } }}
                />
              </Grid>
            </Grid>
          </Box>

          <Divider sx={{ my: 2 }} />

          {/* ── Section 6: Auto-Bet Toggle ── */}
          <Box>
            <FormControlLabel
              control={<Switch checked={betting.autoBet} onChange={(e) => betting.setBetting({ autoBet: e.target.checked })} />}
              label={
                <Box>
                  <Typography sx={{ fontSize: 13, fontWeight: 500 }}>Auto-place bets (simulation mode)</Typography>
                  <Typography sx={{ fontSize: 11, color: 'text.secondary' }}>Automatically track all qualifying predictions as placed bets for P/L tracking.</Typography>
                </Box>
              }
            />
          </Box>
        </TabPanel>

        <TabPanel value={tab} index={2}>
          <Grid container spacing={2}>
            {/* Telegram Section */}
            <Grid item xs={12}>
              <Box display="flex" justifyContent="space-between" alignItems="center" mb={1.5}>
                <Typography variant="subtitle1" fontWeight={600}><Telegram sx={{ verticalAlign: 'middle', mr: 1, fontSize: 18 }} />Telegram Bot <Chip label={`${telegramAccounts.length} accounts`} size="small" sx={{ ml: 1, fontSize: 12, height: 20 }} /></Typography>
                <Button variant="outlined" size="small" startIcon={<Add />} onClick={() => setTelegramDialog(true)} sx={{ fontSize: 13 }}>Add Telegram</Button>
              </Box>
              <TextField fullWidth size="small" label="Bot Token" type={showToken ? 'text' : 'password'} value={telegramToken} onChange={(e) => setTelegramToken(e.target.value)} placeholder="Get from @BotFather" sx={{ mb: 1.5 }} InputProps={{ endAdornment: <IconButton size="small" onClick={() => setShowToken(!showToken)}>{showToken ? <VisibilityOff fontSize="small" /> : <Visibility fontSize="small" />}</IconButton> }} />
              <Paper variant="outlined" sx={{ mb: 2 }}>
                <List disablePadding>
                  {telegramAccounts.map((acc, idx) => (
                    <ListItem key={acc.id} divider={idx < telegramAccounts.length - 1} sx={{ py: 1 }}>
                      <Switch checked={acc.enabled} onChange={() => toggleTelegramAccount(acc.id)} size="small" sx={{ mr: 1 }} />
                      <ListItemText primary={<Typography variant="body2">{acc.name}</Typography>} secondary={<Typography variant="caption">{`Chat ID: ${acc.chatId}`}</Typography>} sx={{ flex: 1 }} />
                      <Box display="flex" alignItems="center" gap={1} sx={{ ml: 2 }}>
                        {acc.enabled && <Chip label="Active" size="small" color="success" sx={{ fontSize: 12, height: 20 }} />}
                        <IconButton size="small" onClick={() => testTelegram(acc.chatId)} color="primary" title="Send Test"><Send sx={{ fontSize: 16 }} /></IconButton>
                        <IconButton size="small" onClick={() => removeTelegramAccount(acc.id)} color="error" title="Delete"><Delete sx={{ fontSize: 16 }} /></IconButton>
                      </Box>
                    </ListItem>
                  ))}
                  {telegramAccounts.length === 0 && <ListItem><ListItemText primary={<Typography variant="body2">No Telegram accounts</Typography>} secondary={<Typography variant="caption">Click 'Add Telegram' to add</Typography>} /></ListItem>}
                </List>
              </Paper>
            </Grid>
            
            {/* Email Section */}
            <Grid item xs={12}>
              <Box display="flex" justifyContent="space-between" alignItems="center" mb={1.5}>
                <Typography variant="subtitle1" fontWeight={600}><Email sx={{ verticalAlign: 'middle', mr: 1, fontSize: 18 }} />Email Notifications <Chip label={`${emailAccounts.length} addresses`} size="small" sx={{ ml: 1, fontSize: 12, height: 20 }} /></Typography>
                <Button variant="outlined" size="small" startIcon={<Add />} onClick={() => setEmailDialog(true)} sx={{ fontSize: 13 }}>Add Email</Button>
              </Box>
              <Paper variant="outlined" sx={{ mb: 2 }}>
                <List disablePadding>
                  {emailAccounts.map((acc, idx) => (
                    <ListItem key={acc.id} divider={idx < emailAccounts.length - 1} sx={{ py: 1 }}>
                      <Switch checked={acc.enabled} onChange={() => toggleEmailAccount(acc.id)} size="small" sx={{ mr: 1 }} />
                      <ListItemText primary={<Typography variant="body2">{acc.email}</Typography>} sx={{ flex: 1 }} />
                      <Box display="flex" alignItems="center" gap={1} sx={{ ml: 2 }}>
                        {acc.enabled && <Chip label="Active" size="small" color="success" sx={{ fontSize: 12, height: 20 }} />}
                        <IconButton size="small" onClick={() => testEmail(acc.email)} color="primary" title="Send Test"><Send sx={{ fontSize: 16 }} /></IconButton>
                        <IconButton size="small" onClick={() => removeEmailAccount(acc.id)} color="error" title="Delete"><Delete sx={{ fontSize: 16 }} /></IconButton>
                      </Box>
                    </ListItem>
                  ))}
                  {emailAccounts.length === 0 && <ListItem><ListItemText primary={<Typography variant="body2">No email addresses</Typography>} secondary={<Typography variant="caption">Click 'Add Email' to add</Typography>} /></ListItem>}
                </List>
              </Paper>
            </Grid>
            
            {/* Notification Preferences */}
            <Grid item xs={12}><Typography variant="subtitle1" fontWeight={600} gutterBottom>Notification Preferences</Typography><TableContainer><Table size="small"><TableHead><TableRow><TableCell sx={{ fontSize: 13, fontWeight: 600 }}>Event</TableCell><TableCell align="center" sx={{ fontSize: 13, fontWeight: 600 }}>Telegram</TableCell><TableCell align="center" sx={{ fontSize: 13, fontWeight: 600 }}>Email</TableCell></TableRow></TableHead><TableBody><TableRow><TableCell sx={{ fontSize: 13 }}>Tier A Predictions (65%+)</TableCell><TableCell align="center"><Switch defaultChecked size="small" /></TableCell><TableCell align="center"><Switch defaultChecked size="small" /></TableCell></TableRow><TableRow><TableCell sx={{ fontSize: 13 }}>Tier B Predictions (60-65%)</TableCell><TableCell align="center"><Switch defaultChecked size="small" /></TableCell><TableCell align="center"><Switch size="small" /></TableCell></TableRow><TableRow><TableCell sx={{ fontSize: 13 }}>Tier C Predictions (55-60%)</TableCell><TableCell align="center"><Switch size="small" /></TableCell><TableCell align="center"><Switch size="small" /></TableCell></TableRow><TableRow><TableCell sx={{ fontSize: 13 }}>Tier D Predictions (&lt;55%)</TableCell><TableCell align="center"><Switch size="small" /></TableCell><TableCell align="center"><Switch size="small" /></TableCell></TableRow><TableRow><TableCell sx={{ fontSize: 13 }}>System Errors</TableCell><TableCell align="center"><Switch defaultChecked size="small" /></TableCell><TableCell align="center"><Switch defaultChecked size="small" /></TableCell></TableRow><TableRow><TableCell sx={{ fontSize: 13 }}>Model Training Complete</TableCell><TableCell align="center"><Switch defaultChecked size="small" /></TableCell><TableCell align="center"><Switch size="small" /></TableCell></TableRow><TableRow><TableCell sx={{ fontSize: 13 }}>Daily Summary</TableCell><TableCell align="center"><Switch size="small" /></TableCell><TableCell align="center"><Switch defaultChecked size="small" /></TableCell></TableRow><TableRow><TableCell sx={{ fontSize: 13 }}>Live Game Alerts</TableCell><TableCell align="center"><Switch defaultChecked size="small" /></TableCell><TableCell align="center"><Switch size="small" /></TableCell></TableRow></TableBody></Table></TableContainer></Grid>
          </Grid>
        </TabPanel>

        <TabPanel value={tab} index={3}>
          {/* ── Section 1: Active Frameworks ── */}
          <Box sx={{ mb: 3 }}>
            <Box display="flex" alignItems="center" gap={1} mb={1.5}>
              <Hub sx={{ fontSize: 18, color: 'primary.main' }} />
              <Typography variant="subtitle1" fontWeight={600} sx={{ fontSize: 15 }}>Active Frameworks</Typography>
            </Box>
            <Typography variant="body2" color="text.secondary" sx={{ fontSize: 12, mb: 1.5 }}>
              Enable/disable frameworks for the meta-ensemble. Predictions combine all active frameworks weighted by performance.
            </Typography>
            <Grid container spacing={1}>
              {([
                { key: 'frameworkH2o', label: 'H2O AutoML', desc: 'GBM, XGBoost, GLM, DRF, Stacked', color: '#2196f3' },
                { key: 'frameworkSklearn', label: 'Sklearn', desc: 'LightGBM, XGBoost, RF, Logistic', color: '#4caf50' },
                { key: 'frameworkAutogluon', label: 'AutoGluon', desc: 'Multi-layer stacking, bagging', color: '#ff9800' },
                { key: 'frameworkDeepLearning', label: 'Deep Learning', desc: 'Neural networks (experimental)', color: '#9c27b0' },
                { key: 'frameworkQuantum', label: 'Quantum ML', desc: 'PennyLane hybrid circuits', color: '#e91e63' },
              ] as const).map(fw => (
                <Grid item xs={6} sm={4} md={2.4} key={fw.key}>
                  <Paper variant="outlined" sx={{ p: 1.5, textAlign: 'center', borderColor: (ml as any)[fw.key] ? fw.color : 'divider', bgcolor: (ml as any)[fw.key] ? `${fw.color}12` : 'transparent', borderWidth: (ml as any)[fw.key] ? 2 : 1 }}>
                    <FormControlLabel control={<Switch size="small" checked={(ml as any)[fw.key]} onChange={(e) => ml.setMLConfig({ [fw.key]: e.target.checked } as any)} sx={{ '& .MuiSwitch-switchBase.Mui-checked': { color: fw.color } }} />}
                      label={<Typography sx={{ fontSize: 12, fontWeight: 600 }}>{fw.label}</Typography>} sx={{ m: 0 }} />
                    <Typography sx={{ fontSize: 10, color: 'text.secondary', mt: 0.5 }}>{fw.desc}</Typography>
                  </Paper>
                </Grid>
              ))}
            </Grid>
          </Box>

          <Divider sx={{ my: 2 }} />

          {/* ── Section 2: H2O AutoML ── */}
          <Box sx={{ mb: 3 }}>
            <Box display="flex" alignItems="center" gap={1} mb={1.5}>
              <Memory sx={{ fontSize: 18, color: 'primary.main' }} />
              <Typography variant="subtitle1" fontWeight={600} sx={{ fontSize: 15 }}>H2O AutoML</Typography>
              {!ml.frameworkH2o && <Chip label="Disabled" size="small" sx={{ fontSize: 10, height: 20 }} />}
            </Box>
            <Grid container spacing={2}>
              <Grid item xs={12} sm={4}>
                <TextField fullWidth size="small" label="Max Memory" value={ml.h2oMaxMem}
                  onChange={(e) => ml.setMLConfig({ h2oMaxMem: e.target.value })}
                  InputLabelProps={{ sx: { fontSize: 13 } }} sx={{ '& input': { fontSize: 13 } }}
                  helperText="e.g. 16g, 32g, 64g" FormHelperTextProps={{ sx: { fontSize: 11 } }} />
              </Grid>
              <Grid item xs={12} sm={4}>
                <TextField fullWidth size="small" label="Max Models" type="number" value={ml.h2oMaxModels}
                  onChange={(e) => ml.setMLConfig({ h2oMaxModels: Math.max(5, parseInt(e.target.value) || 50) })}
                  inputProps={{ min: 5, max: 200 }}
                  InputLabelProps={{ sx: { fontSize: 13 } }} sx={{ '& input': { fontSize: 13 } }}
                  helperText="Models to explore per run" FormHelperTextProps={{ sx: { fontSize: 11 } }} />
              </Grid>
              <Grid item xs={12} sm={4}>
                <TextField fullWidth size="small" label="Max Runtime (sec)" type="number" value={ml.h2oMaxRuntime}
                  onChange={(e) => ml.setMLConfig({ h2oMaxRuntime: Math.max(60, parseInt(e.target.value) || 3600) })}
                  inputProps={{ min: 60, step: 300 }}
                  InputLabelProps={{ sx: { fontSize: 13 } }} sx={{ '& input': { fontSize: 13 } }}
                  helperText={`${Math.round(ml.h2oMaxRuntime / 60)} minutes`} FormHelperTextProps={{ sx: { fontSize: 11 } }} />
              </Grid>
              <Grid item xs={12} sm={4}>
                <TextField fullWidth size="small" label="CV Folds" type="number" value={ml.h2oNfolds}
                  onChange={(e) => ml.setMLConfig({ h2oNfolds: Math.min(10, Math.max(2, parseInt(e.target.value) || 5)) })}
                  inputProps={{ min: 2, max: 10 }}
                  InputLabelProps={{ sx: { fontSize: 13 } }} sx={{ '& input': { fontSize: 13 } }} />
              </Grid>
              <Grid item xs={12} sm={4}>
                <TextField fullWidth size="small" label="Random Seed" type="number" value={ml.h2oSeed}
                  onChange={(e) => ml.setMLConfig({ h2oSeed: parseInt(e.target.value) || 42 })}
                  InputLabelProps={{ sx: { fontSize: 13 } }} sx={{ '& input': { fontSize: 13 } }} />
              </Grid>
            </Grid>
          </Box>

          <Divider sx={{ my: 2 }} />

          {/* ── Section 3: Sklearn ── */}
          <Box sx={{ mb: 3 }}>
            <Box display="flex" alignItems="center" gap={1} mb={1.5}>
              <Science sx={{ fontSize: 18, color: 'primary.main' }} />
              <Typography variant="subtitle1" fontWeight={600} sx={{ fontSize: 15 }}>Sklearn / LightGBM</Typography>
              {!ml.frameworkSklearn && <Chip label="Disabled" size="small" sx={{ fontSize: 10, height: 20 }} />}
            </Box>
            <Grid container spacing={2}>
              <Grid item xs={12} sm={4}>
                <TextField fullWidth size="small" label="N Estimators (trees)" type="number" value={ml.sklearnEstimators}
                  onChange={(e) => ml.setMLConfig({ sklearnEstimators: Math.max(50, parseInt(e.target.value) || 200) })}
                  inputProps={{ min: 50, max: 2000, step: 50 }}
                  InputLabelProps={{ sx: { fontSize: 13 } }} sx={{ '& input': { fontSize: 13 } }}
                  helperText="More trees = slower but better" FormHelperTextProps={{ sx: { fontSize: 11 } }} />
              </Grid>
              <Grid item xs={12} sm={4}>
                <TextField fullWidth size="small" label="Max Depth" type="number" value={ml.sklearnMaxDepth}
                  onChange={(e) => ml.setMLConfig({ sklearnMaxDepth: Math.min(20, Math.max(2, parseInt(e.target.value) || 8)) })}
                  inputProps={{ min: 2, max: 20 }}
                  InputLabelProps={{ sx: { fontSize: 13 } }} sx={{ '& input': { fontSize: 13 } }} />
              </Grid>
              <Grid item xs={12} sm={4}>
                <TextField fullWidth size="small" label="Learning Rate" type="number" value={ml.sklearnLearningRate}
                  onChange={(e) => ml.setMLConfig({ sklearnLearningRate: Math.min(1, Math.max(0.001, parseFloat(e.target.value) || 0.05)) })}
                  inputProps={{ min: 0.001, max: 1, step: 0.01 }}
                  InputLabelProps={{ sx: { fontSize: 13 } }} sx={{ '& input': { fontSize: 13 } }} />
              </Grid>
              <Grid item xs={12} sm={4}>
                <TextField fullWidth size="small" label="CV Folds" type="number" value={ml.sklearnCvFolds}
                  onChange={(e) => ml.setMLConfig({ sklearnCvFolds: Math.min(10, Math.max(2, parseInt(e.target.value) || 3)) })}
                  inputProps={{ min: 2, max: 10 }}
                  InputLabelProps={{ sx: { fontSize: 13 } }} sx={{ '& input': { fontSize: 13 } }} />
              </Grid>
            </Grid>
          </Box>

          <Divider sx={{ my: 2 }} />

          {/* ── Section 4: AutoGluon ── */}
          <Box sx={{ mb: 3 }}>
            <Box display="flex" alignItems="center" gap={1} mb={1.5}>
              <Tune sx={{ fontSize: 18, color: 'primary.main' }} />
              <Typography variant="subtitle1" fontWeight={600} sx={{ fontSize: 15 }}>AutoGluon</Typography>
              {!ml.frameworkAutogluon && <Chip label="Disabled" size="small" sx={{ fontSize: 10, height: 20 }} />}
            </Box>
            <Grid container spacing={2}>
              <Grid item xs={12} sm={4}>
                <FormControl fullWidth size="small">
                  <InputLabel sx={{ fontSize: 13 }}>Presets</InputLabel>
                  <Select value={ml.autogluonPresets} label="Presets"
                    onChange={(e) => ml.setMLConfig({ autogluonPresets: e.target.value as any })}
                    sx={{ fontSize: 13 }}>
                    <MenuItem value="best_quality" sx={{ fontSize: 13 }}>Best Quality (slowest)</MenuItem>
                    <MenuItem value="high_quality" sx={{ fontSize: 13 }}>High Quality</MenuItem>
                    <MenuItem value="good_quality" sx={{ fontSize: 13 }}>Good Quality</MenuItem>
                    <MenuItem value="medium_quality" sx={{ fontSize: 13 }}>Medium Quality (fastest)</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={12} sm={4}>
                <TextField fullWidth size="small" label="Time Limit (sec)" type="number" value={ml.autogluonTimeLimit}
                  onChange={(e) => ml.setMLConfig({ autogluonTimeLimit: Math.max(60, parseInt(e.target.value) || 3600) })}
                  inputProps={{ min: 60, step: 300 }}
                  InputLabelProps={{ sx: { fontSize: 13 } }} sx={{ '& input': { fontSize: 13 } }}
                  helperText={`${Math.round(ml.autogluonTimeLimit / 60)} minutes`} FormHelperTextProps={{ sx: { fontSize: 11 } }} />
              </Grid>
              <Grid item xs={12} sm={4}>
                <TextField fullWidth size="small" label="Bag Folds" type="number" value={ml.autogluonBagFolds}
                  onChange={(e) => ml.setMLConfig({ autogluonBagFolds: Math.min(10, Math.max(2, parseInt(e.target.value) || 8)) })}
                  inputProps={{ min: 2, max: 10 }}
                  InputLabelProps={{ sx: { fontSize: 13 } }} sx={{ '& input': { fontSize: 13 } }} />
              </Grid>
              <Grid item xs={12} sm={4}>
                <TextField fullWidth size="small" label="Stack Levels" type="number" value={ml.autogluonStackLevels}
                  onChange={(e) => ml.setMLConfig({ autogluonStackLevels: Math.min(5, Math.max(0, parseInt(e.target.value) || 2)) })}
                  inputProps={{ min: 0, max: 5 }}
                  InputLabelProps={{ sx: { fontSize: 13 } }} sx={{ '& input': { fontSize: 13 } }}
                  helperText="0 = no stacking" FormHelperTextProps={{ sx: { fontSize: 11 } }} />
              </Grid>
            </Grid>
          </Box>

          <Divider sx={{ my: 2 }} />

          {/* ── Section 5: Walk-Forward Validation ── */}
          <Box sx={{ mb: 3 }}>
            <Box display="flex" alignItems="center" gap={1} mb={1.5}>
              <Timeline sx={{ fontSize: 18, color: 'primary.main' }} />
              <Typography variant="subtitle1" fontWeight={600} sx={{ fontSize: 15 }}>Walk-Forward Validation</Typography>
            </Box>
            <Typography variant="body2" color="text.secondary" sx={{ fontSize: 12, mb: 1.5 }}>
              Sliding window that trains on historical data, validates on the next window, then steps forward. Prevents data leakage.
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={12} sm={4}>
                <TextField fullWidth size="small" label="Training Window (days)" type="number" value={ml.trainingWindowDays}
                  onChange={(e) => ml.setMLConfig({ trainingWindowDays: Math.max(90, parseInt(e.target.value) || 365) })}
                  inputProps={{ min: 90, step: 30 }}
                  InputLabelProps={{ sx: { fontSize: 13 } }} sx={{ '& input': { fontSize: 13 } }}
                  helperText={`~${Math.round(ml.trainingWindowDays / 30)} months of training data`} FormHelperTextProps={{ sx: { fontSize: 11 } }} />
              </Grid>
              <Grid item xs={12} sm={4}>
                <TextField fullWidth size="small" label="Validation Window (days)" type="number" value={ml.validationWindowDays}
                  onChange={(e) => ml.setMLConfig({ validationWindowDays: Math.max(7, parseInt(e.target.value) || 30) })}
                  inputProps={{ min: 7, step: 7 }}
                  InputLabelProps={{ sx: { fontSize: 13 } }} sx={{ '& input': { fontSize: 13 } }} />
              </Grid>
              <Grid item xs={12} sm={4}>
                <TextField fullWidth size="small" label="Step Size (days)" type="number" value={ml.stepSizeDays}
                  onChange={(e) => ml.setMLConfig({ stepSizeDays: Math.max(7, parseInt(e.target.value) || 30) })}
                  inputProps={{ min: 7, step: 7 }}
                  InputLabelProps={{ sx: { fontSize: 13 } }} sx={{ '& input': { fontSize: 13 } }} />
              </Grid>
              <Grid item xs={12} sm={4}>
                <TextField fullWidth size="small" label="Min Training Size (days)" type="number" value={ml.minTrainingSizeDays}
                  onChange={(e) => ml.setMLConfig({ minTrainingSizeDays: Math.max(30, parseInt(e.target.value) || 180) })}
                  inputProps={{ min: 30, step: 30 }}
                  InputLabelProps={{ sx: { fontSize: 13 } }} sx={{ '& input': { fontSize: 13 } }} />
              </Grid>
              <Grid item xs={12} sm={4}>
                <TextField fullWidth size="small" label="Gap Days" type="number" value={ml.gapDays}
                  onChange={(e) => ml.setMLConfig({ gapDays: Math.max(0, parseInt(e.target.value) || 1) })}
                  inputProps={{ min: 0, max: 7 }}
                  InputLabelProps={{ sx: { fontSize: 13 } }} sx={{ '& input': { fontSize: 13 } }}
                  helperText="Gap between train/val to prevent leakage" FormHelperTextProps={{ sx: { fontSize: 11 } }} />
              </Grid>
            </Grid>
          </Box>

          <Divider sx={{ my: 2 }} />

          {/* ── Section 6: Calibration & Signal Tiers ── */}
          <Box sx={{ mb: 3 }}>
            <Box display="flex" alignItems="center" gap={1} mb={1.5}>
              <Speed sx={{ fontSize: 18, color: 'primary.main' }} />
              <Typography variant="subtitle1" fontWeight={600} sx={{ fontSize: 15 }}>Calibration & Signal Tiers</Typography>
            </Box>
            <Grid container spacing={2}>
              <Grid item xs={12} sm={4}>
                <FormControl fullWidth size="small">
                  <InputLabel sx={{ fontSize: 13 }}>Calibration Method</InputLabel>
                  <Select value={ml.calibrationMethod} label="Calibration Method"
                    onChange={(e) => ml.setMLConfig({ calibrationMethod: e.target.value as any })}
                    sx={{ fontSize: 13 }}>
                    <MenuItem value="isotonic" sx={{ fontSize: 13 }}>Isotonic Regression</MenuItem>
                    <MenuItem value="platt" sx={{ fontSize: 13 }}>Platt Scaling</MenuItem>
                    <MenuItem value="temperature" sx={{ fontSize: 13 }}>Temperature Scaling</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={12} sm={4}>
                <TextField fullWidth size="small" label="Calibration CV Folds" type="number" value={ml.calibrationCvFolds}
                  onChange={(e) => ml.setMLConfig({ calibrationCvFolds: Math.min(10, Math.max(2, parseInt(e.target.value) || 5)) })}
                  inputProps={{ min: 2, max: 10 }}
                  InputLabelProps={{ sx: { fontSize: 13 } }} sx={{ '& input': { fontSize: 13 } }} />
              </Grid>
            </Grid>
            <Typography variant="body2" fontWeight={500} sx={{ fontSize: 13, mt: 2, mb: 1 }}>Signal Tier Thresholds</Typography>
            <Grid container spacing={2}>
              <Grid item xs={12} sm={4}>
                <Paper variant="outlined" sx={{ p: 1.5, borderColor: 'success.main', bgcolor: 'rgba(76, 175, 80, 0.06)' }}>
                  <Typography sx={{ fontSize: 12, fontWeight: 600, color: 'success.main', mb: 1 }}>Tier A — High Confidence</Typography>
                  <TextField fullWidth size="small" label="Min Probability" type="number" value={ml.tierAThreshold}
                    onChange={(e) => ml.setMLConfig({ tierAThreshold: Math.min(0.90, Math.max(0.50, parseFloat(e.target.value) || 0.58)) })}
                    inputProps={{ min: 0.50, max: 0.90, step: 0.01 }}
                    InputLabelProps={{ sx: { fontSize: 12 } }} sx={{ '& input': { fontSize: 13 } }}
                    helperText={`≥ ${(ml.tierAThreshold * 100).toFixed(0)}% confidence`} FormHelperTextProps={{ sx: { fontSize: 11 } }} />
                </Paper>
              </Grid>
              <Grid item xs={12} sm={4}>
                <Paper variant="outlined" sx={{ p: 1.5, borderColor: 'info.main', bgcolor: 'rgba(33, 150, 243, 0.06)' }}>
                  <Typography sx={{ fontSize: 12, fontWeight: 600, color: 'info.main', mb: 1 }}>Tier B — Medium Confidence</Typography>
                  <TextField fullWidth size="small" label="Min Probability" type="number" value={ml.tierBThreshold}
                    onChange={(e) => ml.setMLConfig({ tierBThreshold: Math.min(ml.tierAThreshold, Math.max(0.50, parseFloat(e.target.value) || 0.55)) })}
                    inputProps={{ min: 0.50, max: 0.90, step: 0.01 }}
                    InputLabelProps={{ sx: { fontSize: 12 } }} sx={{ '& input': { fontSize: 13 } }}
                    helperText={`${(ml.tierBThreshold * 100).toFixed(0)}%–${(ml.tierAThreshold * 100).toFixed(0)}%`} FormHelperTextProps={{ sx: { fontSize: 11 } }} />
                </Paper>
              </Grid>
              <Grid item xs={12} sm={4}>
                <Paper variant="outlined" sx={{ p: 1.5, borderColor: 'warning.main', bgcolor: 'rgba(255, 152, 0, 0.06)' }}>
                  <Typography sx={{ fontSize: 12, fontWeight: 600, color: 'warning.main', mb: 1 }}>Tier C — Low Confidence</Typography>
                  <TextField fullWidth size="small" label="Min Probability" type="number" value={ml.tierCThreshold}
                    onChange={(e) => ml.setMLConfig({ tierCThreshold: Math.min(ml.tierBThreshold, Math.max(0.50, parseFloat(e.target.value) || 0.52)) })}
                    inputProps={{ min: 0.50, max: 0.90, step: 0.01 }}
                    InputLabelProps={{ sx: { fontSize: 12 } }} sx={{ '& input': { fontSize: 13 } }}
                    helperText={`${(ml.tierCThreshold * 100).toFixed(0)}%–${(ml.tierBThreshold * 100).toFixed(0)}%`} FormHelperTextProps={{ sx: { fontSize: 11 } }} />
                </Paper>
              </Grid>
            </Grid>
          </Box>

          <Divider sx={{ my: 2 }} />

          {/* ── Section 7: Meta-Ensemble & Model Management ── */}
          <Box sx={{ mb: 3 }}>
            <Box display="flex" alignItems="center" gap={1} mb={1.5}>
              <Psychology sx={{ fontSize: 18, color: 'primary.main' }} />
              <Typography variant="subtitle1" fontWeight={600} sx={{ fontSize: 15 }}>Meta-Ensemble & Model Management</Typography>
            </Box>
            <Grid container spacing={2}>
              <Grid item xs={12} sm={3}>
                <TextField fullWidth size="small" label="Min Framework Weight" type="number" value={ml.ensembleMinWeight}
                  onChange={(e) => ml.setMLConfig({ ensembleMinWeight: Math.min(0.5, Math.max(0, parseFloat(e.target.value) || 0.1)) })}
                  inputProps={{ min: 0, max: 0.5, step: 0.05 }}
                  InputLabelProps={{ sx: { fontSize: 13 } }} sx={{ '& input': { fontSize: 13 } }}
                  helperText="Floor for any framework's weight" FormHelperTextProps={{ sx: { fontSize: 11 } }} />
              </Grid>
              <Grid item xs={12} sm={3}>
                <TextField fullWidth size="small" label="Weight Decay" type="number" value={ml.ensembleWeightDecay}
                  onChange={(e) => ml.setMLConfig({ ensembleWeightDecay: Math.min(1, Math.max(0.5, parseFloat(e.target.value) || 0.95)) })}
                  inputProps={{ min: 0.5, max: 1, step: 0.01 }}
                  InputLabelProps={{ sx: { fontSize: 13 } }} sx={{ '& input': { fontSize: 13 } }}
                  helperText="Recency bias for ensemble weights" FormHelperTextProps={{ sx: { fontSize: 11 } }} />
              </Grid>
              <Grid item xs={12} sm={3}>
                <TextField fullWidth size="small" label="Max Models/Sport" type="number" value={ml.maxModelsPerSport}
                  onChange={(e) => ml.setMLConfig({ maxModelsPerSport: Math.min(20, Math.max(1, parseInt(e.target.value) || 5)) })}
                  inputProps={{ min: 1, max: 20 }}
                  InputLabelProps={{ sx: { fontSize: 13 } }} sx={{ '& input': { fontSize: 13 } }} />
              </Grid>
              <Grid item xs={12} sm={3}>
                <TextField fullWidth size="small" label="Model TTL (days)" type="number" value={ml.modelTtlDays}
                  onChange={(e) => ml.setMLConfig({ modelTtlDays: Math.max(7, parseInt(e.target.value) || 90) })}
                  inputProps={{ min: 7, step: 7 }}
                  InputLabelProps={{ sx: { fontSize: 13 } }} sx={{ '& input': { fontSize: 13 } }}
                  helperText="Auto-expire old models after N days" FormHelperTextProps={{ sx: { fontSize: 11 } }} />
              </Grid>
            </Grid>
          </Box>

          <Divider sx={{ my: 2 }} />

          {/* ── Section 8: Training Schedule & Targets ── */}
          <Box>
            <Box display="flex" alignItems="center" gap={1} mb={1.5}>
              <CalendarMonth sx={{ fontSize: 18, color: 'primary.main' }} />
              <Typography variant="subtitle1" fontWeight={600} sx={{ fontSize: 15 }}>Training Schedule & Performance Targets</Typography>
            </Box>
            <Grid container spacing={2}>
              <Grid item xs={12} sm={4}>
                <FormControl fullWidth size="small">
                  <InputLabel sx={{ fontSize: 13 }}>Weekly Retrain Day</InputLabel>
                  <Select value={ml.weeklyRetrainDay} label="Weekly Retrain Day"
                    onChange={(e) => ml.setMLConfig({ weeklyRetrainDay: e.target.value as number })}
                    sx={{ fontSize: 13 }}>
                    {['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'].map((d, i) => (
                      <MenuItem key={i} value={i} sx={{ fontSize: 13 }}>{d}</MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={12} sm={4}>
                <TextField fullWidth size="small" label="Retrain Hour (UTC)" type="number" value={ml.weeklyRetrainHour}
                  onChange={(e) => ml.setMLConfig({ weeklyRetrainHour: Math.min(23, Math.max(0, parseInt(e.target.value) || 4)) })}
                  inputProps={{ min: 0, max: 23 }}
                  InputLabelProps={{ sx: { fontSize: 13 } }} sx={{ '& input': { fontSize: 13 } }}
                  helperText={`${ml.weeklyRetrainHour}:00 UTC`} FormHelperTextProps={{ sx: { fontSize: 11 } }} />
              </Grid>
            </Grid>
            <Typography variant="body2" fontWeight={500} sx={{ fontSize: 13, mt: 2, mb: 1 }}>Performance Targets</Typography>
            <Grid container spacing={2}>
              <Grid item xs={12} sm={3}>
                <TextField fullWidth size="small" label="Target Accuracy" type="number" value={ml.targetAccuracy}
                  onChange={(e) => ml.setMLConfig({ targetAccuracy: Math.min(0.90, Math.max(0.50, parseFloat(e.target.value) || 0.60)) })}
                  inputProps={{ min: 0.50, max: 0.90, step: 0.01 }}
                  InputLabelProps={{ sx: { fontSize: 13 } }} sx={{ '& input': { fontSize: 13 } }}
                  helperText={`${(ml.targetAccuracy * 100).toFixed(0)}%`} FormHelperTextProps={{ sx: { fontSize: 11 } }} />
              </Grid>
              <Grid item xs={12} sm={3}>
                <TextField fullWidth size="small" label="Target AUC" type="number" value={ml.targetAuc}
                  onChange={(e) => ml.setMLConfig({ targetAuc: Math.min(0.95, Math.max(0.50, parseFloat(e.target.value) || 0.60)) })}
                  inputProps={{ min: 0.50, max: 0.95, step: 0.01 }}
                  InputLabelProps={{ sx: { fontSize: 13 } }} sx={{ '& input': { fontSize: 13 } }} />
              </Grid>
              <Grid item xs={12} sm={3}>
                <TextField fullWidth size="small" label="Target Cal. Error" type="number" value={ml.targetCalibrationError}
                  onChange={(e) => ml.setMLConfig({ targetCalibrationError: Math.min(0.20, Math.max(0.01, parseFloat(e.target.value) || 0.05)) })}
                  inputProps={{ min: 0.01, max: 0.20, step: 0.01 }}
                  InputLabelProps={{ sx: { fontSize: 13 } }} sx={{ '& input': { fontSize: 13 } }}
                  helperText="Max expected calibration error" FormHelperTextProps={{ sx: { fontSize: 11 } }} />
              </Grid>
            </Grid>
          </Box>
        </TabPanel>

        <TabPanel value={tab} index={4}>
          <Alert severity="info" sx={{ mb: 2, fontSize: 13 }}>Monitor data completeness per sport. Target: 95%+ for accurate predictions.</Alert>
          <TableContainer><Table size="small"><TableHead><TableRow><TableCell sx={{ fontSize: 13, fontWeight: 600 }}>Sport</TableCell><TableCell sx={{ fontSize: 13, fontWeight: 600 }}>Completeness</TableCell><TableCell align="center" sx={{ fontSize: 13, fontWeight: 600 }}>Complete</TableCell><TableCell align="center" sx={{ fontSize: 13, fontWeight: 600 }}>Missing</TableCell><TableCell align="center" sx={{ fontSize: 13, fontWeight: 600 }}>Bad Data</TableCell><TableCell align="center" sx={{ fontSize: 13, fontWeight: 600 }}>Gaps</TableCell><TableCell sx={{ fontSize: 13, fontWeight: 600 }}>Status</TableCell></TableRow></TableHead><TableBody>
            {[{ sport: 'NBA', complete: 98.5, missing: 1.2, bad: 0.3, gaps: 0, status: 'excellent' },{ sport: 'NFL', complete: 97.2, missing: 2.0, bad: 0.8, gaps: 0, status: 'excellent' },{ sport: 'MLB', complete: 94.1, missing: 4.5, bad: 1.4, gaps: 2, status: 'good' },{ sport: 'NHL', complete: 96.8, missing: 2.5, bad: 0.7, gaps: 0, status: 'excellent' },{ sport: 'NCAAB', complete: 89.3, missing: 8.2, bad: 2.5, gaps: 5, status: 'warning' },{ sport: 'NCAAF', complete: 91.5, missing: 6.0, bad: 2.5, gaps: 3, status: 'warning' }].map((row) => <TableRow key={row.sport} sx={{ '& td': { fontSize: 13 } }}><TableCell><Chip label={row.sport} size="small" sx={{ fontSize: 12, height: 20 }} /></TableCell><TableCell><Box display="flex" alignItems="center" gap={1}><LinearProgress variant="determinate" value={row.complete} sx={{ flex: 1, height: 6, borderRadius: 3 }} /><Typography variant="caption">{row.complete}%</Typography></Box></TableCell><TableCell align="center" sx={{ color: 'success.main' }}>{row.complete}%</TableCell><TableCell align="center" sx={{ color: row.missing > 5 ? 'error.main' : 'warning.main' }}>{row.missing}%</TableCell><TableCell align="center" sx={{ color: row.bad > 1 ? 'error.main' : 'inherit' }}>{row.bad}%</TableCell><TableCell align="center">{row.gaps}</TableCell><TableCell><Chip label={row.status} size="small" color={row.status === 'excellent' ? 'success' : row.status === 'good' ? 'primary' : 'warning'} sx={{ fontSize: 12, height: 20 }} /></TableCell></TableRow>)}
          </TableBody></Table></TableContainer>
          <Box mt={2}><Button variant="outlined" size="small" startIcon={<Refresh />} sx={{ mr: 1, fontSize: 13 }}>Re-fetch Missing Data</Button><Button variant="outlined" size="small" color="warning" sx={{ fontSize: 13 }}>View All Gaps</Button></Box>
        </TabPanel>

        <TabPanel value={tab} index={5}>
          <Alert severity="info" sx={{ mb: 2, fontSize: 13 }}>Manage all data sources: APIs, Scrapers, and WebSockets.</Alert>
          <Typography variant="subtitle1" fontWeight={600} gutterBottom>API Sources</Typography>
          <TableContainer sx={{ mb: 2 }}><Table size="small"><TableHead><TableRow><TableCell sx={{ fontSize: 13, fontWeight: 600 }}>Source</TableCell><TableCell sx={{ fontSize: 13, fontWeight: 600 }}>Status</TableCell><TableCell sx={{ fontSize: 13, fontWeight: 600 }}>Calls / Limit</TableCell><TableCell sx={{ fontSize: 13, fontWeight: 600 }}>Reset</TableCell><TableCell sx={{ fontSize: 13, fontWeight: 600 }}>Latency</TableCell><TableCell sx={{ fontSize: 13, fontWeight: 600 }}>Actions</TableCell></TableRow></TableHead><TableBody>
            {[{ name: 'TheOddsAPI', status: 'active', calls: 450, limit: 500, reset: '23h 15m', latency: '120ms' },{ name: 'Pinnacle RapidAPI', status: 'slow', calls: 890, limit: 1000, reset: '5h 30m', latency: '850ms' },{ name: 'ESPN Hidden API', status: 'active', calls: 0, limit: 0, reset: '-', latency: '95ms' },{ name: 'Sportradar', status: 'error', calls: 0, limit: 100, reset: '-', latency: 'Timeout' }].map((api) => <TableRow key={api.name} sx={{ '& td': { fontSize: 13 } }}><TableCell><Typography variant="body2" fontWeight={500}>{api.name}</Typography></TableCell><TableCell><Chip label={api.status} size="small" color={api.status === 'active' ? 'success' : api.status === 'slow' ? 'warning' : 'error'} sx={{ fontSize: 12, height: 20 }} /></TableCell><TableCell>{api.limit > 0 ? <Box display="flex" alignItems="center" gap={1}><LinearProgress variant="determinate" value={(api.calls / api.limit) * 100} sx={{ flex: 1, height: 5, borderRadius: 3 }} /><Typography variant="caption">{api.calls}/{api.limit}</Typography></Box> : 'Unlimited'}</TableCell><TableCell>{api.reset}</TableCell><TableCell sx={{ color: api.latency === 'Timeout' ? 'error.main' : parseInt(api.latency) > 500 ? 'warning.main' : 'success.main' }}>{api.latency}</TableCell><TableCell><Button size="small" sx={{ fontSize: 12 }}>Config</Button>{api.status === 'error' && <Button size="small" color="warning" sx={{ fontSize: 12 }}>Retry</Button>}</TableCell></TableRow>)}
          </TableBody></Table></TableContainer>
          <Typography variant="subtitle1" fontWeight={600} gutterBottom>Web Scrapers</Typography>
          <TableContainer><Table size="small"><TableHead><TableRow><TableCell sx={{ fontSize: 13, fontWeight: 600 }}>Scraper</TableCell><TableCell sx={{ fontSize: 13, fontWeight: 600 }}>Status</TableCell><TableCell sx={{ fontSize: 13, fontWeight: 600 }}>Last Run</TableCell><TableCell sx={{ fontSize: 13, fontWeight: 600 }}>Success Rate</TableCell><TableCell sx={{ fontSize: 13, fontWeight: 600 }}>Records</TableCell><TableCell sx={{ fontSize: 13, fontWeight: 600 }}>Actions</TableCell></TableRow></TableHead><TableBody>
            {[{ name: 'ESPN Injuries', status: 'active', lastRun: '5 min ago', success: 100, records: 145 },{ name: 'Pro-Football-Ref', status: 'active', lastRun: '1 hour ago', success: 98, records: 2450 },{ name: 'Basketball-Ref', status: 'slow', lastRun: '2 hours ago', success: 85, records: 1200 },{ name: 'Covers.com', status: 'active', lastRun: '10 min ago', success: 100, records: 50 }].map((scraper) => <TableRow key={scraper.name} sx={{ '& td': { fontSize: 13 } }}><TableCell><Typography variant="body2" fontWeight={500}>{scraper.name}</Typography></TableCell><TableCell><Chip label={scraper.status} size="small" color={scraper.status === 'active' ? 'success' : 'warning'} sx={{ fontSize: 12, height: 20 }} /></TableCell><TableCell>{scraper.lastRun}</TableCell><TableCell sx={{ color: scraper.success >= 95 ? 'success.main' : 'warning.main' }}>{scraper.success}%</TableCell><TableCell>{scraper.records}</TableCell><TableCell><Button size="small" startIcon={<PlayArrow sx={{ fontSize: 14 }} />} sx={{ fontSize: 12 }}>Run Now</Button></TableCell></TableRow>)}
          </TableBody></Table></TableContainer>
        </TabPanel>

        <TabPanel value={tab} index={6}>
          <Grid container spacing={2}>
            <Grid item xs={12}><Typography variant="subtitle2" fontWeight={600} gutterBottom sx={{ fontSize: 14 }}>Account</Typography><TextField fullWidth size="small" label="Email" defaultValue="admin@example.com" disabled sx={{ mb: 1 }} InputLabelProps={{ sx: { fontSize: 13 } }} inputProps={{ style: { fontSize: 13 } }} /></Grid>
            <Grid item xs={12}><Typography variant="subtitle2" fontWeight={600} gutterBottom sx={{ fontSize: 14 }}>Change Password</Typography><Grid container spacing={2}><Grid item xs={12} sm={4}><TextField fullWidth size="small" label="Current Password" type="password" InputLabelProps={{ sx: { fontSize: 13 } }} inputProps={{ style: { fontSize: 13 } }} /></Grid><Grid item xs={12} sm={4}><TextField fullWidth size="small" label="New Password" type="password" InputLabelProps={{ sx: { fontSize: 13 } }} inputProps={{ style: { fontSize: 13 } }} /></Grid><Grid item xs={12} sm={4}><TextField fullWidth size="small" label="Confirm Password" type="password" InputLabelProps={{ sx: { fontSize: 13 } }} inputProps={{ style: { fontSize: 13 } }} /></Grid></Grid></Grid>
          </Grid>
        </TabPanel>

        <Box sx={{ px: 3, py: 2, borderTop: 1, borderColor: 'divider' }}><Button variant="contained" size="small" startIcon={<Save />} onClick={handleSave} sx={{ fontSize: 13 }}>Save All Settings</Button></Box>
      </Card>
      
      {/* Add Telegram Dialog */}
      <Dialog open={telegramDialog} onClose={() => setTelegramDialog(false)} maxWidth="sm" fullWidth>
        <DialogTitle sx={{ fontSize: 16, fontWeight: 600 }}>Add Telegram Account</DialogTitle>
        <DialogContent>
          <Box sx={{ pt: 2 }}>
            <TextField fullWidth size="small" label="Account Name" value={newTelegram.name} onChange={(e) => setNewTelegram({ ...newTelegram, name: e.target.value })} placeholder="e.g., Main, Backup" sx={{ mb: 2 }} />
            <TextField fullWidth size="small" label="Chat ID" value={newTelegram.chatId} onChange={(e) => setNewTelegram({ ...newTelegram, chatId: e.target.value })} placeholder="Get from @userinfobot" helperText="Send /start to @userinfobot to get your Chat ID" />
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setTelegramDialog(false)}>Cancel</Button>
          <Button variant="contained" onClick={addTelegramAccount} disabled={!newTelegram.name || !newTelegram.chatId}>Add</Button>
        </DialogActions>
      </Dialog>
      
      {/* Add Email Dialog */}
      <Dialog open={emailDialog} onClose={() => setEmailDialog(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Add Email Address</DialogTitle>
        <DialogContent>
          <Box sx={{ pt: 2 }}>
            <TextField fullWidth label="Email Address" type="email" value={newEmail} onChange={(e) => setNewEmail(e.target.value)} placeholder="your@email.com" />
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setEmailDialog(false)}>Cancel</Button>
          <Button variant="contained" onClick={addEmailAccount} disabled={!newEmail || !newEmail.includes('@')}>Add</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default Settings;