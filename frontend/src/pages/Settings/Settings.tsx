// src/pages/Settings/Settings.tsx
import React, { useState } from 'react';
import {
  Box, Card, CardContent, Typography, Grid, Tabs, Tab, TextField, Button,
  Switch, FormControlLabel, Select, MenuItem, FormControl, InputLabel,
  Table, TableBody, TableCell, TableContainer, TableHead, TableRow,
  Chip, LinearProgress, Alert, Divider, IconButton, Dialog, DialogTitle,
  DialogContent, DialogActions, List, ListItem, ListItemText, Paper
} from '@mui/material';
import { Palette, Casino, Notifications, Psychology, Storage, Security, Api, Save, Telegram, Email, Visibility, VisibilityOff, Refresh, CheckCircle, Warning, PlayArrow, Add, Delete, Send } from '@mui/icons-material';
import { useSettingsStore } from '../../store';
import { TIMEZONES } from '../../types';

interface TabPanelProps { children?: React.ReactNode; index: number; value: number; }
const TabPanel = ({ children, value, index }: TabPanelProps) => <Box hidden={value !== index} sx={{ py: 2, px: 4 }}>{value === index && children}</Box>;

interface TelegramAccount { id: string; name: string; chatId: string; enabled: boolean; }
interface EmailAccount { id: string; email: string; enabled: boolean; }

const Settings: React.FC = () => {
  const [tab, setTab] = useState(0);
  const { theme, setTheme, oddsFormat, setOddsFormat, timezone, setTimezone, timeFormat, setTimeFormat } = useSettingsStore();
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
          <Alert severity="info" sx={{ mb: 2, fontSize: 13 }}>Flat betting: Fixed amount per bet. All records saved for model training.</Alert>
          <Grid container spacing={2}>
            <Grid item xs={12} sm={6}><TextField fullWidth size="small" label="Flat Bet Amount ($)" type="number" defaultValue={100} InputLabelProps={{ sx: { fontSize: 13 } }} inputProps={{ style: { fontSize: 13 } }} /></Grid>
            <Grid item xs={12}><Typography variant="body2" fontWeight={500} gutterBottom sx={{ fontSize: 14 }}>Bet on Tiers:</Typography><FormControlLabel control={<Switch defaultChecked size="small" />} label={<Typography sx={{ fontSize: 13 }}>Tier A (65%+)</Typography>} /><FormControlLabel control={<Switch defaultChecked size="small" />} label={<Typography sx={{ fontSize: 13 }}>Tier B (60-65%)</Typography>} /><FormControlLabel control={<Switch size="small" />} label={<Typography sx={{ fontSize: 13 }}>Tier C (55-60%)</Typography>} /><FormControlLabel control={<Switch size="small" />} label={<Typography sx={{ fontSize: 13 }}>Tier D (&lt;55%)</Typography>} /></Grid>
          </Grid>
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
          <Grid container spacing={2}>
            <Grid item xs={12} sm={6}><TextField fullWidth size="small" label="H2O Max Memory" defaultValue="32g" InputLabelProps={{ sx: { fontSize: 13 } }} inputProps={{ style: { fontSize: 13 } }} /></Grid>
            <Grid item xs={12} sm={6}><TextField fullWidth size="small" label="Max Models per Training" type="number" defaultValue={50} InputLabelProps={{ sx: { fontSize: 13 } }} inputProps={{ style: { fontSize: 13 } }} /></Grid>
            <Grid item xs={12} sm={6}><TextField fullWidth size="small" label="Max Runtime (seconds)" type="number" defaultValue={3600} InputLabelProps={{ sx: { fontSize: 13 } }} inputProps={{ style: { fontSize: 13 } }} /></Grid>
            <Grid item xs={12} sm={6}><TextField fullWidth size="small" label="Min Training Samples" type="number" defaultValue={500} InputLabelProps={{ sx: { fontSize: 13 } }} inputProps={{ style: { fontSize: 13 } }} /></Grid>
            <Grid item xs={12}><Divider sx={{ my: 1 }} /></Grid>
            <Grid item xs={12} sm={4}><TextField fullWidth size="small" label="Training Window (days)" type="number" defaultValue={365} InputLabelProps={{ sx: { fontSize: 13 } }} inputProps={{ style: { fontSize: 13 } }} /></Grid>
            <Grid item xs={12} sm={4}><TextField fullWidth size="small" label="Test Window (days)" type="number" defaultValue={30} InputLabelProps={{ sx: { fontSize: 13 } }} inputProps={{ style: { fontSize: 13 } }} /></Grid>
            <Grid item xs={12} sm={4}><TextField fullWidth size="small" label="Step Size (days)" type="number" defaultValue={30} InputLabelProps={{ sx: { fontSize: 13 } }} inputProps={{ style: { fontSize: 13 } }} /></Grid>
          </Grid>
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