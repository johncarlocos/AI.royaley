// src/App.tsx
import React, { lazy, Suspense } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { ThemeProvider, createTheme, CssBaseline, CircularProgress, Box } from '@mui/material';
import { LocalizationProvider } from '@mui/x-date-pickers/LocalizationProvider';
import { AdapterDateFns } from '@mui/x-date-pickers/AdapterDateFns';
import { useAuthStore, useSettingsStore } from './store';

const Layout = lazy(() => import('./components/Layout/Layout'));
const Dashboard = lazy(() => import('./components/Dashboard/Dashboard'));
const Predictions = lazy(() => import('./pages/Predictions/Predictions'));
const Betting = lazy(() => import('./pages/Betting/Betting'));
const Analytics = lazy(() => import('./pages/Analytics/Analytics'));
const Backtesting = lazy(() => import('./pages/Backtesting/Backtesting'));
const Models = lazy(() => import('./pages/Models/Models'));
const PlayerProps = lazy(() => import('./pages/PlayerProps/PlayerProps'));
const GameProps = lazy(() => import('./pages/GameProps/GameProps'));
const Live = lazy(() => import('./pages/Live/Live'));
const Alerts = lazy(() => import('./pages/Alerts/Alerts'));
const Settings = lazy(() => import('./pages/Settings/Settings'));
const Login = lazy(() => import('./pages/Auth/Login'));

const createAppTheme = (mode: 'light' | 'dark') => createTheme({
  palette: {
    mode,
    primary: { main: '#3b82f6' },
    secondary: { main: '#8b5cf6' },
    success: { main: '#10b981' },
    warning: { main: '#f59e0b' },
    error: { main: '#ef4444' },
    background: mode === 'dark' ? { default: '#0f172a', paper: '#1e293b' } : { default: '#f8fafc', paper: '#ffffff' },
    text: mode === 'dark' ? { primary: '#f1f5f9', secondary: '#94a3b8' } : { primary: '#1e293b', secondary: '#64748b' },
  },
  typography: { fontFamily: '"Inter", "Roboto", sans-serif', h5: { fontWeight: 700 }, h6: { fontWeight: 600 } },
  shape: { borderRadius: 12 },
  components: {
    MuiButton: { styleOverrides: { root: { textTransform: 'none', fontWeight: 600 } } },
    MuiCard: { styleOverrides: { root: { borderRadius: 16 } } },
    MuiChip: { styleOverrides: { root: { fontWeight: 500 } } },
  },
});

const Loading = () => (
  <Box display="flex" justifyContent="center" alignItems="center" minHeight="100vh">
    <CircularProgress />
  </Box>
);

const ProtectedRoute: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { isAuthenticated } = useAuthStore();
  return isAuthenticated ? <>{children}</> : <Navigate to="/login" replace />;
};

const App: React.FC = () => {
  const { theme } = useSettingsStore();

  return (
    <ThemeProvider theme={createAppTheme(theme)}>
      <CssBaseline />
      <LocalizationProvider dateAdapter={AdapterDateFns}>
        <BrowserRouter>
          <Suspense fallback={<Loading />}>
            <Routes>
              <Route path="/login" element={<Login />} />
              <Route path="/" element={<ProtectedRoute><Layout /></ProtectedRoute>}>
                <Route index element={<Dashboard />} />
                <Route path="predictions" element={<Predictions />} />
                <Route path="betting" element={<Betting />} />
                <Route path="analytics" element={<Analytics />} />
                <Route path="backtesting" element={<Backtesting />} />
                <Route path="models" element={<Models />} />
                <Route path="player-props" element={<PlayerProps />} />
                <Route path="game-props" element={<GameProps />} />
                <Route path="live" element={<Live />} />
                <Route path="alerts" element={<Alerts />} />
                <Route path="settings" element={<Settings />} />
              </Route>
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </Suspense>
        </BrowserRouter>
      </LocalizationProvider>
    </ThemeProvider>
  );
};

export default App;
