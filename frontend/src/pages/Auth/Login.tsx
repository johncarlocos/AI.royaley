// src/pages/Auth/Login.tsx
import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box, Card, CardContent, Typography, TextField, Button, Alert,
  Checkbox, FormControlLabel, Divider, CircularProgress
} from '@mui/material';
import { Login as LoginIcon } from '@mui/icons-material';
import { api } from '../../api/client';
import { useAuthStore } from '../../store';

const Login: React.FC = () => {
  const navigate = useNavigate();
  const { login } = useAuthStore();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [rememberMe, setRememberMe] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const data = await api.login(email, password);
      login(data.user || { id: '1', email, role: 'user' }, data.access_token || 'token');
      navigate('/');
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Invalid credentials');
    } finally {
      setLoading(false);
    }
  };

  const handleDemoLogin = () => {
    login({ id: 'demo', email: 'demo@example.com', role: 'admin' }, 'demo-token');
    navigate('/');
  };

  return (
    <Box sx={{ minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', bgcolor: 'background.default', p: 2 }}>
      <Card sx={{ maxWidth: 440, width: '100%' }}>
        <CardContent sx={{ p: 4 }}>
          <Box textAlign="center" mb={4}>
            <Typography variant="h4" sx={{ fontWeight: 'bold', color: 'primary.main', mb: 1 }}>
              AI PRO SPORTS
            </Typography>
            <Typography color="textSecondary">Enterprise Sports Prediction Platform</Typography>
          </Box>

          {error && <Alert severity="error" sx={{ mb: 3 }}>{error}</Alert>}

          <form onSubmit={handleSubmit}>
            <TextField
              fullWidth
              label="Email"
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              margin="normal"
              required
              autoComplete="email"
            />
            <TextField
              fullWidth
              label="Password"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              margin="normal"
              required
              autoComplete="current-password"
            />
            <FormControlLabel
              control={<Checkbox checked={rememberMe} onChange={(e) => setRememberMe(e.target.checked)} />}
              label="Remember me"
              sx={{ mt: 1 }}
            />
            <Button
              fullWidth
              type="submit"
              variant="contained"
              size="large"
              disabled={loading}
              sx={{ mt: 3, mb: 2, height: 48 }}
              startIcon={loading ? <CircularProgress size={20} /> : <LoginIcon />}
            >
              {loading ? 'Signing in...' : 'Sign In'}
            </Button>
          </form>

          <Divider sx={{ my: 3 }}>or</Divider>

          <Button fullWidth variant="outlined" size="large" onClick={handleDemoLogin} sx={{ height: 48 }}>
            Try Demo Mode
          </Button>

          <Typography variant="caption" color="textSecondary" display="block" textAlign="center" mt={3}>
            Demo mode provides full access with sample data
          </Typography>
        </CardContent>
      </Card>
    </Box>
  );
};

export default Login;
