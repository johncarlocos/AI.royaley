// src/api/client.ts
import axios from 'axios';

const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1';

const axiosClient = axios.create({
  baseURL: BASE_URL,
  headers: { 'Content-Type': 'application/json' },
});

axiosClient.interceptors.request.use((config) => {
  const stored = localStorage.getItem('auth-storage');
  if (stored) {
    const { state } = JSON.parse(stored);
    if (state?.token) {
      config.headers.Authorization = `Bearer ${state.token}`;
    }
  }
  return config;
});

export const api = {
  // Auth
  login: async (email: string, password: string) => {
    const { data } = await axiosClient.post('/auth/login', { email, password });
    return data;
  },

  // Predictions
  getPredictions: async (params?: { sport?: string }) => {
    const { data } = await axiosClient.get('/predictions', { params });
    return data;
  },

  // Games
  getGames: async (params?: { sport?: string }) => {
    const { data } = await axiosClient.get('/games', { params });
    return data;
  },

  // Bets
  getBets: async () => {
    const { data } = await axiosClient.get('/betting/history');
    return data;
  },

  updateBettingConfig: async (config: Record<string, unknown>) => {
    const { data } = await axiosClient.put('/betting/config', config);
    return data;
  },

  // Models
  getModels: async () => {
    const { data } = await axiosClient.get('/models');
    return data;
  },

  // Analytics
  getAnalytics: async () => {
    const { data } = await axiosClient.get('/analytics');
    return data;
  },

  // Player Props
  getPlayerProps: async (params?: { sport?: string }) => {
    const { data } = await axiosClient.get('/player-props', { params });
    return data;
  },

  // Health
  getHealth: async () => {
    const { data } = await axiosClient.get('/health/detailed');
    return data;
  },

  // Backtest
  runBacktest: async (config: Record<string, unknown>) => {
    const { data } = await axiosClient.post('/backtest/run', config);
    return data;
  },
};

export default api;
