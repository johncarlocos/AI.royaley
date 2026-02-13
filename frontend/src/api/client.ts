// src/api/client.ts
import axios from 'axios';

const BASE_URL = import.meta.env.VITE_API_URL || '/api/v1';

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

  // Public API (no auth required) - for frontend live data
  getPublicPredictions: async (params?: { sport?: string; signal_tier?: string; per_page?: number; page?: number }) => {
    const { data } = await axiosClient.get('/public/predictions', { params });
    return data;
  },

  // Fetch ALL predictions across pages
  getAllPublicPredictions: async (params?: { sport?: string; signal_tier?: string }) => {
    const perPage = 200;
    let page = 1;
    let allPredictions: any[] = [];
    
    while (true) {
      const { data } = await axiosClient.get('/public/predictions', {
        params: { ...params, per_page: perPage, page },
      });
      const predictions = data?.predictions || [];
      allPredictions = allPredictions.concat(predictions);
      
      // Stop if we got everything or no more data
      const total = data?.total || 0;
      if (allPredictions.length >= total || predictions.length === 0) break;
      page++;
    }
    
    return { predictions: allPredictions, total: allPredictions.length };
  },

  getDashboardStats: async () => {
    const { data } = await axiosClient.get('/public/dashboard/stats');
    return data;
  },

  getBettingSummary: async (params?: { sport?: string; tiers?: string; stake?: number; initial_bankroll?: number }) => {
    const { data } = await axiosClient.get('/public/betting-summary', { params });
    return data;
  },

  // Predictions (authenticated)
  getPredictions: async (params?: { sport?: string }) => {
    const { data } = await axiosClient.get('/predictions', { params });
    return data;
  },

  // Games
  getGames: async (params?: { sport?: string }) => {
    const { data } = await axiosClient.get('/games', { params });
    return data;
  },

  // Live scoreboard (public, no auth)
  getLiveGames: async (sport?: string) => {
    const params: Record<string, string> = {};
    if (sport) params.sport = sport;
    const { data } = await axiosClient.get('/public/live', { params });
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

  // Models (public read, auth for training)
  getModels: async (params?: { sport_code?: string; production_only?: boolean }) => {
    const { data } = await axiosClient.get('/public/models', { params });
    return data;
  },

  getTrainingRuns: async (params?: { sport_code?: string; limit?: number }) => {
    const { data } = await axiosClient.get('/public/models/training-runs', { params });
    return data;
  },

  trainModel: async (config: { sport_code: string; bet_type: string; framework?: string }) => {
    const { data } = await axiosClient.post('/models/train', config);
    return data;
  },

  promoteModel: async (modelId: string) => {
    const { data } = await axiosClient.post(`/public/models/${modelId}/promote`);
    return data;
  },

  deprecateModel: async (modelId: string) => {
    const { data } = await axiosClient.post(`/public/models/${modelId}/deprecate`);
    return data;
  },

  cancelTrainingRun: async (runId: string) => {
    const { data } = await axiosClient.post(`/public/models/training-runs/${runId}/cancel`);
    return data;
  },

  reinforceModel: async (params: { sport_code: string; bet_type: string; framework?: string }) => {
    const { data } = await axiosClient.post('/public/models/reinforce', null, { params });
    return data;
  },

  // Analytics
  getAnalytics: async () => {
    const { data } = await axiosClient.get('/analytics');
    return data;
  },

  // Player Props (public - no auth required)
  getPlayerProps: async (params?: { sport?: string }) => {
    const { data } = await axiosClient.get('/public/player-props', { params });
    return data;
  },

  // Health
  getHealth: async () => {
    const { data } = await axiosClient.get('/health/detailed');
    return data;
  },

  getSystemHealth: async () => {
    const { data } = await axiosClient.get('/public/system-health');
    return data;
  },

  // Backtest
  runBacktest: async (config: Record<string, unknown>) => {
    const { data } = await axiosClient.post('/backtest/run', config);
    return data;
  },
};

export default api;
