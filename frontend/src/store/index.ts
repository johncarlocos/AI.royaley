// src/store/index.ts
import { create } from 'zustand';
import { persist } from 'zustand/middleware';

// Auth Store
interface User {
  id: string;
  email: string;
  role: string;
}

interface AuthState {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  login: (user: User, token: string) => void;
  logout: () => void;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set) => ({
      user: null,
      token: null,
      isAuthenticated: false,
      login: (user, token) => set({ user, token, isAuthenticated: true }),
      logout: () => set({ user: null, token: null, isAuthenticated: false }),
    }),
    { name: 'auth-storage' }
  )
);

// Alert Store
interface AlertState {
  alerts: Array<{ id: string; type: string; message: string; read: boolean }>;
  unreadCount: number;
  addAlert: (alert: { id: string; type: string; message: string }) => void;
  markAllRead: () => void;
  clearAlerts: () => void;
}

export const useAlertStore = create<AlertState>((set) => ({
  alerts: [],
  unreadCount: 0,
  addAlert: (alert) => set((state) => ({
    alerts: [{ ...alert, read: false }, ...state.alerts],
    unreadCount: state.unreadCount + 1,
  })),
  markAllRead: () => set((state) => ({
    alerts: state.alerts.map((a) => ({ ...a, read: true })),
    unreadCount: 0,
  })),
  clearAlerts: () => set({ alerts: [], unreadCount: 0 }),
}));

// Settings Store
interface SettingsState {
  theme: 'light' | 'dark';
  oddsFormat: 'american' | 'decimal' | 'fractional';
  timezone: string;
  timeFormat: '12h' | '24h';
  setTheme: (theme: 'light' | 'dark') => void;
  setOddsFormat: (format: 'american' | 'decimal' | 'fractional') => void;
  setTimezone: (tz: string) => void;
  setTimeFormat: (tf: '12h' | '24h') => void;
}

export const useSettingsStore = create<SettingsState>()(
  persist(
    (set) => ({
      theme: 'dark',
      oddsFormat: 'american',
      timezone: 'America/New_York',
      timeFormat: '12h',
      setTheme: (theme) => set({ theme }),
      setOddsFormat: (oddsFormat) => set({ oddsFormat }),
      setTimezone: (timezone) => set({ timezone }),
      setTimeFormat: (timeFormat) => set({ timeFormat }),
    }),
    { name: 'settings-storage' }
  )
);

// Betting Store
interface BettingState {
  // Bankroll
  initialBankroll: number;
  // Bet Sizing
  betSizing: 'flat' | 'percentage' | 'kelly';
  flatAmount: number;
  percentageAmount: number;
  kellyFraction: 'quarter' | 'half' | 'full';
  // Tier Filters
  tierA: boolean;
  tierB: boolean;
  tierC: boolean;
  // Bet Type Filters
  betSpread: boolean;
  betMoneyline: boolean;
  betTotal: boolean;
  // Risk Controls
  minEdge: number;
  maxDailyBets: number;
  minOdds: number;
  maxOdds: number;
  // Auto
  autoBet: boolean;
  // Setter
  setBetting: (config: Partial<BettingState>) => void;
}

export const useBettingStore = create<BettingState>()(
  persist(
    (set) => ({
      initialBankroll: 10000,
      betSizing: 'flat',
      flatAmount: 100,
      percentageAmount: 2,
      kellyFraction: 'quarter',
      tierA: true,
      tierB: true,
      tierC: false,
      betSpread: true,
      betMoneyline: true,
      betTotal: true,
      minEdge: 1.0,
      maxDailyBets: 20,
      minOdds: -500,
      maxOdds: 500,
      autoBet: false,
      setBetting: (config) => set((state) => ({ ...state, ...config })),
    }),
    { name: 'betting-storage' }
  )
);

// ML Config Store
interface MLConfigState {
  // H2O AutoML
  h2oMaxModels: number;
  h2oMaxRuntime: number;
  h2oMaxMem: string;
  h2oNfolds: number;
  h2oSeed: number;
  // AutoGluon
  autogluonPresets: 'best_quality' | 'high_quality' | 'good_quality' | 'medium_quality';
  autogluonTimeLimit: number;
  autogluonBagFolds: number;
  autogluonStackLevels: number;
  // Sklearn
  sklearnEstimators: number;
  sklearnMaxDepth: number;
  sklearnLearningRate: number;
  sklearnCvFolds: number;
  // Walk-Forward Validation
  trainingWindowDays: number;
  validationWindowDays: number;
  stepSizeDays: number;
  minTrainingSizeDays: number;
  gapDays: number;
  // Calibration
  calibrationMethod: 'isotonic' | 'platt' | 'temperature';
  calibrationCvFolds: number;
  // Signal Tiers
  tierAThreshold: number;
  tierBThreshold: number;
  tierCThreshold: number;
  // Meta-Ensemble
  ensembleMinWeight: number;
  ensembleWeightDecay: number;
  maxModelsPerSport: number;
  modelTtlDays: number;
  // Feature Engineering
  rollingWindows: number[];
  eloBaseRating: number;
  momentumDecay: number;
  // Performance Targets
  targetAccuracy: number;
  targetAuc: number;
  targetCalibrationError: number;
  // Schedule
  weeklyRetrainDay: number;
  weeklyRetrainHour: number;
  // Active Frameworks
  frameworkH2o: boolean;
  frameworkSklearn: boolean;
  frameworkAutogluon: boolean;
  frameworkDeepLearning: boolean;
  frameworkQuantum: boolean;
  // Setter
  setMLConfig: (config: Partial<MLConfigState>) => void;
}

export const useMLConfigStore = create<MLConfigState>()(
  persist(
    (set) => ({
      h2oMaxModels: 50,
      h2oMaxRuntime: 3600,
      h2oMaxMem: '32g',
      h2oNfolds: 5,
      h2oSeed: 42,
      autogluonPresets: 'best_quality',
      autogluonTimeLimit: 3600,
      autogluonBagFolds: 8,
      autogluonStackLevels: 2,
      sklearnEstimators: 200,
      sklearnMaxDepth: 8,
      sklearnLearningRate: 0.05,
      sklearnCvFolds: 3,
      trainingWindowDays: 365,
      validationWindowDays: 30,
      stepSizeDays: 30,
      minTrainingSizeDays: 180,
      gapDays: 1,
      calibrationMethod: 'isotonic',
      calibrationCvFolds: 5,
      tierAThreshold: 0.58,
      tierBThreshold: 0.55,
      tierCThreshold: 0.52,
      ensembleMinWeight: 0.1,
      ensembleWeightDecay: 0.95,
      maxModelsPerSport: 5,
      modelTtlDays: 90,
      rollingWindows: [3, 5, 10, 15, 30],
      eloBaseRating: 1500,
      momentumDecay: 0.9,
      targetAccuracy: 0.60,
      targetAuc: 0.60,
      targetCalibrationError: 0.05,
      weeklyRetrainDay: 0,
      weeklyRetrainHour: 4,
      frameworkH2o: true,
      frameworkSklearn: true,
      frameworkAutogluon: true,
      frameworkDeepLearning: false,
      frameworkQuantum: false,
      setMLConfig: (config) => set((state) => ({ ...state, ...config })),
    }),
    { name: 'ml-config-storage' }
  )
);

// Filter Store
interface FilterState {
  selectedSport: string;
  selectedTier: string;
  setSelectedSport: (sport: string) => void;
  setSelectedTier: (tier: string) => void;
}

export const useFilterStore = create<FilterState>((set) => ({
  selectedSport: 'all',
  selectedTier: 'all',
  setSelectedSport: (selectedSport) => set({ selectedSport }),
  setSelectedTier: (selectedTier) => set({ selectedTier }),
}));