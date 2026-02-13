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