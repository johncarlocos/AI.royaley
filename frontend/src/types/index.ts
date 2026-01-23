// src/types/index.ts

export const SPORTS = [
  { code: 'NBA', name: 'NBA Basketball' },
  { code: 'NFL', name: 'NFL Football' },
  { code: 'MLB', name: 'MLB Baseball' },
  { code: 'NHL', name: 'NHL Hockey' },
  { code: 'NCAAB', name: 'NCAA Basketball' },
  { code: 'NCAAF', name: 'NCAA Football' },
  { code: 'WNBA', name: 'WNBA Basketball' },
  { code: 'CFL', name: 'CFL Football' },
  { code: 'ATP', name: 'ATP Tennis' },
  { code: 'WTA', name: 'WTA Tennis' },
];

export const PROP_TYPES = {
  basketball: [
    { value: 'points', label: 'Points' },
    { value: 'rebounds', label: 'Rebounds' },
    { value: 'assists', label: 'Assists' },
    { value: 'pra', label: 'Pts + Reb + Ast' },
    { value: 'threes', label: '3-Pointers Made' },
  ],
  football: [
    { value: 'passing_yards', label: 'Passing Yards' },
    { value: 'rushing_yards', label: 'Rushing Yards' },
    { value: 'receiving_yards', label: 'Receiving Yards' },
  ],
  baseball: [
    { value: 'strikeouts', label: 'Strikeouts' },
    { value: 'hits', label: 'Hits' },
  ],
  hockey: [
    { value: 'goals', label: 'Goals' },
    { value: 'assists', label: 'Assists' },
  ],
};

export const TIMEZONES = [
  { value: 'America/New_York', label: 'Eastern (ET)' },
  { value: 'America/Chicago', label: 'Central (CT)' },
  { value: 'America/Denver', label: 'Mountain (MT)' },
  { value: 'America/Los_Angeles', label: 'Pacific (PT)' },
  { value: 'UTC', label: 'UTC' },
];

export const BET_TYPES = [
  { value: 'spread', label: 'Spread' },
  { value: 'moneyline', label: 'Moneyline' },
  { value: 'total', label: 'Total (O/U)' },
];

export const SIGNAL_TIERS = [
  { value: 'A', label: 'Tier A (65%+)' },
  { value: 'B', label: 'Tier B (60-65%)' },
  { value: 'C', label: 'Tier C (55-60%)' },
  { value: 'D', label: 'Tier D (<55%)' },
];
