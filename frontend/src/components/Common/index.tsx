// src/components/Common/index.tsx
import React from 'react';
import { Box, Typography, Chip, Alert, AlertTitle } from '@mui/material';
import { WifiOff } from '@mui/icons-material';

// Tier Badge
interface TierBadgeProps {
  tier: string;
}

export const TierBadge: React.FC<TierBadgeProps> = ({ tier }) => {
  const colors: Record<string, { bg: string; color: string }> = {
    'A': { bg: '#4caf50', color: '#fff' },
    'B': { bg: '#2196f3', color: '#fff' },
    'C': { bg: '#ff9800', color: '#fff' },
    'D': { bg: '#9e9e9e', color: '#fff' },
  };
  const style = colors[tier] || colors['D'];

  return (
    <Chip
      label={tier}
      size="small"
      sx={{
        bgcolor: style.bg,
        color: style.color,
        fontWeight: 700,
        minWidth: 28,
        height: 22,
        fontSize: 11,
      }}
    />
  );
};

// Offline Indicator
export const OfflineIndicator: React.FC = () => (
  <Alert
    severity="warning"
    icon={<WifiOff />}
    sx={{
      position: 'fixed',
      bottom: 16,
      left: '50%',
      transform: 'translateX(-50%)',
      zIndex: 9999,
    }}
  >
    <AlertTitle>You are offline</AlertTitle>
    Some features may be unavailable.
  </Alert>
);

// Stat Card
interface StatCardProps {
  title: string;
  value: string | number;
  color?: string;
}

export const StatCard: React.FC<StatCardProps> = ({ title, value, color = 'primary.main' }) => (
  <Box>
    <Typography variant="caption" color="textSecondary">{title}</Typography>
    <Typography variant="h5" fontWeight={700} sx={{ color }}>{value}</Typography>
  </Box>
);

// Table Skeleton
export const TableSkeleton: React.FC = () => (
  <Box p={4} textAlign="center">
    <Typography color="textSecondary">Loading...</Typography>
  </Box>
);
