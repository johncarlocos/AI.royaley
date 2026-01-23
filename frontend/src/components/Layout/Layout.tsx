// src/components/Layout/Layout.tsx
import React, { useState } from 'react';
import { Outlet, useNavigate, useLocation } from 'react-router-dom';
import {
  Box, Drawer, AppBar, Toolbar, Typography, List, ListItem,
  ListItemIcon, ListItemText, ListItemButton, IconButton, Avatar,
  Menu, MenuItem, Divider, Badge, Tooltip, useMediaQuery, useTheme
} from '@mui/material';
import {
  Dashboard, TrendingUp, AccountBalance, Analytics, Science,
  Psychology, SportsBasketball, PlayCircle, MonitorHeart,
  Settings, Menu as MenuIcon, Brightness4, Brightness7, Logout,
  ChevronLeft, SportsScore
} from '@mui/icons-material';
import { useAuthStore, useAlertStore, useSettingsStore } from '../../store';
import { useOnlineStatus } from '../../hooks';
import { OfflineIndicator } from '../Common';

const DRAWER_WIDTH = 240;

const menuItems = [
  { text: 'Dashboard', icon: <Dashboard />, path: '/' },
  { text: 'Predictions', icon: <TrendingUp />, path: '/predictions' },
  { text: 'Betting', icon: <AccountBalance />, path: '/betting' },
  { text: 'Analytics', icon: <Analytics />, path: '/analytics' },
  { text: 'Backtesting', icon: <Science />, path: '/backtesting' },
  { text: 'Models', icon: <Psychology />, path: '/models' },
  { divider: true },
  { text: 'Player Props', icon: <SportsBasketball />, path: '/player-props' },
  { text: 'Game Props', icon: <SportsScore />, path: '/game-props' },
  { text: 'LIVE', icon: <PlayCircle />, path: '/live' },
  { text: 'System Health', icon: <MonitorHeart />, path: '/alerts', badge: true },
  { divider: true },
  { text: 'Settings', icon: <Settings />, path: '/settings' },
];

const Layout: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const muiTheme = useTheme();
  const isMobile = useMediaQuery(muiTheme.breakpoints.down('md'));
  const isOnline = useOnlineStatus();

  const [mobileOpen, setMobileOpen] = useState(false);
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);

  const { user, logout } = useAuthStore();
  const { unreadCount } = useAlertStore();
  const { theme, setTheme } = useSettingsStore();

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  const drawer = (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <Box sx={{ p: 2, display: 'flex', alignItems: 'center' }}>
        <Typography variant="h6" sx={{ fontWeight: 700, background: 'linear-gradient(90deg, #3b82f6, #8b5cf6)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
          LOYALEY
        </Typography>
      </Box>
      <Divider />
      <List sx={{ flex: 1, py: 1 }}>
        {menuItems.map((item, index) => {
          if ('divider' in item) return <Divider key={index} sx={{ my: 1 }} />;
          const isActive = location.pathname === item.path;
          return (
            <ListItem key={item.text} disablePadding sx={{ px: 1 }}>
              <ListItemButton
                onClick={() => { navigate(item.path); if (isMobile) setMobileOpen(false); }}
                sx={{
                  borderRadius: 2,
                  mb: 0.5,
                  bgcolor: isActive ? 'rgba(59, 130, 246, 0.12)' : 'transparent',
                  '&:hover': { bgcolor: 'rgba(59, 130, 246, 0.08)' },
                }}
              >
                <ListItemIcon sx={{ minWidth: 36, color: isActive ? 'primary.main' : 'inherit' }}>
                  {'badge' in item && item.badge && unreadCount > 0 ? (
                    <Badge badgeContent={unreadCount} color="error">{item.icon}</Badge>
                  ) : (
                    item.icon
                  )}
                </ListItemIcon>
                <ListItemText primary={item.text} primaryTypographyProps={{ fontWeight: isActive ? 600 : 400, fontSize: 14 }} />
              </ListItemButton>
            </ListItem>
          );
        })}
      </List>
      <Box sx={{ p: 2, borderTop: 1, borderColor: 'divider' }}>
        <Typography variant="caption" color="textSecondary">v3.0.0</Typography>
      </Box>
    </Box>
  );

  return (
    <Box sx={{ display: 'flex', minHeight: '100vh' }}>
      <AppBar position="fixed" sx={{ width: { md: `calc(100% - ${DRAWER_WIDTH}px)` }, ml: { md: `${DRAWER_WIDTH}px` }, bgcolor: 'background.paper', borderBottom: 1, borderColor: 'divider' }} elevation={0}>
        <Toolbar>
          <IconButton edge="start" onClick={() => setMobileOpen(!mobileOpen)} sx={{ mr: 2, display: { md: 'none' }, color: theme === 'dark' ? 'grey.300' : 'grey.700' }}>
            <MenuIcon />
          </IconButton>
          <Box flex={1} />
          <Tooltip title={theme === 'dark' ? 'Light Mode' : 'Dark Mode'}>
            <IconButton onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')} sx={{ color: theme === 'dark' ? 'grey.300' : 'grey.700' }}>
              {theme === 'dark' ? <Brightness7 /> : <Brightness4 />}
            </IconButton>
          </Tooltip>
          <Tooltip title="System Health">
            <IconButton onClick={() => navigate('/alerts')} sx={{ color: theme === 'dark' ? 'grey.300' : 'grey.700' }}>
              <Badge badgeContent={unreadCount} color="error"><MonitorHeart /></Badge>
            </IconButton>
          </Tooltip>
          <IconButton onClick={(e) => setAnchorEl(e.currentTarget)} sx={{ ml: 1 }}>
            <Avatar sx={{ width: 36, height: 36, bgcolor: 'primary.main' }}>{user?.email?.[0]?.toUpperCase() || 'U'}</Avatar>
          </IconButton>
          <Menu anchorEl={anchorEl} open={Boolean(anchorEl)} onClose={() => setAnchorEl(null)}>
            <MenuItem disabled><Typography variant="body2" color="textSecondary">{user?.email}</Typography></MenuItem>
            <Divider />
            <MenuItem onClick={() => { setAnchorEl(null); navigate('/settings'); }}><ListItemIcon><Settings fontSize="small" /></ListItemIcon>Settings</MenuItem>
            <MenuItem onClick={handleLogout}><ListItemIcon><Logout fontSize="small" /></ListItemIcon>Logout</MenuItem>
          </Menu>
        </Toolbar>
      </AppBar>

      <Box component="nav" sx={{ width: { md: DRAWER_WIDTH }, flexShrink: { md: 0 } }}>
        <Drawer variant="temporary" open={mobileOpen} onClose={() => setMobileOpen(false)} ModalProps={{ keepMounted: true }} sx={{ display: { xs: 'block', md: 'none' }, '& .MuiDrawer-paper': { width: DRAWER_WIDTH } }}>
          {drawer}
        </Drawer>
        <Drawer variant="permanent" sx={{ display: { xs: 'none', md: 'block' }, '& .MuiDrawer-paper': { width: DRAWER_WIDTH } }} open>
          {drawer}
        </Drawer>
      </Box>

      <Box component="main" sx={{ flexGrow: 1, p: 3, width: { md: `calc(100% - ${DRAWER_WIDTH}px)` }, minHeight: '100vh', bgcolor: 'background.default' }}>
        <Toolbar />
        <Outlet />
      </Box>

      {!isOnline && <OfflineIndicator />}
    </Box>
  );
};

export default Layout;
