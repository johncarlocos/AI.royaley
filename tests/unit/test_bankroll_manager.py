"""
Unit tests for Bankroll Manager.
Tests transaction tracking, balance management, and statistics calculation.
"""

import pytest
from decimal import Decimal
from datetime import datetime, timedelta

from app.services.betting.bankroll_manager import (
    BankrollManager,
    TransactionType,
    BankrollTransaction,
    BankrollSnapshot,
    BankrollStats,
)


class TestTransactionType:
    """Tests for TransactionType enum."""
    
    def test_transaction_types_exist(self):
        """Verify all transaction types are defined."""
        assert TransactionType.DEPOSIT.value == "deposit"
        assert TransactionType.WITHDRAWAL.value == "withdrawal"
        assert TransactionType.BET_PLACED.value == "bet_placed"
        assert TransactionType.BET_WON.value == "bet_won"
        assert TransactionType.BET_LOST.value == "bet_lost"
        assert TransactionType.BET_PUSH.value == "bet_push"
        assert TransactionType.ADJUSTMENT.value == "adjustment"
        assert TransactionType.BONUS.value == "bonus"


class TestBankrollTransaction:
    """Tests for BankrollTransaction dataclass."""
    
    def test_transaction_creation(self):
        """Test creating a transaction."""
        tx = BankrollTransaction(
            id="tx_123",
            transaction_type=TransactionType.DEPOSIT,
            amount=Decimal("1000.00"),
            balance_before=Decimal("0.00"),
            balance_after=Decimal("1000.00"),
            description="Initial deposit",
        )
        assert tx.id == "tx_123"
        assert tx.amount == Decimal("1000.00")
        assert tx.balance_after == Decimal("1000.00")
    
    def test_transaction_with_reference(self):
        """Test transaction with bet reference."""
        tx = BankrollTransaction(
            id="tx_456",
            transaction_type=TransactionType.BET_PLACED,
            amount=Decimal("-50.00"),
            balance_before=Decimal("1000.00"),
            balance_after=Decimal("950.00"),
            description="Bet on NBA game",
            reference_id="bet_789",
        )
        assert tx.reference_id == "bet_789"
        assert tx.amount == Decimal("-50.00")


class TestBankrollManager:
    """Tests for BankrollManager class."""
    
    @pytest.fixture
    def manager(self):
        """Create a fresh bankroll manager."""
        return BankrollManager(initial_balance=Decimal("10000.00"))
    
    def test_initial_balance(self, manager):
        """Test initial balance is set correctly."""
        assert manager.balance == Decimal("10000.00")
        assert manager.initial_balance == Decimal("10000.00")
    
    def test_deposit(self, manager):
        """Test deposit increases balance."""
        result = manager.deposit(Decimal("500.00"), "Weekly deposit")
        assert manager.balance == Decimal("10500.00")
        assert result.transaction_type == TransactionType.DEPOSIT
        assert result.amount == Decimal("500.00")
    
    def test_withdrawal(self, manager):
        """Test withdrawal decreases balance."""
        result = manager.withdraw(Decimal("1000.00"), "Withdrawal")
        assert manager.balance == Decimal("9000.00")
        assert result.transaction_type == TransactionType.WITHDRAWAL
        assert result.amount == Decimal("-1000.00")
    
    def test_withdrawal_insufficient_funds(self, manager):
        """Test withdrawal with insufficient funds."""
        with pytest.raises(ValueError, match="Insufficient funds"):
            manager.withdraw(Decimal("15000.00"))
    
    def test_place_bet(self, manager):
        """Test placing a bet."""
        result = manager.place_bet(Decimal("100.00"), bet_id="bet_001")
        assert manager.balance == Decimal("9900.00")
        assert result.transaction_type == TransactionType.BET_PLACED
        assert result.reference_id == "bet_001"
    
    def test_record_win_positive_odds(self, manager):
        """Test recording a win with positive odds."""
        manager.place_bet(Decimal("100.00"), bet_id="bet_001")
        result = manager.record_win(
            stake=Decimal("100.00"),
            odds=150,  # +150
            bet_id="bet_001"
        )
        # Win: stake + (stake * 150/100) = 100 + 150 = 250
        assert manager.balance == Decimal("10150.00")
        assert result.transaction_type == TransactionType.BET_WON
    
    def test_record_win_negative_odds(self, manager):
        """Test recording a win with negative odds."""
        manager.place_bet(Decimal("150.00"), bet_id="bet_002")
        result = manager.record_win(
            stake=Decimal("150.00"),
            odds=-150,  # -150
            bet_id="bet_002"
        )
        # Win: stake + (stake * 100/150) = 150 + 100 = 250
        assert manager.balance == Decimal("10100.00")
    
    def test_record_loss(self, manager):
        """Test recording a loss."""
        manager.place_bet(Decimal("100.00"), bet_id="bet_003")
        result = manager.record_loss(bet_id="bet_003")
        assert manager.balance == Decimal("9900.00")
        assert result.transaction_type == TransactionType.BET_LOST
    
    def test_record_push(self, manager):
        """Test recording a push (stake returned)."""
        manager.place_bet(Decimal("100.00"), bet_id="bet_004")
        result = manager.record_push(
            stake=Decimal("100.00"),
            bet_id="bet_004"
        )
        assert manager.balance == Decimal("10000.00")
        assert result.transaction_type == TransactionType.BET_PUSH
    
    def test_peak_balance_tracking(self, manager):
        """Test peak balance is tracked."""
        assert manager.peak_balance == Decimal("10000.00")
        manager.deposit(Decimal("5000.00"))
        assert manager.peak_balance == Decimal("15000.00")
        manager.withdraw(Decimal("2000.00"))
        assert manager.peak_balance == Decimal("15000.00")  # Still at peak
    
    def test_low_balance_tracking(self, manager):
        """Test low balance is tracked."""
        assert manager.low_balance == Decimal("10000.00")
        manager.withdraw(Decimal("5000.00"))
        assert manager.low_balance == Decimal("5000.00")
        manager.deposit(Decimal("2000.00"))
        assert manager.low_balance == Decimal("5000.00")  # Still at low
    
    def test_transaction_history(self, manager):
        """Test transaction history is maintained."""
        manager.deposit(Decimal("100.00"))
        manager.place_bet(Decimal("50.00"), bet_id="bet_1")
        manager.record_win(Decimal("50.00"), odds=100, bet_id="bet_1")
        
        transactions = manager.get_transactions()
        assert len(transactions) == 3
        assert transactions[0].transaction_type == TransactionType.DEPOSIT
        assert transactions[1].transaction_type == TransactionType.BET_PLACED
        assert transactions[2].transaction_type == TransactionType.BET_WON


class TestBankrollStats:
    """Tests for BankrollStats calculation."""
    
    @pytest.fixture
    def manager_with_history(self):
        """Create manager with betting history."""
        manager = BankrollManager(initial_balance=Decimal("1000.00"))
        
        # Place and resolve several bets
        manager.place_bet(Decimal("100.00"), bet_id="bet_1")
        manager.record_win(Decimal("100.00"), odds=100, bet_id="bet_1")
        
        manager.place_bet(Decimal("100.00"), bet_id="bet_2")
        manager.record_loss(bet_id="bet_2")
        
        manager.place_bet(Decimal("100.00"), bet_id="bet_3")
        manager.record_win(Decimal("100.00"), odds=-110, bet_id="bet_3")
        
        manager.place_bet(Decimal("100.00"), bet_id="bet_4")
        manager.record_push(Decimal("100.00"), bet_id="bet_4")
        
        return manager
    
    def test_stats_calculation(self, manager_with_history):
        """Test statistics are calculated correctly."""
        stats = manager_with_history.get_stats()
        
        assert stats.total_bets == 4
        assert stats.wins == 2
        assert stats.losses == 1
        assert stats.pushes == 1
        assert stats.win_rate == pytest.approx(0.6667, rel=0.01)  # 2/3 (excluding push)
    
    def test_profit_calculation(self, manager_with_history):
        """Test profit is calculated correctly."""
        stats = manager_with_history.get_stats()
        
        # Won: +100 (from +100 odds) + ~90.91 (from -110 odds)
        # Lost: -100
        # Net: ~90.91 profit
        assert stats.net_profit > Decimal("0")
    
    def test_roi_calculation(self, manager_with_history):
        """Test ROI is calculated correctly."""
        stats = manager_with_history.get_stats()
        
        # ROI = net_profit / total_wagered * 100
        assert stats.roi is not None


class TestBankrollSnapshot:
    """Tests for daily snapshot functionality."""
    
    def test_snapshot_creation(self):
        """Test creating a daily snapshot."""
        snapshot = BankrollSnapshot(
            date=datetime.now().date(),
            opening_balance=Decimal("1000.00"),
            closing_balance=Decimal("1050.00"),
            deposits=Decimal("0.00"),
            withdrawals=Decimal("0.00"),
            bets_placed=5,
            bets_won=3,
            bets_lost=2,
            bets_pushed=0,
            amount_wagered=Decimal("500.00"),
            amount_won=Decimal("325.00"),
            amount_lost=Decimal("200.00"),
            net_profit=Decimal("50.00"),
            roi=10.0,
        )
        
        assert snapshot.closing_balance == Decimal("1050.00")
        assert snapshot.net_profit == Decimal("50.00")
        assert snapshot.roi == 10.0


class TestDrawdownCalculation:
    """Tests for drawdown calculation."""
    
    def test_max_drawdown(self):
        """Test maximum drawdown calculation."""
        manager = BankrollManager(initial_balance=Decimal("1000.00"))
        
        # Increase to 1200 (new peak)
        manager.deposit(Decimal("200.00"))
        
        # Decrease to 800 (400 drawdown, 33.33% from peak)
        manager.withdraw(Decimal("400.00"))
        
        stats = manager.get_stats()
        assert stats.max_drawdown_amount == Decimal("400.00")
        assert stats.max_drawdown_percent == pytest.approx(33.33, rel=0.1)
    
    def test_current_drawdown(self):
        """Test current drawdown calculation."""
        manager = BankrollManager(initial_balance=Decimal("1000.00"))
        
        # Increase to 1500 (peak)
        manager.deposit(Decimal("500.00"))
        
        # Decrease to 1200 (300 current drawdown)
        manager.withdraw(Decimal("300.00"))
        
        stats = manager.get_stats()
        assert stats.current_drawdown == Decimal("300.00")


class TestStreakTracking:
    """Tests for win/loss streak tracking."""
    
    def test_win_streak(self):
        """Test winning streak tracking."""
        manager = BankrollManager(initial_balance=Decimal("1000.00"))
        
        for i in range(5):
            manager.place_bet(Decimal("10.00"), bet_id=f"bet_{i}")
            manager.record_win(Decimal("10.00"), odds=100, bet_id=f"bet_{i}")
        
        stats = manager.get_stats()
        assert stats.current_streak == 5
        assert stats.longest_win_streak == 5
    
    def test_loss_streak(self):
        """Test losing streak tracking."""
        manager = BankrollManager(initial_balance=Decimal("1000.00"))
        
        for i in range(3):
            manager.place_bet(Decimal("10.00"), bet_id=f"bet_{i}")
            manager.record_loss(bet_id=f"bet_{i}")
        
        stats = manager.get_stats()
        assert stats.current_streak == -3
        assert stats.longest_loss_streak == 3
    
    def test_streak_reset(self):
        """Test streak resets on opposite result."""
        manager = BankrollManager(initial_balance=Decimal("1000.00"))
        
        # Win 3
        for i in range(3):
            manager.place_bet(Decimal("10.00"), bet_id=f"win_{i}")
            manager.record_win(Decimal("10.00"), odds=100, bet_id=f"win_{i}")
        
        # Lose 1 (resets streak)
        manager.place_bet(Decimal("10.00"), bet_id="loss_0")
        manager.record_loss(bet_id="loss_0")
        
        stats = manager.get_stats()
        assert stats.current_streak == -1
        assert stats.longest_win_streak == 3
