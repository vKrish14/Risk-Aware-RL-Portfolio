"""Tests for the small, assumption-explicit finance calculators."""
import pytest

from src.dashboard.finance_kit import cagr, future_value


def test_future_value_with_zero_rate_and_end_of_year_contributions() -> None:
    assert future_value(1000, 200, 0, 3) == pytest.approx(1600)


def test_future_value_compounds_initial_value_and_contributions() -> None:
    assert future_value(1000, 100, 0.10, 2) == pytest.approx(1420)


def test_cagr_round_trip() -> None:
    assert cagr(1000, 1210, 2) == pytest.approx(0.10)


def test_calculators_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        future_value(-1, 0, 0.05, 4)
    with pytest.raises(ValueError):
        cagr(1000, 1200, 0)
