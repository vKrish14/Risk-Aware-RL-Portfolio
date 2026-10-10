"""Small, assumption-explicit finance calculations for the dashboard."""
from __future__ import annotations


def future_value(
    starting_value: float,
    annual_contribution: float,
    annual_rate: float,
    years: int,
) -> float:
    """Compound an initial value plus end-of-year contributions.

    This is a deterministic teaching calculator. It assumes the same annual
    rate every year, contributions at year-end, no taxes, and no fees.
    """
    if starting_value < 0 or annual_contribution < 0:
        raise ValueError("Starting value and annual contribution must be nonnegative.")
    if not -1 <= annual_rate or years < 0:
        raise ValueError("Annual rate must be at least -100% and years nonnegative.")
    if years == 0:
        return float(starting_value)
    growth = (1 + annual_rate) ** years
    if annual_rate == 0:
        return float(starting_value + annual_contribution * years)
    return float(starting_value * growth + annual_contribution * (growth - 1) / annual_rate)


def cagr(starting_value: float, ending_value: float, years: float) -> float:
    """Return the annualized compound growth rate for a positive-value path."""
    if starting_value <= 0 or ending_value < 0 or years <= 0:
        raise ValueError("Values must be positive (ending may be zero) and years must be positive.")
    if ending_value == 0:
        return -1.0
    return float((ending_value / starting_value) ** (1 / years) - 1)
