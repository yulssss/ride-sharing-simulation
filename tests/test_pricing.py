import pytest

from src.pricing import PricingConfig, calculate_surge_multiplier


def test_surge_pricing_low_demand():
    """Test pricing when supply exceeds demand."""
    config = PricingConfig()
    multiplier, reason = calculate_surge_multiplier(
        num_active_requests=10, num_available_cars=20, config=config
    )
    assert multiplier == 1.0
    assert reason == "Normal pricing"


def test_surge_pricing_high_demand():
    """Test pricing when demand exceeds supply."""
    config = PricingConfig()
    # Ratio = 3.0, should trigger high/extreme surge
    multiplier, reason = calculate_surge_multiplier(
        num_active_requests=30, num_available_cars=10, config=config
    )
    assert multiplier >= config.surge_multiplier_high
    assert "High demand" in reason or "Extreme" in reason


def test_zero_supply():
    """Test pricing when no cars are available."""
    config = PricingConfig()
    multiplier, reason = calculate_surge_multiplier(
        num_active_requests=5, num_available_cars=0, config=config
    )
    assert multiplier == config.surge_multiplier_extreme
    assert reason == "No cars available"


def test_zone_specific_surge():
    """Test that high zone density increases surge."""
    config = PricingConfig()
    zone_demand = {1: 10, 2: 1}  # Zone 1 is hot

    # Global ratio is 1:1 (normal)
    # But checking for Zone 1 should boost it
    multiplier, reason = calculate_surge_multiplier(
        num_active_requests=20,
        num_available_cars=20,
        zone_demand=zone_demand,
        zone_id=1,
        config=config,
    )

    # We expect some zone adjustment > 1.0 even if global is 1.0
    # Note: Exact boost depends on implementation details, but generally > 1
    assert multiplier > 1.0
