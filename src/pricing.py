"""
Dynamic Pricing Module for the ride-pooling service.

Implements surge pricing based on:
- Current demand (number of active requests)
- Current supply (number of available cars)
- Zone-based demand density
- Time of day factors
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

from src.utils import Car, Location, Passenger

# =============================================================================
# Configuration
# =============================================================================


@dataclass
class PricingConfig:
    """Configuration for the pricing model."""

    base_fare: float = 2.50  # Base fare in dollars
    per_km_rate: float = 1.50  # Cost per kilometer
    per_minute_rate: float = 0.25  # Cost per minute of travel
    minimum_fare: float = 5.00  # Minimum fare

    # Surge pricing thresholds
    surge_threshold_low: float = 1.5  # demand/supply ratio for 1.25x surge
    surge_threshold_medium: float = 2.0  # demand/supply ratio for 1.5x surge
    surge_threshold_high: float = 3.0  # demand/supply ratio for 2.0x surge

    # Surge multipliers
    surge_multiplier_low: float = 1.25
    surge_multiplier_medium: float = 1.5
    surge_multiplier_high: float = 2.0
    surge_multiplier_extreme: float = 2.5

    # Time-based adjustments
    rush_hour_multiplier: float = 1.2
    late_night_multiplier: float = 1.3


@dataclass
class PriceEstimate:
    """A price estimate for a ride."""

    base_fare: float
    distance_cost: float
    time_cost: float
    subtotal: float
    surge_multiplier: float
    surge_reason: str
    final_price: float
    estimated_distance_km: float
    estimated_time_minutes: float


# =============================================================================
# Core Pricing Functions
# =============================================================================


def calculate_surge_multiplier(
    num_active_requests: int,
    num_available_cars: int,
    zone_demand: Optional[Dict[int, int]] = None,
    zone_id: Optional[int] = None,
    config: PricingConfig = None,
) -> tuple[float, str]:
    """
    Calculate the surge pricing multiplier based on demand/supply ratio.

    Args:
        num_active_requests: Total active ride requests
        num_available_cars: Number of available cars
        zone_demand: Optional dict of zone_id -> request count for zone-specific pricing
        zone_id: Optional zone ID for zone-specific pricing
        config: Pricing configuration

    Returns:
        Tuple of (multiplier, reason string)
    """
    if config is None:
        config = PricingConfig()

    # Avoid division by zero
    if num_available_cars == 0:
        return config.surge_multiplier_extreme, "No cars available"

    # Calculate demand/supply ratio
    ratio = num_active_requests / num_available_cars

    # Zone-specific adjustment
    zone_adjustment = 1.0
    if zone_demand and zone_id is not None:
        zone_requests = zone_demand.get(zone_id, 0)
        avg_zone_demand = np.mean(list(zone_demand.values())) if zone_demand else 1
        if avg_zone_demand > 0:
            zone_adjustment = min(1.5, zone_requests / avg_zone_demand)

    effective_ratio = ratio * zone_adjustment

    # Determine surge level
    if effective_ratio >= config.surge_threshold_high:
        return (
            config.surge_multiplier_extreme,
            f"Extreme demand (ratio: {effective_ratio:.1f}x)",
        )
    elif effective_ratio >= config.surge_threshold_medium:
        return (
            config.surge_multiplier_high,
            f"Very high demand (ratio: {effective_ratio:.1f}x)",
        )
    elif effective_ratio >= config.surge_threshold_low:
        return (
            config.surge_multiplier_medium,
            f"High demand (ratio: {effective_ratio:.1f}x)",
        )
    elif effective_ratio >= 1.0:
        return (
            config.surge_multiplier_low,
            f"Moderate demand (ratio: {effective_ratio:.1f}x)",
        )
    else:
        return 1.0, "Normal pricing"


def get_time_multiplier(current_time: datetime, config: PricingConfig = None) -> float:
    """
    Get time-of-day pricing adjustment.

    Args:
        current_time: Current datetime
        config: Pricing configuration

    Returns:
        Time-based multiplier
    """
    if config is None:
        config = PricingConfig()

    hour = current_time.hour

    # Rush hours: 7-9 AM, 5-7 PM
    if 7 <= hour <= 9 or 17 <= hour <= 19:
        return config.rush_hour_multiplier

    # Late night: 11 PM - 5 AM
    if hour >= 23 or hour < 5:
        return config.late_night_multiplier

    return 1.0


def calculate_price(
    distance_km: float,
    estimated_time_minutes: float,
    num_active_requests: int,
    num_available_cars: int,
    current_time: datetime = None,
    zone_demand: Dict[int, int] = None,
    zone_id: int = None,
    config: PricingConfig = None,
) -> PriceEstimate:
    """
    Calculate the final price for a ride.

    Args:
        distance_km: Estimated trip distance in kilometers
        estimated_time_minutes: Estimated trip time in minutes
        num_active_requests: Current number of active requests
        num_available_cars: Current number of available cars
        current_time: Current datetime (for time-based pricing)
        zone_demand: Optional zone demand dictionary
        zone_id: Optional zone ID for the pickup location
        config: Pricing configuration

    Returns:
        PriceEstimate with all pricing details
    """
    if config is None:
        config = PricingConfig()

    if current_time is None:
        current_time = datetime.now()

    # Base calculations
    distance_cost = distance_km * config.per_km_rate
    time_cost = estimated_time_minutes * config.per_minute_rate
    subtotal = config.base_fare + distance_cost + time_cost

    # Get surge multiplier
    surge_multiplier, surge_reason = calculate_surge_multiplier(
        num_active_requests, num_available_cars, zone_demand, zone_id, config
    )

    # Apply time-of-day adjustment
    time_multiplier = get_time_multiplier(current_time, config)

    # Combined multiplier (don't stack too aggressively)
    combined_multiplier = min(
        surge_multiplier * time_multiplier, config.surge_multiplier_extreme
    )

    # Calculate final price
    final_price = max(subtotal * combined_multiplier, config.minimum_fare)

    return PriceEstimate(
        base_fare=config.base_fare,
        distance_cost=distance_cost,
        time_cost=time_cost,
        subtotal=subtotal,
        surge_multiplier=combined_multiplier,
        surge_reason=surge_reason,
        final_price=round(final_price, 2),
        estimated_distance_km=distance_km,
        estimated_time_minutes=estimated_time_minutes,
    )


# =============================================================================
# Zone-Based Pricing
# =============================================================================


def calculate_zone_demand(
    passengers: List[Passenger], grid_size: int = 10
) -> Dict[int, int]:
    """
    Calculate demand per zone based on current passengers.

    Zones are defined as groups of nodes (e.g., every 10 nodes = 1 zone).

    Args:
        passengers: List of current passengers
        grid_size: Size of the city grid

    Returns:
        Dictionary mapping zone_id -> demand count
    """
    zone_demand = {}

    for passenger in passengers:
        if not passenger.is_assigned:
            zone_id = passenger.pickup.id // grid_size
            zone_demand[zone_id] = zone_demand.get(zone_id, 0) + 1

    return zone_demand


def identify_hotspots(
    zone_demand: Dict[int, int], threshold_percentile: float = 75
) -> List[int]:
    """
    Identify demand hotspots (zones with above-threshold demand).

    Args:
        zone_demand: Dictionary of zone_id -> demand count
        threshold_percentile: Percentile above which a zone is a hotspot

    Returns:
        List of hotspot zone IDs
    """
    if not zone_demand:
        return []

    values = list(zone_demand.values())
    threshold = np.percentile(values, threshold_percentile)

    return [zone_id for zone_id, demand in zone_demand.items() if demand >= threshold]


# =============================================================================
# Pricing for Multiple Rides
# =============================================================================


def calculate_all_prices(
    passengers: List[Passenger],
    cars: List[Car],
    distances: Dict[int, float],
    current_time: datetime = None,
    config: PricingConfig = None,
) -> Dict[int, PriceEstimate]:
    """
    Calculate prices for all pending ride requests.

    Args:
        passengers: List of passengers
        cars: List of cars
        distances: Dict mapping passenger_id -> estimated trip distance
        current_time: Current datetime
        config: Pricing configuration

    Returns:
        Dictionary mapping passenger_id -> PriceEstimate
    """
    if config is None:
        config = PricingConfig()

    if current_time is None:
        current_time = datetime.now()

    # Calculate supply and demand
    num_active_requests = sum(1 for p in passengers if not p.is_assigned)
    num_available_cars = sum(1 for c in cars if c.is_available)

    # Calculate zone demand
    zone_demand = calculate_zone_demand(passengers)

    prices = {}

    for passenger in passengers:
        if passenger.id in distances:
            distance_km = distances[passenger.id]
            # Estimate time: assume 30 km/h average speed
            estimated_time = (distance_km / 30) * 60  # minutes
            zone_id = passenger.pickup.id // 10

            prices[passenger.id] = calculate_price(
                distance_km=distance_km,
                estimated_time_minutes=estimated_time,
                num_active_requests=num_active_requests,
                num_available_cars=num_available_cars,
                current_time=current_time,
                zone_demand=zone_demand,
                zone_id=zone_id,
                config=config,
            )

    return prices


def print_pricing_summary(prices: Dict[int, PriceEstimate]) -> None:
    """Print a summary of all calculated prices."""
    print("\n" + "=" * 60)
    print("PRICING SUMMARY")
    print("=" * 60)

    if not prices:
        print("No prices calculated.")
        return

    # Calculate statistics
    all_prices = [p.final_price for p in prices.values()]
    all_surges = [p.surge_multiplier for p in prices.values()]

    print(f"  Total Rides Priced: {len(prices)}")
    print(f"  Average Price: ${np.mean(all_prices):.2f}")
    print(f"  Min Price: ${min(all_prices):.2f}")
    print(f"  Max Price: ${max(all_prices):.2f}")
    print(f"  Average Surge: {np.mean(all_surges):.2f}x")

    # Show surge distribution
    surge_counts = {}
    for estimate in prices.values():
        surge_key = f"{estimate.surge_multiplier:.2f}x"
        surge_counts[surge_key] = surge_counts.get(surge_key, 0) + 1

    print("\n  Surge Distribution:")
    for surge, count in sorted(surge_counts.items()):
        print(f"    {surge}: {count} rides")

    print("=" * 60)
