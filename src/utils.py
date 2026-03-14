"""
Utility functions for the ride-pooling service.
- Synthetic city graph generation (grid-based)
- Historical ride data simulation
- Helper functions for distance calculations
"""

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd

# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class Location:
    """A location node in the city graph."""

    id: int
    x: float
    y: float
    name: str = ""

    def __hash__(self):
        return hash(self.id)


@dataclass
class Passenger:
    """A passenger requesting a ride."""

    id: int
    pickup: Location
    dropoff: Location
    request_time: datetime
    is_assigned: bool = False
    assigned_car_id: int = None


@dataclass
class Car:
    """A car in the fleet."""

    id: int
    current_location: Location
    capacity: int = 4
    passengers: List[Passenger] = field(default_factory=list)
    route: List[Location] = field(default_factory=list)
    is_available: bool = True


# =============================================================================
# City Graph Generation
# =============================================================================


def create_city_graph(
    grid_size: int = 10, seed: int = 42
) -> Tuple[nx.Graph, Dict[int, Location]]:
    """
    Create a synthetic city as a grid graph.

    Args:
        grid_size: Size of the grid (grid_size x grid_size nodes)
        seed: Random seed for reproducibility

    Returns:
        G: NetworkX graph representing the city
        locations: Dictionary mapping node_id -> Location
    """
    random.seed(seed)
    np.random.seed(seed)

    G = nx.Graph()
    locations = {}

    node_id = 0
    for i in range(grid_size):
        for j in range(grid_size):
            loc = Location(id=node_id, x=float(i), y=float(j), name=f"Node_{node_id}")
            locations[node_id] = loc
            G.add_node(node_id, pos=(i, j), location=loc)
            node_id += 1

    # Add edges (grid connections + some diagonal shortcuts)
    for i in range(grid_size):
        for j in range(grid_size):
            current = i * grid_size + j

            # Right neighbor
            if j < grid_size - 1:
                right = i * grid_size + (j + 1)
                distance = 1.0 + random.uniform(0, 0.3)  # Add some variance
                G.add_edge(current, right, weight=distance)

            # Bottom neighbor
            if i < grid_size - 1:
                bottom = (i + 1) * grid_size + j
                distance = 1.0 + random.uniform(0, 0.3)
                G.add_edge(current, bottom, weight=distance)

            # Diagonal (bottom-right) with some probability
            if i < grid_size - 1 and j < grid_size - 1 and random.random() < 0.3:
                diagonal = (i + 1) * grid_size + (j + 1)
                distance = 1.414 + random.uniform(0, 0.3)  # sqrt(2) ≈ 1.414
                G.add_edge(current, diagonal, weight=distance)

    return G, locations


def get_shortest_path_distance(G: nx.Graph, from_node: int, to_node: int) -> float:
    """Get the shortest path distance between two nodes."""
    try:
        return nx.shortest_path_length(G, from_node, to_node, weight="weight")
    except nx.NetworkXNoPath:
        return float("inf")


def get_shortest_path(G: nx.Graph, from_node: int, to_node: int) -> List[int]:
    """Get the shortest path between two nodes."""
    try:
        return nx.shortest_path(G, from_node, to_node, weight="weight")
    except nx.NetworkXNoPath:
        return []


# =============================================================================
# Synthetic Data Generation
# =============================================================================


def generate_historical_rides(
    locations: Dict[int, Location],
    num_days: int = 30,
    rides_per_day: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic historical ride data for training the prediction model.

    The data simulates realistic patterns:
    - Morning rush (7-9 AM): High demand from residential to business areas
    - Evening rush (5-7 PM): High demand from business to residential areas
    - Hotspots: Certain areas have higher demand

    Args:
        locations: Dictionary of all locations
        num_days: Number of days of historical data
        rides_per_day: Average number of rides per day
        seed: Random seed

    Returns:
        DataFrame with columns: timestamp, hour, day_of_week, pickup_id, dropoff_id, zone_id
    """
    random.seed(seed)
    np.random.seed(seed)

    location_ids = list(locations.keys())
    num_locations = len(location_ids)

    # Define hotspots (certain nodes have higher demand)
    hotspot_ids = random.sample(location_ids, k=min(10, num_locations // 5))

    rides = []
    start_date = datetime(2025, 12, 1)  # Start from a month ago

    for day in range(num_days):
        current_date = start_date + timedelta(days=day)
        day_of_week = current_date.weekday()

        # More rides on weekdays
        daily_rides = int(rides_per_day * (1.2 if day_of_week < 5 else 0.8))

        for _ in range(daily_rides):
            # Generate hour with realistic distribution (peaks at rush hours)
            hour = _sample_hour_with_rush_pattern()

            # Bias pickups toward hotspots
            if random.random() < 0.6:
                pickup_id = random.choice(hotspot_ids)
            else:
                pickup_id = random.choice(location_ids)

            # Dropoff can be anywhere
            dropoff_id = random.choice(
                [loc for loc in location_ids if loc != pickup_id]
            )

            timestamp = current_date.replace(hour=hour, minute=random.randint(0, 59))

            rides.append(
                {
                    "timestamp": timestamp,
                    "hour": hour,
                    "day_of_week": day_of_week,
                    "pickup_id": pickup_id,
                    "dropoff_id": dropoff_id,
                    "zone_id": pickup_id // 10,  # Group into zones for prediction
                }
            )

    return pd.DataFrame(rides)


def _sample_hour_with_rush_pattern() -> int:
    """Sample an hour with realistic rush-hour patterns."""
    # Probability weights for each hour (higher during rush hours)
    weights = [
        0.5,
        0.3,
        0.2,
        0.2,
        0.3,
        0.5,  # 0-5 AM (low)
        1.0,
        2.5,
        3.0,
        2.0,
        1.5,
        1.5,  # 6-11 AM (morning rush)
        1.8,
        1.5,
        1.5,
        1.5,
        2.0,
        3.0,  # 12-5 PM (lunch + evening rush start)
        2.5,
        2.0,
        1.5,
        1.2,
        1.0,
        0.7,  # 6-11 PM (evening decline)
    ]
    hours = list(range(24))
    return random.choices(hours, weights=weights, k=1)[0]


def generate_current_requests(
    locations: Dict[int, Location],
    num_requests: int = 50,
    current_time: datetime = None,
    seed: int = None,
) -> List[Passenger]:
    """
    Generate current ride requests for the simulation.

    Args:
        locations: Dictionary of all locations
        num_requests: Number of passengers to generate
        current_time: Current simulation time
        seed: Random seed (optional)

    Returns:
        List of Passenger objects
    """
    if seed is not None:
        random.seed(seed)

    if current_time is None:
        current_time = datetime.now()

    location_ids = list(locations.keys())
    passengers = []

    for i in range(num_requests):
        pickup_id = random.choice(location_ids)
        dropoff_id = random.choice([loc for loc in location_ids if loc != pickup_id])

        # Stagger request times slightly
        request_time = current_time + timedelta(seconds=random.randint(0, 300))

        passengers.append(
            Passenger(
                id=i,
                pickup=locations[pickup_id],
                dropoff=locations[dropoff_id],
                request_time=request_time,
            )
        )

    return passengers


def initialize_fleet(
    locations: Dict[int, Location], num_cars: int = 10, seed: int = 42
) -> List[Car]:
    """
    Initialize the car fleet at random positions.

    Args:
        locations: Dictionary of all locations
        num_cars: Number of cars in the fleet
        seed: Random seed

    Returns:
        List of Car objects
    """
    random.seed(seed)
    location_ids = list(locations.keys())

    cars = []
    for i in range(num_cars):
        start_location = locations[random.choice(location_ids)]
        cars.append(Car(id=i, current_location=start_location))

    return cars


# =============================================================================
# Distance Matrix Helpers
# =============================================================================


def compute_distance_matrix(G: nx.Graph, node_ids: List[int]) -> np.ndarray:
    """
    Compute a distance matrix for the given nodes.

    Args:
        G: NetworkX graph
        node_ids: List of node IDs to include

    Returns:
        2D numpy array of distances
    """
    n = len(node_ids)
    matrix = np.zeros((n, n))

    # Precompute all shortest paths
    all_paths = dict(nx.all_pairs_dijkstra_path_length(G, weight="weight"))

    for i, from_node in enumerate(node_ids):
        for j, to_node in enumerate(node_ids):
            if i != j:
                matrix[i][j] = all_paths.get(from_node, {}).get(to_node, float("inf"))

    return matrix


def euclidean_distance(loc1: Location, loc2: Location) -> float:
    """Calculate Euclidean distance between two locations."""
    return np.sqrt((loc1.x - loc2.x) ** 2 + (loc1.y - loc2.y) ** 2)
