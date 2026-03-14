from datetime import datetime

import pytest

from src.routing import solve_vrp_greedy, solve_vrp_ortools
from src.utils import Car, Passenger, create_city_graph


@pytest.fixture
def city_data():
    """Create a 10x10 grid city using proper project utils."""
    G, locations = create_city_graph(grid_size=10, seed=42)
    return G, locations


@pytest.fixture
def fleet(city_data):
    """Create 2 cars at known locations."""
    _, locations = city_data
    # Car 0 at Node 0 (0,0)
    # Car 1 at Node 99 (9,9)
    return [
        Car(id=0, current_location=locations[0]),
        Car(id=1, current_location=locations[99]),
    ]


@pytest.fixture
def passengers(city_data):
    """Create 2 passengers."""
    _, locations = city_data
    # P1: Node 1 -> Node 11 (Near Car 0)
    # P2: Node 98 -> Node 88 (Near Car 1)

    p1 = Passenger(
        id=0, pickup=locations[1], dropoff=locations[11], request_time=datetime.now()
    )
    p2 = Passenger(
        id=1, pickup=locations[98], dropoff=locations[88], request_time=datetime.now()
    )
    return [p1, p2]


def test_greedy_assignment(city_data, fleet, passengers):
    """Test that greedy solver assigns nearest cars."""
    G, _ = city_data
    solution = solve_vrp_greedy(G, fleet, passengers)

    assert len(solution.assignments) == 2, "Should use both cars for optimal locality"
    assert len(solution.unassigned_passengers) == 0

    # Check assignments: P0 should go to Car 0, P1 to Car 1
    car0_assignment = next(a for a in solution.assignments if a.car_id == 0)
    assert 0 in car0_assignment.passengers

    car1_assignment = next(a for a in solution.assignments if a.car_id == 1)
    assert 1 in car1_assignment.passengers


def test_ortools_assignment(city_data, fleet, passengers):
    """Test that OR-Tools finds a valid solution."""
    G, _ = city_data

    # Run OR-Tools
    solution = solve_vrp_ortools(G, fleet, passengers, time_limit_seconds=5)

    assert len(solution.unassigned_passengers) == 0
    # In this obvious split case, it should likely use 2 cars
    assert len(solution.assignments) > 0

    # Optional: Check if it splits the work (global span cost effect)
    # With only 2 requests, it might not trigger strong penalties, but it should be valid.
    car_ids = [a.car_id for a in solution.assignments]
    assert len(set(car_ids)) == len(car_ids)
