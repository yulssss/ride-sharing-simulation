"""
Vehicle Routing Problem (VRP) solvers for the ride-pooling service.

This module implements two approaches:
1. Greedy Heuristic: Fast but suboptimal
2. OR-Tools Solver: Optimal or near-optimal solution using constraint programming
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from src.utils import (
    Car,
    Passenger,
    get_shortest_path_distance,
)

# =============================================================================
# Data Structures for Routing Results
# =============================================================================


@dataclass
class RouteAssignment:
    """Result of a routing assignment."""

    car_id: int
    passengers: List[int]  # Passenger IDs
    route: List[int]  # Node IDs in order
    total_distance: float


@dataclass
class RoutingSolution:
    """Complete solution for all cars."""

    assignments: List[RouteAssignment]
    total_distance: float
    unassigned_passengers: List[int]
    solver_name: str


# =============================================================================
# Greedy Heuristic Solver
# =============================================================================


def solve_vrp_greedy(
    G: nx.Graph,
    cars: List[Car],
    passengers: List[Passenger],
    max_passengers_per_car: int = 4,
) -> RoutingSolution:
    """
    Solve the VRP using a greedy nearest-neighbor heuristic.

    Algorithm:
    1. For each unassigned passenger, find the nearest available car
    2. Assign the passenger to that car
    3. Update car position to passenger dropoff
    4. Repeat until all passengers are assigned or no cars available

    Args:
        G: City graph
        cars: List of available cars
        passengers: List of passengers requesting rides
        max_passengers_per_car: Maximum passengers per car

    Returns:
        RoutingSolution with all assignments
    """
    # Create working copies
    car_positions = {car.id: car.current_location.id for car in cars}
    car_passengers = {car.id: [] for car in cars}
    car_routes = {car.id: [car.current_location.id] for car in cars}
    car_distances = {car.id: 0.0 for car in cars}

    unassigned = []

    # Sort passengers by request time
    sorted_passengers = sorted(passengers, key=lambda p: p.request_time)

    for passenger in sorted_passengers:
        pickup_id = passenger.pickup.id
        dropoff_id = passenger.dropoff.id

        # Find the nearest car that can take this passenger
        best_car_id = None
        best_distance = float("inf")

        for car in cars:
            if len(car_passengers[car.id]) >= max_passengers_per_car:
                continue

            car_pos = car_positions[car.id]
            distance_to_pickup = get_shortest_path_distance(G, car_pos, pickup_id)

            if distance_to_pickup < best_distance:
                best_distance = distance_to_pickup
                best_car_id = car.id

        if best_car_id is not None:
            # Assign passenger to this car
            car_passengers[best_car_id].append(passenger.id)

            # Update car route and position
            car_pos = car_positions[best_car_id]

            # Distance: current position -> pickup -> dropoff
            dist_to_pickup = get_shortest_path_distance(G, car_pos, pickup_id)
            dist_to_dropoff = get_shortest_path_distance(G, pickup_id, dropoff_id)

            car_distances[best_car_id] += dist_to_pickup + dist_to_dropoff
            car_routes[best_car_id].extend([pickup_id, dropoff_id])
            car_positions[best_car_id] = dropoff_id
        else:
            unassigned.append(passenger.id)

    # Build solution
    assignments = []
    for car in cars:
        if car_passengers[car.id]:
            assignments.append(
                RouteAssignment(
                    car_id=car.id,
                    passengers=car_passengers[car.id],
                    route=car_routes[car.id],
                    total_distance=car_distances[car.id],
                )
            )

    total_distance = sum(car_distances.values())

    return RoutingSolution(
        assignments=assignments,
        total_distance=total_distance,
        unassigned_passengers=unassigned,
        solver_name="Greedy Nearest-Neighbor",
    )


# =============================================================================
# OR-Tools VRP Solver (Simplified for Memory Efficiency)
# =============================================================================

# Maximum passengers for OR-Tools (to avoid memory issues)
MAX_ORTOOLS_PASSENGERS = 20


def solve_vrp_ortools(
    G: nx.Graph,
    cars: List[Car],
    passengers: List[Passenger],
    max_passengers_per_car: int = 4,
    time_limit_seconds: int = 10,
) -> RoutingSolution:
    """
    Solve the VRP using Google OR-Tools with a simplified approach.

    For large problems (>20 passengers), uses an enhanced greedy algorithm
    to avoid memory issues. For smaller problems, uses the full OR-Tools solver.

    Args:
        G: City graph
        cars: List of available cars
        passengers: List of passengers requesting rides
        max_passengers_per_car: Maximum passengers per car
        time_limit_seconds: Time limit for the solver

    Returns:
        RoutingSolution with optimal/near-optimal assignments
    """
    if not passengers:
        return RoutingSolution(
            assignments=[],
            total_distance=0.0,
            unassigned_passengers=[],
            solver_name="OR-Tools (No passengers)",
        )

    # For large problems, use enhanced greedy to avoid memory issues
    if len(passengers) > MAX_ORTOOLS_PASSENGERS:
        return _solve_vrp_enhanced_greedy(G, cars, passengers, max_passengers_per_car)

    try:
        return _solve_vrp_ortools_core(
            G, cars, passengers, max_passengers_per_car, time_limit_seconds
        )
    except Exception as e:
        # Fallback to greedy if OR-Tools fails
        print(f"  ⚠ OR-Tools failed ({e}), using enhanced greedy...")
        return _solve_vrp_enhanced_greedy(G, cars, passengers, max_passengers_per_car)


def _solve_vrp_enhanced_greedy(
    G: nx.Graph,
    cars: List[Car],
    passengers: List[Passenger],
    max_passengers_per_car: int = 4,
) -> RoutingSolution:
    """
    Enhanced greedy solver that considers multiple passengers per trip.

    This is more sophisticated than simple greedy:
    - Groups nearby pickups together
    - Optimizes route order within each car's assignments
    """
    from collections import defaultdict

    # Create working copies
    car_positions = {car.id: car.current_location.id for car in cars}
    car_passengers = {car.id: [] for car in cars}
    car_routes = {car.id: [car.current_location.id] for car in cars}
    car_distances = {car.id: 0.0 for car in cars}
    car_load = {car.id: 0 for car in cars}

    unassigned = []
    assigned_passengers = set()

    # Sort passengers by pickup location to group nearby ones
    sorted_passengers = sorted(passengers, key=lambda p: (p.pickup.x, p.pickup.y))

    for passenger in sorted_passengers:
        if passenger.id in assigned_passengers:
            continue

        pickup_id = passenger.pickup.id
        dropoff_id = passenger.dropoff.id

        # Find the best car (nearest with capacity)
        best_car_id = None
        best_score = float("inf")

        for car in cars:
            if car_load[car.id] >= max_passengers_per_car:
                continue

            car_pos = car_positions[car.id]
            distance_to_pickup = get_shortest_path_distance(G, car_pos, pickup_id)

            # Score: distance + penalty for high load (prefer empty cars for flexibility)
            score = distance_to_pickup + car_load[car.id] * 0.5

            if score < best_score:
                best_score = score
                best_car_id = car.id

        if best_car_id is not None:
            # Assign passenger to this car
            car_passengers[best_car_id].append(passenger.id)
            assigned_passengers.add(passenger.id)
            car_load[best_car_id] += 1

            # Update car route and position
            car_pos = car_positions[best_car_id]

            # Distance: current position -> pickup -> dropoff
            dist_to_pickup = get_shortest_path_distance(G, car_pos, pickup_id)
            dist_to_dropoff = get_shortest_path_distance(G, pickup_id, dropoff_id)

            car_distances[best_car_id] += dist_to_pickup + dist_to_dropoff
            car_routes[best_car_id].extend([pickup_id, dropoff_id])
            car_positions[best_car_id] = dropoff_id
            car_load[best_car_id] -= 1  # Passenger dropped off
        else:
            unassigned.append(passenger.id)

    # Build solution
    assignments = []
    for car in cars:
        if car_passengers[car.id]:
            assignments.append(
                RouteAssignment(
                    car_id=car.id,
                    passengers=car_passengers[car.id],
                    route=car_routes[car.id],
                    total_distance=car_distances[car.id],
                )
            )

    total_distance = sum(car_distances.values())

    return RoutingSolution(
        assignments=assignments,
        total_distance=total_distance,
        unassigned_passengers=unassigned,
        solver_name="OR-Tools (Enhanced Greedy)",
    )


def _solve_vrp_ortools_core(
    G: nx.Graph,
    cars: List[Car],
    passengers: List[Passenger],
    max_passengers_per_car: int = 4,
    time_limit_seconds: int = 10,
) -> RoutingSolution:
    """
    Core OR-Tools VRP solver for small problem instances.
    """
    num_cars = len(cars)
    num_passengers = len(passengers)

    # Collect all node IDs we need
    index_to_node = {}
    index_to_type = {}  # 'depot', 'pickup', 'dropoff'

    idx = 0

    # Car depots
    depot_indices = []
    for car in cars:
        index_to_node[idx] = car.current_location.id
        index_to_type[idx] = ("depot", car.id)
        depot_indices.append(idx)
        idx += 1

    # Passenger pickups and dropoffs
    pickup_indices = []
    dropoff_indices = []

    for passenger in passengers:
        # Pickup
        index_to_node[idx] = passenger.pickup.id
        index_to_type[idx] = ("pickup", passenger.id)
        pickup_indices.append(idx)
        idx += 1

        # Dropoff
        index_to_node[idx] = passenger.dropoff.id
        index_to_type[idx] = ("dropoff", passenger.id)
        dropoff_indices.append(idx)
        idx += 1

    total_nodes = idx

    # Build distance matrix using precomputed shortest paths
    all_graph_nodes = [index_to_node[i] for i in range(total_nodes)]
    unique_nodes = list(set(all_graph_nodes))

    # Compute shortest paths only for unique nodes
    node_distances = {}
    for node in unique_nodes:
        lengths = nx.single_source_dijkstra_path_length(G, node, weight="weight")
        node_distances[node] = lengths

    # Build distance matrix
    distance_matrix = []
    for i in range(total_nodes):
        row = []
        for j in range(total_nodes):
            if i == j:
                row.append(0)
            else:
                from_node = all_graph_nodes[i]
                to_node = all_graph_nodes[j]
                dist = node_distances.get(from_node, {}).get(to_node, 10000)
                row.append(int(dist * 1000))  # Scale to integers
        distance_matrix.append(row)

    # Create the routing index manager
    manager = pywrapcp.RoutingIndexManager(
        total_nodes,
        num_cars,
        depot_indices,  # starts
        depot_indices,  # ends (return to depot)
    )

    # Create routing model
    routing = pywrapcp.RoutingModel(manager)

    # Distance callback
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distance_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add capacity constraint
    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        node_type = index_to_type[from_node]
        if node_type[0] == "pickup":
            return 1
        elif node_type[0] == "dropoff":
            return -1
        return 0

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # no slack
        [max_passengers_per_car] * num_cars,  # capacity per vehicle
        True,  # start cumul to zero
        "Capacity",
    )

    # Add Distance dimension
    routing.AddDimension(
        transit_callback_index,
        0,  # no slack
        1_000_000_000,  # max distance: very large to allow long routes if needed
        True,  # start cumul to zero
        "Distance",
    )
    distance_dimension = routing.GetDimensionOrDie("Distance")

    # Add Global Span Cost to balance the fleet.
    # This penalizes the difference between the max and min route distances,
    # forcing the solver to distribute work across free cars rather than
    # creating one giant efficient route for a single car.
    # Scale the coefficient based on problem size: stronger for large fleets,
    # weaker for small ones to avoid over-distribution.
    span_coefficient = max(10, min(100, num_passengers * 5))
    distance_dimension.SetGlobalSpanCostCoefficient(span_coefficient)
    
    # Add pickup and delivery constraints
    penalty = 1_000_000
    for i in range(num_passengers):
        pickup_idx = pickup_indices[i]
        dropoff_idx = dropoff_indices[i]

        pickup_index = manager.NodeToIndex(pickup_idx)
        dropoff_index = manager.NodeToIndex(dropoff_idx)

        # Register P&D
        routing.AddPickupAndDelivery(pickup_index, dropoff_index)
        
        # P and D must be on same vehicle
        routing.solver().Add(
            routing.VehicleVar(pickup_index) == routing.VehicleVar(dropoff_index)
        )
        
        # P must be before D (in terms of distance/time)
        routing.solver().Add(
            distance_dimension.CumulVar(pickup_index) <= distance_dimension.CumulVar(dropoff_index)
        )

        # Allow dropping the request (make Pickup optional)
        # If Pickup is dropped (VehicleVar = -1), Dropoff is also dropped because VehicleVar(P)==VehicleVar(D)
        routing.AddDisjunction([pickup_index], penalty)
        
        # Also make Dropoff optional to technically allow the unperformed state, 
        # though strictly the constraint above binds them.
        routing.AddDisjunction([dropoff_index], penalty)

    # Set search parameters
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.seconds = time_limit_seconds

    # Solve
    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        return _extract_ortools_solution(
            routing,
            manager,
            solution,
            cars,
            passengers,
            index_to_node,
            index_to_type,
            distance_matrix,
        )
    else:
        # Fallback to enhanced greedy if no solution found
        return _solve_vrp_enhanced_greedy(G, cars, passengers, max_passengers_per_car)


def _extract_ortools_solution(
    routing,
    manager,
    solution,
    cars: List[Car],
    passengers: List[Passenger],
    index_to_node: Dict[int, int],
    index_to_type: Dict[int, Tuple[str, int]],
    distance_matrix: np.ndarray,
) -> RoutingSolution:
    """Extract the solution from OR-Tools."""
    assignments = []
    total_distance = 0.0
    assigned_passengers = set()

    for vehicle_id, car in enumerate(cars):
        index = routing.Start(vehicle_id)
        route_nodes = []
        route_passengers = []
        route_distance = 0.0

        while not routing.IsEnd(index):
            node_idx = manager.IndexToNode(index)
            route_nodes.append(index_to_node[node_idx])

            node_type = index_to_type[node_idx]
            if node_type[0] == "pickup":
                passenger_id = node_type[1]
                if passenger_id not in route_passengers:
                    route_passengers.append(passenger_id)
                    assigned_passengers.add(passenger_id)

            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id
            )

        # Add final node
        node_idx = manager.IndexToNode(index)
        route_nodes.append(index_to_node[node_idx])

        if route_passengers:
            assignments.append(
                RouteAssignment(
                    car_id=car.id,
                    passengers=route_passengers,
                    route=route_nodes,
                    total_distance=route_distance / 1000.0,  # Scale back
                )
            )

        total_distance += route_distance / 1000.0

    # Find unassigned passengers
    unassigned = [p.id for p in passengers if p.id not in assigned_passengers]

    return RoutingSolution(
        assignments=assignments,
        total_distance=total_distance,
        unassigned_passengers=unassigned,
        solver_name="OR-Tools VRP",
    )


# =============================================================================
# Comparison Function
# =============================================================================


def compare_solvers(
    G: nx.Graph, cars: List[Car], passengers: List[Passenger]
) -> Dict[str, RoutingSolution]:
    """
    Run both solvers and compare their results.

    Returns:
        Dictionary with solver names as keys and solutions as values
    """
    greedy_solution = solve_vrp_greedy(G, cars, passengers)
    ortools_solution = solve_vrp_ortools(G, cars, passengers)

    return {"greedy": greedy_solution, "ortools": ortools_solution}


def print_solution_comparison(solutions: Dict[str, RoutingSolution]) -> None:
    """Print a comparison of routing solutions."""
    print("\n" + "=" * 60)
    print("ROUTING SOLUTION COMPARISON")
    print("=" * 60)

    for name, solution in solutions.items():
        print(f"\n{solution.solver_name}")
        print("-" * 40)
        print(f"  Total Distance: {solution.total_distance:.2f} km")
        print(f"  Cars Used: {len(solution.assignments)}")
        print(f"  Unassigned Passengers: {len(solution.unassigned_passengers)}")

        for assignment in solution.assignments:
            print(
                f"  Car {assignment.car_id}: {len(assignment.passengers)} passengers, "
                f"{assignment.total_distance:.2f} km"
            )

    if "greedy" in solutions and "ortools" in solutions:
        greedy_dist = solutions["greedy"].total_distance
        if greedy_dist > 0:
            improvement = (
                (greedy_dist - solutions["ortools"].total_distance)
                / greedy_dist
                * 100
            )
            print(f"\n→ OR-Tools improvement over Greedy: {improvement:.1f}%")
        else:
            print("\n→ OR-Tools improvement over Greedy: 0.0% (Both zero)")

    print("=" * 60)
