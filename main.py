#!/usr/bin/env python3
"""
Smart Ride-Pooling Service - Main Simulation

This script orchestrates the complete ride-pooling simulation:
1. Generate a synthetic city graph
2. Create historical data and train demand prediction model
3. Generate current ride requests
4. Solve the routing problem (Greedy vs OR-Tools comparison)
5. Calculate dynamic pricing
6. Visualize results and generate dashboard

Usage:
    python main.py [--num-cars NUM] [--num-passengers NUM] [--grid-size SIZE]
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt
import numpy as np

from src.prediction import (
    DemandPredictor,
    print_prediction_summary,
    print_repositioning_recommendations,
    recommend_repositioning,
)
from src.pricing import (
    PricingConfig,
    calculate_all_prices,
    calculate_surge_multiplier,
    calculate_zone_demand,
    identify_hotspots,
    print_pricing_summary,
)
from src.routing import (
    compare_solvers,
    print_solution_comparison,
)
from src.utils import (
    create_city_graph,
    generate_current_requests,
    generate_historical_rides,
    get_shortest_path_distance,
    initialize_fleet,
)
from src.visualization import (
    create_dashboard,
    save_dashboard,
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Smart Ride-Pooling Service Simulation"
    )
    parser.add_argument(
        "--num-cars",
        type=int,
        default=40,
        help="Number of cars in the fleet (default: 40)",
    )
    parser.add_argument(
        "--num-passengers",
        type=int,
        default=40,
        help="Number of passengers requesting rides (default: 40)",
    )
    parser.add_argument(
        "--grid-size", type=int, default=40, help="Size of the city grid (default: 40x40)"
    )
    parser.add_argument(
        "--no-viz", action="store_true", help="Disable visualization (text output only)"
    )
    parser.add_argument(
        "--save-dashboard",
        type=str,
        default=None,
        help="Save dashboard to specified filename",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    return parser.parse_args()


def main():
    """Run the complete ride-pooling simulation."""
    args = parse_args()

    print("=" * 70)
    print("🚗  SMART RIDE-POOLING SERVICE SIMULATION  🚗")
    print("=" * 70)
    print("\nConfiguration:")
    print(f"  • Fleet Size: {args.num_cars} cars")
    print(f"  • Passengers: {args.num_passengers}")
    print(
        f"  • City Grid: {args.grid_size}x{args.grid_size} ({args.grid_size**2} nodes)"
    )
    print(f"  • Random Seed: {args.seed}")

    # =========================================================================
    # STEP 1: Create City Graph
    # =========================================================================
    print("\n" + "-" * 70)
    print("📍 STEP 1: Generating City Map...")

    G, locations = create_city_graph(grid_size=args.grid_size, seed=args.seed)

    print(
        f"  ✓ Created city with {G.number_of_nodes()} intersections "
        f"and {G.number_of_edges()} roads"
    )

    # =========================================================================
    # STEP 2: Generate Historical Data & Train Prediction Model
    # =========================================================================
    print("\n" + "-" * 70)
    print("📊 STEP 2: Training Demand Prediction Model...")

    # Generate synthetic historical data
    historical_data = generate_historical_rides(
        locations, num_days=500, rides_per_day=500, seed=args.seed
    )
    print(f"  ✓ Generated {len(historical_data)} historical ride records")

    # Train the prediction model
    predictor = DemandPredictor(model_type="random_forest")
    metrics = predictor.train(historical_data)

    print(
        f"  ✓ Model trained with R² = {metrics['r2']:.3f}, MAE = {metrics['mae']:.2f}"
    )

    # =========================================================================
    # STEP 3: Initialize Fleet and Generate Current Requests
    # =========================================================================
    print("\n" + "-" * 70)
    print("🚕 STEP 3: Initializing Fleet & Generating Ride Requests...")

    cars = initialize_fleet(locations, num_cars=args.num_cars, seed=args.seed)
    print(f"  ✓ Deployed {len(cars)} cars at random positions")

    current_time = datetime.now()
    passengers = generate_current_requests(
        locations,
        num_requests=args.num_passengers,
        current_time=current_time,
        seed=args.seed,
    )
    print(f"  ✓ Received {len(passengers)} ride requests")

    # =========================================================================
    # STEP 4: Predict Future Demand
    # =========================================================================
    print("\n" + "-" * 70)
    print("🔮 STEP 4: Predicting Future Demand...")

    num_zones = args.grid_size
    predictions = predictor.predict_all_zones(
        num_zones=num_zones, timestamp=current_time, horizon_hours=2
    )

    # Get current car positions by zone
    current_car_positions = {
        car.id: car.current_location.id // args.grid_size for car in cars
    }

    # Get repositioning recommendations
    reposition_recs = recommend_repositioning(
        predictions, current_car_positions, locations, grid_size=args.grid_size, top_n=3
    )

    print_prediction_summary(predictions, metrics)
    print_repositioning_recommendations(reposition_recs)

    # =========================================================================
    # STEP 5: Solve Vehicle Routing Problem
    # =========================================================================
    print("\n" + "-" * 70)
    print("🗺️  STEP 5: Solving Vehicle Routing Problem...")
    print("   (Comparing Greedy Heuristic vs OR-Tools Solver)")

    # Run both solvers
    routing_solutions = compare_solvers(G, cars, passengers)

    print_solution_comparison(routing_solutions)

    # =========================================================================
    # STEP 6: Calculate Dynamic Pricing
    # =========================================================================
    print("\n" + "-" * 70)
    print("💰 STEP 6: Calculating Dynamic Pricing...")

    # Calculate distances for each passenger's trip
    distances = {}
    for passenger in passengers:
        dist = get_shortest_path_distance(G, passenger.pickup.id, passenger.dropoff.id)
        distances[passenger.id] = dist

    # Calculate prices
    pricing_config = PricingConfig()
    prices = calculate_all_prices(
        passengers, cars, distances, current_time=current_time, config=pricing_config
    )

    print_pricing_summary(prices)

    # Identify hotspots
    zone_demand = calculate_zone_demand(passengers, grid_size=args.grid_size)
    hotspots = identify_hotspots(zone_demand)

    if hotspots:
        print(f"\n🔥 Surge Pricing Active in Zones: {hotspots}")

    # =========================================================================
    # STEP 7: Visualization
    # =========================================================================
    if not args.no_viz:
        print("\n" + "-" * 70)
        print("📈 STEP 7: Generating Visualizations...")

        # Calculate zone prices for visualization
        zone_prices = {}
        num_available = sum(1 for c in cars if c.is_available)
        for zone_id, demand in zone_demand.items():
            multiplier, _ = calculate_surge_multiplier(
                demand,
                num_available // args.grid_size or 1,
                zone_demand,
                zone_id,
                pricing_config,
            )
            zone_prices[zone_id] = multiplier

        # Create comprehensive dashboard
        fig = create_dashboard(
            G,
            locations,
            cars,
            passengers,
            routing_solutions,
            demand_predictions=predictions,
            zone_prices=zone_prices,
            grid_size=args.grid_size,
        )

        if args.save_dashboard:
            save_dashboard(fig, args.save_dashboard)
        else:
            print("  ✓ Dashboard ready. Displaying...")
            plt.show()

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("📋 SIMULATION SUMMARY")
    print("=" * 70)

    greedy_dist = routing_solutions["greedy"].total_distance
    ortools_dist = routing_solutions["ortools"].total_distance
    improvement = (
        (greedy_dist - ortools_dist) / greedy_dist * 100 if greedy_dist > 0 else 0
    )

    avg_price = np.mean([p.final_price for p in prices.values()]) if prices else 0.0
    avg_surge = np.mean([p.surge_multiplier for p in prices.values()]) if prices else 0.0

    print(f"""
    ROUTING:
    --------
    • Greedy Total Distance:   {greedy_dist:.1f} km
    • OR-Tools Total Distance: {ortools_dist:.1f} km
    • OR-Tools Improvement:    {improvement:.1f}%
    • Unassigned (Greedy):     {len(routing_solutions["greedy"].unassigned_passengers)}
    • Unassigned (OR-Tools):   {len(routing_solutions["ortools"].unassigned_passengers)}

    PRICING:
    --------
    • Average Fare:            ${avg_price:.2f}
    • Average Surge:           {avg_surge:.2f}x
    • Hotspot Zones:           {len(hotspots)}

    PREDICTION:
    -----------
    • Model R² Score:          {metrics["r2"]:.3f}
    • Repositioning Recs:      {len(reposition_recs)}
    """)

    print("=" * 70)
    print("✅ Simulation Complete!")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    exit(main())
