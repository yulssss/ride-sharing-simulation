"""
Visualization Module for the ride-pooling service.

Provides matplotlib-based visualizations for:
- City graph with nodes and edges
- Car positions and routes
- Demand heatmaps
- Pricing zones
- Algorithm comparison charts
"""

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

from src.prediction import DemandPrediction
from src.routing import RoutingSolution
from src.utils import Car, Location, Passenger

# =============================================================================
# Color Schemes
# =============================================================================

# Custom colormap for demand heatmap
DEMAND_CMAP = LinearSegmentedColormap.from_list(
    "demand", ["#2ecc71", "#f1c40f", "#e74c3c"], N=256
)

# Car colors
CAR_COLORS = [
    "#3498db",
    "#9b59b6",
    "#1abc9c",
    "#e67e22",
    "#e74c3c",
    "#2980b9",
    "#8e44ad",
    "#16a085",
    "#d35400",
    "#c0392b",
]


def _infer_grid_size_from_locations(locations: Dict[int, Location]) -> int:
    """Infer grid size from (x, y) locations; falls back to 10."""
    if not locations:
        return 10
    max_x = max(loc.x for loc in locations.values())
    max_y = max(loc.y for loc in locations.values())
    inferred = int(max(max_x, max_y)) + 1
    return max(1, inferred)


def _top_k_cells(grid: np.ndarray, k: int) -> List[Tuple[int, int]]:
    """Return (row, col) indices of top-k values in a 2D grid."""
    if k <= 0:
        return []
    flat = grid.ravel()
    if flat.size == 0:
        return []
    k = min(k, flat.size)
    # argpartition gives indices of the k largest elements in arbitrary order
    idxs = np.argpartition(flat, -k)[-k:]
    rows, cols = np.unravel_index(idxs, grid.shape)
    return list(zip(rows.tolist(), cols.tolist()))


def _fit_cell_label(grid_size: int, text: str) -> Tuple[str, int]:
    """Heuristic: shrink cell label text to fit smaller cells.

    Matplotlib doesn't know the final pixel size of each cell easily without a
    full layout pass, so we use a reliable heuristic based on grid_size and
    label length.

    Returns:
        (possibly shortened text, fontsize)
    """
    grid_size = max(1, int(grid_size))

    # Base font scales down as the grid grows.
    # 10 -> ~9, 20 -> ~7, 40 -> ~5
    base = int(max(4, min(10, round(180 / (grid_size + 4)))))

    # Further shrink based on label length.
    length = max(1, len(text))
    # Allow ~4 chars at base size, shrink if longer.
    length_factor = min(1.0, 4.0 / float(length))
    fontsize = max(3, int(round(base * length_factor)))

    # If we're still too dense, shorten common patterns.
    if grid_size >= 25:
        # Remove trailing 'x'/'×' if present.
        if text.endswith("x") or text.endswith("×"):
            text = text[:-1]

    return text, fontsize


def _auto_viz_style(
    *,
    n_nodes: int,
    n_edges: int,
    n_cars: int,
    n_passengers: int,
    grid_size: int,
) -> Dict[str, object]:
    """Return scale-aware visualization parameters."""
    # Treat 20x20-ish maps as already visually dense in Matplotlib.
    dense_graph = n_nodes >= 400 or grid_size >= 20
    very_dense_graph = n_nodes >= 900 or grid_size >= 30
    many_agents = (n_cars + n_passengers) >= 60

    node_size = 150
    node_alpha = 1.0
    node_edgecolors = "#7f8c8d"
    node_linewidths = 1.5
    edge_width = 1.5
    edge_alpha = 0.7

    if dense_graph:
        node_size = 22
        node_alpha = 0.35
        node_edgecolors = "none"
        node_linewidths = 0.0
        edge_width = 0.35
        edge_alpha = 0.12
    if very_dense_graph:
        node_size = 10
        node_alpha = 0.22
        edge_width = 0.25
        edge_alpha = 0.08

    show_car_labels = n_cars <= 12
    label_fontsize = 8
    if n_cars > 25:
        label_fontsize = 6

    # Passenger overlay
    draw_trip_arrows = (n_passengers <= 18) and (not dense_graph)
    trip_arrow_limit = 12
    trip_arrow_alpha = 0.25 if dense_graph else 0.45
    # Scale markers relative to node size so overlays don't dwarf intersections.
    pickup_size = 100 if not dense_graph else max(14, int(node_size * 1.6))
    dropoff_size = 80 if not dense_graph else max(12, int(node_size * 1.35))
    pickup_lw = 2 if not dense_graph else 0.7
    dropoff_lw = 2 if not dense_graph else 0.7

    car_size = 300 if not dense_graph else max(18, int(node_size * 2.6))
    car_lw = 3 if not dense_graph else 0.9

    # Routes
    show_route_legend = (n_cars <= 10) and (n_passengers <= 25) and (not dense_graph)
    legend_max_items = 8
    route_linewidth = 3 if not dense_graph else 1.2
    route_alpha = 0.7 if not dense_graph else 0.55
    draw_route_arrows = (not dense_graph) and (n_passengers <= 20)
    arrows_per_route = 3 if (not dense_graph) else 0

    # Heatmaps
    heatmap_show_labels = grid_size <= 15
    heatmap_max_labels = 30
    gridline_lw = 2 if grid_size <= 15 else 0.6
    tick_step = 1 if grid_size <= 12 else (5 if grid_size <= 50 else 10)

    return {
        "node_size": node_size,
        "node_alpha": node_alpha,
        "node_edgecolors": node_edgecolors,
        "node_linewidths": node_linewidths,
        "edge_width": edge_width,
        "edge_alpha": edge_alpha,
        "show_car_labels": show_car_labels,
        "label_fontsize": label_fontsize,
        "draw_trip_arrows": draw_trip_arrows,
        "trip_arrow_limit": trip_arrow_limit,
        "trip_arrow_alpha": trip_arrow_alpha,
        "pickup_size": pickup_size,
        "dropoff_size": dropoff_size,
        "pickup_lw": pickup_lw,
        "dropoff_lw": dropoff_lw,
        "car_size": car_size,
        "car_lw": car_lw,
        "show_route_legend": show_route_legend,
        "legend_max_items": legend_max_items,
        "route_linewidth": route_linewidth,
        "route_alpha": route_alpha,
        "draw_route_arrows": draw_route_arrows,
        "arrows_per_route": arrows_per_route,
        "heatmap_show_labels": heatmap_show_labels,
        "heatmap_max_labels": heatmap_max_labels,
        "gridline_lw": gridline_lw,
        "tick_step": tick_step,
        "dense_graph": dense_graph,
        "very_dense_graph": very_dense_graph,
        "many_agents": many_agents,
    }


# =============================================================================
# City Graph Visualization
# =============================================================================


def plot_city_graph(
    G: nx.Graph,
    locations: Dict[int, Location],
    ax: Optional[plt.Axes] = None,
    show_labels: bool = False,
    title: str = "City Map",
) -> plt.Axes:
    """
    Plot the city graph.

    Args:
        G: NetworkX graph
        locations: Dictionary of locations
        ax: Matplotlib axes (creates new figure if None)
        show_labels: Whether to show node labels
        title: Plot title

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    grid_size = _infer_grid_size_from_locations(locations)
    style = _auto_viz_style(
        n_nodes=G.number_of_nodes(),
        n_edges=G.number_of_edges(),
        n_cars=0,
        n_passengers=0,
        grid_size=grid_size,
    )

    # Get positions
    pos = {node_id: (loc.x, loc.y) for node_id, loc in locations.items()}

    # Draw edges
    edge_collection = nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        edge_color="#bdc3c7",
        width=style["edge_width"],
        alpha=style["edge_alpha"],
    )
    if edge_collection is not None:
        try:
            edge_collection.set_rasterized(True)
        except Exception:
            pass

    # Draw nodes
    node_collection = nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_color="#ecf0f1",
        node_size=style["node_size"],
        edgecolors=style["node_edgecolors"],
        linewidths=style["node_linewidths"],
        alpha=style["node_alpha"],
    )
    if node_collection is not None:
        try:
            node_collection.set_rasterized(True)
        except Exception:
            pass

    if show_labels:
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=8, font_color="#2c3e50")

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    return ax


def plot_cars_and_passengers(
    G: nx.Graph,
    locations: Dict[int, Location],
    cars: List[Car],
    passengers: List[Passenger],
    ax: Optional[plt.Axes] = None,
    title: str = "Fleet & Passengers",
) -> plt.Axes:
    """
    Plot cars and passengers on the city map.

    Args:
        G: NetworkX graph
        locations: Dictionary of locations
        cars: List of cars
        passengers: List of passengers
        ax: Matplotlib axes
        title: Plot title

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    grid_size = _infer_grid_size_from_locations(locations)
    style = _auto_viz_style(
        n_nodes=G.number_of_nodes(),
        n_edges=G.number_of_edges(),
        n_cars=len(cars),
        n_passengers=len(passengers),
        grid_size=grid_size,
    )

    # Plot base city
    plot_city_graph(G, locations, ax=ax, title=title)

    # Plot passengers (pickup = green, dropoff = red)
    # For large scenarios, avoid drawing every trip arrow.
    passengers_sorted = sorted(passengers, key=lambda p: int(p.is_assigned))
    arrow_count = 0
    for passenger in passengers_sorted:
        pickup_pos = (passenger.pickup.x, passenger.pickup.y)
        dropoff_pos = (passenger.dropoff.x, passenger.dropoff.y)

        if style["draw_trip_arrows"] and arrow_count < style["trip_arrow_limit"]:
            # Draw line from pickup to dropoff
            ax.annotate(
                "",
                xy=dropoff_pos,
                xytext=pickup_pos,
                arrowprops=dict(
                    arrowstyle="->",
                    color="#95a5a6",
                    lw=0.8,
                    alpha=style["trip_arrow_alpha"],
                    mutation_scale=8,
                ),
            )
            arrow_count += 1

        # Pickup point
        color = "#27ae60" if not passenger.is_assigned else "#bdc3c7"
        ax.scatter(
            *pickup_pos,
            c=color,
            s=style["pickup_size"],
            marker="^",
            edgecolors="white",
            linewidths=style["pickup_lw"],
            zorder=5,
        )

        # Dropoff point
        ax.scatter(
            *dropoff_pos,
            c="#e74c3c",
            s=style["dropoff_size"],
            marker="v",
            edgecolors="white",
            linewidths=style["dropoff_lw"],
            zorder=5,
            alpha=0.7,
        )

    # Plot cars
    for i, car in enumerate(cars):
        car_pos = (car.current_location.x, car.current_location.y)
        color = CAR_COLORS[i % len(CAR_COLORS)]

        ax.scatter(
            *car_pos,
            c=color,
            s=style["car_size"],
            marker="s",
            edgecolors="white",
            linewidths=style["car_lw"],
            zorder=10,
        )
        if style["show_car_labels"]:
            ax.annotate(
                f"C{car.id}",
                car_pos,
                ha="center",
                va="center",
                fontsize=style["label_fontsize"],
                fontweight="bold",
                color="white",
                zorder=11,
            )

    # Legend
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="^",
            color="none",
            markerfacecolor="#27ae60",
            markeredgecolor="white",
            markersize=8,
            label="Pickup (waiting)",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="none",
            markerfacecolor="#bdc3c7",
            markeredgecolor="white",
            markersize=8,
            label="Pickup (assigned)",
        ),
        Line2D(
            [0],
            [0],
            marker="v",
            color="none",
            markerfacecolor="#e74c3c",
            markeredgecolor="white",
            markersize=7,
            label="Dropoff",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor="#3498db",
            markeredgecolor="white",
            markersize=9,
            label="Car",
        ),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        borderaxespad=0.0,
        framealpha=0.95,
        facecolor="white",
        edgecolor="#ecf0f1",
        fontsize=8,
        handletextpad=0.4,
        labelspacing=0.35,
    )

    return ax


# =============================================================================
# Route Visualization
# =============================================================================


def plot_routes(
    G: nx.Graph,
    locations: Dict[int, Location],
    solution: RoutingSolution,
    ax: Optional[plt.Axes] = None,
    title: str = None,
) -> plt.Axes:
    """
    Plot the routing solution on the city map.

    Args:
        G: NetworkX graph
        locations: Dictionary of locations
        solution: Routing solution to visualize
        ax: Matplotlib axes
        title: Plot title (defaults to solver name)

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Plot base city
    plot_city_graph(
        G, locations, ax=ax, title=title or f"{solution.solver_name} Solution"
    )

    grid_size = _infer_grid_size_from_locations(locations)
    style = _auto_viz_style(
        n_nodes=G.number_of_nodes(),
        n_edges=G.number_of_edges(),
        n_cars=max(1, len(solution.assignments)),
        n_passengers=sum(len(a.passengers) for a in solution.assignments),
        grid_size=grid_size,
    )

    # Plot each car's route
    legend_items = 0
    for i, assignment in enumerate(solution.assignments):
        color = CAR_COLORS[assignment.car_id % len(CAR_COLORS)]

        # Draw route as connected lines
        route_coords = [
            (locations[node].x, locations[node].y)
            for node in assignment.route
            if node in locations
        ]

        if len(route_coords) >= 2:
            xs, ys = zip(*route_coords)
            label = None
            if style["show_route_legend"] and legend_items < style["legend_max_items"]:
                label = (
                    f"Car {assignment.car_id} ({assignment.total_distance:.1f} km)"
                )
                legend_items += 1
            ax.plot(
                xs,
                ys,
                c=color,
                linewidth=style["route_linewidth"],
                alpha=style["route_alpha"],
                label=label,
            )

            # Mark route direction with a few arrows (avoid per-segment arrows)
            if style["draw_route_arrows"] and style["arrows_per_route"] > 0:
                steps = min(style["arrows_per_route"], max(1, len(route_coords) - 1))
                indices = np.linspace(0, len(route_coords) - 2, steps, dtype=int)
                for j in indices:
                    start = route_coords[j]
                    end = route_coords[j + 1]
                    mid = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
                    dx = end[0] - start[0]
                    dy = end[1] - start[1]
                    ax.annotate(
                        "",
                        xy=(mid[0] + dx * 0.12, mid[1] + dy * 0.12),
                        xytext=mid,
                        arrowprops=dict(
                            arrowstyle="->",
                            color=color,
                            lw=1.0,
                            alpha=0.55,
                            mutation_scale=8,
                        ),
                    )

    # Add summary annotation
    summary_text = (
        f"Total Distance: {solution.total_distance:.1f} km\n"
        f"Cars Used: {len(solution.assignments)}\n"
        f"Unassigned: {len(solution.unassigned_passengers)}"
    )
    ax.annotate(
        summary_text,
        xy=(0.02, 0.98),
        xycoords="axes fraction",
        verticalalignment="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )

    if style["show_route_legend"]:
        ax.legend(loc="lower right", fontsize=9)

    return ax


def plot_route_comparison(
    G: nx.Graph,
    locations: Dict[int, Location],
    solutions: Dict[str, RoutingSolution],
    figsize: Tuple[int, int] = (16, 7),
) -> plt.Figure:
    """
    Plot side-by-side comparison of routing solutions.

    Args:
        G: NetworkX graph
        locations: Dictionary of locations
        solutions: Dictionary of solver_name -> RoutingSolution
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    n_solutions = len(solutions)
    fig, axes = plt.subplots(1, n_solutions, figsize=figsize)

    if n_solutions == 1:
        axes = [axes]

    for ax, (name, solution) in zip(axes, solutions.items()):
        plot_routes(G, locations, solution, ax=ax)

    plt.tight_layout()
    return fig


# =============================================================================
# Demand Heatmap
# =============================================================================


def plot_demand_heatmap(
    locations: Dict[int, Location],
    demand_per_zone: Dict[int, float],
    grid_size: int = 10,
    ax: Optional[plt.Axes] = None,
    title: str = "Demand Heatmap",
) -> plt.Axes:
    """
    Plot a heatmap of demand across zones.

    Args:
        locations: Dictionary of locations
        demand_per_zone: Dictionary of zone_id -> demand value
        grid_size: Size of the city grid
        ax: Matplotlib axes
        title: Plot title

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Create heatmap grid
    zone_grid = np.zeros((grid_size, grid_size))

    for zone_id, demand in demand_per_zone.items():
        row = zone_id // grid_size
        col = zone_id % grid_size
        if row < grid_size and col < grid_size:
            zone_grid[row, col] = demand

    # Plot heatmap
    im = ax.imshow(
        zone_grid, cmap=DEMAND_CMAP, origin="lower", interpolation="nearest", alpha=0.7
    )

    style = _auto_viz_style(
        n_nodes=len(locations),
        n_edges=0,
        n_cars=0,
        n_passengers=0,
        grid_size=grid_size,
    )

    # Add grid lines
    ax.set_xticks(np.arange(-0.5, grid_size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid_size, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=style["gridline_lw"])

    # Reduce tick density for large grids
    step = int(style["tick_step"]) if style["tick_step"] else 1
    ax.set_xticks(np.arange(0, grid_size, step))
    ax.set_yticks(np.arange(0, grid_size, step))

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Predicted Demand", fontsize=10)

    # Add zone labels (limit for large grids)
    if style["heatmap_show_labels"]:
        for zone_id, demand in demand_per_zone.items():
            row = zone_id // grid_size
            col = zone_id % grid_size
            if row < grid_size and col < grid_size and demand > 0:
                label, fontsize = _fit_cell_label(grid_size, f"{demand:.0f}")
                ax.text(
                    col,
                    row,
                    label,
                    ha="center",
                    va="center",
                    fontsize=fontsize,
                    fontweight="bold",
                    color="white",
                    clip_on=True,
                )
    else:
        # Show only the top-k cells
        top_cells = _top_k_cells(zone_grid, int(style["heatmap_max_labels"]))
        for row, col in top_cells:
            val = zone_grid[row, col]
            if val <= 0:
                continue
            label, fontsize = _fit_cell_label(grid_size, f"{val:.0f}")
            ax.text(
                col,
                row,
                label,
                ha="center",
                va="center",
                fontsize=fontsize,
                fontweight="bold",
                color="white",
                clip_on=True,
            )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("X Zone")
    ax.set_ylabel("Y Zone")

    return ax


def plot_predictions_over_time(
    predictions: Dict[int, List[DemandPrediction]],
    top_n_zones: int = 5,
    ax: Optional[plt.Axes] = None,
    title: str = "Demand Predictions Over Time",
) -> plt.Axes:
    """
    Plot predicted demand over time for top zones.

    Args:
        predictions: Dictionary of zone_id -> list of predictions
        top_n_zones: Number of top zones to show
        ax: Matplotlib axes
        title: Plot title

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # Find top zones by first prediction
    zone_demand = {}
    for zone_id, preds in predictions.items():
        if preds:
            zone_demand[zone_id] = preds[0].predicted_demand

    top_zones = sorted(zone_demand.items(), key=lambda x: x[1], reverse=True)[
        :top_n_zones
    ]

    # Plot each zone
    for i, (zone_id, _) in enumerate(top_zones):
        preds = predictions[zone_id]
        times = [p.timestamp for p in preds]
        demands = [p.predicted_demand for p in preds]

        ax.plot(
            times,
            demands,
            marker="o",
            linewidth=2,
            label=f"Zone {zone_id}",
            color=CAR_COLORS[i % len(CAR_COLORS)],
        )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Time")
    ax.set_ylabel("Predicted Demand")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.xticks(rotation=45)

    return ax


# =============================================================================
# Pricing Visualization
# =============================================================================


def plot_surge_pricing(
    zone_prices: Dict[int, float],
    grid_size: int = 10,
    ax: Optional[plt.Axes] = None,
    title: str = "Surge Pricing Map",
) -> plt.Axes:
    """
    Plot surge pricing across zones.

    Args:
        zone_prices: Dictionary of zone_id -> surge multiplier
        grid_size: Size of the city grid
        ax: Matplotlib axes
        title: Plot title

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Create price grid
    price_grid = np.ones((grid_size, grid_size))  # Base multiplier = 1.0

    for zone_id, multiplier in zone_prices.items():
        row = zone_id // grid_size
        col = zone_id % grid_size
        if row < grid_size and col < grid_size:
            price_grid[row, col] = multiplier

    # Custom colormap for surge (green = normal, red = high surge)
    surge_cmap = LinearSegmentedColormap.from_list(
        "surge", ["#2ecc71", "#f1c40f", "#e74c3c", "#8e44ad"], N=256
    )

    im = ax.imshow(
        price_grid,
        cmap=surge_cmap,
        origin="lower",
        interpolation="nearest",
        vmin=1.0,
        vmax=2.5,
        alpha=0.8,
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Surge Multiplier", fontsize=10)

    style = _auto_viz_style(
        n_nodes=grid_size * grid_size,
        n_edges=0,
        n_cars=0,
        n_passengers=0,
        grid_size=grid_size,
    )

    # Reduce tick density for large grids
    step = int(style["tick_step"]) if style["tick_step"] else 1
    ax.set_xticks(np.arange(0, grid_size, step))
    ax.set_yticks(np.arange(0, grid_size, step))

    # Add multiplier labels (limit for large grids)
    if grid_size <= 15:
        label_items = [(z, m) for z, m in zone_prices.items() if m > 1.0]
    else:
        # Label only significant surge cells
        label_items = [(z, m) for z, m in zone_prices.items() if m >= 1.25]
    for zone_id, multiplier in label_items[: int(style["heatmap_max_labels"])]:
        row = zone_id // grid_size
        col = zone_id % grid_size
        if row < grid_size and col < grid_size:
            raw_label = f"{multiplier:.1f}×"
            label, fontsize = _fit_cell_label(grid_size, raw_label)
            ax.text(
                col,
                row,
                label,
                ha="center",
                va="center",
                fontsize=fontsize,
                fontweight="bold",
                color="white" if multiplier > 1.5 else "black",
                clip_on=True,
            )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("X Zone")
    ax.set_ylabel("Y Zone")
    ax.grid(True, alpha=0.3)

    return ax


# =============================================================================
# Dashboard
# =============================================================================


def create_dashboard(
    G: nx.Graph,
    locations: Dict[int, Location],
    cars: List[Car],
    passengers: List[Passenger],
    routing_solutions: Dict[str, RoutingSolution],
    demand_predictions: Dict[int, List[DemandPrediction]] = None,
    zone_prices: Dict[int, float] = None,
    grid_size: Optional[int] = None,
    figsize: Tuple[int, int] = (20, 16),
) -> plt.Figure:
    """
    Create a comprehensive dashboard showing all aspects of the simulation.

    Args:
        G: NetworkX graph
        locations: Dictionary of locations
        cars: List of cars
        passengers: List of passengers
        routing_solutions: Dictionary of routing solutions
        demand_predictions: Demand predictions per zone
        zone_prices: Surge multipliers per zone
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    if grid_size is None:
        grid_size = _infer_grid_size_from_locations(locations)

    fig = plt.figure(figsize=figsize)

    # Create grid
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # 1. Cars and Passengers (top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    plot_cars_and_passengers(
        G, locations, cars, passengers, ax=ax1, title="Current Fleet & Passengers"
    )

    # 2. Greedy Solution (top-middle)
    if "greedy" in routing_solutions:
        ax2 = fig.add_subplot(gs[0, 1])
        plot_routes(G, locations, routing_solutions["greedy"], ax=ax2)

    # 3. OR-Tools Solution (top-right)
    if "ortools" in routing_solutions:
        ax3 = fig.add_subplot(gs[0, 2])
        plot_routes(G, locations, routing_solutions["ortools"], ax=ax3)

    # 4. Demand Heatmap (bottom-left)
    if demand_predictions:
        ax4 = fig.add_subplot(gs[1, 0])
        demand_per_zone = {
            z: preds[0].predicted_demand
            for z, preds in demand_predictions.items()
            if preds
        }
        plot_demand_heatmap(
            locations,
            demand_per_zone,
            grid_size=grid_size,
            ax=ax4,
            title="Predicted Demand (Next Hour)",
        )

    # 5. Surge Pricing (bottom-middle)
    if zone_prices:
        ax5 = fig.add_subplot(gs[1, 1])
        plot_surge_pricing(zone_prices, grid_size=grid_size, ax=ax5)

    # 6. Algorithm Comparison (bottom-right)
    ax6 = fig.add_subplot(gs[1, 2])
    plot_algorithm_comparison(routing_solutions, ax=ax6)

    fig.suptitle(
        "Smart Ride-Pooling Service Dashboard", fontsize=18, fontweight="bold", y=1.02
    )

    return fig


def plot_algorithm_comparison(
    solutions: Dict[str, RoutingSolution], ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot bar chart comparing algorithm performance.

    Args:
        solutions: Dictionary of routing solutions
        ax: Matplotlib axes

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    names = []
    distances = []
    cars_used = []
    unassigned = []

    for name, solution in solutions.items():
        names.append(solution.solver_name.replace(" ", "\n"))
        distances.append(solution.total_distance)
        cars_used.append(len(solution.assignments))
        unassigned.append(len(solution.unassigned_passengers))

    x = np.arange(len(names))
    width = 0.25

    ax.bar(x - width, distances, width, label="Total Distance (km)", color="#3498db")
    ax.bar(
        x, [c * 10 for c in cars_used], width, label="Cars Used (×10)", color="#2ecc71"
    )
    ax.bar(
        x + width,
        [u * 5 for u in unassigned],
        width,
        label="Unassigned (×5)",
        color="#e74c3c",
    )

    ax.set_ylabel("Value")
    ax.set_title("Algorithm Comparison", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")

    # Add actual values as annotations
    for i, (d, c, u) in enumerate(zip(distances, cars_used, unassigned)):
        ax.annotate(f"{d:.0f}", (i - width, d + 1), ha="center", fontsize=8)
        ax.annotate(f"{c}", (i, c * 10 + 1), ha="center", fontsize=8)
        ax.annotate(f"{u}", (i + width, u * 5 + 1), ha="center", fontsize=8)

    return ax


def save_dashboard(fig: plt.Figure, filename: str = "dashboard.png", dpi: int = 150):
    """Save the dashboard figure to a file."""
    fig.savefig(filename, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"Dashboard saved to {filename}")
