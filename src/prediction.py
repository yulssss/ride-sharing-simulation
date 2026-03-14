"""
Demand Prediction Module for the ride-pooling service.

Uses machine learning to predict future demand based on:
- Historical ride data (time of day, day of week, zone)
- Time series forecasting for proactive fleet repositioning
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.utils import Location

# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class DemandPrediction:
    """Prediction for a specific zone and time."""

    zone_id: int
    timestamp: datetime
    predicted_demand: float
    confidence: float  # 0-1 confidence score


@dataclass
class FleetRepositionRecommendation:
    """Recommendation for repositioning a car."""

    car_id: int
    target_zone_id: int
    target_location_id: int
    priority: float  # Higher = more urgent
    reason: str


# =============================================================================
# Feature Engineering
# =============================================================================


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare features for the demand prediction model.

    Args:
        df: DataFrame with columns: timestamp, hour, day_of_week, pickup_id, zone_id

    Returns:
        DataFrame with engineered features
    """
    features = df.copy()

    # Time-based features
    features["hour_sin"] = np.sin(2 * np.pi * features["hour"] / 24)
    features["hour_cos"] = np.cos(2 * np.pi * features["hour"] / 24)
    features["day_sin"] = np.sin(2 * np.pi * features["day_of_week"] / 7)
    features["day_cos"] = np.cos(2 * np.pi * features["day_of_week"] / 7)

    # Is weekend
    features["is_weekend"] = (features["day_of_week"] >= 5).astype(int)

    # Rush hour flags
    features["is_morning_rush"] = (
        (features["hour"] >= 7) & (features["hour"] <= 9)
    ).astype(int)
    features["is_evening_rush"] = (
        (features["hour"] >= 17) & (features["hour"] <= 19)
    ).astype(int)

    return features


def aggregate_demand(df: pd.DataFrame, time_window_minutes: int = 60) -> pd.DataFrame:
    """
    Aggregate ride data into demand counts per zone per time window.

    Args:
        df: DataFrame with ride data
        time_window_minutes: Aggregation window in minutes

    Returns:
        DataFrame with demand counts
    """
    # Round timestamps to the time window
    df = df.copy()
    df["time_bucket"] = df["timestamp"].dt.floor(f"{time_window_minutes}min")

    # Aggregate by zone and time bucket
    demand = (
        df.groupby(["zone_id", "time_bucket"])
        .agg({"pickup_id": "count", "hour": "first", "day_of_week": "first"})
        .reset_index()
    )

    demand.rename(columns={"pickup_id": "demand_count"}, inplace=True)

    return demand


# =============================================================================
# Demand Prediction Model
# =============================================================================


class DemandPredictor:
    """Machine learning model for demand prediction."""

    def __init__(self, model_type: str = "random_forest"):
        """
        Initialize the demand predictor.

        Args:
            model_type: 'random_forest', 'gradient_boosting', or 'linear'
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = [
            "zone_id",
            "hour",
            "day_of_week",
            "hour_sin",
            "hour_cos",
            "day_sin",
            "day_cos",
            "is_weekend",
            "is_morning_rush",
            "is_evening_rush",
        ]
        self.is_trained = False
        self.training_metrics = {}

    def _create_model(self):
        """Create the underlying ML model."""
        if self.model_type == "random_forest":
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1,
            )
        elif self.model_type == "gradient_boosting":
            return GradientBoostingRegressor(
                n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
            )
        else:
            return LinearRegression()

    def train(
        self, historical_data: pd.DataFrame, test_size: float = 0.2
    ) -> Dict[str, float]:
        """
        Train the demand prediction model.

        Args:
            historical_data: DataFrame with historical ride data
            test_size: Fraction of data to use for testing

        Returns:
            Dictionary with training metrics
        """
        # Aggregate demand
        demand_df = aggregate_demand(historical_data)

        # Prepare features
        demand_df = prepare_features(demand_df)

        # Ensure all feature columns exist
        for col in self.feature_columns:
            if col not in demand_df.columns:
                demand_df[col] = 0

        X = demand_df[self.feature_columns].values
        y = demand_df["demand_count"].values

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model
        self.model = self._create_model()
        self.model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test_scaled)

        self.training_metrics = {
            "mae": mean_absolute_error(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "r2": r2_score(y_test, y_pred),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
        }

        self.is_trained = True

        return self.training_metrics

    def predict(
        self, zone_id: int, timestamp: datetime, num_predictions: int = 1
    ) -> List[DemandPrediction]:
        """
        Predict demand for a zone at a specific time.

        Args:
            zone_id: Zone to predict for
            timestamp: Time to predict for
            num_predictions: Number of future time steps to predict

        Returns:
            List of DemandPrediction objects
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        predictions = []

        for i in range(num_predictions):
            pred_time = timestamp + timedelta(hours=i)

            # Create feature vector
            hour = pred_time.hour
            day_of_week = pred_time.weekday()

            features = {
                "zone_id": zone_id,
                "hour": hour,
                "day_of_week": day_of_week,
                "hour_sin": np.sin(2 * np.pi * hour / 24),
                "hour_cos": np.cos(2 * np.pi * hour / 24),
                "day_sin": np.sin(2 * np.pi * day_of_week / 7),
                "day_cos": np.cos(2 * np.pi * day_of_week / 7),
                "is_weekend": 1 if day_of_week >= 5 else 0,
                "is_morning_rush": 1 if 7 <= hour <= 9 else 0,
                "is_evening_rush": 1 if 17 <= hour <= 19 else 0,
            }

            X = np.array([[features[col] for col in self.feature_columns]])
            X_scaled = self.scaler.transform(X)

            pred_demand = max(0, self.model.predict(X_scaled)[0])

            # Estimate confidence based on model R² and time distance
            confidence = max(0.5, self.training_metrics.get("r2", 0.5) - (i * 0.1))

            predictions.append(
                DemandPrediction(
                    zone_id=zone_id,
                    timestamp=pred_time,
                    predicted_demand=pred_demand,
                    confidence=confidence,
                )
            )

        return predictions

    def predict_all_zones(
        self, num_zones: int, timestamp: datetime, horizon_hours: int = 1
    ) -> Dict[int, List[DemandPrediction]]:
        """
        Predict demand for all zones.

        Args:
            num_zones: Number of zones in the city
            timestamp: Current time
            horizon_hours: How many hours ahead to predict

        Returns:
            Dictionary mapping zone_id -> list of predictions
        """
        all_predictions = {}

        for zone_id in range(num_zones):
            all_predictions[zone_id] = self.predict(zone_id, timestamp, horizon_hours)

        return all_predictions


# =============================================================================
# Fleet Repositioning
# =============================================================================


def recommend_repositioning(
    predictions: Dict[int, List[DemandPrediction]],
    current_car_positions: Dict[int, int],  # car_id -> zone_id
    locations: Dict[int, Location],
    grid_size: int = 10,
    top_n: int = 5,
) -> List[FleetRepositionRecommendation]:
    """
    Recommend car repositioning based on demand predictions.

    Args:
        predictions: Demand predictions per zone
        current_car_positions: Current zone for each car
        locations: All locations in the city
        grid_size: Size of the city grid
        top_n: Number of recommendations to return

    Returns:
        List of repositioning recommendations
    """
    # Calculate demand-supply gap per zone
    zone_gaps = {}

    for zone_id, preds in predictions.items():
        if preds:
            predicted_demand = preds[0].predicted_demand
            # Count cars currently in this zone
            cars_in_zone = sum(
                1 for z in current_car_positions.values() if z == zone_id
            )
            gap = predicted_demand - cars_in_zone
            zone_gaps[zone_id] = gap

    # Find zones with highest positive gap (need more cars)
    hot_zones = sorted(zone_gaps.items(), key=lambda x: x[1], reverse=True)
    hot_zones = [(z, g) for z, g in hot_zones if g > 0]

    # Find zones with negative gap (excess cars)
    cold_zones = sorted(zone_gaps.items(), key=lambda x: x[1])
    cold_zones = [(z, g) for z, g in cold_zones if g < 0]

    recommendations = []

    # Match cars from cold zones to hot zones
    for hot_zone, hot_gap in hot_zones[:top_n]:
        if not cold_zones:
            break

        # Find a car to move
        for car_id, car_zone in current_car_positions.items():
            if any(car_zone == z for z, _ in cold_zones):
                # Find a target location in the hot zone
                target_location_id = hot_zone * grid_size  # First node in zone

                recommendations.append(
                    FleetRepositionRecommendation(
                        car_id=car_id,
                        target_zone_id=hot_zone,
                        target_location_id=target_location_id,
                        priority=hot_gap,
                        reason=f"High predicted demand in zone {hot_zone}",
                    )
                )

                # Remove this car from consideration
                cold_zones = [(z, g) for z, g in cold_zones if z != car_zone]
                break

    return recommendations[:top_n]


# =============================================================================
# Utility Functions
# =============================================================================


def print_prediction_summary(
    predictions: Dict[int, List[DemandPrediction]], training_metrics: Dict[str, float]
) -> None:
    """Print a summary of predictions and model performance."""
    print("\n" + "=" * 60)
    print("DEMAND PREDICTION SUMMARY")
    print("=" * 60)

    print("\nModel Performance:")
    print(f"  MAE: {training_metrics.get('mae', 0):.2f}")
    print(f"  RMSE: {training_metrics.get('rmse', 0):.2f}")
    print(f"  R²: {training_metrics.get('r2', 0):.3f}")

    # Find hotspots (zones with highest predicted demand)
    if predictions:
        zone_demands = {}
        for zone_id, preds in predictions.items():
            if preds:
                zone_demands[zone_id] = preds[0].predicted_demand

        sorted_zones = sorted(zone_demands.items(), key=lambda x: x[1], reverse=True)

        print("\nTop 5 Predicted Hotspots (next hour):")
        for zone_id, demand in sorted_zones[:5]:
            print(f"  Zone {zone_id}: {demand:.1f} expected rides")

    print("=" * 60)


def print_repositioning_recommendations(
    recommendations: List[FleetRepositionRecommendation],
) -> None:
    """Print fleet repositioning recommendations."""
    print("\n" + "=" * 60)
    print("FLEET REPOSITIONING RECOMMENDATIONS")
    print("=" * 60)

    if not recommendations:
        print("No repositioning recommended at this time.")
    else:
        for rec in recommendations:
            print(f"\n  Car {rec.car_id}:")
            print(
                f"    Move to Zone {rec.target_zone_id} (Node {rec.target_location_id})"
            )
            print(f"    Priority: {rec.priority:.1f}")
            print(f"    Reason: {rec.reason}")

    print("=" * 60)
