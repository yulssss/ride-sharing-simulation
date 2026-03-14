from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.prediction import DemandPredictor, aggregate_demand, prepare_features


@pytest.fixture
def sample_ride_data():
    """Create a small DataFrame of ride data with enough variety for aggregation."""
    base_time = datetime(2024, 1, 1, 8, 0)  # Monday 8am
    data = []

    # Generate data for 10 hours to ensure sufficient aggregated samples
    for h in range(10):
        current_hour_time = base_time + timedelta(hours=h)
        
        # Ensure we have data for different zones to create multiple training examples
        for z in [1, 2]:
            # Vary number of rides slightly
            num_rides = 5 + (h % 3) + z
            
            for i in range(num_rides):
                data.append(
                    {
                        "timestamp": current_hour_time + timedelta(minutes=i * 2),
                        "hour": current_hour_time.hour,
                        "day_of_week": 0,  # Monday
                        "pickup_id": i + (h * 100),
                        "zone_id": z,
                        "dropoff_id": 100 + z,
                    }
                )
    return pd.DataFrame(data)


def test_feature_engineering(sample_ride_data):
    """Test feature creation logic."""
    features = prepare_features(sample_ride_data)

    assert "hour_sin" in features.columns
    assert "is_weekend" in features.columns
    assert "is_morning_rush" in features.columns

    # Check 8am is morning rush
    assert features[features["hour"] == 8]["is_morning_rush"].all() == 1

    # Check monday is not weekend
    assert features["is_weekend"].max() == 0


def test_demand_aggregation(sample_ride_data):
    """Test that rides are counted correctly per bucket."""
    agg = aggregate_demand(sample_ride_data, time_window_minutes=60)

    # We expect buckets for 8am and 9am
    assert not agg.empty
    assert "demand_count" in agg.columns

    # Zone 1 at 8am calculation:
    # h=0 (8am), z=1
    # num_rides = 5 + (0 % 3) + 1 = 6
    zone1_8am = agg[(agg["zone_id"] == 1) & (agg["hour"] == 8)]
    if not zone1_8am.empty:
        assert zone1_8am.iloc[0]["demand_count"] == 6


def test_training_pipeline(sample_ride_data):
    """Test that the model trains without error."""
    # We need enough data to split, so dupe it a bit
    big_data = pd.concat([sample_ride_data] * 5, ignore_index=True)

    predictor = DemandPredictor(model_type="random_forest")

    # Should run without error
    # Use train() directly with DataFrame
    metrics = predictor.train(big_data)

    assert "r2" in metrics
    assert "mae" in metrics
    assert predictor.model is not None


def test_prediction_output(sample_ride_data):
    """Test that predict_all_zones returns valid structure."""
    big_data = pd.concat([sample_ride_data] * 5, ignore_index=True)
    predictor = DemandPredictor()
    predictor.train(big_data)

    preds = predictor.predict_all_zones(
        num_zones=5, timestamp=datetime(2024, 1, 1, 10, 0), horizon_hours=1
    )

    assert len(preds) > 0
    # preds is Dict[zone_id, List[DemandPrediction]]
    # Check predictions for zone 0
    if 0 in preds and preds[0]:
        assert preds[0][0].zone_id == 0
        assert preds[0][0].predicted_demand >= 0
