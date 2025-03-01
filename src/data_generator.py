import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_sample_data(days=30, frequency='1H'):
    """
    Generate sample energy consumption data with temperature

    Parameters:
    - days: Number of days of data to generate
    - frequency: Time frequency ('1H' for hourly)

    Returns:
    - DataFrame with timestamp, energy consumption, and temperature
    """
    # Create timestamp range
    end_date = datetime.now().replace(minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=days)
    timestamps = pd.date_range(start=start_date, end=end_date, freq=frequency)

    # Generate base consumption with daily and weekly patterns
    n_points = len(timestamps)
    base_load = 50 + np.random.normal(0, 5, n_points)

    # Add daily pattern (higher during day, lower at night)
    hour_pattern = 20 * np.sin(np.pi * timestamps.hour / 12)

    # Add weekly pattern (lower on weekends)
    weekend_mask = timestamps.dayofweek.isin([5, 6])
    weekly_pattern = np.where(weekend_mask, -10, 0)

    # Generate temperature data (simulated daily pattern)
    base_temp = 20 + 5 * np.sin(2 * np.pi * np.arange(n_points) / (24 * 7))  # Weekly cycle
    daily_temp = 5 * np.sin(2 * np.pi * timestamps.hour / 24)  # Daily cycle
    temperature = base_temp + daily_temp + np.random.normal(0, 1, n_points)  # Add noise

    # Add temperature effect to energy consumption
    temp_effect = 0.5 * (temperature - 20)  # Consumption increases with temperature difference from 20°C
    energy_consumption = base_load + hour_pattern + weekly_pattern + temp_effect
    energy_consumption = np.maximum(energy_consumption, 0)  # Ensure non-negative values

    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'energy_consumption': energy_consumption,
        'temperature': temperature
    })

    return df