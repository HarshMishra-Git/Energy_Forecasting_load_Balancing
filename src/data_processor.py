import pandas as pd
from config import REQUIRED_COLUMNS, DATE_FORMAT

class DataProcessor:
    def __init__(self):
        self.data = None

    def validate_data(self, df):
        """Validate uploaded data format"""
        missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        return True

    def preprocess_data(self, df):
        """Preprocess the input dataframe"""
        df = df.copy()

        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], format=DATE_FORMAT)

        # Sort by timestamp
        df = df.sort_values('timestamp')

        # Handle missing values
        df['energy_consumption'] = df['energy_consumption'].interpolate()

        # Remove outliers using IQR method
        Q1 = df['energy_consumption'].quantile(0.25)
        Q3 = df['energy_consumption'].quantile(0.75)
        IQR = Q3 - Q1
        df = df[
            (df['energy_consumption'] >= Q1 - 1.5 * IQR) &
            (df['energy_consumption'] <= Q3 + 1.5 * IQR)
        ]

        # Add time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month

        self.data = df
        return df

    def create_features(self, weather_data=None):
        """Create features for modeling"""
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")

        df = self.data.copy()

        # Create lag features
        df['consumption_lag_1h'] = df['energy_consumption'].shift(1)
        df['consumption_lag_24h'] = df['energy_consumption'].shift(24)

        # Add weather data if available
        if weather_data is not None:
            df = pd.merge(df, weather_data, on='timestamp', how='left')

        # Drop rows with NaN values
        df = df.dropna()

        return df