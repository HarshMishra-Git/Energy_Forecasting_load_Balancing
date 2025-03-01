import os

# Weather API configuration
WEATHER_API_KEY = os.getenv("WEATHERBIT_API_KEY", "8fe73ff057924b7aa58eb2b6acde1f53")
WEATHER_API_BASE_URL = "https://api.weatherbit.io/v2.0/history/hourly"

# Model parameters
FORECAST_HORIZON = 24  # hours
TRAIN_TEST_SPLIT = 0.8

# Load balancing parameters
MAX_LOAD_THRESHOLD = 0.85
MIN_LOAD_THRESHOLD = 0.15

# Data processing
REQUIRED_COLUMNS = ['timestamp', 'energy_consumption']
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'