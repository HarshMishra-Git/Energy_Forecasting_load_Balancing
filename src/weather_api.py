import requests
import pandas as pd
from config import WEATHER_API_KEY, WEATHER_API_BASE_URL

class WeatherAPI:
    def __init__(self):
        self.api_key = WEATHER_API_KEY
        self.base_url = WEATHER_API_BASE_URL

    def get_historical_weather(self, lat, lon, start_date, end_date):
        """Fetch historical weather data for the given location and date range"""
        try:
            params = {
                'lat': lat,
                'lon': lon,
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d'),
                'key': self.api_key
            }
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            weather_data = response.json()

            # Process weather data
            df_weather = pd.DataFrame(weather_data['data'])
            df_weather['timestamp'] = pd.to_datetime(df_weather['timestamp_local'])
            df_weather = df_weather[['timestamp', 'temp', 'rh', 'weather']]
            df_weather.rename(columns={'temp': 'temperature', 'rh': 'humidity'}, inplace=True)

            return df_weather
        except requests.RequestException as e:
            raise Exception(f"Error fetching weather data: {e}")