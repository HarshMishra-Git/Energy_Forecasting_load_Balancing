import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

class WeatherImpactAnalyzer:
    def __init__(self):
        self.model = LinearRegression()
        self.temperature_sensitivity = None
        
    def analyze_temperature_impact(self, df):
        """Analyze the relationship between temperature and energy consumption"""
        if 'temperature' not in df.columns:
            raise ValueError("Temperature data not available")
            
        X = df[['temperature']].values
        y = df['energy_consumption'].values
        
        self.model.fit(X, y)
        self.temperature_sensitivity = self.model.coef_[0]
        
        return {
            'sensitivity': self.temperature_sensitivity,
            'baseline': self.model.intercept_
        }
        
    def simulate_temperature_change(self, df, delta_temp):
        """Simulate energy consumption with temperature change"""
        if self.temperature_sensitivity is None:
            self.analyze_temperature_impact(df)
            
        adjusted_consumption = (
            df['energy_consumption'].values + 
            self.temperature_sensitivity * delta_temp
        )
        
        return pd.DataFrame({
            'timestamp': df['timestamp'],
            'energy_consumption': adjusted_consumption
        })
