import numpy as np
from config import MAX_LOAD_THRESHOLD, MIN_LOAD_THRESHOLD

class LoadBalancer:
    def __init__(self):
        self.total_capacity = None
        
    def set_capacity(self, capacity):
        """Set the total capacity of the grid"""
        self.total_capacity = capacity
        
    def calculate_load_distribution(self, demand_forecast):
        """Calculate optimal load distribution"""
        if self.total_capacity is None:
            raise ValueError("Total capacity not set")
            
        # Calculate load percentage
        load_percentage = demand_forecast / self.total_capacity
        
        # Identify periods of high and low load
        high_load = load_percentage > MAX_LOAD_THRESHOLD
        low_load = load_percentage < MIN_LOAD_THRESHOLD
        
        # Calculate suggested redistribution
        redistribution = np.zeros_like(demand_forecast)
        redistribution[high_load] = -1 * (demand_forecast[high_load] - self.total_capacity * MAX_LOAD_THRESHOLD)
        redistribution[low_load] = self.total_capacity * MIN_LOAD_THRESHOLD - demand_forecast[low_load]
        
        return {
            'load_percentage': load_percentage,
            'high_load_periods': high_load,
            'low_load_periods': low_load,
            'suggested_redistribution': redistribution
        }