import random

class EnergyRecommender:
    def __init__(self):
        self.tips = [
            "Turn off lights when not in use.",
            "Use energy-efficient LED bulbs.",
            "Unplug electronics when not in use.",
            "Use a programmable thermostat.",
            "Wash clothes in cold water.",
            "Air dry your clothes.",
            "Use a power strip for electronics.",
            "Set your refrigerator temperature to 37°F.",
            "Use natural light during the day.",
            "Insulate your home to save energy."
        ]

    def get_eco_friendly_tips(self, n=5):
        """Get a list of eco-friendly tips"""
        return random.sample(self.tips, n)