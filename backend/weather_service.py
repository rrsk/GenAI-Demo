"""
Weather service for weather-based health recommendations
"""
import httpx
from typing import Optional, Dict
from .config import WEATHERAPI_KEY


class WeatherService:
    """Fetches weather data and provides weather-based health insights"""
    
    BASE_URL = "https://api.weatherapi.com/v1"
    
    def __init__(self):
        self.api_key = WEATHERAPI_KEY
    
    async def get_current_weather(self, location: str) -> Optional[Dict]:
        """Get current weather for a location"""
        if not self.api_key:
            # Return mock data if no API key
            return self._get_mock_weather(location)
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.BASE_URL}/current.json",
                    params={
                        "key": self.api_key,
                        "q": location
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return self._parse_weather_data(data)
                return self._get_mock_weather(location)
                
        except Exception as e:
            print(f"Weather API error: {e}")
            return self._get_mock_weather(location)
    
    def _parse_weather_data(self, data: Dict) -> Dict:
        """Parse WeatherAPI.com response"""
        current = data["current"]
        location_data = data["location"]
        return {
            "temperature": current["temp_c"],
            "feels_like": current["feelslike_c"],
            "humidity": current["humidity"],
            "pressure": current["pressure_mb"],
            "condition": current["condition"]["text"],
            "description": current["condition"]["text"].lower(),
            "wind_speed": current["wind_kph"] / 3.6,  # Convert to m/s for consistency
            "clouds": current["cloud"],
            "location": location_data["name"],
            "country": location_data["country"]
        }
    
    def _get_mock_weather(self, location: str) -> Dict:
        """Return mock weather data for demo purposes"""
        import random
        conditions = ["Clear", "Clouds", "Rain", "Snow", "Humid"]
        return {
            "temperature": random.randint(5, 35),
            "feels_like": random.randint(3, 37),
            "humidity": random.randint(30, 90),
            "pressure": random.randint(1010, 1030),
            "condition": random.choice(conditions),
            "description": "mock weather data",
            "wind_speed": random.randint(0, 20),
            "clouds": random.randint(0, 100),
            "location": location or "Unknown",
            "country": "US"
        }
    
    def get_weather_health_impacts(self, weather: Dict) -> Dict:
        """Analyze weather impacts on health"""
        impacts = []
        recommendations = []
        
        temp = weather.get("temperature", 20)
        humidity = weather.get("humidity", 50)
        condition = weather.get("condition", "Clear")
        pressure = weather.get("pressure", 1015)
        
        # Temperature impacts
        if temp < 5:
            impacts.append("cold_stress")
            recommendations.append("Increase calorie intake for thermogenesis. Warm soups and stews recommended.")
            recommendations.append("Consider vitamin D supplementation due to reduced sun exposure.")
        elif temp > 30:
            impacts.append("heat_stress")
            recommendations.append("Increase hydration. Add electrolytes to water.")
            recommendations.append("Choose lighter, water-rich foods like cucumber and watermelon.")
        
        # Humidity impacts
        if humidity > 80:
            impacts.append("high_humidity")
            recommendations.append("Higher humidity can affect breathing. Stay hydrated and cool.")
            recommendations.append("Anti-inflammatory foods help with humidity-related joint discomfort.")
        elif humidity < 30:
            impacts.append("low_humidity")
            recommendations.append("Low humidity can cause dehydration. Increase water intake.")
            recommendations.append("Consider omega-3s for skin health in dry conditions.")
        
        # Weather condition impacts
        if condition in ["Rain", "Clouds", "Overcast"]:
            impacts.append("low_light")
            recommendations.append("Overcast conditions may affect mood. Consider vitamin D and mood-boosting foods.")
        
        # Barometric pressure
        if pressure < 1000:
            impacts.append("low_pressure")
            recommendations.append("Low pressure can trigger headaches/migraines. Stay hydrated, avoid trigger foods.")
        elif pressure > 1025:
            impacts.append("high_pressure")
            recommendations.append("High pressure generally favorable for health and energy.")
        
        # Seasonal considerations
        if temp < 10 and condition in ["Clear", "Clouds"]:
            impacts.append("cold_dry_season")
            recommendations.append("Boost immune system with citrus, ginger, garlic, and warm broths.")
        
        return {
            "weather_summary": weather,
            "health_impacts": impacts,
            "recommendations": recommendations,
            "hydration_multiplier": self._calculate_hydration_needs(temp, humidity),
            "calorie_adjustment": self._calculate_calorie_adjustment(temp)
        }
    
    def _calculate_hydration_needs(self, temp: float, humidity: float) -> float:
        """Calculate hydration needs multiplier based on weather"""
        base = 1.0
        if temp > 25:
            base += (temp - 25) * 0.05
        if humidity < 40:
            base += 0.1
        if humidity > 80:
            base += 0.05
        return round(min(base, 1.8), 2)
    
    def _calculate_calorie_adjustment(self, temp: float) -> int:
        """Calculate calorie adjustment based on temperature"""
        if temp < 5:
            return 200  # Extra calories for cold
        elif temp > 30:
            return -100  # Slightly less in heat
        return 0


# Singleton instance
_weather_service = None

def get_weather_service() -> WeatherService:
    """Get or create WeatherService singleton"""
    global _weather_service
    if _weather_service is None:
        _weather_service = WeatherService()
    return _weather_service
