"""
Recommendation Engine for WellnessAI

This module implements a Python-based recommendation system that:
1. Analyzes health metrics using rule-based logic
2. Integrates with ML predictions for personalized insights
3. Generates structured recommendations for meal, activity, and lifestyle
4. Provides the foundation for LLM-enhanced natural language responses

Key GenAI Concepts Demonstrated:
- Feature engineering for health data
- Rule-based expert systems
- Integration of ML predictions with domain knowledge
- Structured data preparation for LLM context
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import math


class HealthStatus(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    MODERATE = "moderate"
    POOR = "poor"
    CRITICAL = "critical"


class QueryIntent(Enum):
    MEAL_PLAN = "meal_plan"
    HEADACHE = "headache"
    FATIGUE = "fatigue"
    SLEEP = "sleep"
    EXERCISE = "exercise"
    RECOVERY = "recovery"
    GENERAL = "general"
    STRESS = "stress"


@dataclass
class Recommendation:
    """Structured recommendation output"""
    category: str
    title: str
    description: str
    priority: int  # 1 = highest
    actions: List[str]
    foods: List[str] = None
    avoid: List[str] = None


class HealthRecommendationEngine:
    """
    Rule-based Health Recommendation Engine
    
    This engine analyzes health data and generates personalized recommendations
    using domain knowledge encoded as rules. It integrates with ML predictions
    for enhanced accuracy.
    """
    
    def __init__(self):
        # Nutrition knowledge base
        self.nutrition_db = {
            "energy_boost": {
                "foods": [
                    "Oatmeal with berries and nuts",
                    "Greek yogurt with honey",
                    "Bananas with almond butter",
                    "Whole grain toast with avocado",
                    "Eggs with spinach"
                ],
                "nutrients": ["Iron", "B-vitamins", "Complex carbs", "Protein"],
                "avoid": ["Refined sugars", "Heavy fatty foods", "Excessive caffeine"]
            },
            "sleep_support": {
                "foods": [
                    "Chamomile tea",
                    "Tart cherry juice",
                    "Warm milk with turmeric",
                    "Almonds and walnuts",
                    "Kiwi fruit",
                    "Turkey or chicken"
                ],
                "nutrients": ["Tryptophan", "Magnesium", "Melatonin precursors"],
                "avoid": ["Caffeine after 2pm", "Alcohol", "Heavy meals before bed", "Spicy foods"]
            },
            "headache_relief": {
                "foods": [
                    "Water (hydration first)",
                    "Ginger tea",
                    "Leafy greens (magnesium)",
                    "Fatty fish (omega-3s)",
                    "Peppermint tea"
                ],
                "nutrients": ["Magnesium", "Riboflavin", "Omega-3", "Water"],
                "avoid": ["Aged cheese", "Processed meats", "MSG", "Artificial sweeteners", "Alcohol"]
            },
            "recovery_optimization": {
                "foods": [
                    "Salmon with vegetables",
                    "Quinoa bowl with lean protein",
                    "Bone broth",
                    "Berries and dark chocolate",
                    "Sweet potato with chicken"
                ],
                "nutrients": ["Protein (1.6-2g/kg)", "Omega-3s", "Antioxidants", "Complex carbs"],
                "avoid": ["Excessive alcohol", "Fried foods", "High sodium processed foods"]
            },
            "stress_reduction": {
                "foods": [
                    "Dark chocolate (70%+ cacao)",
                    "Blueberries",
                    "Avocado",
                    "Green tea (L-theanine)",
                    "Fatty fish",
                    "Fermented foods (yogurt, kimchi)"
                ],
                "nutrients": ["Omega-3s", "Magnesium", "B-vitamins", "Probiotics"],
                "avoid": ["Caffeine excess", "Alcohol", "Refined sugars", "Processed foods"]
            }
        }
        
        # Activity recommendations based on recovery
        self.activity_matrix = {
            (90, 100): {
                "intensity": "High",
                "workouts": ["HIIT", "CrossFit", "Heavy lifting", "Sprint intervals"],
                "duration": "45-75 min",
                "message": "Your body is primed for peak performance!"
            },
            (70, 89): {
                "intensity": "Moderate-High",
                "workouts": ["Strength training", "Running", "Cycling", "Swimming"],
                "duration": "40-60 min",
                "message": "Great recovery - mix intensity with some rest."
            },
            (50, 69): {
                "intensity": "Moderate",
                "workouts": ["Light strength", "Yoga", "Brisk walking", "Light cycling"],
                "duration": "30-45 min",
                "message": "Focus on technique and endurance today."
            },
            (30, 49): {
                "intensity": "Low",
                "workouts": ["Yoga", "Stretching", "Walking", "Swimming (easy)"],
                "duration": "20-30 min",
                "message": "Active recovery - gentle movement only."
            },
            (0, 29): {
                "intensity": "Rest",
                "workouts": ["Stretching", "Meditation", "Light walk", "Foam rolling"],
                "duration": "15-20 min",
                "message": "Your body needs rest. Prioritize recovery today."
            }
        }
    
    def classify_intent(self, message: str) -> QueryIntent:
        """Classify user message intent for targeted recommendations"""
        message_lower = message.lower()
        
        intent_keywords = {
            QueryIntent.MEAL_PLAN: ["meal", "eat", "food", "diet", "nutrition", "breakfast", "lunch", "dinner", "snack", "plan"],
            QueryIntent.HEADACHE: ["headache", "migraine", "head pain", "head hurts"],
            QueryIntent.FATIGUE: ["tired", "fatigue", "exhausted", "energy", "low energy", "sleepy", "drowsy"],
            QueryIntent.SLEEP: ["sleep", "insomnia", "can't sleep", "sleeping", "rest", "bedtime"],
            QueryIntent.EXERCISE: ["workout", "exercise", "train", "activity", "gym", "run", "fitness"],
            QueryIntent.RECOVERY: ["recovery", "recover", "sore", "muscle", "rest day"],
            QueryIntent.STRESS: ["stress", "anxious", "anxiety", "overwhelmed", "tense", "calm"]
        }
        
        for intent, keywords in intent_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                return intent
        
        return QueryIntent.GENERAL
    
    def analyze_health_status(self, health_context: Dict) -> Dict[str, HealthStatus]:
        """Analyze overall health status from metrics"""
        metrics = health_context.get("recent_metrics", {})
        statuses = {}
        
        # Recovery status
        recovery = metrics.get("avg_recovery_score", 65)
        if recovery >= 80:
            statuses["recovery"] = HealthStatus.EXCELLENT
        elif recovery >= 60:
            statuses["recovery"] = HealthStatus.GOOD
        elif recovery >= 40:
            statuses["recovery"] = HealthStatus.MODERATE
        else:
            statuses["recovery"] = HealthStatus.POOR
        
        # Sleep status
        sleep_hours = metrics.get("avg_sleep_hours", 7)
        sleep_efficiency = metrics.get("avg_sleep_efficiency", 80)
        if sleep_hours >= 7.5 and sleep_efficiency >= 85:
            statuses["sleep"] = HealthStatus.EXCELLENT
        elif sleep_hours >= 7 and sleep_efficiency >= 75:
            statuses["sleep"] = HealthStatus.GOOD
        elif sleep_hours >= 6:
            statuses["sleep"] = HealthStatus.MODERATE
        else:
            statuses["sleep"] = HealthStatus.POOR
        
        # HRV/Stress status
        hrv = metrics.get("avg_hrv", 100)
        if hrv >= 120:
            statuses["stress"] = HealthStatus.EXCELLENT  # Low stress
        elif hrv >= 80:
            statuses["stress"] = HealthStatus.GOOD
        elif hrv >= 50:
            statuses["stress"] = HealthStatus.MODERATE
        else:
            statuses["stress"] = HealthStatus.POOR  # High stress
        
        # Activity status
        strain = metrics.get("avg_strain", 10)
        workout_days = metrics.get("workout_days", 3)
        if 8 <= strain <= 15 and workout_days >= 4:
            statuses["activity"] = HealthStatus.EXCELLENT
        elif 5 <= strain <= 18 and workout_days >= 3:
            statuses["activity"] = HealthStatus.GOOD
        elif workout_days >= 2:
            statuses["activity"] = HealthStatus.MODERATE
        else:
            statuses["activity"] = HealthStatus.POOR
        
        return statuses
    
    def calculate_calorie_needs(self, profile: Dict, metrics: Dict) -> Dict:
        """Calculate personalized calorie and macro needs"""
        # Harris-Benedict equation for BMR
        weight = profile.get("weight_kg", 70)
        height = profile.get("height_cm", 170)
        age = profile.get("age", 30)
        gender = profile.get("gender", "Male")
        
        if gender == "Male":
            bmr = 88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * age)
        else:
            bmr = 447.593 + (9.247 * weight) + (3.098 * height) - (4.330 * age)
        
        # Activity multiplier based on strain
        strain = metrics.get("avg_strain", 10)
        if strain >= 15:
            multiplier = 1.725  # Very active
        elif strain >= 10:
            multiplier = 1.55   # Moderately active
        elif strain >= 5:
            multiplier = 1.375  # Lightly active
        else:
            multiplier = 1.2    # Sedentary
        
        tdee = bmr * multiplier
        
        return {
            "bmr": round(bmr),
            "tdee": round(tdee),
            "protein_g": round(weight * 1.6),  # 1.6g per kg
            "carbs_g": round((tdee * 0.45) / 4),  # 45% from carbs
            "fat_g": round((tdee * 0.25) / 9),  # 25% from fat
            "fiber_g": 30,
            "water_liters": round(weight * 0.033, 1)
        }
    
    def get_activity_recommendation(self, recovery_score: float) -> Dict:
        """Get activity recommendation based on recovery"""
        for (low, high), rec in self.activity_matrix.items():
            if low <= recovery_score <= high:
                return rec
        return self.activity_matrix[(50, 69)]  # Default moderate
    
    def generate_meal_plan(
        self, 
        profile: Dict, 
        metrics: Dict, 
        focus_area: str = "general"
    ) -> Dict:
        """Generate a personalized meal plan"""
        nutrition = self.calculate_calorie_needs(profile, metrics)
        
        # Base meal structure
        meal_plan = {
            "daily_targets": nutrition,
            "breakfast": {
                "time": "7:00-8:00 AM",
                "calories": round(nutrition["tdee"] * 0.25),
                "options": [
                    "Greek yogurt parfait with granola and mixed berries",
                    "Oatmeal with banana, almonds, and honey",
                    "Eggs with whole grain toast and avocado"
                ]
            },
            "lunch": {
                "time": "12:00-1:00 PM",
                "calories": round(nutrition["tdee"] * 0.35),
                "options": [
                    "Grilled chicken salad with quinoa",
                    "Salmon bowl with brown rice and vegetables",
                    "Turkey wrap with hummus and greens"
                ]
            },
            "dinner": {
                "time": "6:00-7:00 PM",
                "calories": round(nutrition["tdee"] * 0.30),
                "options": [
                    "Baked fish with sweet potato and broccoli",
                    "Lean steak with roasted vegetables",
                    "Chicken stir-fry with brown rice"
                ]
            },
            "snacks": {
                "calories": round(nutrition["tdee"] * 0.10),
                "options": [
                    "Apple with almond butter",
                    "Handful of mixed nuts",
                    "Protein smoothie"
                ]
            }
        }
        
        # Add focus-specific recommendations
        if focus_area in self.nutrition_db:
            db_entry = self.nutrition_db[focus_area]
            meal_plan["focus_foods"] = db_entry["foods"][:3]
            meal_plan["foods_to_avoid"] = db_entry["avoid"][:3]
            meal_plan["key_nutrients"] = db_entry["nutrients"]
        
        return meal_plan
    
    def generate_recommendations(
        self,
        health_context: Dict,
        intent: QueryIntent,
        weather_data: Optional[Dict] = None,
        predictions: Optional[Dict] = None
    ) -> List[Recommendation]:
        """Generate prioritized recommendations based on health data and intent"""
        recommendations = []
        metrics = health_context.get("recent_metrics", {})
        profile = health_context.get("profile", {})
        risks = health_context.get("health_risks", [])
        
        # Analyze current status
        statuses = self.analyze_health_status(health_context)
        
        # Intent-specific recommendations
        if intent == QueryIntent.MEAL_PLAN:
            meal_plan = self.generate_meal_plan(profile, metrics)
            recommendations.append(Recommendation(
                category="Nutrition",
                title="Personalized Meal Plan",
                description=f"Based on your metrics, you need approximately {meal_plan['daily_targets']['tdee']} calories/day",
                priority=1,
                actions=[
                    f"Aim for {meal_plan['daily_targets']['protein_g']}g protein daily",
                    f"Drink at least {meal_plan['daily_targets']['water_liters']}L water",
                    "Eat meals at consistent times"
                ],
                foods=meal_plan["breakfast"]["options"] + meal_plan["lunch"]["options"][:2]
            ))
        
        elif intent == QueryIntent.HEADACHE:
            db = self.nutrition_db["headache_relief"]
            recommendations.append(Recommendation(
                category="Symptom Relief",
                title="Headache Management",
                description="Natural approaches to relieve your headache",
                priority=1,
                actions=[
                    "Hydrate immediately - drink 500ml water",
                    "Rest in a quiet, dark room for 15-20 minutes",
                    "Apply cold compress to forehead or temples",
                    "Consider magnesium-rich foods"
                ],
                foods=db["foods"],
                avoid=db["avoid"]
            ))
        
        elif intent == QueryIntent.FATIGUE:
            db = self.nutrition_db["energy_boost"]
            recovery = metrics.get("avg_recovery_score", 65)
            recommendations.append(Recommendation(
                category="Energy",
                title="Combat Fatigue",
                description=f"Your recovery is at {recovery}% - here's how to boost energy",
                priority=1,
                actions=[
                    "Take a 10-15 minute power nap if possible",
                    "Get natural sunlight for 10+ minutes",
                    "Do 5 minutes of light stretching",
                    "Check hydration - aim for clear urine"
                ],
                foods=db["foods"],
                avoid=db["avoid"]
            ))
        
        elif intent == QueryIntent.SLEEP:
            db = self.nutrition_db["sleep_support"]
            sleep_hours = metrics.get("avg_sleep_hours", 6)
            recommendations.append(Recommendation(
                category="Sleep",
                title="Improve Sleep Quality",
                description=f"You're averaging {sleep_hours:.1f} hours - let's optimize this",
                priority=1,
                actions=[
                    "Set a consistent bedtime (tonight: 10:30 PM)",
                    "Dim lights 1 hour before bed",
                    "No screens 30 minutes before sleep",
                    "Keep bedroom cool (65-68°F / 18-20°C)"
                ],
                foods=db["foods"],
                avoid=db["avoid"]
            ))
        
        elif intent == QueryIntent.EXERCISE:
            recovery = metrics.get("avg_recovery_score", 65)
            activity_rec = self.get_activity_recommendation(recovery)
            recommendations.append(Recommendation(
                category="Activity",
                title=f"{activity_rec['intensity']} Intensity Day",
                description=activity_rec["message"],
                priority=1,
                actions=[
                    f"Workout duration: {activity_rec['duration']}",
                    f"Recommended: {', '.join(activity_rec['workouts'][:2])}",
                    "Warm up for 5-10 minutes first",
                    "Post-workout: protein within 30 minutes"
                ]
            ))
        
        elif intent == QueryIntent.STRESS:
            db = self.nutrition_db["stress_reduction"]
            hrv = metrics.get("avg_hrv", 80)
            recommendations.append(Recommendation(
                category="Stress Management",
                title="Reduce Stress & Anxiety",
                description=f"Your HRV ({hrv}ms) indicates {'high' if hrv < 60 else 'moderate' if hrv < 90 else 'good'} stress response",
                priority=1,
                actions=[
                    "Practice 4-7-8 breathing (4s inhale, 7s hold, 8s exhale)",
                    "Take a 10-minute walk outside",
                    "Try 5 minutes of guided meditation",
                    "Reduce caffeine intake today"
                ],
                foods=db["foods"],
                avoid=db["avoid"]
            ))
        
        # Add risk-based recommendations
        for risk in risks[:2]:
            if risk.get("severity") in ["high", "critical"]:
                recommendations.append(Recommendation(
                    category="Health Alert",
                    title=f"⚠️ {risk.get('risk', 'Health Concern')}",
                    description=risk.get("dietary_impact", ""),
                    priority=2,
                    actions=[risk.get("recommendation", "Monitor and adjust as needed")]
                ))
        
        # Add weather-based recommendations
        if weather_data:
            weather_impacts = weather_data.get("health_impacts", [])
            weather_recs = weather_data.get("recommendations", [])
            if weather_impacts or weather_recs:
                recommendations.append(Recommendation(
                    category="Weather Adaptation",
                    title="Weather-Based Adjustments",
                    description=f"Current: {weather_data.get('weather_summary', {}).get('temperature', 'N/A')}°C, {weather_data.get('weather_summary', {}).get('condition', '')}",
                    priority=3,
                    actions=weather_recs[:3] if weather_recs else ["Stay hydrated", "Adjust activity for conditions"]
                ))
        
        # Sort by priority
        recommendations.sort(key=lambda x: x.priority)
        
        return recommendations
    
    def format_recommendations_for_llm(self, recommendations: List[Recommendation]) -> str:
        """Format recommendations as structured text for LLM context"""
        if not recommendations:
            return "No specific recommendations generated. Provide general wellness advice."
        
        parts = []
        for rec in recommendations:
            section = f"### {rec.title} ({rec.category})\n"
            section += f"{rec.description}\n\n"
            section += "**Actions:**\n"
            for action in rec.actions:
                section += f"- {action}\n"
            
            if rec.foods:
                section += "\n**Recommended Foods:**\n"
                for food in rec.foods[:5]:
                    section += f"- {food}\n"
            
            if rec.avoid:
                section += "\n**Avoid:**\n"
                for item in rec.avoid[:3]:
                    section += f"- {item}\n"
            
            parts.append(section)
        
        return "\n".join(parts)
    
    def format_health_context_for_llm(self, health_context: Dict, forecasts: List[Dict] = None) -> str:
        """Format health context as structured text for LLM, including forecasts"""
        parts = []
        
        # Profile summary
        if profile := health_context.get("profile"):
            parts.append(f"""**User Profile:**
- {profile.get('age', 'N/A')} years old, {profile.get('gender', 'N/A')}
- {profile.get('weight_kg', 'N/A')} kg, {profile.get('height_cm', 'N/A')} cm
- Fitness Level: {profile.get('fitness_level', 'N/A')}
- Primary Sport: {profile.get('primary_sport', 'None')}""")
        
        # Current metrics
        if metrics := health_context.get("recent_metrics"):
            parts.append(f"""**Current Health Metrics:**
- Recovery Score: {metrics.get('avg_recovery_score', 'N/A')}%
- Sleep: {metrics.get('avg_sleep_hours', 'N/A')} hours ({metrics.get('avg_sleep_efficiency', 'N/A')}% efficiency)
- HRV: {metrics.get('avg_hrv', 'N/A')} ms
- Resting HR: {metrics.get('avg_resting_hr', 'N/A')} bpm
- Daily Strain: {metrics.get('avg_strain', 'N/A')}""")
        
        # LSTM Forecasts (3-day predictions)
        if forecasts:
            forecast_lines = ["**3-Day Health Forecast (LSTM Predictions):**"]
            for day in forecasts[:3]:
                forecast_lines.append(
                    f"- Day {day.get('day', '?')} ({day.get('date', 'N/A')}): "
                    f"Recovery {day.get('recovery_score', 0):.0f}%, "
                    f"Sleep {day.get('sleep_hours', 0):.1f}h, "
                    f"HRV {day.get('hrv', 0):.0f}ms"
                )
            
            # Add forecast insights
            if len(forecasts) >= 2:
                avg_recovery = sum(f.get('recovery_score', 65) for f in forecasts[:3]) / min(3, len(forecasts))
                if avg_recovery < 50:
                    forecast_lines.append("**Alert:** Low recovery predicted - recommend reducing activity intensity")
                elif avg_recovery > 75:
                    forecast_lines.append("**Outlook:** Strong recovery ahead - good time for challenging workouts")
            
            parts.append("\n".join(forecast_lines))
        
        # Health risks
        if risks := health_context.get("health_risks"):
            risk_text = "\n".join([f"- {r.get('risk', 'Unknown')}: {r.get('dietary_impact', '')}" for r in risks[:3]])
            parts.append(f"**Health Concerns:**\n{risk_text}")
        
        return "\n\n".join(parts)
    
    def generate_proactive_recommendations(self, forecasts: List[Dict]) -> List[Recommendation]:
        """Generate proactive recommendations based on LSTM predictions"""
        recommendations = []
        
        if not forecasts:
            return recommendations
        
        for day in forecasts[:3]:
            day_num = day.get('day', 1)
            date = day.get('date', 'upcoming')
            recovery = day.get('recovery_score', 65)
            sleep_hours = day.get('sleep_hours', 7)
            
            # Low recovery warning
            if recovery < 50:
                db = self.nutrition_db.get("recovery_optimization", {})
                recommendations.append(Recommendation(
                    category="Forecast Alert",
                    title=f"Day {day_num}: Low Recovery Predicted ({recovery:.0f}%)",
                    description=f"On {date}, your recovery may be low. Plan accordingly.",
                    priority=1,
                    actions=[
                        "Consider lighter activities or rest day",
                        "Prioritize sleep tonight",
                        "Focus on anti-inflammatory foods",
                        "Stay well hydrated"
                    ],
                    foods=db.get("foods", [])[:3],
                    avoid=db.get("avoid", [])[:2]
                ))
            
            # Sleep deficit warning
            if sleep_hours < 6:
                db = self.nutrition_db.get("sleep_support", {})
                recommendations.append(Recommendation(
                    category="Forecast Alert",
                    title=f"Day {day_num}: Sleep May Be Insufficient ({sleep_hours:.1f}h)",
                    description=f"Predicted low sleep on {date}. Take preventive action.",
                    priority=2,
                    actions=[
                        "Set earlier bedtime reminder",
                        "Avoid caffeine after 2pm",
                        "Create calm evening routine"
                    ],
                    foods=db.get("foods", [])[:3],
                    avoid=db.get("avoid", [])[:2]
                ))
        
        return recommendations


# Singleton instance
_recommendation_engine = None

def get_recommendation_engine() -> HealthRecommendationEngine:
    """Get or create recommendation engine singleton"""
    global _recommendation_engine
    if _recommendation_engine is None:
        _recommendation_engine = HealthRecommendationEngine()
    return _recommendation_engine
