"""
Pydantic models for API requests and responses
"""
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import date
from enum import Enum


class Gender(str, Enum):
    MALE = "Male"
    FEMALE = "Female"


class FitnessLevel(str, Enum):
    BEGINNER = "Beginner"
    INTERMEDIATE = "Intermediate"
    ADVANCED = "Advanced"
    ELITE = "Elite"


class UserProfile(BaseModel):
    """User health profile"""
    user_id: str
    age: int = Field(ge=18, le=100)
    gender: Gender
    weight_kg: float = Field(ge=30, le=300)
    height_cm: float = Field(ge=100, le=250)
    fitness_level: FitnessLevel
    primary_sport: Optional[str] = None
    dietary_preferences: Optional[List[str]] = None
    health_conditions: Optional[List[str]] = None
    location: Optional[str] = None  # For weather-based recommendations


class HealthMetrics(BaseModel):
    """Current health metrics from Whoop data"""
    recovery_score: float = Field(ge=0, le=100)
    day_strain: float = Field(ge=0, le=21)
    sleep_hours: float = Field(ge=0, le=24)
    sleep_efficiency: float = Field(ge=0, le=100)
    hrv: float = Field(ge=0)
    resting_heart_rate: float = Field(ge=30, le=200)
    respiratory_rate: float = Field(ge=5, le=40)
    skin_temp_deviation: float = Field(ge=-5, le=5)
    deep_sleep_hours: float = Field(ge=0, le=12)
    rem_sleep_hours: float = Field(ge=0, le=12)
    calories_burned: float = Field(ge=0)


class WeatherData(BaseModel):
    """Weather information for recommendations"""
    temperature: float
    humidity: float
    condition: str
    uv_index: Optional[float] = None
    air_quality_index: Optional[int] = None


class ChatMessage(BaseModel):
    """Chat message from user"""
    message: str
    user_id: Optional[str] = None
    include_weather: bool = False
    location: Optional[str] = None


class HealthConcern(BaseModel):
    """User's health concern or symptom"""
    concern_type: str  # headache, fatigue, insomnia, etc.
    severity: int = Field(ge=1, le=10)
    duration: str  # "acute", "chronic"
    notes: Optional[str] = None


class MealPlan(BaseModel):
    """Generated meal plan"""
    date: date
    breakfast: dict
    lunch: dict
    dinner: dict
    snacks: List[dict]
    total_calories: int
    macros: dict
    reasoning: str


class HealthRecommendation(BaseModel):
    """AI-generated health recommendation"""
    category: str  # "nutrition", "sleep", "exercise", "recovery"
    recommendation: str
    priority: str  # "high", "medium", "low"
    scientific_basis: Optional[str] = None


class ChatResponse(BaseModel):
    """Response from the AI assistant"""
    message: str
    recommendations: Optional[List[HealthRecommendation]] = None
    meal_plan: Optional[MealPlan] = None
    insights: Optional[dict] = None
