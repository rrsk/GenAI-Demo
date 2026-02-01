"""
Health data analysis service using Whoop data
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timedelta

from .config import WHOOP_DATA_PATH

try:
    from .ml_service import get_ml_service
    from .lstm_service import get_lstm_service
except ImportError:
    get_ml_service = None
    get_lstm_service = None


class HealthAnalyzer:
    """Analyzes Whoop health data to extract insights and patterns"""
    
    def __init__(self):
        self.df = None
        self._load_data()
    
    def _load_data(self):
        """Load Whoop dataset"""
        if WHOOP_DATA_PATH.exists():
            self.df = pd.read_csv(WHOOP_DATA_PATH)
            self.df['date'] = pd.to_datetime(self.df['date'])
        else:
            raise FileNotFoundError(f"Whoop data not found at {WHOOP_DATA_PATH}")
    
    def get_user_profile(self, user_id: str) -> Optional[Dict]:
        """Get user profile from data"""
        if self.df is None:
            return None
        
        user_data = self.df[self.df['user_id'] == user_id]
        if user_data.empty:
            return None
        
        latest = user_data.iloc[-1]
        return {
            "user_id": user_id,
            "age": int(latest['age']),
            "gender": latest['gender'],
            "weight_kg": float(latest['weight_kg']),
            "height_cm": float(latest['height_cm']),
            "fitness_level": latest['fitness_level'],
            "primary_sport": latest['primary_sport']
        }
    
    def get_recent_health_metrics(self, user_id: str, days: int = 7) -> Dict:
        """Get recent health metrics for a user"""
        if self.df is None:
            return {}
        
        user_data = self.df[self.df['user_id'] == user_id].tail(days)
        if user_data.empty:
            return {}
        
        return {
            "avg_recovery_score": round(user_data['recovery_score'].mean(), 1),
            "avg_sleep_hours": round(user_data['sleep_hours'].mean(), 2),
            "avg_sleep_efficiency": round(user_data['sleep_efficiency'].mean(), 1),
            "avg_hrv": round(user_data['hrv'].mean(), 1),
            "avg_resting_hr": round(user_data['resting_heart_rate'].mean(), 1),
            "avg_strain": round(user_data['day_strain'].mean(), 2),
            "avg_deep_sleep": round(user_data['deep_sleep_hours'].mean(), 2),
            "avg_rem_sleep": round(user_data['rem_sleep_hours'].mean(), 2),
            "avg_respiratory_rate": round(user_data['respiratory_rate'].mean(), 1),
            "avg_skin_temp_deviation": round(user_data['skin_temp_deviation'].mean(), 2),
            "total_calories": int(user_data['calories_burned'].sum()),
            "workout_days": int(user_data['workout_completed'].sum()),
            "trend_recovery": self._calculate_trend(user_data['recovery_score']),
            "trend_sleep": self._calculate_trend(user_data['sleep_hours']),
            "trend_hrv": self._calculate_trend(user_data['hrv'])
        }
    
    def _calculate_trend(self, series: pd.Series) -> str:
        """Calculate trend direction"""
        if len(series) < 2:
            return "stable"
        
        first_half = series.iloc[:len(series)//2].mean()
        second_half = series.iloc[len(series)//2:].mean()
        
        diff_pct = ((second_half - first_half) / first_half) * 100 if first_half != 0 else 0
        
        if diff_pct > 5:
            return "improving"
        elif diff_pct < -5:
            return "declining"
        return "stable"
    
    def analyze_sleep_patterns(self, user_id: str, days: int = 30) -> Dict:
        """Analyze sleep patterns and identify issues"""
        user_data = self.df[self.df['user_id'] == user_id].tail(days)
        if user_data.empty:
            return {}
        
        issues = []
        recommendations = []
        
        avg_sleep = user_data['sleep_hours'].mean()
        avg_efficiency = user_data['sleep_efficiency'].mean()
        avg_deep = user_data['deep_sleep_hours'].mean()
        avg_rem = user_data['rem_sleep_hours'].mean()
        avg_wake_ups = user_data['wake_ups'].mean()
        avg_time_to_sleep = user_data['time_to_fall_asleep_min'].mean()
        
        # Check for sleep issues
        if avg_sleep < 7:
            issues.append("insufficient_sleep_duration")
            recommendations.append("Aim for 7-9 hours of sleep. Consider earlier bedtime.")
        
        if avg_efficiency < 75:
            issues.append("low_sleep_efficiency")
            recommendations.append("Improve sleep hygiene. Reduce screen time before bed.")
        
        if avg_deep < 1.0:
            issues.append("insufficient_deep_sleep")
            recommendations.append("Increase magnesium-rich foods. Avoid alcohol before bed.")
        
        if avg_rem < 1.2:
            issues.append("insufficient_rem_sleep")
            recommendations.append("Maintain consistent sleep schedule. Reduce late caffeine.")
        
        if avg_wake_ups > 2:
            issues.append("frequent_awakenings")
            recommendations.append("Check room temperature (65-68°F). Limit fluids before bed.")
        
        if avg_time_to_sleep > 20:
            issues.append("difficulty_falling_asleep")
            recommendations.append("Practice relaxation techniques. Consider chamomile tea.")
        
        return {
            "avg_sleep_hours": round(avg_sleep, 2),
            "avg_efficiency": round(avg_efficiency, 1),
            "avg_deep_sleep": round(avg_deep, 2),
            "avg_rem_sleep": round(avg_rem, 2),
            "avg_wake_ups": round(avg_wake_ups, 1),
            "avg_time_to_sleep": round(avg_time_to_sleep, 1),
            "issues": issues,
            "recommendations": recommendations
        }
    
    def analyze_recovery_patterns(self, user_id: str, days: int = 30) -> Dict:
        """Analyze recovery patterns and stress indicators"""
        user_data = self.df[self.df['user_id'] == user_id].tail(days)
        if user_data.empty:
            return {}
        
        issues = []
        recommendations = []
        
        avg_recovery = user_data['recovery_score'].mean()
        avg_hrv = user_data['hrv'].mean()
        hrv_baseline = user_data['hrv_baseline'].iloc[-1]
        avg_rhr = user_data['resting_heart_rate'].mean()
        rhr_baseline = user_data['rhr_baseline'].iloc[-1]
        avg_strain = user_data['day_strain'].mean()
        avg_skin_temp = user_data['skin_temp_deviation'].mean()
        
        # Analyze recovery issues
        if avg_recovery < 50:
            issues.append("low_recovery")
            recommendations.append("Reduce training intensity. Focus on rest and nutrition.")
        
        if avg_hrv < hrv_baseline * 0.8:
            issues.append("low_hrv")
            recommendations.append("HRV below baseline indicates stress. Prioritize recovery.")
        
        if avg_rhr > rhr_baseline * 1.1:
            issues.append("elevated_resting_hr")
            recommendations.append("Elevated RHR may indicate overtraining or illness.")
        
        if avg_strain > 15 and avg_recovery < 66:
            issues.append("overtraining_risk")
            recommendations.append("High strain with low recovery. Consider deload week.")
        
        if avg_skin_temp > 0.5:
            issues.append("elevated_skin_temp")
            recommendations.append("Elevated temperature may indicate illness brewing.")
        
        return {
            "avg_recovery": round(avg_recovery, 1),
            "avg_hrv": round(avg_hrv, 1),
            "hrv_vs_baseline": round((avg_hrv / hrv_baseline) * 100, 1) if hrv_baseline else 100,
            "avg_rhr": round(avg_rhr, 1),
            "rhr_vs_baseline": round((avg_rhr / rhr_baseline) * 100, 1) if rhr_baseline else 100,
            "avg_strain": round(avg_strain, 2),
            "avg_skin_temp_deviation": round(avg_skin_temp, 2),
            "issues": issues,
            "recommendations": recommendations
        }
    
    def get_activity_summary(self, user_id: str, days: int = 30) -> Dict:
        """Get activity and workout summary"""
        user_data = self.df[self.df['user_id'] == user_id].tail(days)
        if user_data.empty:
            return {}
        
        workout_days = user_data[user_data['workout_completed'] == 1]
        
        return {
            "total_workouts": len(workout_days),
            "workout_frequency": round(len(workout_days) / days * 7, 1),  # per week
            "avg_workout_duration": round(workout_days['activity_duration_min'].mean(), 1) if not workout_days.empty else 0,
            "avg_workout_strain": round(workout_days['activity_strain'].mean(), 2) if not workout_days.empty else 0,
            "total_calories": int(user_data['calories_burned'].sum()),
            "avg_daily_calories": int(user_data['calories_burned'].mean()),
            "activity_types": workout_days['activity_type'].value_counts().to_dict() if not workout_days.empty else {},
            "preferred_workout_time": workout_days['workout_time_of_day'].mode().iloc[0] if not workout_days.empty and not workout_days['workout_time_of_day'].mode().empty else "N/A"
        }
    
    def identify_health_risks(self, user_id: str) -> List[Dict]:
        """Identify potential health risks based on data patterns"""
        sleep_analysis = self.analyze_sleep_patterns(user_id)
        recovery_analysis = self.analyze_recovery_patterns(user_id)
        activity = self.get_activity_summary(user_id)
        
        risks = []
        
        # Sleep-related risks
        if "insufficient_sleep_duration" in sleep_analysis.get("issues", []):
            risks.append({
                "category": "sleep",
                "risk": "Sleep Deprivation",
                "severity": "high",
                "description": "Consistently getting less than 7 hours affects immunity, cognition, and metabolism.",
                "dietary_impact": "May increase cravings for high-sugar foods and disrupt hunger hormones."
            })
        
        # Recovery risks
        if "overtraining_risk" in recovery_analysis.get("issues", []):
            risks.append({
                "category": "activity",
                "risk": "Overtraining Syndrome",
                "severity": "medium",
                "description": "High training load without adequate recovery can lead to burnout.",
                "dietary_impact": "Increase protein intake and anti-inflammatory foods."
            })
        
        # HRV-related risks
        if "low_hrv" in recovery_analysis.get("issues", []):
            risks.append({
                "category": "stress",
                "risk": "Chronic Stress",
                "severity": "medium",
                "description": "Low HRV indicates autonomic nervous system imbalance.",
                "dietary_impact": "Consider magnesium, omega-3s, and adaptogens."
            })
        
        # Temperature deviation
        if "elevated_skin_temp" in recovery_analysis.get("issues", []):
            risks.append({
                "category": "immunity",
                "risk": "Immune Challenge",
                "severity": "high",
                "description": "Elevated skin temperature may indicate onset of illness.",
                "dietary_impact": "Boost vitamin C, zinc, and hydration. Consider bone broth."
            })
        
        return risks
    
    def get_comprehensive_health_context(self, user_id: str) -> Dict:
        """Get comprehensive health context for AI recommendations, including LSTM forecast"""
        profile = self.get_user_profile(user_id)
        recent_metrics = self.get_recent_health_metrics(user_id)
        sleep_patterns = self.analyze_sleep_patterns(user_id)
        recovery_patterns = self.analyze_recovery_patterns(user_id)
        activity_summary = self.get_activity_summary(user_id)
        health_risks = self.identify_health_risks(user_id)
        lstm_forecast = None
        if get_ml_service and get_lstm_service:
            try:
                ml = get_ml_service()
                lstm_forecast = get_lstm_service().get_forecast(user_id, ml.df)
            except Exception:
                pass
        return {
            "profile": profile,
            "recent_metrics": recent_metrics,
            "sleep_analysis": sleep_patterns,
            "recovery_analysis": recovery_patterns,
            "activity_summary": activity_summary,
            "health_risks": health_risks,
            "lstm_forecast": lstm_forecast,
        }
    
    def get_all_user_ids(self, limit: int = 100) -> List[str]:
        """Get list of all unique user IDs"""
        if self.df is None:
            return []
        return self.df['user_id'].unique()[:limit].tolist()


# Singleton instance
_analyzer_instance = None

def get_health_analyzer() -> HealthAnalyzer:
    """Get or create HealthAnalyzer singleton"""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = HealthAnalyzer()
    return _analyzer_instance
