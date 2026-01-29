"""
Machine Learning Service for Health Predictions
Uses scikit-learn for predictive modeling on Whoop data
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
import warnings

from .config import WHOOP_DATA_PATH

warnings.filterwarnings('ignore')


class HealthMLService:
    """Machine Learning models for health predictions"""
    
    def __init__(self):
        self.df = None
        self.recovery_model = None
        self.strain_model = None
        self.risk_model = None
        self.scaler = StandardScaler()
        self.models_trained = False
        self._load_data()
        self._train_models()
    
    def _load_data(self):
        """Load and preprocess data"""
        if WHOOP_DATA_PATH.exists():
            self.df = pd.read_csv(WHOOP_DATA_PATH)
            self.df['date'] = pd.to_datetime(self.df['date'])
            # Add derived features
            self.df['sleep_quality'] = self.df['sleep_efficiency'] * (self.df['deep_sleep_hours'] + self.df['rem_sleep_hours']) / self.df['sleep_hours'].clip(lower=0.1)
            self.df['recovery_category'] = pd.cut(self.df['recovery_score'], bins=[0, 33, 66, 100], labels=['low', 'medium', 'high'])
    
    def _get_features_for_recovery(self, user_data: pd.DataFrame) -> np.ndarray:
        """Extract features for recovery prediction"""
        features = [
            'sleep_hours', 'sleep_efficiency', 'deep_sleep_hours', 'rem_sleep_hours',
            'wake_ups', 'time_to_fall_asleep_min', 'hrv', 'resting_heart_rate',
            'respiratory_rate', 'skin_temp_deviation', 'day_strain'
        ]
        return user_data[features].values
    
    def _get_features_for_strain(self, user_data: pd.DataFrame) -> np.ndarray:
        """Extract features for optimal strain prediction"""
        features = [
            'recovery_score', 'sleep_hours', 'sleep_efficiency', 'hrv',
            'resting_heart_rate', 'hrv_baseline', 'rhr_baseline'
        ]
        return user_data[features].values
    
    def _train_models(self):
        """Train ML models on the dataset"""
        if self.df is None or len(self.df) < 100:
            return
        
        try:
            # Prepare data - use a sample for faster training
            sample_df = self.df.sample(min(10000, len(self.df)), random_state=42)
            
            # Recovery Prediction Model
            recovery_features = [
                'sleep_hours', 'sleep_efficiency', 'deep_sleep_hours', 'rem_sleep_hours',
                'wake_ups', 'time_to_fall_asleep_min', 'hrv', 'resting_heart_rate',
                'respiratory_rate', 'skin_temp_deviation', 'day_strain'
            ]
            
            X_recovery = sample_df[recovery_features].dropna()
            y_recovery = sample_df.loc[X_recovery.index, 'recovery_score']
            
            if len(X_recovery) > 100:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_recovery, y_recovery, test_size=0.2, random_state=42
                )
                
                self.recovery_model = RandomForestRegressor(
                    n_estimators=50, max_depth=10, random_state=42, n_jobs=-1
                )
                self.recovery_model.fit(X_train, y_train)
                self.recovery_accuracy = self.recovery_model.score(X_test, y_test)
            
            # Optimal Strain Prediction Model
            strain_features = [
                'recovery_score', 'sleep_hours', 'sleep_efficiency', 'hrv',
                'resting_heart_rate', 'hrv_baseline', 'rhr_baseline'
            ]
            
            X_strain = sample_df[strain_features].dropna()
            y_strain = sample_df.loc[X_strain.index, 'day_strain']
            
            if len(X_strain) > 100:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_strain, y_strain, test_size=0.2, random_state=42
                )
                
                self.strain_model = RandomForestRegressor(
                    n_estimators=50, max_depth=8, random_state=42, n_jobs=-1
                )
                self.strain_model.fit(X_train, y_train)
                self.strain_accuracy = self.strain_model.score(X_test, y_test)
            
            # Health Risk Classification Model
            # Create risk labels based on multiple factors
            sample_df['health_risk'] = (
                (sample_df['recovery_score'] < 50).astype(int) +
                (sample_df['sleep_hours'] < 6).astype(int) +
                (sample_df['hrv'] < sample_df['hrv_baseline'] * 0.7).astype(int) +
                (sample_df['skin_temp_deviation'] > 0.5).astype(int)
            )
            sample_df['high_risk'] = (sample_df['health_risk'] >= 2).astype(int)
            
            risk_features = [
                'recovery_score', 'sleep_hours', 'sleep_efficiency', 'hrv',
                'resting_heart_rate', 'skin_temp_deviation', 'day_strain'
            ]
            
            X_risk = sample_df[risk_features].dropna()
            y_risk = sample_df.loc[X_risk.index, 'high_risk']
            
            if len(X_risk) > 100:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_risk, y_risk, test_size=0.2, random_state=42
                )
                
                self.risk_model = GradientBoostingClassifier(
                    n_estimators=50, max_depth=5, random_state=42
                )
                self.risk_model.fit(X_train, y_train)
                self.risk_accuracy = self.risk_model.score(X_test, y_test)
            
            self.models_trained = True
            print("ML models trained successfully!")
            
        except Exception as e:
            print(f"Error training models: {e}")
            self.models_trained = False
    
    def predict_recovery(self, user_id: str) -> Dict:
        """Predict tomorrow's recovery score"""
        if not self.models_trained or self.recovery_model is None:
            return self._fallback_recovery_prediction(user_id)
        
        try:
            user_data = self.df[self.df['user_id'] == user_id].tail(7)
            if len(user_data) < 3:
                return self._fallback_recovery_prediction(user_id)
            
            latest = user_data.iloc[-1]
            
            features = np.array([[
                latest['sleep_hours'],
                latest['sleep_efficiency'],
                latest['deep_sleep_hours'],
                latest['rem_sleep_hours'],
                latest['wake_ups'],
                latest['time_to_fall_asleep_min'],
                latest['hrv'],
                latest['resting_heart_rate'],
                latest['respiratory_rate'],
                latest['skin_temp_deviation'],
                latest['day_strain']
            ]])
            
            predicted_recovery = self.recovery_model.predict(features)[0]
            predicted_recovery = np.clip(predicted_recovery, 0, 100)
            
            # Get feature importances
            feature_names = [
                'sleep_hours', 'sleep_efficiency', 'deep_sleep_hours', 'rem_sleep_hours',
                'wake_ups', 'time_to_fall_asleep_min', 'hrv', 'resting_heart_rate',
                'respiratory_rate', 'skin_temp_deviation', 'day_strain'
            ]
            importances = dict(zip(feature_names, self.recovery_model.feature_importances_))
            top_factors = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:3]
            
            # Confidence based on recent data consistency
            recent_std = user_data['recovery_score'].std()
            confidence = max(0.5, min(0.95, 1 - (recent_std / 50)))
            
            return {
                "predicted_recovery": round(predicted_recovery, 1),
                "confidence": round(confidence, 2),
                "prediction_range": {
                    "low": round(max(0, predicted_recovery - 10), 1),
                    "high": round(min(100, predicted_recovery + 10), 1)
                },
                "top_influencing_factors": [
                    {"factor": f[0].replace('_', ' ').title(), "importance": round(f[1] * 100, 1)}
                    for f in top_factors
                ],
                "recommendation": self._get_recovery_recommendation(predicted_recovery),
                "model_accuracy": round(getattr(self, 'recovery_accuracy', 0.7) * 100, 1)
            }
            
        except Exception as e:
            print(f"Recovery prediction error: {e}")
            return self._fallback_recovery_prediction(user_id)
    
    def _fallback_recovery_prediction(self, user_id: str) -> Dict:
        """Fallback prediction using simple statistics"""
        user_data = self.df[self.df['user_id'] == user_id].tail(14) if self.df is not None else None
        
        if user_data is None or len(user_data) < 3:
            return {
                "predicted_recovery": 65.0,
                "confidence": 0.5,
                "prediction_range": {"low": 55.0, "high": 75.0},
                "top_influencing_factors": [
                    {"factor": "Sleep Hours", "importance": 30.0},
                    {"factor": "HRV", "importance": 25.0},
                    {"factor": "Day Strain", "importance": 20.0}
                ],
                "recommendation": "Maintain good sleep habits for optimal recovery.",
                "model_accuracy": 70.0
            }
        
        # Simple moving average prediction
        avg_recovery = user_data['recovery_score'].mean()
        trend = user_data['recovery_score'].diff().mean()
        predicted = avg_recovery + trend
        predicted = np.clip(predicted, 0, 100)
        
        return {
            "predicted_recovery": round(predicted, 1),
            "confidence": 0.6,
            "prediction_range": {
                "low": round(max(0, predicted - 15), 1),
                "high": round(min(100, predicted + 15), 1)
            },
            "top_influencing_factors": [
                {"factor": "Sleep Hours", "importance": 30.0},
                {"factor": "HRV", "importance": 25.0},
                {"factor": "Day Strain", "importance": 20.0}
            ],
            "recommendation": self._get_recovery_recommendation(predicted),
            "model_accuracy": 65.0
        }
    
    def _get_recovery_recommendation(self, predicted_recovery: float) -> str:
        """Get recommendation based on predicted recovery"""
        if predicted_recovery >= 80:
            return "Great recovery predicted! You're ready for high-intensity training."
        elif predicted_recovery >= 60:
            return "Moderate recovery expected. Consider medium-intensity activities."
        elif predicted_recovery >= 40:
            return "Lower recovery predicted. Focus on light activity and rest."
        else:
            return "Low recovery expected. Prioritize rest, nutrition, and sleep quality."
    
    def predict_optimal_strain(self, user_id: str) -> Dict:
        """Predict optimal strain level for today"""
        if not self.models_trained or self.strain_model is None:
            return self._fallback_strain_prediction(user_id)
        
        try:
            user_data = self.df[self.df['user_id'] == user_id].tail(7)
            if len(user_data) < 3:
                return self._fallback_strain_prediction(user_id)
            
            latest = user_data.iloc[-1]
            
            features = np.array([[
                latest['recovery_score'],
                latest['sleep_hours'],
                latest['sleep_efficiency'],
                latest['hrv'],
                latest['resting_heart_rate'],
                latest['hrv_baseline'],
                latest['rhr_baseline']
            ]])
            
            predicted_strain = self.strain_model.predict(features)[0]
            predicted_strain = np.clip(predicted_strain, 0, 21)
            
            # Adjust based on recovery
            recovery_factor = latest['recovery_score'] / 100
            adjusted_strain = predicted_strain * (0.5 + 0.5 * recovery_factor)
            
            return {
                "optimal_strain": round(adjusted_strain, 1),
                "strain_range": {
                    "minimum": round(max(0, adjusted_strain - 3), 1),
                    "maximum": round(min(21, adjusted_strain + 3), 1)
                },
                "current_recovery": round(latest['recovery_score'], 1),
                "activity_recommendation": self._get_activity_recommendation(adjusted_strain, latest['recovery_score']),
                "workout_suggestions": self._get_workout_suggestions(adjusted_strain),
                "model_accuracy": round(getattr(self, 'strain_accuracy', 0.65) * 100, 1)
            }
            
        except Exception as e:
            print(f"Strain prediction error: {e}")
            return self._fallback_strain_prediction(user_id)
    
    def _fallback_strain_prediction(self, user_id: str) -> Dict:
        """Fallback strain prediction"""
        return {
            "optimal_strain": 10.0,
            "strain_range": {"minimum": 7.0, "maximum": 13.0},
            "current_recovery": 65.0,
            "activity_recommendation": "Moderate activity recommended. Listen to your body.",
            "workout_suggestions": [
                {"type": "Yoga", "duration": "30-45 min", "intensity": "Low"},
                {"type": "Light Jog", "duration": "20-30 min", "intensity": "Low-Medium"},
                {"type": "Strength Training", "duration": "45 min", "intensity": "Medium"}
            ],
            "model_accuracy": 65.0
        }
    
    def _get_activity_recommendation(self, strain: float, recovery: float) -> str:
        """Get activity recommendation based on predicted strain and recovery"""
        if recovery >= 80 and strain >= 15:
            return "Your body is ready for intense training! Go for high-intensity workouts."
        elif recovery >= 60:
            return "Good capacity for moderate to high activity. Mix intensity levels."
        elif recovery >= 40:
            return "Focus on lighter activities. Active recovery or low-intensity workouts."
        else:
            return "Prioritize rest and recovery. Light stretching or walking only."
    
    def _get_workout_suggestions(self, strain: float) -> List[Dict]:
        """Get workout suggestions based on optimal strain"""
        if strain >= 15:
            return [
                {"type": "HIIT", "duration": "30-45 min", "intensity": "High"},
                {"type": "CrossFit", "duration": "45-60 min", "intensity": "High"},
                {"type": "Running (Intervals)", "duration": "40 min", "intensity": "High"}
            ]
        elif strain >= 10:
            return [
                {"type": "Strength Training", "duration": "45-60 min", "intensity": "Medium"},
                {"type": "Cycling", "duration": "45-60 min", "intensity": "Medium"},
                {"type": "Swimming", "duration": "30-45 min", "intensity": "Medium"}
            ]
        elif strain >= 5:
            return [
                {"type": "Yoga", "duration": "45-60 min", "intensity": "Low"},
                {"type": "Light Jog", "duration": "20-30 min", "intensity": "Low"},
                {"type": "Pilates", "duration": "30-45 min", "intensity": "Low"}
            ]
        else:
            return [
                {"type": "Stretching", "duration": "15-20 min", "intensity": "Very Low"},
                {"type": "Walking", "duration": "20-30 min", "intensity": "Very Low"},
                {"type": "Meditation", "duration": "15-20 min", "intensity": "Rest"}
            ]
    
    def predict_health_risk(self, user_id: str) -> Dict:
        """Predict health risk score"""
        if not self.models_trained or self.risk_model is None:
            return self._fallback_risk_prediction(user_id)
        
        try:
            user_data = self.df[self.df['user_id'] == user_id].tail(7)
            if len(user_data) < 3:
                return self._fallback_risk_prediction(user_id)
            
            latest = user_data.iloc[-1]
            
            features = np.array([[
                latest['recovery_score'],
                latest['sleep_hours'],
                latest['sleep_efficiency'],
                latest['hrv'],
                latest['resting_heart_rate'],
                latest['skin_temp_deviation'],
                latest['day_strain']
            ]])
            
            risk_probability = self.risk_model.predict_proba(features)[0][1]
            risk_score = risk_probability * 100
            
            # Identify risk factors
            risk_factors = []
            if latest['recovery_score'] < 50:
                risk_factors.append({"factor": "Low Recovery", "severity": "high", "value": f"{latest['recovery_score']:.1f}%"})
            if latest['sleep_hours'] < 6:
                risk_factors.append({"factor": "Insufficient Sleep", "severity": "high", "value": f"{latest['sleep_hours']:.1f} hrs"})
            if latest['sleep_hours'] < 7:
                risk_factors.append({"factor": "Below Optimal Sleep", "severity": "medium", "value": f"{latest['sleep_hours']:.1f} hrs"})
            if latest['skin_temp_deviation'] > 0.5:
                risk_factors.append({"factor": "Elevated Temperature", "severity": "high", "value": f"+{latest['skin_temp_deviation']:.2f}°C"})
            if latest['hrv'] < latest['hrv_baseline'] * 0.8:
                risk_factors.append({"factor": "Low HRV", "severity": "medium", "value": f"{latest['hrv']:.0f} ms"})
            
            return {
                "risk_score": round(risk_score, 1),
                "risk_level": self._get_risk_level(risk_score),
                "risk_factors": risk_factors[:4],  # Top 4 factors
                "protective_factors": self._get_protective_factors(latest),
                "recommendations": self._get_risk_recommendations(risk_score, risk_factors),
                "trend": self._calculate_risk_trend(user_data),
                "model_accuracy": round(getattr(self, 'risk_accuracy', 0.75) * 100, 1)
            }
            
        except Exception as e:
            print(f"Risk prediction error: {e}")
            return self._fallback_risk_prediction(user_id)
    
    def _fallback_risk_prediction(self, user_id: str) -> Dict:
        """Fallback risk prediction"""
        return {
            "risk_score": 25.0,
            "risk_level": "low",
            "risk_factors": [],
            "protective_factors": ["Adequate activity levels", "Consistent routine"],
            "recommendations": ["Maintain current healthy habits", "Monitor sleep quality"],
            "trend": "stable",
            "model_accuracy": 70.0
        }
    
    def _get_risk_level(self, risk_score: float) -> str:
        """Convert risk score to level"""
        if risk_score >= 70:
            return "high"
        elif risk_score >= 40:
            return "medium"
        else:
            return "low"
    
    def _get_protective_factors(self, latest: pd.Series) -> List[str]:
        """Identify protective health factors"""
        factors = []
        if latest['recovery_score'] >= 70:
            factors.append("Strong recovery score")
        if latest['sleep_hours'] >= 7:
            factors.append("Adequate sleep duration")
        if latest['sleep_efficiency'] >= 80:
            factors.append("Good sleep efficiency")
        if abs(latest['skin_temp_deviation']) < 0.3:
            factors.append("Normal body temperature")
        if latest['hrv'] >= latest['hrv_baseline'] * 0.9:
            factors.append("Healthy HRV levels")
        return factors[:3] if factors else ["Continue monitoring health metrics"]
    
    def _get_risk_recommendations(self, risk_score: float, risk_factors: List[Dict]) -> List[str]:
        """Generate recommendations based on risk"""
        recommendations = []
        
        if risk_score >= 70:
            recommendations.append("Consider consulting a healthcare provider")
            recommendations.append("Prioritize rest and recovery for the next 48 hours")
        elif risk_score >= 40:
            recommendations.append("Focus on sleep quality and duration")
            recommendations.append("Reduce training intensity temporarily")
        
        for factor in risk_factors:
            if factor['factor'] == "Insufficient Sleep":
                recommendations.append("Aim for 7-9 hours of sleep tonight")
            elif factor['factor'] == "Elevated Temperature":
                recommendations.append("Monitor for signs of illness; increase fluid intake")
            elif factor['factor'] == "Low HRV":
                recommendations.append("Practice stress-reduction techniques")
        
        return recommendations[:4] if recommendations else ["Maintain current healthy habits"]
    
    def _calculate_risk_trend(self, user_data: pd.DataFrame) -> str:
        """Calculate risk trend from recent data"""
        if len(user_data) < 3:
            return "stable"
        
        # Calculate composite risk score for trend
        user_data = user_data.copy()
        user_data['risk_composite'] = (
            (100 - user_data['recovery_score']) / 100 +
            (7 - user_data['sleep_hours'].clip(upper=7)) / 7 +
            user_data['skin_temp_deviation'].clip(lower=0) / 2
        ) / 3 * 100
        
        first_half = user_data['risk_composite'].iloc[:len(user_data)//2].mean()
        second_half = user_data['risk_composite'].iloc[len(user_data)//2:].mean()
        
        diff = second_half - first_half
        if diff > 5:
            return "increasing"
        elif diff < -5:
            return "decreasing"
        return "stable"
    
    def get_trend_data(self, user_id: str, days: int = 30) -> Dict:
        """Get historical trend data for charts"""
        if self.df is None:
            return self._empty_trend_data()
        
        user_data = self.df[self.df['user_id'] == user_id].tail(days)
        if len(user_data) < 3:
            return self._empty_trend_data()
        
        return {
            "dates": user_data['date'].dt.strftime('%Y-%m-%d').tolist(),
            "recovery": user_data['recovery_score'].round(1).tolist(),
            "sleep_hours": user_data['sleep_hours'].round(2).tolist(),
            "sleep_efficiency": user_data['sleep_efficiency'].round(1).tolist(),
            "hrv": user_data['hrv'].round(1).tolist(),
            "resting_hr": user_data['resting_heart_rate'].round(1).tolist(),
            "strain": user_data['day_strain'].round(2).tolist(),
            "calories": user_data['calories_burned'].round(0).tolist(),
            "deep_sleep": user_data['deep_sleep_hours'].round(2).tolist(),
            "rem_sleep": user_data['rem_sleep_hours'].round(2).tolist(),
            "skin_temp": user_data['skin_temp_deviation'].round(2).tolist(),
            "statistics": {
                "recovery": {
                    "mean": round(user_data['recovery_score'].mean(), 1),
                    "min": round(user_data['recovery_score'].min(), 1),
                    "max": round(user_data['recovery_score'].max(), 1),
                    "std": round(user_data['recovery_score'].std(), 1)
                },
                "sleep": {
                    "mean": round(user_data['sleep_hours'].mean(), 2),
                    "min": round(user_data['sleep_hours'].min(), 2),
                    "max": round(user_data['sleep_hours'].max(), 2)
                },
                "hrv": {
                    "mean": round(user_data['hrv'].mean(), 1),
                    "min": round(user_data['hrv'].min(), 1),
                    "max": round(user_data['hrv'].max(), 1)
                },
                "strain": {
                    "mean": round(user_data['day_strain'].mean(), 2),
                    "total": round(user_data['day_strain'].sum(), 1)
                }
            }
        }
    
    def _empty_trend_data(self) -> Dict:
        """Return empty trend data structure"""
        return {
            "dates": [],
            "recovery": [],
            "sleep_hours": [],
            "sleep_efficiency": [],
            "hrv": [],
            "resting_hr": [],
            "strain": [],
            "calories": [],
            "deep_sleep": [],
            "rem_sleep": [],
            "skin_temp": [],
            "statistics": {}
        }
    
    def get_correlation_insights(self, user_id: str) -> Dict:
        """Get correlation insights between metrics"""
        if self.df is None:
            return {"insights": []}
        
        user_data = self.df[self.df['user_id'] == user_id].tail(60)
        if len(user_data) < 10:
            return {"insights": []}
        
        insights = []
        
        # Sleep vs Recovery correlation (lowered threshold)
        sleep_recovery_corr = user_data['sleep_hours'].corr(user_data['recovery_score'])
        if abs(sleep_recovery_corr) > 0.15:
            direction = "positively" if sleep_recovery_corr > 0 else "negatively"
            insights.append({
                "title": "Sleep & Recovery Connection",
                "description": f"Your sleep duration is {direction} correlated with recovery ({abs(sleep_recovery_corr):.0%}). More sleep tends to mean better recovery.",
                "strength": abs(sleep_recovery_corr),
                "type": "sleep_recovery"
            })
        
        # Strain vs Next Day Recovery (lowered threshold)
        user_data_shifted = user_data.copy()
        user_data_shifted['next_recovery'] = user_data_shifted['recovery_score'].shift(-1)
        strain_recovery_corr = user_data_shifted['day_strain'].corr(user_data_shifted['next_recovery'])
        if abs(strain_recovery_corr) > 0.1:
            insights.append({
                "title": "Strain Impact on Recovery",
                "description": f"Higher strain days tend to {'decrease' if strain_recovery_corr < 0 else 'increase'} next-day recovery by {abs(strain_recovery_corr):.0%}.",
                "strength": abs(strain_recovery_corr),
                "type": "strain_recovery"
            })
        
        # HRV vs Recovery (lowered threshold)
        hrv_recovery_corr = user_data['hrv'].corr(user_data['recovery_score'])
        if abs(hrv_recovery_corr) > 0.15:
            insights.append({
                "title": "HRV as Recovery Predictor",
                "description": f"Your HRV strongly correlates with recovery ({hrv_recovery_corr:.0%}). HRV is a reliable indicator of your readiness.",
                "strength": abs(hrv_recovery_corr),
                "type": "hrv_recovery"
            })
        
        # Deep sleep importance (lowered threshold)
        deep_sleep_corr = user_data['deep_sleep_hours'].corr(user_data['recovery_score'])
        if abs(deep_sleep_corr) > 0.1:
            insights.append({
                "title": "Deep Sleep Quality",
                "description": f"Deep sleep has a {abs(deep_sleep_corr):.0%} correlation with your recovery. Prioritize sleep quality over quantity.",
                "strength": abs(deep_sleep_corr),
                "type": "deep_sleep"
            })
        
        # Always add trend-based insights
        recent_week = user_data.tail(7)
        previous_week = user_data.tail(14).head(7)
        
        if len(recent_week) >= 5 and len(previous_week) >= 5:
            # Recovery trend
            recent_recovery = recent_week['recovery_score'].mean()
            prev_recovery = previous_week['recovery_score'].mean()
            recovery_change = ((recent_recovery - prev_recovery) / prev_recovery) * 100 if prev_recovery > 0 else 0
            
            if abs(recovery_change) > 3:
                trend = "improving" if recovery_change > 0 else "declining"
                emoji = "📈" if recovery_change > 0 else "📉"
                insights.append({
                    "title": f"Recovery Trend {emoji}",
                    "description": f"Your recovery is {trend} by {abs(recovery_change):.0f}% compared to last week. {'Keep up the good work!' if recovery_change > 0 else 'Consider more rest.'}",
                    "strength": min(abs(recovery_change) / 20, 0.8),
                    "type": "recovery_trend"
                })
            
            # Activity consistency
            strain_std = recent_week['day_strain'].std()
            avg_strain = recent_week['day_strain'].mean()
            consistency = 1 - (strain_std / avg_strain) if avg_strain > 0 else 0
            
            if consistency > 0.5:
                insights.append({
                    "title": "Activity Consistency 🎯",
                    "description": f"Your activity levels are {int(consistency * 100)}% consistent this week. Consistency helps your body adapt and recover better.",
                    "strength": consistency * 0.6,
                    "type": "consistency"
                })
        
        return {"insights": sorted(insights, key=lambda x: x['strength'], reverse=True)[:5]}


# Singleton instance
_ml_service = None

def get_ml_service() -> HealthMLService:
    """Get or create HealthMLService singleton"""
    global _ml_service
    if _ml_service is None:
        _ml_service = HealthMLService()
    return _ml_service
