"""
LSTM Health Forecasting Service
Uses trained PyTorch LSTM model for 7-day health predictions
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime, timedelta
import warnings

from .config import BASE_DIR
from .health_analyzer import get_health_analyzer

warnings.filterwarnings('ignore')


class HealthLSTM(nn.Module):
    """LSTM Model Architecture for Health Forecasting"""
    
    def __init__(self, input_size=8, hidden_size=128, num_layers=2, forecast_horizon=7):
        super(HealthLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # Output layer - Sequential with dropout matching training architecture
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, input_size * forecast_horizon)
        )
    
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        
        # Take the last output
        last_output = lstm_out[:, -1, :]
        
        # Generate forecast
        forecast = self.fc(last_output)
        
        # Reshape to (batch, forecast_horizon, input_size)
        forecast = forecast.view(-1, self.forecast_horizon, x.size(2))
        
        return forecast


class LSTMHealthPredictor:
    """LSTM-based health forecasting service"""
    
    def __init__(self):
        self.model = None
        self.scalers = None
        self.device = None
        self.model_loaded = False
        
        # Model configuration (must match training)
        self.sequence_length = 21  # 21 days of history
        self.forecast_horizon = 7  # 7 days forecast
        self.num_features = 8
        
        # Feature names (MUST match training order)
        self.feature_names = [
            'recovery_score',
            'sleep_hours',
            'sleep_efficiency',
            'hrv',
            'resting_heart_rate',
            'day_strain',
            'deep_sleep_hours',
            'rem_sleep_hours'
        ]
        
        # Model paths
        self.model_path = BASE_DIR / "backend" / "models" / "lstm_health.pt"
        self.scalers_path = BASE_DIR / "backend" / "models" / "lstm_scalers.joblib"
        
        # Health analyzer for data access
        self.health_analyzer = get_health_analyzer()
    
    def load_model(self):
        """Load LSTM model and scalers"""
        if self.model_loaded:
            return
        
        try:
            # Detect device
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"[LSTM] Loading model on {self.device}...")
            
            # Load PyTorch checkpoint
            if not self.model_path.exists():
                raise FileNotFoundError(f"LSTM model not found at {self.model_path}")
            
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Extract model parameters from checkpoint
            hidden_size = checkpoint.get('hidden_size', 128)
            num_layers = checkpoint.get('num_layers', 2)
            seq_len = checkpoint.get('seq_len', 21)
            forecast_days = checkpoint.get('forecast_days', 7)
            
            # Instantiate model with correct architecture
            self.model = HealthLSTM(
                input_size=self.num_features,
                hidden_size=hidden_size,
                num_layers=num_layers,
                forecast_horizon=forecast_days
            )
            
            # Load state dict
            self.model.load_state_dict(checkpoint['model_state'])
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            # Load scalers
            if not self.scalers_path.exists():
                raise FileNotFoundError(f"Scalers not found at {self.scalers_path}")
            
            scalers_dict = joblib.load(self.scalers_path)
            # Extract the input scaler (scaler_x)
            self.scalers = scalers_dict.get('scaler_x') if isinstance(scalers_dict, dict) else scalers_dict
            
            self.model_loaded = True
            print(f"[SUCCESS] LSTM model loaded successfully!")
            print(f"   - Device: {self.device}")
            print(f"   - Features: {self.num_features}")
            print(f"   - Hidden size: {hidden_size}")
            print(f"   - Num layers: {num_layers}")
            print(f"   - Sequence length: {seq_len} days")
            print(f"   - Forecast horizon: {forecast_days} days")
            
        except Exception as e:
            print(f"[ERROR] Error loading LSTM model: {e}")
            import traceback
            traceback.print_exc()
            self.model_loaded = False
            raise
    
    def prepare_sequence(self, user_id: str) -> Optional[np.ndarray]:
        """
        Prepare input sequence for LSTM prediction
        Returns shape: (1, sequence_length, num_features)
        """
        try:
            # Get user data from health analyzer
            df = self.health_analyzer.df
            if df is None:
                return None
            
            user_data = df[df['user_id'] == user_id].tail(self.sequence_length)
            
            # Check if we have enough data
            if len(user_data) < self.sequence_length:
                print(f"[WARNING] Insufficient data for {user_id}: {len(user_data)}/{self.sequence_length} days")
                return None
            
            # Extract features in correct order
            features = user_data[self.feature_names].values
            
            # Handle missing values
            if np.isnan(features).any():
                # Forward fill missing values
                features = pd.DataFrame(features, columns=self.feature_names).fillna(method='ffill').fillna(method='bfill').values
            
            # Scale features
            if self.scalers is not None:
                features = self.scalers.transform(features)
            
            # Reshape for LSTM: (batch_size=1, sequence_length, num_features)
            sequence = features.reshape(1, self.sequence_length, self.num_features)
            
            return sequence
            
        except Exception as e:
            print(f"Error preparing sequence: {e}")
            return None
    
    def predict_forecast(self, user_id: str) -> Dict:
        """Generate 7-day health forecast"""
        # Lazy load model
        if not self.model_loaded:
            self.load_model()
        
        if not self.model_loaded:
            return self._fallback_forecast(user_id)
        
        try:
            # Prepare input sequence
            sequence = self.prepare_sequence(user_id)
            if sequence is None:
                return self._fallback_forecast(user_id)
            
            # Convert to tensor
            input_tensor = torch.FloatTensor(sequence).to(self.device)
            
            # Generate predictions
            with torch.no_grad():
                forecast = self.model(input_tensor)
            
            # Convert back to numpy
            forecast_np = forecast.cpu().numpy()
            
            # Inverse transform to original scale
            if self.scalers is not None:
                # Reshape for inverse transform
                forecast_reshaped = forecast_np.reshape(-1, self.num_features)
                forecast_unscaled = self.scalers.inverse_transform(forecast_reshaped)
            else:
                forecast_unscaled = forecast_np.reshape(-1, self.num_features)
            
            # Format response
            predictions = self._format_predictions(forecast_unscaled, user_id)
            
            return {
                "status": "success",
                "user_id": user_id,
                "model": "LSTM",
                "forecast_horizon": self.forecast_horizon,
                "predictions": predictions,
                "metadata": {
                    "sequence_length": self.sequence_length,
                    "features_used": self.feature_names,
                    "device": str(self.device),
                    "model_loaded": self.model_loaded
                }
            }
            
        except Exception as e:
            print(f"LSTM prediction error: {e}")
            return self._fallback_forecast(user_id)
    
    def _format_predictions(self, forecast: np.ndarray, user_id: str) -> List[Dict]:
        """Format forecast array into structured predictions"""
        predictions = []
        
        # Get current date from user's latest data
        df = self.health_analyzer.df
        user_data = df[df['user_id'] == user_id].tail(1)
        last_date = pd.to_datetime(user_data.iloc[0]['date']) if not user_data.empty else datetime.now()
        
        for day_idx in range(self.forecast_horizon):
            forecast_date = last_date + timedelta(days=day_idx + 1)
            
            day_prediction = {
                "day": day_idx + 1,
                "date": forecast_date.strftime('%Y-%m-%d'),
                "recovery_score": float(np.clip(forecast[day_idx, 0], 0, 100).round(1)),
                "sleep_hours": float(np.clip(forecast[day_idx, 1], 0, 12).round(2)),
                "sleep_efficiency": float(np.clip(forecast[day_idx, 2], 0, 100).round(1)),
                "hrv": float(np.clip(forecast[day_idx, 3], 0, 200).round(1)),
                "resting_heart_rate": float(np.clip(forecast[day_idx, 4], 30, 120).round(1)),
                "day_strain": float(np.clip(forecast[day_idx, 5], 0, 21).round(2)),
                "deep_sleep_hours": float(np.clip(forecast[day_idx, 6], 0, 5).round(2)),
                "rem_sleep_hours": float(np.clip(forecast[day_idx, 7], 0, 5).round(2))
            }
            
            predictions.append(day_prediction)
        
        return predictions
    
    def _fallback_forecast(self, user_id: str) -> Dict:
        """Fallback forecast using simple statistics"""
        try:
            df = self.health_analyzer.df
            if df is None:
                return self._empty_forecast(user_id)
            
            user_data = df[df['user_id'] == user_id].tail(7)
            if user_data.empty:
                return self._empty_forecast(user_id)
            
            # Use recent averages for forecast
            predictions = []
            last_date = pd.to_datetime(user_data.iloc[-1]['date'])
            
            for day_idx in range(self.forecast_horizon):
                forecast_date = last_date + timedelta(days=day_idx + 1)
                
                predictions.append({
                    "day": day_idx + 1,
                    "date": forecast_date.strftime('%Y-%m-%d'),
                    "recovery_score": float(user_data['recovery_score'].mean().round(1)),
                    "sleep_hours": float(user_data['sleep_hours'].mean().round(2)),
                    "sleep_efficiency": float(user_data['sleep_efficiency'].mean().round(1)),
                    "hrv": float(user_data['hrv'].mean().round(1)),
                    "resting_heart_rate": float(user_data['resting_heart_rate'].mean().round(1)),
                    "day_strain": float(user_data['day_strain'].mean().round(2)),
                    "deep_sleep_hours": float(user_data['deep_sleep_hours'].mean().round(2)),
                    "rem_sleep_hours": float(user_data['rem_sleep_hours'].mean().round(2))
                })
            
            return {
                "status": "fallback",
                "user_id": user_id,
                "model": "Statistical Average",
                "forecast_horizon": self.forecast_horizon,
                "predictions": predictions,
                "message": "Using statistical averages (LSTM model unavailable or insufficient data)"
            }
            
        except Exception as e:
            print(f"Fallback forecast error: {e}")
            return self._empty_forecast(user_id)
    
    def _empty_forecast(self, user_id: str) -> Dict:
        """Empty forecast when no data available"""
        return {
            "status": "error",
            "user_id": user_id,
            "model": "None",
            "forecast_horizon": 0,
            "predictions": [],
            "message": "Insufficient data for forecast generation"
        }
    
    def get_model_status(self) -> Dict:
        """Get LSTM model status and metadata"""
        return {
            "model_loaded": self.model_loaded,
            "device": str(self.device) if self.device else "not_loaded",
            "model_path": str(self.model_path),
            "model_exists": self.model_path.exists(),
            "scalers_exists": self.scalers_path.exists(),
            "sequence_length": self.sequence_length,
            "forecast_horizon": self.forecast_horizon,
            "num_features": self.num_features,
            "feature_names": self.feature_names
        }


# Singleton instance
_lstm_service = None

def get_lstm_service() -> LSTMHealthPredictor:
    """Get or create LSTMHealthPredictor singleton"""
    global _lstm_service
    if _lstm_service is None:
        _lstm_service = LSTMHealthPredictor()
    return _lstm_service
