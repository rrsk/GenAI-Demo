"""
WellnessAI - Main FastAPI Application
A GenAI-powered health and meal planning assistant
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict
from pathlib import Path
import uvicorn

from .models import ChatMessage, ChatResponse, UserProfile, PreferenceUpdate
from .health_analyzer import get_health_analyzer
from .weather_service import get_weather_service
from .ai_service import get_ai_service
from .ml_service import get_ml_service
from .lstm_service import get_lstm_service
from .web_search_service import get_web_search_service
from .user_preferences_service import get_preferences_service
from .config import HOST, PORT
import re

# Initialize FastAPI app
app = FastAPI(
    title="WellnessAI",
    description="AI-powered health and meal planning assistant using Whoop biometric data",
    version="1.0.0"
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store conversation history per user
conversation_store: Dict[str, List[Dict]] = {}


class ChatRequest(BaseModel):
    """Chat request from frontend"""
    message: str
    user_id: str = "USER_00001"
    include_weather: bool = True
    location: Optional[str] = "New York"


class UserListResponse(BaseModel):
    """Response with list of users"""
    users: List[str]
    total: int


class HealthContextResponse(BaseModel):
    """Response with health context"""
    profile: Optional[Dict]
    recent_metrics: Optional[Dict]
    sleep_analysis: Optional[Dict]
    recovery_analysis: Optional[Dict]
    activity_summary: Optional[Dict]
    health_risks: Optional[List[Dict]]


@app.get("/styles.css")
async def serve_css():
    """Serve the CSS file with no-cache headers"""
    css_path = Path(__file__).parent.parent / "frontend" / "styles.css"
    if css_path.exists():
        return FileResponse(
            css_path, 
            media_type="text/css",
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
        )
    raise HTTPException(status_code=404, detail="CSS not found")


@app.get("/app.js")
async def serve_js():
    """Serve the JS file with no-cache headers"""
    js_path = Path(__file__).parent.parent / "frontend" / "app.js"
    if js_path.exists():
        return FileResponse(
            js_path, 
            media_type="application/javascript",
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
        )
    raise HTTPException(status_code=404, detail="JS not found")


@app.get("/")
async def root():
    """Serve the frontend"""
    frontend_path = Path(__file__).parent.parent / "frontend" / "index.html"
    if frontend_path.exists():
        return FileResponse(frontend_path, media_type="text/html")
    return {"message": "WellnessAI API is running. Frontend not found."}


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "WellnessAI"}


@app.get("/api/ai-status")
async def ai_status():
    """Get status of AI service components"""
    ai_service = get_ai_service()
    web_search = get_web_search_service()
    return {
        "service": "WellnessAI AI",
        "status": ai_service.get_service_status(),
        "mode": "local_llm" if ai_service.use_local_llm else "api_only",
        "web_search": "available"
    }


# ============================================
# Web Search Endpoints
# ============================================

class WebSearchRequest(BaseModel):
    """Web search request"""
    query: str
    search_type: Optional[str] = "general"  # general, health, nutrition, symptom


@app.post("/api/web-search")
async def web_search(request: WebSearchRequest):
    """
    Search the web for information
    
    Demonstrates Retrieval-Augmented Generation (RAG) capability
    """
    search_service = get_web_search_service()
    
    if request.search_type == "health":
        results = await search_service.search_health_topic(request.query)
    elif request.search_type == "nutrition":
        results = await search_service.search_nutrition_info(request.query)
    elif request.search_type == "symptom":
        results = await search_service.search_symptom(request.query)
    else:
        results = await search_service.search(request.query)
    
    return results


@app.get("/api/web-search")
async def web_search_get(query: str, search_type: str = "general"):
    """
    Search the web (GET endpoint for easy testing)
    """
    search_service = get_web_search_service()
    
    if search_type == "health":
        results = await search_service.search_health_topic(query)
    elif search_type == "nutrition":
        results = await search_service.search_nutrition_info(query)
    elif search_type == "symptom":
        results = await search_service.search_symptom(query)
    else:
        results = await search_service.search(query)
    
    return results


@app.get("/api/system-status")
async def system_status():
    """
    Get comprehensive system status (State of Union)
    Shows all components and their status
    """
    ai_service = get_ai_service()
    health_analyzer = get_health_analyzer()
    ml_service = get_ml_service()
    weather_service = get_weather_service()
    lstm_service = get_lstm_service()
    
    return {
        "system": "WellnessAI",
        "version": "1.0.0",
        "components": {
            "local_llm": {
                "status": "active" if ai_service.use_local_llm else "disabled",
                "model": ai_service.get_service_status().get("local_llm_model", "none"),
                "device": ai_service.get_service_status().get("local_llm_device", "cpu")
            },
            "recommendation_engine": {
                "status": "active",
                "capabilities": ["meal_planning", "activity_suggestions", "health_alerts"]
            },
            "ml_predictions": {
                "status": "active",
                "models": ["recovery_predictor", "strain_predictor", "risk_classifier"]
            },
            "lstm_forecasting": {
                "status": "active" if lstm_service.get_model_status()["model_exists"] else "unavailable",
                "model": "PyTorch LSTM",
                "forecast_horizon": "7 days",
                "sequence_length": "21 days"
            },
            "health_analyzer": {
                "status": "active",
                "data_source": "Whoop biometric data",
                "user_count": len(health_analyzer.get_all_user_ids(100))
            },
            "weather_service": {
                "status": "active",
                "provider": "OpenWeatherMap"
            },
            "web_search": {
                "status": "active",
                "provider": "DuckDuckGo",
                "capabilities": ["general", "health", "nutrition", "symptom"]
            }
        },
        "genai_features": [
            "Local LLM (TinyLlama/BioGPT/Phi-2)",
            "Prompt Engineering",
            "Intent Classification",
            "Context Augmentation (Weather + Health)",
            "Web Search (RAG)",
            "ML Predictions (Scikit-learn)",
            "LSTM Health Forecasting (PyTorch)"
        ]
    }


@app.get("/api/users", response_model=UserListResponse)
async def get_users(limit: int = 50):
    """Get list of available user IDs"""
    analyzer = get_health_analyzer()
    users = analyzer.get_all_user_ids(limit)
    return UserListResponse(users=users, total=len(users))


@app.get("/api/users/{user_id}/profile")
async def get_user_profile(user_id: str):
    """Get user profile"""
    analyzer = get_health_analyzer()
    profile = analyzer.get_user_profile(user_id)
    if not profile:
        raise HTTPException(status_code=404, detail="User not found")
    return profile


@app.get("/api/users/{user_id}/health-context", response_model=HealthContextResponse)
async def get_health_context(user_id: str):
    """Get comprehensive health context for a user"""
    analyzer = get_health_analyzer()
    context = analyzer.get_comprehensive_health_context(user_id)
    if not context.get("profile"):
        raise HTTPException(status_code=404, detail="User not found")
    return context


@app.get("/api/users/{user_id}/metrics")
async def get_user_metrics(user_id: str, days: int = 7):
    """Get recent health metrics"""
    analyzer = get_health_analyzer()
    metrics = analyzer.get_recent_health_metrics(user_id, days)
    if not metrics:
        raise HTTPException(status_code=404, detail="User not found")
    return metrics


@app.get("/api/users/{user_id}/sleep-analysis")
async def get_sleep_analysis(user_id: str, days: int = 30):
    """Get sleep pattern analysis"""
    analyzer = get_health_analyzer()
    analysis = analyzer.analyze_sleep_patterns(user_id, days)
    if not analysis:
        raise HTTPException(status_code=404, detail="User not found")
    return analysis


@app.get("/api/users/{user_id}/recovery-analysis")
async def get_recovery_analysis(user_id: str, days: int = 30):
    """Get recovery pattern analysis"""
    analyzer = get_health_analyzer()
    analysis = analyzer.analyze_recovery_patterns(user_id, days)
    if not analysis:
        raise HTTPException(status_code=404, detail="User not found")
    return analysis


@app.get("/api/users/{user_id}/activity-summary")
async def get_activity_summary(user_id: str, days: int = 30):
    """Get activity and workout summary"""
    analyzer = get_health_analyzer()
    summary = analyzer.get_activity_summary(user_id, days)
    if not summary:
        raise HTTPException(status_code=404, detail="User not found")
    return summary


@app.get("/api/users/{user_id}/health-risks")
async def get_health_risks(user_id: str):
    """Get identified health risks"""
    analyzer = get_health_analyzer()
    risks = analyzer.identify_health_risks(user_id)
    return {"risks": risks}


@app.get("/api/weather")
async def get_weather(location: str = "New York"):
    """Get weather data and health impacts"""
    weather_service = get_weather_service()
    weather = await weather_service.get_current_weather(location)
    if weather:
        impacts = weather_service.get_weather_health_impacts(weather)
        return impacts
    return {"error": "Could not fetch weather data"}


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Main chat endpoint for AI health assistant"""
    analyzer = get_health_analyzer()
    weather_service = get_weather_service()
    ai_service = get_ai_service()
    
    # Get user health context
    health_context = analyzer.get_comprehensive_health_context(request.user_id)
    
    # Get weather data if requested
    weather_data = None
    if request.include_weather and request.location:
        weather = await weather_service.get_current_weather(request.location)
        if weather:
            weather_data = weather_service.get_weather_health_impacts(weather)
    
    # Get conversation history
    history = conversation_store.get(request.user_id, [])
    
    # Generate AI response (returns dict with message and optional ui_components)
    result = await ai_service.generate_response(
        user_message=request.message,
        health_context=health_context,
        weather_data=weather_data,
        conversation_history=history,
        user_id=request.user_id,
    )
    response_message = result.get("message", "")
    ui_components = result.get("ui_components")

    # Update conversation history
    if request.user_id not in conversation_store:
        conversation_store[request.user_id] = []
    conversation_store[request.user_id].append({"role": "user", "content": request.message})
    conversation_store[request.user_id].append({"role": "assistant", "content": response_message})
    if len(conversation_store[request.user_id]) > 20:
        conversation_store[request.user_id] = conversation_store[request.user_id][-20:]

    return {
        "response": response_message,
        "ui_components": ui_components,
        "user_id": request.user_id,
        "health_summary": {
            "recovery_score": health_context.get("recent_metrics", {}).get("avg_recovery_score"),
            "sleep_hours": health_context.get("recent_metrics", {}).get("avg_sleep_hours"),
            "hrv": health_context.get("recent_metrics", {}).get("avg_hrv"),
            "health_risks": len(health_context.get("health_risks", []))
        },
        "weather": weather_data.get("weather_summary") if weather_data else None,
    }


@app.post("/api/chat/clear")
async def clear_conversation(user_id: str = "USER_00001"):
    """Clear conversation history for a user"""
    if user_id in conversation_store:
        del conversation_store[user_id]
    return {"message": "Conversation cleared", "user_id": user_id}


def _validate_user_id(user_id: str) -> str:
    """Validate user_id to prevent path traversal / injection."""
    if not re.match(r"^[A-Za-z0-9_-]+$", user_id) or len(user_id) > 64:
        raise HTTPException(status_code=400, detail="Invalid user_id format")
    return user_id


@app.get("/api/users/{user_id}/preferences")
async def get_preferences(user_id: str):
    """Get user preferences for the learning system."""
    user_id = _validate_user_id(user_id)
    prefs_service = get_preferences_service()
    try:
        prefs = prefs_service.load_preferences(user_id)
        return {"status": "ok", "preferences": prefs}
    except Exception as e:
        return {"status": "error", "message": str(e), "preferences": {}}


@app.post("/api/users/{user_id}/preferences")
async def save_preference(user_id: str, update: PreferenceUpdate):
    """Save user preference from UI component interaction."""
    user_id = _validate_user_id(user_id)
    prefs_service = get_preferences_service()
    try:
        prefs_service.save_preference(
            user_id=user_id,
            question_id=update.question_id,
            category=update.category,
            selected_options=update.selected_options,
            skipped=update.skipped,
        )
        return {"status": "ok"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save preference: {e}")


@app.post("/api/meal-plan")
async def generate_meal_plan(
    user_id: str = "USER_00001",
    days: int = 1,
    dietary_preferences: Optional[List[str]] = None
):
    """Generate a personalized meal plan"""
    analyzer = get_health_analyzer()
    ai_service = get_ai_service()
    
    health_context = analyzer.get_comprehensive_health_context(user_id)
    
    preferences = {}
    if dietary_preferences:
        preferences["dietary_restrictions"] = dietary_preferences
    
    meal_plan = await ai_service.generate_meal_plan(
        health_context=health_context,
        preferences=preferences if preferences else None,
        days=days
    )
    
    return {"meal_plan": meal_plan, "user_id": user_id}


# ============ ML & Prediction Endpoints ============

@app.get("/api/users/{user_id}/predictions/recovery")
async def predict_recovery(user_id: str):
    """Predict tomorrow's recovery score using ML model"""
    ml_service = get_ml_service()
    prediction = ml_service.predict_recovery(user_id)
    return prediction


@app.get("/api/users/{user_id}/predictions/strain")
async def predict_optimal_strain(user_id: str):
    """Predict optimal strain level for today"""
    ml_service = get_ml_service()
    prediction = ml_service.predict_optimal_strain(user_id)
    return prediction


@app.get("/api/users/{user_id}/predictions/risk")
async def predict_health_risk(user_id: str):
    """Predict health risk score"""
    ml_service = get_ml_service()
    prediction = ml_service.predict_health_risk(user_id)
    return prediction


@app.get("/api/users/{user_id}/predictions/all")
async def get_all_predictions(user_id: str):
    """Get all predictions for a user"""
    ml_service = get_ml_service()
    return {
        "user_id": user_id,
        "recovery": ml_service.predict_recovery(user_id),
        "strain": ml_service.predict_optimal_strain(user_id),
        "risk": ml_service.predict_health_risk(user_id)
    }


# ============ LSTM Forecast Endpoints ============

@app.get("/api/users/{user_id}/predictions/lstm-forecast")
async def predict_lstm_forecast(user_id: str, days: int = 7):
    """
    Get LSTM 7-day health forecast
    
    Uses trained PyTorch LSTM model to predict health metrics for the next 7 days
    based on 21 days of historical data.
    """
    lstm_service = get_lstm_service()
    forecast = lstm_service.predict_forecast(user_id)
    return forecast


@app.get("/api/users/{user_id}/predictions/lstm-status")
async def get_lstm_model_status(user_id: str):
    """Get LSTM model status and metadata"""
    lstm_service = get_lstm_service()
    status = lstm_service.get_model_status()
    
    # Check if user has enough data
    analyzer = get_health_analyzer()
    df = analyzer.df
    if df is not None:
        user_data = df[df['user_id'] == user_id]
        status['user_data_days'] = len(user_data)
        status['has_sufficient_data'] = len(user_data) >= 21
    
    return status


@app.get("/api/lstm-status")
async def get_lstm_system_status():
    """Get LSTM system-wide status"""
    lstm_service = get_lstm_service()
    return {
        "service": "LSTM Health Forecasting",
        "status": lstm_service.get_model_status()
    }


# ============ Chart Data Endpoints ============

@app.get("/api/users/{user_id}/charts/trends")
async def get_trend_data(user_id: str, days: int = 30):
    """Get historical trend data for charts"""
    ml_service = get_ml_service()
    return ml_service.get_trend_data(user_id, days)


@app.get("/api/users/{user_id}/charts/correlations")
async def get_correlation_insights(user_id: str):
    """Get correlation insights between health metrics"""
    ml_service = get_ml_service()
    return ml_service.get_correlation_insights(user_id)


@app.get("/api/users/{user_id}/dashboard")
async def get_dashboard_data(user_id: str, days: int = 30, forecast_days: int = 3):
    """Get comprehensive dashboard data including predictions, trends, and LSTM forecasts"""
    analyzer = get_health_analyzer()
    ml_service = get_ml_service()
    lstm_service = get_lstm_service()
    
    # Get LSTM forecast
    forecast_response = lstm_service.predict_forecast(user_id)
    forecast_data = []
    forecast_enabled = False
    
    if forecast_response.get('status') == 'success':
        forecast_data = forecast_response.get('predictions', [])[:forecast_days]
        forecast_enabled = True
    
    return {
        "user_id": user_id,
        "profile": analyzer.get_user_profile(user_id),
        "recent_metrics": analyzer.get_recent_health_metrics(user_id, 7),
        "trends": ml_service.get_trend_data(user_id, days),
        "predictions": {
            "recovery": ml_service.predict_recovery(user_id),
            "strain": ml_service.predict_optimal_strain(user_id),
            "risk": ml_service.predict_health_risk(user_id)
        },
        "forecasts": forecast_data,
        "forecast_metadata": {
            "enabled": forecast_enabled,
            "model": forecast_response.get('model', 'N/A'),
            "days": len(forecast_data),
            "status": forecast_response.get('status')
        },
        "correlations": ml_service.get_correlation_insights(user_id),
        "health_risks": analyzer.identify_health_risks(user_id)
    }


# Mount static files for frontend
frontend_path = Path(__file__).parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")


def run_server():
    """Run the server"""
    uvicorn.run(app, host=HOST, port=PORT)


if __name__ == "__main__":
    run_server()
