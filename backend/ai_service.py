"""
AI Service for WellnessAI
Integrates Local LLM + Recommendation Engine for personalized health advice

Architecture:
1. User Query → Intent Classification (Recommendation Engine)
2. Health Data → Structured Analysis (Recommendation Engine) 
3. Structured Recommendations → Natural Language (Local LLM)
4. Optional: External APIs as fallback (OpenAI/Anthropic)

This demonstrates a complete GenAI pipeline:
- Data preprocessing and feature engineering
- Rule-based expert system
- Local LLM deployment
- Prompt engineering
- Fallback handling
"""

import json
import os
import re
from typing import Dict, Optional, List, Any
from pydantic import ValidationError

from .recommendation_engine import get_recommendation_engine, QueryIntent
from .local_llm_service import get_local_llm_service, get_grounding_llm_service
from .models import UIComponent
from .config import USE_STATE_OF_UNION

# Optional external API imports (fallback only)
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

from .config import OPENAI_API_KEY, ANTHROPIC_API_KEY, OPENAI_MODEL, ANTHROPIC_MODEL


class AIService:
    """
    AI-powered health and nutrition recommendations
    
    Priority order for response generation:
    1. Local LLM (HuggingFace Transformers) - Primary, no API dependency
    2. Recommendation Engine Only - Fast fallback with rule-based logic
    3. External APIs - Optional fallback if configured
    """
    
    def __init__(self, use_local_llm: bool = True, preload_model: bool = False):
        """
        Initialize AI Service
        
        Args:
            use_local_llm: Whether to use local LLM (default True)
            preload_model: Whether to load model immediately (default False)
        """
        self.use_local_llm = use_local_llm
        self.recommendation_engine = get_recommendation_engine()
        
        # Local LLM service
        self.local_llm = None
        if use_local_llm:
            self.local_llm = get_local_llm_service()
            if preload_model:
                self.local_llm.load_model()
        
        # External API clients (optional fallback)
        self.openai_client = None
        self.anthropic_client = None
        
        if OPENAI_AVAILABLE and OPENAI_API_KEY:
            self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
        if ANTHROPIC_AVAILABLE and ANTHROPIC_API_KEY:
            self.anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        
        # System prompt for external APIs
        self.system_prompt = """You are WellnessAI, an expert health and nutrition advisor. 
Provide personalized, science-based recommendations using the health data provided.
Be concise, supportive, and actionable. Use the pre-computed recommendations as your guide."""
        
        print("[AIService] Initialized")
        print(f"  - Local LLM: {'Enabled' if use_local_llm else 'Disabled'}")
        if use_local_llm and USE_STATE_OF_UNION:
            print("  - State-of-union: BioGPT grounds context → TinyLlama responds")
        print(f"  - OpenAI API: {'Available' if self.openai_client else 'Not configured'}")
        print(f"  - Anthropic API: {'Available' if self.anthropic_client else 'Not configured'}")
    
    def _parse_llm_response(self, raw_response: str) -> Dict[str, Any]:
        """Extract optional JSON block from LLM response and validate ui_components."""
        out = {"message": raw_response, "ui_components": None}
        json_match = re.search(r"```json\s*(.*?)\s*```", raw_response, re.DOTALL)
        if not json_match:
            return out
        try:
            data = json.loads(json_match.group(1).strip())
            msg = data.get("message")
            if msg:
                out["message"] = msg
            comps = data.get("ui_components")
            if comps and isinstance(comps, list) and len(comps) > 0:
                validated = []
                for c in comps[:1]:  # at most one component
                    try:
                        validated.append(UIComponent(**c).model_dump())
                    except (ValidationError, TypeError):
                        break
                if validated:
                    out["ui_components"] = validated
        except (json.JSONDecodeError, ValidationError) as e:
            print(f"[AIService] JSON parse error: {e}")
        return out

    async def generate_response(
        self,
        user_message: str,
        health_context: Dict,
        weather_data: Optional[Dict] = None,
        conversation_history: Optional[List[Dict]] = None,
        user_id: str = "USER_00001",
    ) -> Dict[str, Any]:
        """
        Generate AI response. Returns dict with "message" (str) and optional "ui_components" (list).
        """
        # Step 1: Classify intent
        intent = self.recommendation_engine.classify_intent(user_message)

        # Step 2: Structured recommendations
        recommendations = self.recommendation_engine.generate_recommendations(
            health_context=health_context,
            intent=intent,
            weather_data=weather_data,
        )

        # Step 3: Get LSTM forecasts for context
        forecasts = []
        try:
            from .lstm_service import get_lstm_service
            lstm_service = get_lstm_service()
            forecast_response = lstm_service.predict_forecast(user_id)
            if forecast_response.get('status') == 'success':
                forecasts = forecast_response.get('predictions', [])[:3]
        except Exception as e:
            print(f"[AIService] LSTM forecast unavailable: {e}")

        # Step 4: Format context with forecasts and inject preferences
        health_context_text = self.recommendation_engine.format_health_context_for_llm(health_context, forecasts)
        recommendations_text = self.recommendation_engine.format_recommendations_for_llm(recommendations)

        try:
            from .user_preferences_service import get_preferences_service
            prefs_service = get_preferences_service()
            prefs_context = prefs_service.get_context_string(user_id)
            health_context_text += "\n\n" + prefs_context
            can_ask = prefs_service.can_ask_question(user_id)
            existing = (prefs_service.load_preferences(user_id).get("preferences") or {})
            if can_ask and not existing:
                health_context_text += """

If you need to learn one preference (dietary, schedule, goals, or feedback), reply with ONLY a JSON block in this exact format:
```json
{"message": "Your short message here", "ui_components": [{"type": "multi_select", "question_id": "dietary_restrictions", "prompt": "Do you have any dietary restrictions?", "category": "dietary", "options": [{"id": "vegetarian", "label": "Vegetarian", "emoji": ""}, {"id": "vegan", "label": "Vegan", "emoji": ""}, {"id": "none", "label": "No restrictions", "emoji": ""}], "allow_skip": true}]}
```
Ask only ONE question. Otherwise reply with normal text only."""
        except Exception as e:
            print(f"[AIService] Preferences unavailable: {e}")
            can_ask = False

        if weather_data:
            weather = weather_data.get("weather_summary", {})
            health_context_text += f"\n\n**Weather:** {weather.get('temperature', 'N/A')}°C, {weather.get('condition', 'Unknown')}"

        # Step 3b (state-of-union): BioGPT produces evidence-based prompt from stats + query; TinyLlama uses it for the final response
        if USE_STATE_OF_UNION and self.use_local_llm and self.local_llm:
            try:
                grounding_service = get_grounding_llm_service()
                grounding = grounding_service.generate_grounding(health_context_text, user_message)
                if grounding and len(grounding.strip()) > 20:
                    health_context_text += "\n\n**Evidence-based framing (from clinical analysis):**\n" + grounding.strip()
            except Exception as e:
                print(f"[AIService] State-of-union grounding error: {e}")

        # Step 4: Generate raw response
        response = None
        if self.use_local_llm and self.local_llm:
            try:
                response = self.local_llm.generate_response(
                    user_message=user_message,
                    health_context=health_context_text,
                    recommendations=recommendations_text,
                )
                if response and len(response) > 20:
                    parsed = self._parse_llm_response(response)
                    message = self._format_response(parsed["message"], intent)
                    if parsed.get("ui_components"):
                        try:
                            from .user_preferences_service import get_preferences_service
                            get_preferences_service().record_question_asked(
                                user_id, parsed["ui_components"][0]["question_id"]
                            )
                        except Exception:
                            pass
                    return {"message": message, "ui_components": parsed.get("ui_components")}
            except Exception as e:
                print(f"[AIService] Local LLM error: {e}")

        if not response and (self.openai_client or self.anthropic_client):
            try:
                response = await self._call_external_api(
                    user_message,
                    health_context_text,
                    recommendations_text,
                    conversation_history,
                )
                if response:
                    parsed = self._parse_llm_response(response)
                    msg = parsed["message"]
                    if not msg.startswith("#"):
                        msg = "## WellnessAI Analysis\n\n" + msg
                    return {"message": msg, "ui_components": parsed.get("ui_components")}
            except Exception as e:
                print(f"[AIService] External API error: {e}")

        fallback = self._generate_rule_based_response(
            user_message, health_context, weather_data, recommendations
        )
        return {"message": fallback, "ui_components": None}
    
    def _format_response(self, response: str, intent: QueryIntent) -> str:
        """Format LLM response with appropriate headers"""
        # Add section header if not present
        if not response.startswith("#"):
            headers = {
                QueryIntent.MEAL_PLAN: "## 🍽️ Your Personalized Meal Plan\n\n",
                QueryIntent.HEADACHE: "## 🩹 Headache Relief Recommendations\n\n",
                QueryIntent.FATIGUE: "## ⚡ Energy Boost Strategy\n\n",
                QueryIntent.SLEEP: "## 😴 Sleep Optimization Plan\n\n",
                QueryIntent.EXERCISE: "## 🏃 Activity Recommendations\n\n",
                QueryIntent.RECOVERY: "## 💪 Recovery Plan\n\n",
                QueryIntent.STRESS: "## 🧘 Stress Management\n\n",
                QueryIntent.GENERAL: "## WellnessAI Analysis\n\n"
            }
            response = headers.get(intent, "## WellnessAI Analysis\n\n") + response
        
        return response
    
    async def _call_external_api(
        self,
        user_message: str,
        health_context: str,
        recommendations: str,
        conversation_history: Optional[List[Dict]]
    ) -> Optional[str]:
        """Call external API (OpenAI or Anthropic) as fallback"""
        
        combined_prompt = f"""Based on the following health data and pre-computed recommendations, 
provide a personalized, conversational response to the user's question.

## User's Health Data:
{health_context}

## Pre-Computed Recommendations:
{recommendations}

## User's Question:
{user_message}

Provide a helpful, personalized response that incorporates the recommendations naturally."""
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": combined_prompt}
        ]
        
        # Add conversation history
        if conversation_history:
            messages = messages[:1] + conversation_history[-6:] + messages[1:]
        
        # Try OpenAI
        if self.openai_client:
            try:
                response = self.openai_client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=1500
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"[AIService] OpenAI error: {e}")
        
        # Try Anthropic
        if self.anthropic_client:
            try:
                system = messages[0]["content"]
                other_messages = [m for m in messages if m["role"] != "system"]
                
                response = self.anthropic_client.messages.create(
                    model=ANTHROPIC_MODEL,
                    max_tokens=1500,
                    system=system,
                    messages=other_messages
                )
                return response.content[0].text
            except Exception as e:
                print(f"[AIService] Anthropic error: {e}")
        
        return None
    
    def _generate_rule_based_response(
        self,
        user_message: str,
        health_context: Dict,
        weather_data: Optional[Dict],
        recommendations: list
    ) -> str:
        """Generate response using only rule-based recommendations (no LLM)"""
        response_parts = ["## WellnessAI Analysis\n"]
        
        # Profile intro
        if profile := health_context.get("profile"):
            response_parts.append(
                f"Based on your profile ({profile.get('age', 'N/A')} year old "
                f"{profile.get('gender', '')}, {profile.get('fitness_level', '')} level), "
                f"here are my recommendations:\n"
            )
        
        # Add each recommendation
        for rec in recommendations[:3]:
            response_parts.append(f"\n### {rec.title}\n")
            response_parts.append(f"{rec.description}\n")
            
            if rec.actions:
                response_parts.append("\n**What to do:**")
                for action in rec.actions[:4]:
                    response_parts.append(f"\n- {action}")
            
            if rec.foods:
                response_parts.append("\n\n**Recommended foods:**")
                for food in rec.foods[:4]:
                    response_parts.append(f"\n- {food}")
            
            if rec.avoid:
                response_parts.append("\n\n**Avoid:**")
                for item in rec.avoid[:3]:
                    response_parts.append(f"\n- {item}")
        
        # Weather context
        if weather_data:
            weather = weather_data.get("weather_summary", {})
            response_parts.append(f"\n\n### Weather Considerations")
            response_parts.append(f"\nCurrent: {weather.get('temperature', 'N/A')}°C, {weather.get('condition', '')}")
            
            for rec in weather_data.get("recommendations", [])[:2]:
                response_parts.append(f"\n- {rec}")
        
        # Health risks
        if risks := health_context.get("health_risks"):
            response_parts.append("\n\n### Priority Health Focus:")
            for risk in risks[:2]:
                response_parts.append(f"\n- **{risk.get('risk', '')}**: {risk.get('dietary_impact', '')}")
        
        return "".join(response_parts)
    
    async def generate_meal_plan(
        self,
        health_context: Dict,
        preferences: Optional[Dict] = None,
        days: int = 1
    ) -> str:
        """Generate a detailed meal plan"""
        metrics = health_context.get("recent_metrics", {})
        profile = health_context.get("profile", {})
        
        # Use recommendation engine to generate meal plan
        meal_plan = self.recommendation_engine.generate_meal_plan(
            profile=profile,
            metrics=metrics,
            focus_area="general"
        )
        
        # Format response
        response = f"""## 🍽️ Your {days}-Day Personalized Meal Plan

### Daily Targets
- **Calories:** {meal_plan['daily_targets']['tdee']} kcal
- **Protein:** {meal_plan['daily_targets']['protein_g']}g
- **Carbs:** {meal_plan['daily_targets']['carbs_g']}g  
- **Fat:** {meal_plan['daily_targets']['fat_g']}g
- **Water:** {meal_plan['daily_targets']['water_liters']}L

---

### 🌅 Breakfast ({meal_plan['breakfast']['time']}) - {meal_plan['breakfast']['calories']} kcal
"""
        for option in meal_plan['breakfast']['options']:
            response += f"- {option}\n"
        
        response += f"""
### 🌞 Lunch ({meal_plan['lunch']['time']}) - {meal_plan['lunch']['calories']} kcal
"""
        for option in meal_plan['lunch']['options']:
            response += f"- {option}\n"
        
        response += f"""
### 🌙 Dinner ({meal_plan['dinner']['time']}) - {meal_plan['dinner']['calories']} kcal
"""
        for option in meal_plan['dinner']['options']:
            response += f"- {option}\n"
        
        response += f"""
### 🥜 Snacks - {meal_plan['snacks']['calories']} kcal
"""
        for option in meal_plan['snacks']['options']:
            response += f"- {option}\n"
        
        # Add focus foods if available
        if meal_plan.get('focus_foods'):
            response += "\n### 💡 Focus Foods for Your Goals\n"
            for food in meal_plan['focus_foods']:
                response += f"- {food}\n"
        
        return response
    
    def get_service_status(self) -> Dict:
        """Get status of AI service components"""
        status = {
            "recommendation_engine": "active",
            "local_llm": "disabled",
            "openai_api": "not_configured",
            "anthropic_api": "not_configured"
        }
        
        if self.use_local_llm and self.local_llm:
            info = self.local_llm.get_model_info()
            status["local_llm"] = "loaded" if info["is_loaded"] else "available"
            status["local_llm_model"] = info["model_name"]
            status["local_llm_device"] = info["device"]
        
        if self.openai_client:
            status["openai_api"] = "configured"
        if self.anthropic_client:
            status["anthropic_api"] = "configured"
        
        return status


# Singleton instance
_ai_service = None

def get_ai_service(use_local_llm: bool = True) -> AIService:
    """Get or create AIService singleton"""
    global _ai_service
    if _ai_service is None:
        _ai_service = AIService(use_local_llm=use_local_llm)
    return _ai_service
