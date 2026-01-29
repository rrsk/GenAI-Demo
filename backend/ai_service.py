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

import os
from typing import Dict, Optional, List
from .recommendation_engine import get_recommendation_engine, QueryIntent
from .local_llm_service import get_local_llm_service

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
        print(f"  - OpenAI API: {'Available' if self.openai_client else 'Not configured'}")
        print(f"  - Anthropic API: {'Available' if self.anthropic_client else 'Not configured'}")
    
    async def generate_response(
        self,
        user_message: str,
        health_context: Dict,
        weather_data: Optional[Dict] = None,
        conversation_history: Optional[List[Dict]] = None
    ) -> str:
        """
        Generate AI response for user health query
        
        Pipeline:
        1. Classify user intent
        2. Generate structured recommendations
        3. Format context for LLM
        4. Generate natural language response
        """
        # Step 1: Classify intent
        intent = self.recommendation_engine.classify_intent(user_message)
        
        # Step 2: Generate structured recommendations
        recommendations = self.recommendation_engine.generate_recommendations(
            health_context=health_context,
            intent=intent,
            weather_data=weather_data
        )
        
        # Step 3: Format for LLM
        health_context_text = self.recommendation_engine.format_health_context_for_llm(health_context)
        recommendations_text = self.recommendation_engine.format_recommendations_for_llm(recommendations)
        
        # Add weather context if available
        if weather_data:
            weather = weather_data.get("weather_summary", {})
            health_context_text += f"\n\n**Weather:** {weather.get('temperature', 'N/A')}°C, {weather.get('condition', 'Unknown')}"
        
        # Step 4: Generate response
        response = None
        
        # Try Local LLM first
        if self.use_local_llm and self.local_llm:
            try:
                response = self.local_llm.generate_response(
                    user_message=user_message,
                    health_context=health_context_text,
                    recommendations=recommendations_text
                )
                if response and len(response) > 50:
                    return self._format_response(response, intent)
            except Exception as e:
                print(f"[AIService] Local LLM error: {e}")
        
        # Fallback to external APIs if available
        if not response and (self.openai_client or self.anthropic_client):
            try:
                response = await self._call_external_api(
                    user_message,
                    health_context_text,
                    recommendations_text,
                    conversation_history
                )
                if response:
                    return response
            except Exception as e:
                print(f"[AIService] External API error: {e}")
        
        # Final fallback: Rule-based response
        return self._generate_rule_based_response(
            user_message,
            health_context,
            weather_data,
            recommendations
        )
    
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
