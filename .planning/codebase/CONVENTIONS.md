# Coding Conventions

**Analysis Date:** 2026-01-25

## Naming Patterns

**Files:**
- Python backend modules use lowercase with underscores: `ai_service.py`, `health_analyzer.py`, `weather_service.py`
- JavaScript frontend uses camelCase: `app.js`
- Classes/Services suffixed with descriptive names: `AIService`, `HealthAnalyzer`, `WeatherService`

**Functions:**
- Python: snake_case for all functions: `get_user_profile()`, `analyze_sleep_patterns()`, `_build_context_message()`, `_parse_weather_data()`
- Private/internal functions prefixed with underscore: `_load_data()`, `_calculate_trend()`, `_generate_fallback_response()`
- JavaScript: camelCase for functions: `loadUsers()`, `sendChatMessage()`, `updateHealthStats()`, `formatMessage()`

**Variables:**
- Python: snake_case: `current_user_id`, `avg_recovery_score`, `health_context`, `conversation_store`
- JavaScript: camelCase: `currentUserId`, `isLoading`, `chatMessages`, `recoveryScore`
- Dictionary keys in JSON responses use snake_case: `avg_recovery_score`, `health_risks`, `recent_metrics`

**Types/Classes:**
- Python: PascalCase for all classes: `AIService`, `HealthAnalyzer`, `WeatherService`, `ChatRequest`, `UserProfile`, `HealthMetrics`
- Enum values use descriptive UPPERCASE: `Gender.MALE`, `FitnessLevel.BEGINNER`, `Gender.FEMALE`, `FitnessLevel.ELITE`
- Pydantic models use PascalCase: `ChatMessage`, `HealthConcern`, `MealPlan`, `HealthRecommendation`, `ChatResponse`

## Code Style

**Formatting:**
- Python: 4-space indentation (implicit from code)
- JavaScript: 4-space indentation for consistency
- Line wrapping: Python functions wrap parameters across multiple lines when needed (see `ai_service.py:42-47`, `main.py:227-231`)
- No explicit linting or formatting tool configured (no `.eslintrc`, `.prettierrc`, `black`, `flake8`)

**Linting:**
- No linting tools configured in the codebase
- Code follows implicit style patterns observed across files
- Implicit adherence to clean code practices without automated enforcement

## Import Organization

**Order (Python):**
1. Standard library imports: `import os`, `from pathlib import Path`, `from typing import Optional, List`
2. Third-party framework imports: `from fastapi import FastAPI`, `from pydantic import BaseModel`, `import pandas as pd`
3. Internal relative imports: `from .models import ChatMessage`, `from .config import HOST, PORT`

**Examples:**
- `backend/main.py`: Lines 5-18 show ordered imports (standard lib → frameworks → local modules)
- `backend/ai_service.py`: Lines 4-8 follow same pattern (json, typing → openai, anthropic → config)

**Path Aliases:**
- No path aliases configured
- Use relative imports for internal modules: `from .models import`, `from .config import`

## Error Handling

**Patterns:**
- Try-except blocks with specific exception handling and fallback responses
- `ai_service.py:142-154`: OpenAI calls wrapped with exception handler returning error message
- `ai_service.py:156-174`: Anthropic calls with try-except, returns formatted error string
- `weather_service.py:23-41`: Async weather fetch with exception handling, falls back to mock data
- `main.py:88-94`: HTTP endpoint validation with explicit `HTTPException` for 404 cases
- Frontend `app.js:97-111`: Async API calls with try-catch, console.error logging, user-facing error notifications

**Error Response Format:**
- Python API: Return structured error messages or `HTTPException(status_code=..., detail="message")`
- JavaScript: Console error logging + user notification via `showNotification(message, 'error')`

## Logging

**Framework:** Built-in `print()` statements and `console` in JavaScript

**Patterns:**
- `ai_service.py:153`: `print(f"OpenAI API error: {e}")`
- `weather_service.py:40`: `print(f"Weather API error: {e}")`
- `app.js:108`: `console.error('Failed to load users:', error)`
- `app.js:122`: `console.error('Failed to load user data:', error)`
- Simple f-string formatting for error context
- Console logging in frontend for debugging

## Comments

**When to Comment:**
- Module-level docstrings for all files: Every Python file starts with triple quotes
- Class-level docstrings: All classes have descriptive docstrings
- Complex functions have docstrings explaining purpose, not implementation details
- No inline comments observed; logic is self-documenting through clear naming

**Documentation Strings:**
- Python uses docstrings (triple-quoted strings)
- JavaScript lacks formal documentation comments - no JSDoc annotations
- Frontend functions have minimal comments, relying on clear names like `updateHealthStats()`, `formatMessage()`

**Examples:**
- `backend/models.py:1-3`: Module docstring explaining Pydantic models
- `backend/ai_service.py:1-3`: Module docstring for AI service
- `backend/health_analyzer.py:13-14`: Class docstring with clear purpose statement
- `backend/main.py:40-41`: Concise docstring for Pydantic model classes

## Function Design

**Size:**
- Typical functions: 10-50 lines
- API endpoints: 5-15 lines, delegating to service classes
- Service methods: 15-30 lines for data processing
- Large functions permitted for complex logic (e.g., `ai_service.py:78-140` for context building with multiple sections)

**Parameters:**
- Use type hints consistently: `async def generate_response(self, user_message: str, health_context: Dict, weather_data: Optional[Dict] = None, conversation_history: Optional[List[Dict]] = None) -> str:`
- Optional parameters with sensible defaults: `days: int = 7`, `limit: int = 50`
- Pydantic BaseModel for request bodies: `ChatRequest(BaseModel)` with field validation

**Return Values:**
- Type-hinted return types: `-> str`, `-> Dict`, `-> Optional[Dict]`, `-> List[str]`
- Pydantic models for API responses: `response_model=UserListResponse`
- Consistent return format within service (dicts for data, strings for text)

## Module Design

**Exports:**
- Singleton factory functions at module end: `get_health_analyzer()`, `get_ai_service()`, `get_weather_service()`
- Pattern: Global `_instance = None`, then `def get_service() -> ServiceClass`
- All services exported as singletons to maintain state across requests

**Barrel Files:**
- No barrel files (index.py) for grouping exports
- Direct imports from specific modules: `from .models import ChatMessage`
- `backend/__init__.py` is empty

**Service Classes:**
- Each service is self-contained: `AIService`, `HealthAnalyzer`, `WeatherService`
- Shared configuration in `config.py`
- Models separated in `models.py`
- Main FastAPI app in `main.py` orchestrates services

## Data Validation

**Pydantic Models:**
- All request bodies validated via Pydantic: `ChatRequest`, `UserProfile`
- Field constraints with validation: `age: int = Field(ge=18, le=100)`, `weight_kg: float = Field(ge=30, le=300)`
- Optional fields clearly marked: `Optional[str] = None`, `Optional[List[str]] = None`
- Enum types for constrained values: `Gender(str, Enum)` with MALE, FEMALE options

**Backend Validation:**
- No explicit middleware validation layer
- Validation happens at Pydantic model level
- Frontend performs basic validation (non-empty inputs, proper format)

## Async/Await Patterns

**Python (FastAPI/async):**
- API endpoints marked `async` when calling async services
- `main.py:167`: `async def chat(request: ChatRequest):`
- Service methods marked `async`: `ai_service.py:42-48`: `async def generate_response(...) -> str:`
- Async external calls: `weather_service.py:17-41` uses `async with httpx.AsyncClient()`

**JavaScript:**
- All fetch calls wrapped in async functions
- `app.js:140-187`: `async function sendChatMessage()` with await on fetch
- Error handling with try-catch in async context

---

*Convention analysis: 2026-01-25*
