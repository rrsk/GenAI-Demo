# Architecture

**Analysis Date:** 2026-01-25

## Pattern Overview

**Overall:** Layered service-oriented architecture with clear separation between data processing, business logic, and API exposure. Backend services are designed as stateless processors with singleton instances for resource management.

**Key Characteristics:**
- Service layer pattern with dependency injection via singleton factories
- Request-response API with stateful conversation management
- Fallback mechanism for degraded AI service availability
- Context-aware processing combining multiple data sources

## Layers

**API Layer (HTTP/REST):**
- Purpose: Expose health and AI services via HTTP endpoints
- Location: `backend/main.py` (FastAPI application)
- Contains: Route handlers, request/response models, endpoint definitions
- Depends on: AIService, HealthAnalyzer, WeatherService
- Used by: Frontend (JavaScript), external API consumers

**Service Layer (Business Logic):**
- Purpose: Implement core business logic for health analysis, weather processing, and AI response generation
- Location: `backend/ai_service.py`, `backend/health_analyzer.py`, `backend/weather_service.py`
- Contains: Data processing, recommendation generation, external API integration
- Depends on: Configuration, data sources, external APIs (OpenAI, Anthropic, OpenWeatherMap)
- Used by: API layer, orchestrated within chat endpoint

**Data/Models Layer:**
- Purpose: Define data structures, validation, and persistence patterns
- Location: `backend/models.py`, `backend/config.py`
- Contains: Pydantic models for request/response validation, configuration management
- Depends on: None (foundational)
- Used by: All service and API layers

**Data Source Layer:**
- Purpose: Provide access to health metrics and user profiles
- Location: CSV file at `Whoop Data/whoop_fitness_dataset_100k.csv` (read via HealthAnalyzer)
- Contains: User profiles, biometric measurements, sleep/recovery/activity data
- Depends on: None (file system)
- Used by: HealthAnalyzer service

**Frontend Layer (Client-side):**
- Purpose: Provide user interface for chat, health monitoring, and weather display
- Location: `frontend/index.html`, `frontend/app.js`, `frontend/styles.css`
- Contains: DOM manipulation, API client logic, state management
- Depends on: Backend API
- Used by: End users via web browser

## Data Flow

**Chat Flow (Primary User Interaction):**

1. User submits message via frontend (`frontend/app.js:sendChatMessage()`)
2. Frontend sends POST to `/api/chat` with user message, user ID, location, weather flag
3. API endpoint (`backend/main.py:chat()`) receives ChatRequest
4. HealthAnalyzer retrieves comprehensive health context for user:
   - User profile (age, gender, fitness level, etc.)
   - Recent metrics (recovery, sleep, HRV, strain - 7-day average)
   - Sleep pattern analysis (30-day window)
   - Recovery pattern analysis (30-day window)
   - Activity summary (30-day aggregation)
   - Health risks (derived from analyses)
5. WeatherService optionally fetches weather data:
   - Calls OpenWeatherMap API if location provided and API key available
   - Falls back to mock data if API unavailable
   - Calculates health impacts (temperature, humidity, pressure effects)
6. Conversation history retrieved from in-memory store
7. AIService constructs messages:
   - System prompt defining WellnessAI persona and capabilities
   - Context message with formatted health data
   - Prior conversation history (last 10 messages)
   - Current user message
8. AIService attempts to call OpenAI first, falls back to Anthropic, then to fallback response generator
9. Response stored in conversation history (in-memory, limited to last 20 messages per user)
10. Frontend displays response, updates health metrics display

**Health Analysis Flow (Data Processing):**

1. HealthAnalyzer loads CSV at initialization (`_load_data()`)
2. Filters data by user_id for requested operations
3. Calculates metrics:
   - Averages over specified time window (default 7-30 days)
   - Trend calculation comparing first half vs second half of window
   - Issue detection via threshold comparisons
4. Returns structured Dict responses to API layer

**Weather Integration Flow:**

1. Frontend sends GET to `/api/weather?location=<location>`
2. WeatherService.get_current_weather() called
3. If API key available: calls OpenWeatherMap weather endpoint
4. Parses response into normalized weather format
5. Calculates health impacts and recommendations based on:
   - Temperature (cold/heat stress)
   - Humidity (hydration needs)
   - Weather condition (light/mood effects)
   - Barometric pressure (headache risk)
6. Returns impacts array, recommendations, hydration multiplier, calorie adjustment

**State Management:**

- **Conversation History:** In-memory Dict[user_id, List[messages]] in `backend/main.py:conversation_store`
  - Persisted only for session duration
  - Limited to last 20 messages per user to manage memory
  - Cleared via `/api/chat/clear` endpoint
- **Service Singletons:** Global instances created once per service (AIService, HealthAnalyzer, WeatherService)
  - Created on first access via factory functions
  - Maintain connections and loaded data across requests
- **User Sessions:** No formal session management; identified by user_id string

## Key Abstractions

**HealthAnalyzer:**
- Purpose: Encapsulates Whoop data access and health analysis algorithms
- Examples: `backend/health_analyzer.py`
- Pattern: Singleton service with pandas-based data analysis
- Methods: `get_user_profile()`, `analyze_sleep_patterns()`, `analyze_recovery_patterns()`, `get_comprehensive_health_context()`
- Key insight: Separates data loading (`_load_data()`) from analysis methods, allowing flexible window sizes and trend calculations

**AIService:**
- Purpose: Abstracts AI provider selection and message construction
- Examples: `backend/ai_service.py`
- Pattern: Singleton service with dual-provider support (OpenAI/Anthropic)
- Methods: `generate_response()`, `generate_meal_plan()`, internal fallback generator
- Key insight: Graceful degradation through provider fallback and rule-based fallback response generation

**WeatherService:**
- Purpose: Abstracts weather data source and health impact analysis
- Examples: `backend/weather_service.py`
- Pattern: Singleton service with fallback to mock data
- Methods: `get_current_weather()`, `get_weather_health_impacts()`, helper calculators
- Key insight: Health-aware weather analysis (calculates hydration/calorie adjustments based on conditions)

**ChatRequest/HealthContextResponse:**
- Purpose: Validate and standardize API payloads
- Pattern: Pydantic BaseModel with field validation
- Usage: Automatic request validation, response serialization, OpenAPI schema generation

## Entry Points

**Backend Entry Point:**
- Location: `run.py`
- Triggers: `python run.py` command
- Responsibilities: Starts FastAPI/Uvicorn server on configured HOST:PORT

**API Entry Points (REST):**

1. **GET `/api/health`** - Health check
   - Location: `backend/main.py:health_check()`
   - Triggers: Healthcheck monitoring
   - Responsibilities: Returns service status

2. **POST `/api/chat`** - Main chat endpoint
   - Location: `backend/main.py:chat()`
   - Triggers: User message submission
   - Responsibilities: Orchestrates health analyzer → weather service → AI service, manages conversation history

3. **GET `/api/users/{user_id}/health-context`** - Comprehensive health data
   - Location: `backend/main.py:get_health_context()`
   - Triggers: Frontend initialization, user selection change
   - Responsibilities: Retrieves all health data for dashboard display

4. **GET `/api/weather`** - Weather data with health impacts
   - Location: `backend/main.py:get_weather()`
   - Triggers: Manual weather update, chat with weather enabled
   - Responsibilities: Fetches weather and calculates health adaptation recommendations

5. **POST `/api/meal-plan`** - Meal plan generation
   - Location: `backend/main.py:generate_meal_plan()`
   - Triggers: User request via chat interface (not explicitly exposed as standalone)
   - Responsibilities: Generates personalized meal plan via AIService

**Frontend Entry Point:**
- Location: `frontend/index.html`
- Triggers: Browser navigation to http://localhost:8000
- Responsibilities: Serves HTML, loads app.js and styles.css

## Error Handling

**Strategy:** Defensive with fallback mechanisms at each integration point.

**Patterns:**

1. **API Level:**
   - HTTPException with status codes for client/user errors (404 for not found)
   - Try-catch blocks returning error objects in JSON responses
   - Example: `backend/main.py:get_user_profile()` returns 404 if user not found

2. **Service Level:**
   - Graceful degradation in AIService: OpenAI → Anthropic → Fallback generator
   - Mock data fallback in WeatherService when API unavailable
   - Example: `backend/ai_service.py:generate_response()` catches exceptions and returns user-friendly error message

3. **Data Level:**
   - HealthAnalyzer.get_user_profile() returns None if user not found, API layer converts to 404
   - Empty DataFrame handling returns empty Dict instead of throwing

4. **Frontend:**
   - Fetch error handling with try-catch, user notifications via console
   - Default values for missing metrics (e.g., N/A for trends)
   - Example: `frontend/app.js:loadUserData()` catches and logs errors, shows notification

## Cross-Cutting Concerns

**Logging:** Printf-style print() statements for API errors and debugging
- Used in: `backend/ai_service.py` (API failures), `backend/weather_service.py` (API errors)
- Future consideration: Structured logging framework (e.g., Python logging module)

**Validation:** Pydantic model validation on API boundaries
- Ensures type safety and range constraints
- Examples: `models.py:UserProfile` enforces age 18-100, weight 30-300kg
- Automatic error responses for validation failures (422 Unprocessable Entity)

**Authentication:** None currently implemented
- All endpoints accessible without authentication
- User identification via user_id parameter (not cryptographically secured)
- Future consideration: JWT tokens, API keys, or session management

**Configuration Management:**
- Environment variables loaded via python-dotenv in `backend/config.py`
- Centralized config file for API keys, model names, data paths
- Supports HOST, PORT, AI provider selection via env vars
