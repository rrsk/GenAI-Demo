# External Integrations

**Analysis Date:** 2026-01-25

## APIs & External Services

**AI & Language Models:**
- OpenAI GPT-4 - Primary AI provider for health recommendations and meal planning
  - SDK/Client: `openai` 1.12.0+
  - Model: `gpt-4-turbo-preview` (configured in `backend/config.py`)
  - Auth: `OPENAI_API_KEY` environment variable
  - Usage: Chat responses, meal plan generation, health analysis
  - Implementation: `backend/ai_service.py` lines 142-154

- Anthropic Claude - Fallback AI provider
  - SDK/Client: `anthropic` 0.18.0+
  - Model: `claude-3-opus-20240229`
  - Auth: `ANTHROPIC_API_KEY` environment variable
  - Usage: Generates responses if OpenAI is unavailable
  - Implementation: `backend/ai_service.py` lines 156-174
  - Auto-fallback: System attempts OpenAI first, then Anthropic, then generates mock response

**Weather & Environmental:**
- OpenWeatherMap - Real-time weather data and health impacts
  - SDK/Client: `httpx` (async HTTP client)
  - API Endpoint: `https://api.openweathermap.org/data/2.5/weather`
  - Auth: `OPENWEATHER_API_KEY` environment variable
  - Units: Metric (Celsius, kmh)
  - Fallback: Mock weather data generated if API unavailable (see `backend/weather_service.py` lines 58-73)
  - Usage: Weather health impacts, hydration adjustments, calorie modifications
  - Implementation: `backend/weather_service.py` lines 17-41

## Data Storage

**Databases:**
- Not applicable - this application does not use a database

**File Storage:**
- Local filesystem only
- CSV-based health data: `Whoop Data/whoop_fitness_dataset_100k.csv`
  - Format: CSV with 100,000 health records
  - Client: Pandas DataFrame
  - Loaded at startup in `backend/health_analyzer.py` lines 20-26
  - Never modified (read-only)

**In-Memory Storage:**
- Conversation history: Python dictionary `conversation_store` in `backend/main.py` lines 36-37
  - Per-user conversation tracking (keeps last 20 messages)
  - Session-scoped (cleared on server restart)
  - No persistence between sessions

**Caching:**
- Application-level singleton caching for services:
  - `HealthAnalyzer` singleton: `backend/health_analyzer.py` lines 287-295
  - `WeatherService` singleton: `backend/weather_service.py` lines 151-159
  - `AIService` singleton: `backend/ai_service.py` lines 354-362
- No external cache system (Redis, Memcached)

## Authentication & Identity

**Auth Provider:**
- Custom/None - No user authentication system
- Default user: `USER_00001` (hardcoded in API requests)
- User selection: Frontend allows switching between available Whoop dataset users
- Auth approach: Simple user_id parameter in requests (no validation, session, or authorization)

**Implementation:**
- `backend/main.py` line 43: Default user_id = "USER_00001"
- Frontend `frontend/app.js` line 9: `currentUserId = 'USER_00001'`
- User switching via dropdown (populated from dataset unique user IDs)

## Monitoring & Observability

**Error Tracking:**
- Not integrated - No external error tracking service
- Logging approach: Print statements to console (not production-ready)
- Error handling: Try-catch blocks with console logging in:
  - `backend/ai_service.py` lines 152-154 (OpenAI errors)
  - `backend/ai_service.py` lines 172-174 (Anthropic errors)
  - `backend/weather_service.py` lines 39-41 (Weather API errors)

**Logs:**
- Standard output to console via Uvicorn logging
- Log level: "info" (configurable in `run.py` line 37)
- No structured logging or log aggregation

## CI/CD & Deployment

**Hosting:**
- Not deployed - Development/local only
- Can be deployed to any platform supporting Python 3.9+
- Containerizable (no Docker setup present)

**CI Pipeline:**
- None detected - No CI configuration files (no GitHub Actions, GitLab CI, etc.)

**Development Server:**
- Uvicorn with auto-reload enabled (see `run.py` line 36)
- Hot reload on file changes

## Environment Configuration

**Required env vars:**
- `OPENAI_API_KEY` (optional, but recommended for AI features)
- `ANTHROPIC_API_KEY` (optional, fallback AI)
- `OPENWEATHER_API_KEY` (optional, real weather data)
- `HOST` (optional, default "0.0.0.0")
- `PORT` (optional, default 8000)

**Optional env vars:**
- `HOST` - Server bind address (default: "0.0.0.0")
- `PORT` - Server port (default: 8000)

**Secrets location:**
- `.env` file in project root (user-created, not committed)
- Loaded by `python-dotenv` in `backend/config.py` lines 9-19
- Defaults to empty strings if not provided (APIs fall back to mock/error states)

## Webhooks & Callbacks

**Incoming:**
- None detected - No webhook endpoints

**Outgoing:**
- None detected - No outbound webhook calls

## API Contract

**Frontend-to-Backend Communication:**
- Base URL: `http://localhost:8000/api` (hardcoded in `frontend/app.js` line 6)
- Method: Fetch API (modern JavaScript)
- Format: JSON request/response bodies
- CORS: Allowed from all origins (configured in `backend/main.py` lines 28-34)

**Endpoints:**
- `POST /api/chat` - Main AI conversation endpoint
- `POST /api/chat/clear` - Clear conversation history
- `GET /api/users` - List available users
- `GET /api/users/{user_id}/profile` - Get user profile
- `GET /api/users/{user_id}/health-context` - Comprehensive health data
- `GET /api/users/{user_id}/metrics` - Recent 7-day metrics
- `GET /api/users/{user_id}/sleep-analysis` - 30-day sleep patterns
- `GET /api/users/{user_id}/recovery-analysis` - 30-day recovery analysis
- `GET /api/users/{user_id}/activity-summary` - Activity statistics
- `GET /api/users/{user_id}/health-risks` - Identified health risks
- `GET /api/weather` - Current weather and health impacts
- `POST /api/meal-plan` - Generate personalized meal plan
- `GET /api/health` - Health check endpoint

## Data Privacy

**Current Implementation:**
- All health data processing is local (in-memory)
- AI providers (OpenAI/Anthropic) receive anonymized health context
- No personal health data stored externally
- No persistent storage of conversations
- No user authentication or data isolation

---

*Integration audit: 2026-01-25*
