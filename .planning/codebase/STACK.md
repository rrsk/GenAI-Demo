# Technology Stack

**Analysis Date:** 2026-01-25

## Languages

**Primary:**
- Python 3.13.7 - Backend server, health data analysis, AI orchestration
- JavaScript (Vanilla ES6+) - Frontend UI and client interactions
- HTML5 - Frontend markup
- CSS3 - Frontend styling and animations

## Runtime

**Environment:**
- Python 3.9+ (specified in README, currently running 3.13.7)
- CPython standard distribution

**Package Manager:**
- pip (Python package manager)
- Lockfile: `requirements.txt` (present)
- Virtual environment: `venv/` (present)

## Frameworks

**Core:**
- FastAPI 0.109.0+ - REST API backend framework with automatic OpenAPI documentation
- Uvicorn 0.27.0+ - ASGI server for running FastAPI application
- Pydantic 2.6.0+ - Data validation and serialization for API models

**Data Processing:**
- Pandas 2.2.0+ - Health data analysis and CSV processing
- NumPy 1.26.0+ - Numerical computing for statistical calculations
- scikit-learn 1.4.0+ - Machine learning utilities for trend analysis

**Frontend:**
- Vanilla JavaScript (no framework dependencies)
- Fetch API for HTTP requests
- DOM manipulation without libraries

## Key Dependencies

**Critical:**
- openai 1.12.0+ - OpenAI GPT-4 API client (primary AI provider)
- anthropic 0.18.0+ - Anthropic Claude API client (fallback AI provider)
- httpx 0.26.0+ - Async HTTP client for weather API calls
- python-dotenv 1.0.0+ - Environment variable management for API keys
- python-multipart 0.0.6 - Multipart form data handling for file uploads
- aiofiles 23.2.1 - Async file I/O operations

**Infrastructure:**
- pandas 2.2.0+ - CSV data loading and manipulation
- numpy 1.26.0+ - Statistical computations and array operations
- scikit-learn 1.4.0+ - ML utilities for data analysis

## Configuration

**Environment:**
- Configuration via `.env` file (not committed, user-created)
- Environment variables loaded by `python-dotenv` in `backend/config.py`
- Required keys: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` (at least one), `OPENWEATHER_API_KEY` (optional)

**Build:**
- No build step required
- Direct Python script execution
- Uvicorn handles live reload with `--reload` flag in development

**Entry Points:**
- `run.py` - Primary entry point, starts Uvicorn server
- `backend/main.py` - FastAPI app definition

## Platform Requirements

**Development:**
- Python 3.9+
- pip for dependency installation
- Virtual environment tool (venv)
- OpenAI API key (optional, falls back to Anthropic or mock responses)
- Anthropic API key (optional, provides fallback AI)
- OpenWeather API key (optional, provides real weather data, returns mocks without it)

**Production:**
- Python 3.9+ runtime
- Environment variables configured for API keys
- Network access to OpenAI, Anthropic, and OpenWeather APIs
- CSV data file at `Whoop Data/whoop_fitness_dataset_100k.csv` (100k records)
- Listens on configurable host (default 0.0.0.0) and port (default 8000)

**Data:**
- Whoop fitness dataset: `Whoop Data/whoop_fitness_dataset_100k.csv` (100,000 user records with daily health metrics)
- Pre-loaded into memory via Pandas on server startup
- Contains: user_id, age, gender, weight, height, fitness_level, recovery_score, hrv, sleep_hours, strain, calories, etc.

## Server Configuration

**Default:**
- HOST: 0.0.0.0 (configurable via `HOST` env var)
- PORT: 8000 (configurable via `PORT` env var)
- CORS: Enabled for all origins (`*`) to allow frontend requests
- Reload: Enabled in development mode for hot reloading

**API Documentation:**
- Automatic OpenAPI/Swagger docs at `/docs`
- ReDoc documentation at `/redoc`

---

*Stack analysis: 2026-01-25*
