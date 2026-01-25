# Testing Patterns

**Analysis Date:** 2026-01-25

## Test Framework

**Runner:**
- Not detected - No test framework configured
- No `pytest.ini`, `setup.cfg`, or test runner configuration files present
- No test files found in the codebase

**Assertion Library:**
- Not applicable - No testing framework installed

**Run Commands:**
- Not applicable - No test infrastructure present

## Current State: No Tests

The codebase has **zero test coverage**. There are:
- No `test_*.py` files
- No `*_test.py` files
- No `.test.js` files
- No test directory structure
- No testing dependencies in `requirements.txt` (no pytest, unittest, pytest-cov, etc.)
- No JavaScript test libraries (Jest, Vitest, Mocha, etc.)

## Recommended Testing Structure

When tests are added, follow these patterns:

### Test File Organization

**Location:** Co-located with source code

**Structure:**
```
backend/
├── models.py
├── test_models.py           # Unit tests for models
├── ai_service.py
├── test_ai_service.py       # Unit tests for AI service
├── health_analyzer.py
├── test_health_analyzer.py  # Unit tests for analyzer
├── main.py
└── test_main.py             # Integration tests for endpoints
```

**Naming:** Follow `test_*.py` convention matching module being tested

### Test Suite Organization Pattern

For Python (when using pytest):
```python
import pytest
from unittest.mock import Mock, patch
from backend.health_analyzer import HealthAnalyzer

class TestHealthAnalyzer:
    """Tests for HealthAnalyzer service"""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance for testing"""
        with patch('backend.health_analyzer.pd.read_csv'):
            yield HealthAnalyzer()

    def test_get_user_profile_returns_dict(self, analyzer):
        """Test that get_user_profile returns correct structure"""
        # arrange, act, assert pattern
        pass
```

### Mocking Strategy

**Framework:** Use `unittest.mock` (Python standard library)

**Pattern - Mocking External APIs:**
```python
# Mock OpenAI/Anthropic calls (from ai_service.py)
with patch('backend.ai_service.OpenAI') as mock_openai:
    mock_client = Mock()
    mock_openai.return_value = mock_client

    # Mock API response
    mock_response = Mock()
    mock_response.choices[0].message.content = "Test response"
    mock_client.chat.completions.create.return_value = mock_response

    # Test code here
```

**Pattern - Mocking File/CSV Loading (from health_analyzer.py):**
```python
# Mock pandas read_csv for HealthAnalyzer
with patch('backend.health_analyzer.pd.read_csv') as mock_read:
    mock_df = Mock()
    mock_read.return_value = mock_df

    analyzer = HealthAnalyzer()
    # assertions about initialization
```

**Pattern - Mocking HTTP Requests (from weather_service.py):**
```python
# Mock httpx async client for weather
with patch('backend.weather_service.httpx.AsyncClient') as mock_client:
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"main": {...}}

    # Configure async context manager
    mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
```

**What to Mock:**
- External API calls (OpenAI, Anthropic, OpenWeatherMap)
- File system operations (CSV loading in HealthAnalyzer)
- HTTP requests to third-party services
- Database connections (if database added later)

**What NOT to Mock:**
- Internal service methods (test actual behavior)
- Pydantic model validation
- Data transformation logic
- Business logic in analyzer classes

### Test Types

**Unit Tests:**
- Test individual functions/methods in isolation
- Focus: `models.py` (Pydantic validation), individual service methods
- Example: Testing `_calculate_trend()` in `health_analyzer.py:75-89`
- Use mocks for dependencies

**Integration Tests:**
- Test service interactions: HealthAnalyzer + AIService + WeatherService
- Focus: Main FastAPI endpoints in `main.py`
- Mock external APIs, test full request → response flow
- Example: POST `/api/chat` with mocked health data and AI response

**E2E Tests (if added):**
- Test complete user flows
- Can use test database with real data
- Test frontend + backend integration
- Not currently present

### Async Testing

**Pattern for testing async functions:**
```python
import pytest

@pytest.mark.asyncio
async def test_generate_response_returns_string():
    """Test async AI response generation"""
    ai_service = AIService()

    with patch.object(ai_service, '_call_openai') as mock_call:
        mock_call.return_value = "Test response"

        result = await ai_service.generate_response(
            user_message="How are you?",
            health_context={},
            weather_data=None,
            conversation_history=None
        )

        assert isinstance(result, str)
        assert result == "Test response"
```

**pytest-asyncio plugin required:** Add to requirements-dev.txt

### Error Testing

**Pattern for testing error handling:**
```python
import pytest
from fastapi import HTTPException

def test_get_user_profile_raises_404_for_missing_user():
    """Test that missing user returns 404"""
    analyzer = HealthAnalyzer()

    with patch.object(analyzer, 'df', None):
        result = analyzer.get_user_profile("INVALID_USER")
        assert result is None

def test_chat_endpoint_returns_400_for_empty_message(client):
    """Test API validation rejects empty messages"""
    response = client.post('/api/chat', json={
        'message': '',
        'user_id': 'USER_00001'
    })
    assert response.status_code == 400
```

**Test patterns for each service:**

**AIService (ai_service.py) - Mock External APIs:**
- Test fallback response when no API keys present
- Test message building with health context
- Test error handling for API failures
- Validate system prompt inclusion

**HealthAnalyzer (health_analyzer.py) - Mock Data Loading:**
- Test trend calculation logic
- Test issue identification (sleep, recovery, health risks)
- Test metric averaging and aggregation
- Validate data filtering by user/date range

**WeatherService (weather_service.py) - Mock HTTP Calls:**
- Test API response parsing
- Test mock weather generation (fallback)
- Test health impact calculation
- Test hydration/calorie adjustments

**FastAPI Endpoints (main.py) - Integration:**
- Test all GET/POST endpoints return correct status codes
- Test request validation via Pydantic
- Test conversation history management
- Test CORS headers present

### Test Data/Fixtures

**Test data location:** Would be in `backend/tests/fixtures/` (when tests added)

**Pattern:**
```python
# backend/tests/fixtures.py
import pytest

@pytest.fixture
def sample_health_context():
    """Sample health data for testing"""
    return {
        "profile": {
            "user_id": "TEST_USER_001",
            "age": 30,
            "gender": "Male",
            "weight_kg": 75.0,
            "height_cm": 180.0,
            "fitness_level": "Intermediate"
        },
        "recent_metrics": {
            "avg_recovery_score": 75.5,
            "avg_sleep_hours": 7.2,
            "avg_hrv": 45.3,
            "trend_recovery": "improving"
        }
    }

@pytest.fixture
def sample_chat_request():
    """Sample chat request"""
    return {
        "message": "I'm feeling tired",
        "user_id": "TEST_USER_001",
        "include_weather": True,
        "location": "New York"
    }
```

### Coverage

**Requirements:** Not enforced - No coverage tools configured

**When adding tests:**
- Target minimum 70% coverage for backend services
- Priority: Service classes (AIService, HealthAnalyzer, WeatherService)
- API endpoints secondary (rely on service tests)
- Non-critical: Config loading, singleton initialization

**View Coverage (when configured):**
```bash
pytest --cov=backend --cov-report=html
# View htmlcov/index.html in browser
```

---

## Implementation Roadmap

When adding tests to this codebase:

1. **Install testing framework:**
   ```
   pytest>=7.0.0
   pytest-asyncio>=0.21.0
   pytest-cov>=4.0.0
   pytest-mock>=3.10.0
   ```

2. **Create test structure:**
   ```
   backend/
   └── tests/
       ├── __init__.py
       ├── conftest.py           # Shared fixtures
       ├── fixtures.py           # Test data
       ├── test_models.py
       ├── test_ai_service.py
       ├── test_health_analyzer.py
       ├── test_weather_service.py
       └── test_main.py
   ```

3. **Add test configuration (pytest.ini):**
   ```ini
   [pytest]
   asyncio_mode = auto
   testpaths = backend/tests
   python_files = test_*.py
   ```

4. **Start with integration tests** for main endpoints before unit tests

5. **Add pre-commit hook** to run tests before commits

---

*Testing analysis: 2026-01-25*
