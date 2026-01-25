# Codebase Concerns

**Analysis Date:** 2026-01-25

## Security Considerations

**CORS Configuration Too Permissive:**
- Risk: Any website can make requests to API
- Files: `backend/main.py` (lines 28-34)
- Current: `allow_origins=["*"]`, `allow_credentials=True`
- Fix: Whitelist specific frontend domain

**No Authentication:**
- Risk: All endpoints accessible without authentication
- Files: `backend/main.py` (all endpoints)
- Current: User identification via user_id parameter only
- Recommendation: Implement JWT tokens or session management

**Missing Input Validation:**
- Risk: User messages passed directly to AI without sanitization
- Files: `backend/ai_service.py`, `backend/main.py`
- Recommendation: Add message length limits, detect injection attempts

## Performance Bottlenecks

**CSV Data Loading:**
- Problem: 100k-record CSV loaded entirely into memory on startup
- Files: `backend/health_analyzer.py` (lines 16-26)
- Recommendation: Migrate to database with proper indexing

**In-Memory Conversation Storage:**
- Problem: Unbounded conversation history in memory
- Files: `backend/main.py` (lines 36-37)
- Limit: Server memory exhaustion with many users
- Recommendation: Implement database persistence

**AI API Fallback Chain:**
- Problem: If OpenAI fails, tries Anthropic, then fallback - adds latency
- Files: `backend/ai_service.py` (lines 142-174)
- Recommendation: Implement timeout-based fallback

## Tech Debt

**No Test Suite:**
- Issue: Zero test coverage
- Impact: Cannot detect regressions safely
- Priority: High

**Print-Based Logging:**
- Issue: Uses `print()` instead of structured logging
- Files: `backend/ai_service.py`, `backend/weather_service.py`
- Recommendation: Implement Python logging module

**Hardcoded Default User:**
- Issue: Default user "USER_00001" throughout codebase
- Files: `backend/main.py`, `frontend/app.js`
- Recommendation: Implement proper user management

**Session State in Memory Only:**
- Issue: Conversation history lost on server restart
- Recommendation: Add database persistence

## Fragile Areas

**Health Analyzer Data Dependency:**
- Files: `backend/health_analyzer.py`
- Risk: App crashes if Whoop CSV is missing or corrupted
- Recommendation: Add graceful degradation

**Weather API Fallback:**
- Files: `backend/weather_service.py`
- Risk: Falls back to mock data silently - users get fake weather
- Recommendation: Indicate mock vs real data in response

**AI Response Parsing:**
- Files: `backend/ai_service.py`
- Risk: Assumes specific message format from APIs
- Recommendation: Add defensive checks

## Missing Critical Features

- **Authentication & Authorization**
- **Error Recovery Mechanisms** (retry logic, circuit breaker)
- **Data Persistence** (database storage)
- **Rate Limiting**
- **Proper Logging**

## Test Coverage Gaps

- All functionality untested
- No test infrastructure
- Priority areas: HealthAnalyzer, AIService, WeatherService

---

*Concerns audit: 2026-01-25*
