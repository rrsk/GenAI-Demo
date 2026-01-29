# WellnessAI - Personal Health & Nutrition Assistant

An AI-powered health and meal planning assistant that uses Whoop biometric data to provide personalized nutrition recommendations, predictive health analytics, and actionable insights.

**Now with Local LLM Support** - Runs entirely offline using HuggingFace Transformers (TinyLlama, BioGPT, Phi-2)!

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green?style=for-the-badge)
![BioGPT](https://img.shields.io/badge/BioGPT-Medical_LLM-red?style=for-the-badge)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-orange?style=for-the-badge)

## Key GenAI Concepts Demonstrated

This project showcases several important GenAI/ML concepts for MTech coursework:

| Concept | Implementation |
|---------|----------------|
| **Domain-Specific LLM** | BioGPT - Medical LLM trained on PubMed biomedical literature |
| **Knowledge Distillation** | Smaller specialized model outperforming larger general models |
| **Transfer Learning** | Biomedical knowledge applied to health recommendations |
| **Prompt Engineering** | Medical context framing for clinical-style responses |
| **ML Predictions** | Scikit-learn models for recovery, strain, and risk prediction |
| **Recommendation Engine** | Rule-based expert system + ML integration |
| **Data Pipeline** | Feature engineering from Whoop biometric data |
| **API Integration** | OpenWeather for environmental context |

## Features

### Core Capabilities

| Feature | Description |
|---------|-------------|
| **Local AI Chat** | Natural language health Q&A powered by local LLM (no API needed) |
| **Recommendation Engine** | Python-based health recommendations with domain knowledge |
| **Personalized Meal Plans** | Daily/weekly nutrition plans based on your biometrics |
| **ML Predictions** | Predict tomorrow's recovery score and optimal strain |
| **Health Risk Analysis** | Early warning system for health issues |
| **Interactive Dashboard** | Real-time charts and trend visualization |
| **Weather Integration** | Nutrition adjustments based on weather conditions |

### Health Metrics Analyzed

- **Recovery Score** (0-100%): Daily readiness indicator
- **HRV** (Heart Rate Variability): Stress and recovery marker
- **Sleep Analysis**: Duration, efficiency, deep/REM sleep
- **Day Strain** (0-21): Activity and exertion level
- **Resting Heart Rate**: Cardiovascular health indicator
- **Skin Temperature**: Illness detection

## Quick Start

### Prerequisites

- **Python 3.9+** (Recommended: Python 3.11)
- **pip** (Python package manager)
- **~4GB disk space** (for local LLM model)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd "GenAI Demo"
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # Create virtual environment
   python3 -m venv venv
   
   # Activate it
   # On macOS/Linux:
   source venv/bin/activate
   
   # On Windows:
   venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the LLM model (first time only):**
   ```bash
   # Download default model (TinyLlama - best for chat, ~2GB)
   python scripts/download_model.py tinyllama
   
   # Or see all available models:
   python scripts/download_model.py --list
   
   # Download medical domain model (for technical health terminology):
   python scripts/download_model.py biogpt
   ```

5. **Configure environment variables (optional):**
   ```bash
   # Copy the example file
   cp .env.example .env
   
   # Add OpenWeather API key for weather features
   # OPENWEATHER_API_KEY=your_key_here
   ```

6. **Run the application:**
   ```bash
   python run.py
   ```

7. **Open in browser:**
   Navigate to [http://localhost:8000](http://localhost:8000)

## Local LLM Models

### Medical Domain Models (Recommended)

These models are trained on biomedical literature and provide better understanding of health metrics:

| Model | Size | RAM | Speed | Domain Knowledge | Best For |
|-------|------|-----|-------|------------------|----------|
| `biogpt` | 3GB | 6GB | Medium | Excellent | Default, health apps |
| `biogpt-medtext` | 661MB | 3GB | Fast | Very Good | Low-resource, demos |
| `biomedlm` | 5GB | 8GB | Slow | Best | Highest accuracy |

### General Purpose Models

| Model | Size | RAM | Speed | Quality | Best For |
|-------|------|-----|-------|---------|----------|
| `tinyllama` | 2GB | 4GB | Fast | Good | General chat |
| `phi2` | 5GB | 8GB | Medium | Better | Quality responses |
| `flan-t5` | 1GB | 3GB | Fast | Good | Low-resource |
| `mistral` | 15GB | 16GB+ | Slow | Best | GPU workstations |

```bash
# Download the default medical model (BioGPT)
python scripts/download_model.py biogpt

# Or download a lightweight medical model
python scripts/download_model.py biogpt-medtext

# Set model via environment variable
export WELLNESS_LLM_MODEL=biogpt
python run.py
```

### Why BioGPT for Health Applications?

BioGPT is a domain-specific language model trained on 15 million PubMed abstracts. This gives it:

- **Medical Terminology Understanding**: Knows HRV, recovery scores, sleep stages
- **Evidence-Based Responses**: Trained on peer-reviewed medical literature
- **Health Context Awareness**: Better understanding of symptom-diet correlations
- **Smaller but Specialized**: 1.5B parameters outperforming larger general models on health tasks

## Project Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      WellnessAI Architecture                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│    User Query                                                    │
│         │                                                        │
│         ▼                                                        │
│    ┌─────────────────────┐                                       │
│    │   Intent Classifier  │ ◄── Recommendation Engine            │
│    └──────────┬──────────┘                                       │
│               │                                                  │
│         ┌─────┴─────┐                                           │
│         ▼           ▼                                           │
│    ┌─────────┐ ┌──────────────┐                                 │
│    │ Health  │ │    Weather   │                                 │
│    │ Context │ │    Context   │                                 │
│    └────┬────┘ └──────┬───────┘                                 │
│         │             │                                         │
│         └──────┬──────┘                                         │
│                ▼                                                 │
│    ┌──────────────────────────┐                                 │
│    │  Recommendation Engine   │ ◄── Python rules + ML models    │
│    │  - Meal Planning         │                                 │
│    │  - Activity Suggestions  │                                 │
│    │  - Health Alerts         │                                 │
│    └───────────┬──────────────┘                                 │
│                │                                                 │
│                ▼                                                 │
│    ┌──────────────────────────┐                                 │
│    │      Local LLM           │ ◄── HuggingFace Transformers    │
│    │  (TinyLlama/Phi-2/etc)   │                                 │
│    └───────────┬──────────────┘                                 │
│                │                                                 │
│                ▼                                                 │
│    Natural Language Response                                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
GenAI Demo/
├── backend/
│   ├── __init__.py              # Package marker
│   ├── main.py                  # FastAPI application & routes
│   ├── config.py                # Configuration & environment
│   ├── models.py                # Pydantic data models
│   ├── health_analyzer.py       # Whoop data analysis
│   ├── weather_service.py       # OpenWeather API integration
│   ├── ai_service.py            # AI orchestration layer
│   ├── local_llm_service.py     # HuggingFace LLM integration
│   ├── recommendation_engine.py # Python-based recommendations
│   └── ml_service.py            # Scikit-learn ML predictions
├── frontend/
│   ├── index.html               # Main HTML (Chat + Dashboard)
│   ├── styles.css               # Modern dark theme styles
│   └── app.js                   # Frontend JavaScript & Chart.js
├── scripts/
│   └── download_model.py        # Model download utility
├── models/                      # Downloaded LLM models (gitignored)
├── Whoop Data/
│   └── whoop_fitness_dataset_100k.csv
├── .env.example                 # Environment variables template
├── requirements.txt             # Python dependencies
├── run.py                       # Application entry point
└── README.md
```

## API Reference

### Chat & AI

| Endpoint | Method | Description |
|----------|--------|-------------|
| `POST /api/chat` | POST | Send message to AI assistant |
| `POST /api/chat/clear` | POST | Clear conversation history |
| `POST /api/meal-plan` | POST | Generate personalized meal plan |
| `GET /api/ai-status` | GET | Check AI service status |

### User Health Data

| Endpoint | Method | Description |
|----------|--------|-------------|
| `GET /api/users` | GET | List available user profiles |
| `GET /api/users/{user_id}/profile` | GET | Get user profile |
| `GET /api/users/{user_id}/health-context` | GET | Comprehensive health data |
| `GET /api/users/{user_id}/health-risks` | GET | Identified health risks |

### ML Predictions

| Endpoint | Method | Description |
|----------|--------|-------------|
| `GET /api/users/{user_id}/predictions/recovery` | GET | Predict tomorrow's recovery |
| `GET /api/users/{user_id}/predictions/strain` | GET | Predict optimal strain |
| `GET /api/users/{user_id}/predictions/risk` | GET | Health risk prediction |
| `GET /api/users/{user_id}/dashboard` | GET | All dashboard data combined |

### Charts & Weather

| Endpoint | Method | Description |
|----------|--------|-------------|
| `GET /api/users/{user_id}/charts/trends` | GET | Historical trend data |
| `GET /api/users/{user_id}/charts/correlations` | GET | Metric correlations |
| `GET /api/weather?location=city` | GET | Weather & health impacts |

## Usage Examples

### Chat with the AI (Local LLM)

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "I have frequent headaches. What dietary changes can help?",
    "user_id": "USER_00001",
    "include_weather": true,
    "location": "New York"
  }'
```

### Check AI Status

```bash
curl http://localhost:8000/api/ai-status
```

Response:
```json
{
  "service": "WellnessAI AI",
  "mode": "local_llm",
  "status": {
    "recommendation_engine": "active",
    "local_llm": "loaded",
    "local_llm_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "local_llm_device": "cpu"
  }
}
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENWEATHER_API_KEY` | For weather | - | OpenWeather API key |
| `WELLNESS_LLM_MODEL` | No | `tinyllama` | Local LLM model choice |
| `USE_LOCAL_LLM` | No | `true` | Enable/disable local LLM |
| `PRELOAD_LLM` | No | `false` | Preload model on startup |
| `OPENAI_API_KEY` | No | - | Optional API fallback |
| `ANTHROPIC_API_KEY` | No | - | Optional API fallback |

## Development

### Running in Development Mode

```bash
# With auto-reload
python run.py

# Or directly with uvicorn
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### Testing the Local LLM

```bash
# Python REPL test
python -c "
from backend.local_llm_service import get_local_llm_service
llm = get_local_llm_service()
llm.load_model()
print(llm.generate_response('What should I eat for energy?', 'Recovery: 75%', 'Eat protein-rich foods'))
"
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `OutOfMemoryError` | Use a smaller model: `export WELLNESS_LLM_MODEL=flan-t5` |
| Model download fails | Check internet connection; try `python scripts/download_model.py` again |
| Slow first response | Model loads on first query; set `PRELOAD_LLM=true` for faster first response |
| `ModuleNotFoundError: torch` | Run `pip install torch transformers accelerate` |

## Technology Stack

| Layer | Technology |
|-------|------------|
| Backend | Python 3.9+, FastAPI, Pydantic |
| Local AI | HuggingFace Transformers, PyTorch |
| ML | Scikit-learn (RandomForest, GradientBoosting) |
| Data | Pandas, NumPy |
| Frontend | HTML5, CSS3, JavaScript, Chart.js |

## Contributing

1. Create a feature branch: `git checkout -b feature/my-feature`
2. Make your changes
3. Test locally: `python run.py`
4. Commit: `git commit -m "Add my feature"`
5. Push: `git push origin feature/my-feature`
6. Open a Pull Request

## License

MIT License - See LICENSE file for details.

---

**Built for GenAI MTech coursework** - Demonstrating local LLM deployment, recommendation systems, and health data analysis.
