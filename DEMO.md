# WellnessAI - Demo Guide & Presentation Script

## Project Overview

**WellnessAI** is a GenAI-powered health and nutrition assistant that demonstrates key concepts in Generative AI, Machine Learning, and modern software architecture.

### What This Project Demonstrates

| GenAI Concept | How It's Implemented |
|---------------|---------------------|
| **Local LLM Deployment** | HuggingFace Transformers running TinyLlama/BioGPT locally without API dependencies |
| **Domain-Specific Models** | BioGPT trained on 15M PubMed abstracts for medical understanding |
| **Prompt Engineering** | Structured prompts with health context, recommendations, and user queries |
| **ML Predictions** | Scikit-learn RandomForest & GradientBoosting for health predictions |
| **Recommendation Systems** | Rule-based expert system combined with ML predictions |
| **Data Pipeline** | Feature engineering from 100K Whoop biometric records |
| **Context Augmentation** | Weather data integrated into health recommendations |

---

## Quick Demo Setup

```bash
# 1. Activate environment
cd "GenAI Demo"
source venv/bin/activate

# 2. Verify model is downloaded
python scripts/download_model.py --list

# 3. Start the server (with TinyLlama for best chat experience)
WELLNESS_LLM_MODEL=tinyllama python run.py

# 4. Open browser
open http://localhost:8000
```

---

## Demo Script (5-Minute Presentation)

### Part 1: Introduction (30 seconds)

> "This is WellnessAI - a personal health assistant that runs entirely locally using a downloaded language model. Unlike ChatGPT or other cloud APIs, this runs on your own machine with no internet dependency for AI inference."

**Show**: The running application in the browser

---

### Part 2: Health Data Analysis (1 minute)

> "The system ingests biometric data from Whoop fitness trackers - 100,000 records including recovery scores, HRV, sleep metrics, and activity strain."

**Demo Actions**:
1. Point to the sidebar showing **Today's Snapshot**
2. Explain each metric:
   - **Energy (Recovery %)**: How recovered the body is (77%)
   - **Sleep**: Hours of sleep (6.0 hrs - needs attention)
   - **Stress (HRV)**: Heart rate variability (112ms - very relaxed)
   - **Activity (Strain)**: Exertion level (8.2 pts)

> "Notice the system automatically detected a health alert: Sleep Deprivation. This is ML-driven risk analysis."

---

### Part 3: Local LLM Chat (2 minutes)

> "Now let me demonstrate the GenAI capabilities. I'll ask the AI for health advice."

**Demo Prompts** (copy and paste these):

#### Prompt 1: Headache & Nutrition
```
I've been having headaches lately and feel tired. What foods should I eat to feel better?
```

**Expected Response**: The LLM will generate personalized nutrition advice including:
- Berries for antioxidants
- Leafy greens for magnesium
- Nuts and seeds for zinc
- Hydration recommendations

#### Prompt 2: Sleep Improvement
```
My sleep quality is poor. What dietary changes can help me sleep better?
```

**Expected Response**: Evidence-based sleep nutrition advice

#### Prompt 3: HRV Understanding
```
What is HRV and how can I improve mine through diet and lifestyle?
```

**Expected Response**: Medical explanation of heart rate variability

> "Notice these responses are generated in real-time by a local language model - TinyLlama with 1.1 billion parameters, running entirely on CPU."

---

### Part 4: Dashboard & ML Predictions (1 minute)

> "Let me show you the predictive ML capabilities."

**Demo Actions**:
1. Click **Dashboard** view
2. Point to **Tomorrow's Outlook**: "This uses a RandomForestRegressor to predict tomorrow's recovery score"
3. Show **Activity Recommendations**: "The system suggests appropriate activities based on recovery prediction"
4. Expand a chart: "Historical trends are visualized with Chart.js"

> "The ML models are trained on the Whoop dataset using scikit-learn. We use:
> - RandomForestRegressor for recovery prediction
> - GradientBoostingClassifier for health risk classification"

---

### Part 5: Architecture Explanation (30 seconds)

> "The architecture follows a pipeline approach..."

**Show the AI Status endpoint**:
```bash
curl http://localhost:8000/api/ai-status | python -m json.tool
```

> "1. User query comes in
> 2. Intent classification determines what type of request (meal plan, headache, sleep, etc.)
> 3. Health context is extracted from the user's Whoop data
> 4. Recommendation engine generates structured advice using rules + ML
> 5. Local LLM converts structured recommendations into natural language"

---

## Technical Deep Dive

### Key Files to Discuss

| File | Purpose | GenAI Concept |
|------|---------|---------------|
| `backend/local_llm_service.py` | Loads and runs HuggingFace models | Local LLM deployment |
| `backend/recommendation_engine.py` | Rule-based health recommendations | Expert systems |
| `backend/ai_service.py` | Orchestrates intent → context → LLM | Pipeline architecture |
| `backend/ml_service.py` | Scikit-learn prediction models | ML predictions |

### Prompt Engineering Example

```python
# From local_llm_service.py - TinyLlama chat format
prompt = f"""<|system|>
You are WellnessAI, a helpful health and nutrition assistant. 
You provide personalized advice based on biometric data from fitness wearables.
Be concise, practical, and supportive.</s>
<|user|>
## User's Health Data:
{health_context}

## Recommended Actions (from analysis):
{recommendations}

## User's Question:
{user_message}

Please provide a helpful, personalized response.</s>
<|assistant|>
"""
```

---

## Available Models Comparison

| Model | Parameters | Specialty | Response Time |
|-------|-----------|-----------|---------------|
| **TinyLlama** | 1.1B | General chat, conversational | ~10-15 sec |
| **BioGPT** | 390M | Medical terminology, PubMed-trained | ~8-12 sec |
| **Phi-2** | 2.7B | Reasoning, quality responses | ~30-45 sec |

**Demo switching models**:
```bash
# Stop current server (Ctrl+C)

# Start with BioGPT (medical model)
WELLNESS_LLM_MODEL=biogpt python run.py

# Start with TinyLlama (chat model - recommended for demo)
WELLNESS_LLM_MODEL=tinyllama python run.py
```

---

## API Demonstration

### Chat API
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What foods help with energy?",
    "user_id": "USER_00001",
    "include_weather": false
  }'
```

### Health Context API
```bash
curl http://localhost:8000/api/users/USER_00001/health-context | python -m json.tool
```

### ML Predictions API
```bash
curl http://localhost:8000/api/users/USER_00001/predictions/recovery | python -m json.tool
```

### AI Status API
```bash
curl http://localhost:8000/api/ai-status | python -m json.tool
```

---

## Key Talking Points

### 1. Why Local LLM?
- **Privacy**: Health data never leaves the device
- **No API costs**: No per-token charges
- **Offline capability**: Works without internet
- **Educational**: Demonstrates actual model deployment

### 2. Why Domain-Specific Models?
- **BioGPT** trained on 15M PubMed abstracts understands medical terminology
- Smaller specialized models can outperform larger general models on specific tasks
- Demonstrates **transfer learning** and **domain adaptation**

### 3. Hybrid Architecture
- **Rule-based engine**: Reliable, interpretable recommendations
- **ML predictions**: Data-driven insights
- **LLM generation**: Natural language interface
- Best of all worlds: reliability + intelligence + usability

---

## Troubleshooting During Demo

| Problem | Quick Fix |
|---------|-----------|
| Slow response | "The model is loading... first query takes longer" |
| Out of memory | `WELLNESS_LLM_MODEL=flan-t5 python run.py` (smallest model) |
| Server not responding | Check terminal for errors, restart with `python run.py` |

---

## Q&A Preparation

**Q: Why not use ChatGPT API?**
> "This demonstrates local deployment - a key skill for privacy-sensitive applications and understanding how LLMs actually work."

**Q: How accurate are the health recommendations?**
> "This is a proof-of-concept. The recommendation engine uses evidence-based rules, and the LLM is trained on medical literature, but it's not a replacement for professional medical advice."

**Q: Can this scale?**
> "For production, you'd use model quantization (INT8/INT4), GPU acceleration, or a serving framework like vLLM. This demo prioritizes simplicity and CPU compatibility."

**Q: What's the model size?**
> "TinyLlama: 1.1B parameters (~2GB), BioGPT: 390M parameters (~3GB). These are 'small' by LLM standards but still powerful for focused tasks."

---

## Files for Code Review

If asked to show code:

1. **LLM Loading**: `backend/local_llm_service.py` lines 104-220
2. **Prompt Engineering**: `backend/local_llm_service.py` lines 229-345
3. **Recommendation Engine**: `backend/recommendation_engine.py`
4. **ML Predictions**: `backend/ml_service.py`
5. **API Orchestration**: `backend/ai_service.py`

---

## Demo Checklist

- [ ] Virtual environment activated
- [ ] Model downloaded (`python scripts/download_model.py --list`)
- [ ] Server running (`python run.py`)
- [ ] Browser open at http://localhost:8000
- [ ] Terminal visible for API demos
- [ ] This DEMO.md open for reference

---

*Good luck with your presentation!*
