# Running TinyLlama and BioGPT (Local LLM) – Setup Guide

This guide gets **TinyLlama** and **BioGPT** working for the WellnessAI chat so the full app (FastAPI + frontend or Streamlit) uses your local model with no external API.

**State-of-union (default):** BioGPT takes the user’s current stats and status and produces an **evidence-based prompt**; TinyLlama then uses that prompt plus the user’s query to give a **grounded, research-aware suggestion**. This keeps responses data- and research-grounded while staying conversational.

---

## 1. What You Need

### Hardware (CPU is enough; GPU is optional)

| Model        | RAM (CPU)   | GPU (optional) | Notes                    |
|-------------|-------------|----------------|---------------------------|
| **TinyLlama** | 4–6 GB      | —              | Fast, good for chat       |
| **BioGPT**    | 6–8 GB      | —              | Medical/PubMed terminology |
| **BioGPT-MedText** | ~4 GB | —              | Lighter BioGPT, very fast  |

- **Recommended:** 8 GB RAM for TinyLlama alone; **12 GB+ RAM** for **state-of-union** (TinyLlama + BioGPT both loaded).
- **GPU:** Not required. If you have a CUDA GPU (e.g. 6 GB+ VRAM), inference will be faster; the app will use it automatically.

### Where to run

- **Local:** Mac (Apple Silicon or Intel), Linux, or Windows with the RAM above.
- **Cloud (if you want TinyLlama/BioGPT in the cloud):**
  - **RunPod / Lambda / Vast.ai:** Rent a GPU or CPU VM, clone repo, install deps, run backend (see below). Use the VM’s public URL as `API_BASE_URL` in Streamlit if you use Streamlit elsewhere.
  - **Google Cloud / AWS / Azure:** Small CPU VM (e.g. 8 GB RAM) or GPU instance; same steps as local.

---

## 2. Quick Setup (Local)

### Step 1: Clone and venv

```bash
cd "GenAI Demo"
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 2: Download the model (one-time)

**TinyLlama (conversational, ~2 GB):**

```bash
python scripts/download_model.py tinyllama
```

**BioGPT (medical, ~3 GB):**

```bash
python scripts/download_model.py biogpt
```

**BioGPT-MedText (lighter, ~661 MB):**

```bash
python scripts/download_model.py biogpt-medtext
```

List all options:

```bash
python scripts/download_model.py --list
```

### Step 3: Configure environment

Copy and edit `.env`:

```bash
cp .env.example .env
```

In `.env` set:

```env
# Response model: TinyLlama (conversational final answer)
WELLNESS_LLM_MODEL=tinyllama
USE_LOCAL_LLM=true

# State-of-union: BioGPT grounds context, TinyLlama responds (recommended)
USE_STATE_OF_UNION=true
WELLNESS_GROUNDING_MODEL=biogpt

# Optional: load model at startup (slower start, faster first reply)
PRELOAD_LLM=false
```

With **state-of-union** on, you need both models: **BioGPT** (or `biogpt-medtext`) for grounding and **TinyLlama** for the final response. Download both: `python scripts/download_model.py biogpt` and `python scripts/download_model.py tinyllama`.

### Step 4: Run the backend (with local LLM)

**Option A – run script (recommended):**

```bash
./scripts/run_with_local_llm.sh
```

Or with a specific model:

```bash
WELLNESS_LLM_MODEL=biogpt ./scripts/run_with_local_llm.sh
```

**Option B – manual:**

```bash
export USE_LOCAL_LLM=true
export WELLNESS_LLM_MODEL=tinyllama
python run.py
```

Backend will be at **http://localhost:8000**. First chat may be slow while the model loads.

### Step 5: Open the app

- **HTML frontend:** Open **http://localhost:8000** in a browser.
- **Streamlit (same machine):** In another terminal:

  ```bash
  export API_BASE_URL=http://localhost:8000
  streamlit run streamlit_app.py
  ```

  Then open the URL Streamlit prints (e.g. http://localhost:8501). Streamlit will use the backend where TinyLlama/BioGPT are running.

---

## 3. Switching Between TinyLlama and BioGPT

- In **.env** set `WELLNESS_LLM_MODEL=tinyllama` or `WELLNESS_LLM_MODEL=biogpt` (or `biogpt-medtext`).
- Restart the backend (`python run.py` or `./scripts/run_with_local_llm.sh`).
- No need to change frontend or Streamlit; they already use the backend.

---

## 4. Optional: Preload model at startup

To avoid a slow first message, load the model when the server starts:

In `.env`:

```env
PRELOAD_LLM=true
```

Startup will take longer; first chat will be fast.

---

## 5. Optional: Docker (consistent environment)

A Dockerfile is provided so TinyLlama/BioGPT run in a fixed environment (see `Dockerfile.local-llm`). Build and run:

```bash
docker build -f Dockerfile.local-llm -t wellness-ai-llm .
docker run --rm -p 8000:8000 -e WELLNESS_LLM_MODEL=tinyllama wellness-ai-llm
```

Use `WELLNESS_LLM_MODEL=biogpt` in `-e` to use BioGPT. Ensure the machine has enough RAM (e.g. 8 GB for TinyLlama, 12 GB for BioGPT).

---

## 6. Troubleshooting

| Issue | What to do |
|-------|------------|
| Out of memory | Use `biogpt-medtext` or `tinyllama`; close other apps; or use a machine with more RAM. |
| First message very slow | Normal: model loads on first use. Set `PRELOAD_LLM=true` to load at startup. |
| “PyTorch not installed” | Run `pip install torch transformers accelerate`. |
| Model not found | Run `python scripts/download_model.py tinyllama` (or `biogpt`) once. |
| Streamlit doesn’t use local LLM | Run the **backend** with `USE_LOCAL_LLM=true` and point Streamlit at it with `API_BASE_URL=http://localhost:8000`. |

---

## Summary

- **TinyLlama:** best for general chat, low RAM, CPU-only.
- **BioGPT / BioGPT-MedText:** best for medical/health terminology.
- Use **LOCAL_LLM_SETUP.md** + **run_with_local_llm.sh** + optional **Dockerfile.local-llm** for a repeatable setup where TinyLlama and BioGPT work fine.
