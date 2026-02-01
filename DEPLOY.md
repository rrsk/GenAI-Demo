# Deploying WellnessAI

You can run the app locally or deploy it. Two UIs are available:

- **FastAPI + HTML/JS** — `run.py` (backend) + `frontend/` (served at `/`)
- **Streamlit** — `streamlit run streamlit_app.py` (single entrypoint, deploy-friendly)

Streamlit works well with **Hugging Face Spaces** and the **Hugging Face Inference API** (see Option 1b).

---

## Option 1a: Deploy on Streamlit (Community Cloud)

Easiest way to get a public URL.

### 1. Push repo to GitHub

Ensure the repo has:

- `streamlit_app.py` (root)
- `requirements.txt`
- `backend/`
- `Whoop Data/whoop_fitness_dataset_100k.csv` (or use API mode; see below)

### 2. Connect to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io).
2. Sign in with GitHub and **New app**.
3. **Repository:** `your-username/GenAI-Demo` (or your repo name).
4. **Branch:** `main`.
5. **Main file path:** `streamlit_app.py`.
6. **Advanced:** add any env vars (see below).

### 3. Two run modes

**A) Standalone (no separate backend)**  
- Leave **API_BASE_URL** unset.  
- Streamlit runs everything in-process (Whoop data, ML, LSTM, rule-based chat).  
- Chat does **not** use the local LLM (too heavy for free tier).  
- Ensure `Whoop Data/whoop_fitness_dataset_100k.csv` is in the repo (or add it via [Streamlit’s “Include data”](https://docs.streamlit.io/streamlit-community-cloud/get-started/share-your-app#add-your-repo-to-streamlit-cloud) / Git LFS).

**B) API mode (backend elsewhere)**  
- Deploy the FastAPI backend first (e.g. Railway or Render).  
- In Streamlit Cloud, set **API_BASE_URL** = `https://your-backend.up.railway.app` (no trailing slash).  
- Streamlit only does UI and calls your API; no Whoop data or ML needed in the Streamlit app.

### 4. Optional secrets (Streamlit Cloud)

In **Settings → Secrets** (or `.streamlit/secrets.toml` in repo):

```toml
WEATHERAPI_KEY = "your-key"
OPENAI_API_KEY = "optional-for-chat"
```

---

## Option 1b: Deploy on Hugging Face Spaces (Streamlit + HF API)

Hugging Face Spaces supports Streamlit and the **Hugging Face Inference API**, so you can use HF models for chat without running a local LLM.

### 1. Create a Space

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space).
2. Pick **Streamlit** as the SDK (or **Docker** and use the Streamlit template if Streamlit SDK is deprecated in your region).
3. Create the Space (e.g. `your-username/wellness-ai`).

### 2. Add your app

- **If using Streamlit SDK:** Your Space expects an `app.py` at the root. Either:
  - Rename/copy `streamlit_app.py` to `app.py` in the Space repo, or
  - In the Space repo, add `app.py` that only contains: `import streamlit_app` (and expose the same UI), or
  - Clone this repo into the Space and set the Space’s **App file** to `streamlit_app.py` if the UI allows it.
- **If using Docker:** Use a Dockerfile that runs `streamlit run streamlit_app.py` (see Option 4). Put `streamlit_app.py`, `backend/`, data, and `requirements.txt` in the image.

Easiest: create a **new** HF Space repo, copy `streamlit_app.py` → `app.py`, copy `backend/`, `Whoop Data/` (or a sample CSV), and a `requirements.txt` that includes `streamlit`, `pandas`, `scikit-learn`, `torch`, `joblib`, `httpx`, etc.

### 3. Use Hugging Face Inference API for chat (optional)

Set the **HF_TOKEN** secret in your Space (Settings → Repository secrets). The Streamlit app can then call the [Hugging Face Inference API](https://huggingface.co/inference-api) for text generation instead of a local model:

- In the app, when **HF_TOKEN** is set and no local LLM is used, call `https://api-inference.huggingface.co/models/<model_id>` with your token for chat responses.
- This keeps the Space lightweight (no GPU needed for inference) and uses HF’s API.

### 4. Port and secrets

- Use the default **port 8501** for Streamlit; do not override it in config.
- Add secrets: **HF_TOKEN**, optionally **WEATHERAPI_KEY**, **OPENAI_API_KEY**.

---

## Option 2: Run Streamlit locally

```bash
# Install
pip install -r requirements.txt

# Run (standalone: no backend)
streamlit run streamlit_app.py

# Or with backend (start backend first in another terminal)
# Terminal 1:
python run.py
# Terminal 2:
export API_BASE_URL=http://localhost:8000
streamlit run streamlit_app.py
```

Open the URL shown (usually `http://localhost:8501`).

---

## Option 3: Deploy FastAPI backend (for API mode)

Use this when you want the HTML/JS frontend or when Streamlit is in API mode.

### Railway

1. New Project → Deploy from GitHub (this repo).
2. **Root directory:** repo root.
3. **Build command:** `pip install -r requirements.txt` (or leave default).
4. **Start command:** `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
5. Add env vars in **Variables** (e.g. `WEATHERAPI_KEY`).  
6. Deploy; use the generated URL as **API_BASE_URL** in Streamlit.

### Render

1. New → Web Service; connect this repo.
2. **Build command:** `pip install -r requirements.txt`
3. **Start command:** `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
4. Set env vars in **Environment**.  
5. Use the service URL as **API_BASE_URL** in Streamlit.

### Fly.io

```bash
fly launch
# Set in fly.toml: cmd = "uvicorn backend.main:app --host 0.0.0.0 --port 8080"
fly deploy
```

Use the Fly URL as **API_BASE_URL**.

---

## Option 4: Docker (single image)

Example **Dockerfile** (backend + Streamlit in one image; run one or the other via CMD):

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Default: run Streamlit (standalone)
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

To run FastAPI instead:

```bash
docker run -p 8000:8000 your-image uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

---

## Summary

| Goal                         | Action                                                                 |
|-----------------------------|------------------------------------------------------------------------|
| Public app, minimal setup   | Deploy **Streamlit** on Streamlit Cloud (standalone or API mode).    |
| Local Streamlit             | `streamlit run streamlit_app.py` (optional: `API_BASE_URL` + backend). |
| Public API + HTML frontend  | Deploy **FastAPI** (Railway/Render/Fly), then open the backend URL.    |
| One container               | Use **Docker** with CMD for either Streamlit or FastAPI.               |
