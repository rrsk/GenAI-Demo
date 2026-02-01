"""
WellnessAI - Streamlit App
Deploy-friendly UI: works with FastAPI backend (API mode) or standalone (in-process).
Set API_BASE_URL to point at your deployed backend, or leave unset for standalone.
"""
import os
import sys
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd

API_BASE_URL = os.environ.get("API_BASE_URL", "").rstrip("/")
USE_API = bool(API_BASE_URL)
HF_TOKEN = os.environ.get("HF_TOKEN", "")  # Hugging Face API token (Spaces / Inference API)
HF_CHAT_MODEL = os.environ.get("HF_CHAT_MODEL", "HuggingFaceH4/zephyr-7b-beta")  # or microsoft/DialoGPT-medium

# Page config
st.set_page_config(
    page_title="WellnessAI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom style
st.markdown("""
<style>
    .stApp { background: linear-gradient(180deg, #0f1612 0%, #1a2a1f 100%); }
    .metric-card { background: rgba(26, 42, 31, 0.8); border-radius: 12px; padding: 1rem; border-left: 4px solid #4ade80; }
    h1, h2, h3 { color: #f0fdf4 !important; }
    .stChatMessage { background: #132019 !important; }
</style>
""", unsafe_allow_html=True)


def get_users_via_api():
    import httpx
    r = httpx.get(f"{API_BASE_URL}/api/users", timeout=10)
    r.raise_for_status()
    return r.json().get("users", ["USER_00001"])


def get_dashboard_via_api(user_id: str, days: int):
    import httpx
    r = httpx.get(f"{API_BASE_URL}/api/users/{user_id}/dashboard", params={"days": days}, timeout=30)
    r.raise_for_status()
    return r.json()


def chat_via_api(user_id: str, message: str, location: str = "New York"):
    import httpx
    r = httpx.post(
        f"{API_BASE_URL}/api/chat",
        json={"message": message, "user_id": user_id, "include_weather": True, "location": location},
        timeout=60,
    )
    r.raise_for_status()
    return r.json().get("response", "")


def get_users_standalone():
    from backend.health_analyzer import get_health_analyzer
    return get_health_analyzer().get_all_user_ids()


def get_dashboard_standalone(user_id: str, days: int):
    from backend.health_analyzer import get_health_analyzer
    from backend.ml_service import get_ml_service
    analyzer = get_health_analyzer()
    ml = get_ml_service()
    return {
        "user_id": user_id,
        "profile": analyzer.get_user_profile(user_id),
        "recent_metrics": analyzer.get_recent_health_metrics(user_id, 7),
        "trends": ml.get_trend_data(user_id, days),
        "predictions": {
            "recovery": ml.predict_recovery(user_id),
            "strain": ml.predict_optimal_strain(user_id),
            "risk": ml.predict_health_risk(user_id),
        },
        "correlations": ml.get_correlation_insights(user_id),
        "health_risks": analyzer.identify_health_risks(user_id),
    }


def _chat_via_hf_inference_api(user_id: str, message: str, context_text: str) -> str:
    """Use Hugging Face Inference API for chat (e.g. on HF Spaces)."""
    import httpx
    if not HF_TOKEN:
        return ""
    prompt = f"""You are WellnessAI, a health and nutrition advisor. Use this context to give a short, helpful reply.

Context:
{context_text[:3000]}

User: {message}
Assistant:"""
    try:
        r = httpx.post(
            f"https://api-inference.huggingface.co/models/{HF_CHAT_MODEL}",
            headers={"Authorization": f"Bearer {HF_TOKEN}"},
            json={"inputs": prompt, "parameters": {"max_new_tokens": 256, "temperature": 0.7}},
            timeout=30,
        )
        if r.status_code != 200:
            return ""
        out = r.json()
        if isinstance(out, list) and len(out) and "generated_text" in out[0]:
            return out[0]["generated_text"].replace(prompt, "").strip()
        if isinstance(out, dict) and "generated_text" in out:
            return out["generated_text"].replace(prompt, "").strip()
        return ""
    except Exception:
        return ""


async def chat_standalone(user_id: str, message: str, location: str = "New York"):
    from backend.recommendation_engine import get_recommendation_engine
    from backend.health_analyzer import get_health_analyzer
    from backend.weather_service import get_weather_service
    analyzer = get_health_analyzer()
    engine = get_recommendation_engine()
    health_context = analyzer.get_comprehensive_health_context(user_id)
    context_text = engine.format_health_context_for_llm(health_context)
    weather_data = None
    if location:
        weather_svc = get_weather_service()
        w = await weather_svc.get_current_weather(location)
        if w:
            weather_data = weather_svc.get_weather_health_impacts(w)
            if weather_data and weather_data.get("weather_summary"):
                context_text += f"\n\nWeather: {weather_data['weather_summary'].get('temperature', 'N/A')}°C, {weather_data['weather_summary'].get('condition', '')}"
    # Prefer Hugging Face Inference API when token is set (e.g. on HF Spaces)
    if HF_TOKEN:
        reply = _chat_via_hf_inference_api(user_id, message, context_text)
        if reply:
            return reply
    # Fallback: rule-based + optional OpenAI/Anthropic via backend AI service
    from backend.ai_service import get_ai_service
    ai = get_ai_service(use_local_llm=False)
    result = await ai.generate_response(
        user_message=message,
        health_context=health_context,
        weather_data=weather_data,
        user_id=user_id,
    )
    return result.get("message", "")


def run_chat_standalone(user_id: str, message: str, location: str):
    import asyncio
    return asyncio.run(chat_standalone(user_id, message, location))


# Sidebar
with st.sidebar:
    st.title("🌿 WellnessAI")
    st.caption("Health assistant with LSTM forecasts")
    if USE_API:
        st.success("API mode")
        st.caption(f"Backend: {API_BASE_URL[:40]}...")
    else:
        st.info("Standalone mode")
    if HF_TOKEN and not USE_API:
        st.caption("🤗 Chat: Hugging Face Inference API")
    st.divider()
    try:
        users = get_users_via_api() if USE_API else get_users_standalone()
    except Exception as e:
        st.error(f"Could not load users: {e}")
        users = ["USER_00001"]
    user_id = st.selectbox("User", users, key="user_select")
    days = st.selectbox("Chart days", [7, 14, 30], index=2, key="days_select")
    st.divider()
    location = st.text_input("Location (for chat)", "New York", key="location")

# Main
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "💬 Chat", "ℹ️ Deploy"])

with tab1:
    st.header("Dashboard")
    try:
        data = get_dashboard_via_api(user_id, days) if USE_API else get_dashboard_standalone(user_id, days)
    except Exception as e:
        st.error(f"Failed to load dashboard: {e}")
        st.stop()
    profile = data.get("profile") or {}
    metrics = data.get("recent_metrics") or {}
    trends = data.get("trends") or {}
    predictions = data.get("predictions") or {}
    correlations = data.get("correlations") or {}
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Recovery %", metrics.get("avg_recovery_score", "—"))
    with col2:
        st.metric("Sleep (hrs)", metrics.get("avg_sleep_hours", "—"))
    with col3:
        st.metric("HRV (ms)", metrics.get("avg_hrv", "—"))
    with col4:
        st.metric("Strain", metrics.get("avg_strain", "—"))
    st.subheader("Predictions")
    pred_rec = predictions.get("recovery") or {}
    pred_strain = predictions.get("strain") or {}
    st.write(f"**Next recovery:** {pred_rec.get('predicted_score', '—')}% — {pred_rec.get('recommendation', '')}")
    st.write(f"**Optimal strain:** {pred_strain.get('recommended_strain', '—')} — {pred_strain.get('recommendation', '')}")
    if trends.get("dates"):
        st.subheader("Trends & LSTM Forecast")
        df_trend = pd.DataFrame({
            "date": trends["dates"],
            "recovery": trends.get("recovery", []),
            "strain": trends.get("strain", []),
            "hrv": trends.get("hrv", []),
            "calories": trends.get("calories", []),
        })
        forecast = trends.get("lstm_forecast")
        if forecast and forecast.get("dates"):
            df_forecast = pd.DataFrame({
                "date": forecast["dates"],
                "recovery": forecast.get("recovery", []),
                "strain": forecast.get("strain", []),
                "hrv": forecast.get("hrv", []),
                "calories": forecast.get("calories", []),
            })
            df_forecast["date"] = "🔮 " + df_forecast["date"].astype(str)
            df_combined = pd.concat([df_trend, df_forecast], ignore_index=True)
        else:
            df_combined = df_trend
        if len(df_combined) > 0:
            cols = [c for c in ["recovery", "strain"] if c in df_combined.columns]
            if cols:
                st.line_chart(df_combined.set_index("date")[cols], height=300)
            if "hrv" in df_combined.columns:
                st.line_chart(df_combined.set_index("date")[["hrv"]], height=250)
    insights = correlations.get("insights", [])
    if insights:
        st.subheader("Insights")
        for i in insights[:5]:
            st.markdown(f"- **{i.get('title', '')}** — {i.get('description', '')}")

with tab2:
    st.header("Chat")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    if prompt := st.chat_input("Ask about meals, sleep, recovery..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        response = ""
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    if USE_API:
                        response = chat_via_api(user_id, prompt, location)
                    else:
                        response = run_chat_standalone(user_id, prompt, location)
                except Exception as e:
                    response = str(e)
                st.markdown(response or "No response.")
        st.session_state.messages.append({"role": "assistant", "content": response or "No response."})
        st.rerun()
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

with tab3:
    st.header("Deploy")
    st.markdown("""
    **Run locally (Streamlit):**
    ```bash
    pip install streamlit
    streamlit run streamlit_app.py
    ```
    **Run with backend (API mode):** set env and run Streamlit:
    ```bash
    export API_BASE_URL=http://localhost:8000
    streamlit run streamlit_app.py
    ```
    **Deploy on Streamlit Community Cloud:**  
    Push to GitHub → [share.streamlit.io](https://share.streamlit.io) → Main file: `streamlit_app.py`. Set **API_BASE_URL** for API mode, or use standalone + **HF_TOKEN** for Hugging Face Inference API chat.

    **Deploy on Hugging Face Spaces (Streamlit + HF API):**
    1. Create a Space at [huggingface.co/new-space](https://huggingface.co/new-space), SDK **Streamlit** (or Docker + Streamlit template).
    2. Add `streamlit_app.py` as `app.py`, plus `backend/`, data, and `requirements.txt`.
    3. In Space **Settings → Repository secrets**, add **HF_TOKEN** (your Hugging Face token). Chat will use the [Hugging Face Inference API](https://huggingface.co/inference-api).
    4. Optional: **HF_CHAT_MODEL** (e.g. `HuggingFaceH4/zephyr-7b-beta`), **WEATHERAPI_KEY**.

    **Deploy FastAPI backend** (for API mode):
    - **Railway:** Connect repo, set start command `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`.
    - **Render:** New Web Service, build command `pip install -r requirements.txt`, start `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`.
    - **Fly.io:** Use a Dockerfile or `fly launch` with a Procfile.
    """)
