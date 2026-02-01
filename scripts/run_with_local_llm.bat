@echo off
REM Run WellnessAI backend with local LLM (TinyLlama or BioGPT).
REM Usage: scripts\run_with_local_llm.bat
REM        set WELLNESS_LLM_MODEL=biogpt && scripts\run_with_local_llm.bat

cd /d "%~dp0\.."
if not defined WELLNESS_LLM_MODEL set WELLNESS_LLM_MODEL=tinyllama
set USE_LOCAL_LLM=true

echo Starting WellnessAI with local LLM: %WELLNESS_LLM_MODEL%
python run.py
