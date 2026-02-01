#!/usr/bin/env bash
# Run WellnessAI backend with local LLM (TinyLlama or BioGPT).
# Usage: ./scripts/run_with_local_llm.sh
#        WELLNESS_LLM_MODEL=biogpt ./scripts/run_with_local_llm.sh

set -e
cd "$(dirname "$0")/.."

export USE_LOCAL_LLM=true
export WELLNESS_LLM_MODEL=${WELLNESS_LLM_MODEL:-tinyllama}

echo "Starting WellnessAI with local LLM: $WELLNESS_LLM_MODEL"
exec python run.py
