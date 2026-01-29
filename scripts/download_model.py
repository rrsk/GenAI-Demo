#!/usr/bin/env python3
"""
Model Download Script for WellnessAI

This script downloads and caches the local LLM model for offline use.
Run this before starting the server to ensure the model is available.

Usage:
    python scripts/download_model.py [model_key]
    python scripts/download_model.py --list
    
HEALTH/MEDICAL DOMAIN MODELS (Recommended for WellnessAI):
    - biogpt (default): Microsoft BioGPT - Medical domain, PubMed-trained, ~3GB
    - biogpt-medtext: Quantized BioGPT - Very fast, ~661MB
    - biomedlm: Stanford BioMedLM - Highest accuracy, ~5GB

GENERAL PURPOSE MODELS:
    - tinyllama: TinyLlama-1.1B - Fast, runs on CPU, ~2GB
    - phi2: Microsoft Phi-2 - Better quality, ~5GB
    - flan-t5: Google Flan-T5-base - Efficient, ~1GB
    - mistral: Mistral-7B - Best quality, needs GPU, ~15GB
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def download_model(model_key: str = "biogpt"):
    """Download and cache the specified model"""
    
    print("=" * 60)
    print("WellnessAI - Local LLM Model Downloader")
    print("=" * 60)
    
    # Model configurations
    models = {
        # HEALTH/MEDICAL DOMAIN MODELS (Recommended)
        "biogpt": {
            "name": "microsoft/biogpt",
            "size": "~3GB",
            "description": "Medical domain LLM trained on 15M PubMed abstracts (RECOMMENDED)",
            "domain": "medical"
        },
        "biogpt-medtext": {
            "name": "AventIQ-AI/BioGPT-MedText",
            "size": "~661MB",
            "description": "Quantized BioGPT, very fast, optimized for medical text",
            "domain": "medical"
        },
        "biomedlm": {
            "name": "stanford-crfm/BioMedLM",
            "size": "~5GB",
            "description": "Stanford 2.7B biomedical model, highest accuracy",
            "domain": "medical"
        },
        # GENERAL PURPOSE MODELS
        "tinyllama": {
            "name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "size": "~2GB",
            "description": "Fast 1.1B model, runs on CPU",
            "domain": "general"
        },
        "phi2": {
            "name": "microsoft/phi-2",
            "size": "~5GB",
            "description": "Quality 2.7B model, needs 6GB+ RAM",
            "domain": "general"
        },
        "flan-t5": {
            "name": "google/flan-t5-base",
            "size": "~1GB",
            "description": "Efficient encoder-decoder model",
            "domain": "general"
        },
        "mistral": {
            "name": "mistralai/Mistral-7B-Instruct-v0.2",
            "size": "~15GB",
            "description": "Best quality, needs GPU",
            "domain": "general"
        }
    }
    
    if model_key not in models:
        print(f"\nError: Unknown model '{model_key}'")
        print(f"Available models: {', '.join(models.keys())}")
        sys.exit(1)
    
    model_info = models[model_key]
    
    print(f"\nSelected Model: {model_key}")
    print(f"  - HuggingFace ID: {model_info['name']}")
    print(f"  - Approximate Size: {model_info['size']}")
    print(f"  - Description: {model_info['description']}")
    
    # Create cache directory
    cache_dir = project_root / "models"
    cache_dir.mkdir(exist_ok=True)
    print(f"\nCache Directory: {cache_dir}")
    
    # Check for required packages
    try:
        import torch
        print(f"\nPyTorch Version: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("\nError: PyTorch not installed. Run: pip install torch")
        sys.exit(1)
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import transformers
        print(f"Transformers Version: {transformers.__version__}")
    except ImportError:
        print("\nError: Transformers not installed. Run: pip install transformers")
        sys.exit(1)
    
    # Download the model
    print("\n" + "-" * 60)
    print("Downloading model and tokenizer...")
    print("This may take a while depending on your internet connection.")
    print("-" * 60 + "\n")
    
    try:
        # Download tokenizer
        print("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_info["name"],
            cache_dir=str(cache_dir),
            trust_remote_code=True
        )
        print("✓ Tokenizer downloaded successfully")
        
        # Download model
        print("\nDownloading model weights...")
        if model_key == "flan-t5":
            from transformers import T5ForConditionalGeneration
            model = T5ForConditionalGeneration.from_pretrained(
                model_info["name"],
                cache_dir=str(cache_dir),
                trust_remote_code=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_info["name"],
                cache_dir=str(cache_dir),
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
        print("✓ Model downloaded successfully")
        
        # Get model size
        param_count = sum(p.numel() for p in model.parameters())
        print(f"\nModel Parameters: {param_count:,}")
        
        # Save a marker file
        marker_file = cache_dir / f".{model_key}_ready"
        marker_file.write_text(f"Model: {model_info['name']}\nParameters: {param_count:,}")
        
        print("\n" + "=" * 60)
        print("✓ MODEL DOWNLOAD COMPLETE")
        print("=" * 60)
        print(f"\nThe model is cached in: {cache_dir}")
        print(f"\nTo use this model, set the environment variable:")
        print(f"  export WELLNESS_LLM_MODEL={model_key}")
        print("\nOr start the server and it will use the default (biogpt).")
        
        if model_info.get("domain") == "medical":
            print("\n💊 This is a MEDICAL DOMAIN model trained on biomedical literature.")
            print("   It provides better understanding of health metrics and medical terminology.")
        
    except Exception as e:
        print(f"\n✗ Error downloading model: {e}")
        print("\nTroubleshooting:")
        print("  1. Check your internet connection")
        print("  2. Ensure you have enough disk space")
        print("  3. Try a smaller model (flan-t5 or tinyllama)")
        sys.exit(1)


def list_available_models():
    """List all available models and their status"""
    cache_dir = Path(__file__).parent.parent / "models"
    
    # Medical domain models (recommended)
    medical_models = {
        "biogpt": ("microsoft/biogpt", "~3GB", "Medical, PubMed-trained (RECOMMENDED)"),
        "biogpt-medtext": ("AventIQ-AI/BioGPT-MedText", "~661MB", "Medical, quantized, very fast"),
        "biomedlm": ("stanford-crfm/BioMedLM", "~5GB", "Medical, Stanford, highest accuracy"),
    }
    
    # General purpose models
    general_models = {
        "tinyllama": ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "~2GB", "General, fast, CPU-friendly"),
        "phi2": ("microsoft/phi-2", "~5GB", "General, high quality"),
        "flan-t5": ("google/flan-t5-base", "~1GB", "General, efficient"),
        "mistral": ("mistralai/Mistral-7B-Instruct-v0.2", "~15GB", "General, best quality, GPU")
    }
    
    print("\n" + "=" * 75)
    print("HEALTH/MEDICAL DOMAIN MODELS (Recommended for WellnessAI)")
    print("=" * 75)
    print(f"{'Key':<16} {'Size':<10} {'Downloaded':<12} {'Description'}")
    print("-" * 75)
    
    for key, (name, size, desc) in medical_models.items():
        marker = cache_dir / f".{key}_ready"
        downloaded = "✓ Yes" if marker.exists() else "No"
        print(f"{key:<16} {size:<10} {downloaded:<12} {desc}")
    
    print("\n" + "=" * 75)
    print("GENERAL PURPOSE MODELS")
    print("=" * 75)
    print(f"{'Key':<16} {'Size':<10} {'Downloaded':<12} {'Description'}")
    print("-" * 75)
    
    for key, (name, size, desc) in general_models.items():
        marker = cache_dir / f".{key}_ready"
        downloaded = "✓ Yes" if marker.exists() else "No"
        print(f"{key:<16} {size:<10} {downloaded:<12} {desc}")
    
    print("-" * 75)
    print("\nUsage: python scripts/download_model.py <model_key>")
    print("Example: python scripts/download_model.py biogpt")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--list":
            list_available_models()
        else:
            download_model(sys.argv[1])
    else:
        # Default to biogpt (medical domain model)
        print("No model specified, using default: biogpt (medical domain)")
        print("Use --list to see all available models")
        print()
        download_model("biogpt")
