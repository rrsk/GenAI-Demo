"""
Local LLM Service using HuggingFace Transformers
Runs a local language model for health recommendations without API dependencies

Supported Models (in order of recommendation):

HEALTH/MEDICAL DOMAIN MODELS (Recommended):
1. BioGPT (1.5B) - Microsoft's biomedical GPT trained on 15M PubMed abstracts
2. BioGPT-MedText - Quantized BioGPT optimized for medical text (~661MB)
3. BioMedLM (2.7B) - Stanford's biomedical model, highest accuracy

GENERAL PURPOSE MODELS:
4. TinyLlama-1.1B - Fast, runs on CPU
5. Phi-2 (2.7B) - Better quality, needs more RAM
6. Mistral-7B - Best quality, needs GPU

This demonstrates key GenAI concepts:
- Domain-specific model deployment (BioGPT for health)
- Knowledge distillation (smaller models, specialized training)
- Transfer learning (PubMed knowledge → health recommendations)
- Prompt engineering for medical domain
- Model quantization for CPU efficiency
"""

import os
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Try to import torch - may not be installed yet
TORCH_AVAILABLE = False
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    print("[LocalLLM] PyTorch not installed. Run: pip install torch transformers accelerate")

# Model configurations
MODEL_CONFIGS = {
    # ==========================================
    # HEALTH/MEDICAL DOMAIN MODELS (Recommended)
    # ==========================================
    "biogpt": {
        "name": "microsoft/biogpt",
        "description": "1.5B biomedical GPT trained on 15M PubMed abstracts",
        "requires_gpu": False,
        "max_new_tokens": 400,
        "template": "biogpt",
        "domain": "medical"
    },
    "biogpt-medtext": {
        "name": "AventIQ-AI/BioGPT-MedText",
        "description": "Quantized BioGPT for medical text (~661MB, very fast)",
        "requires_gpu": False,
        "max_new_tokens": 256,
        "template": "biogpt",
        "domain": "medical"
    },
    "biomedlm": {
        "name": "stanford-crfm/BioMedLM",
        "description": "2.7B Stanford biomedical model, highest accuracy on medical benchmarks",
        "requires_gpu": False,
        "max_new_tokens": 512,
        "template": "biomedlm",
        "domain": "medical"
    },
    # ==========================================
    # GENERAL PURPOSE MODELS
    # ==========================================
    "tinyllama": {
        "name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "description": "Fast 1.1B model, runs on CPU",
        "requires_gpu": False,
        "max_new_tokens": 512,
        "template": "tinyllama",
        "domain": "general"
    },
    "phi2": {
        "name": "microsoft/phi-2",
        "description": "Quality 2.7B model, needs 6GB+ RAM",
        "requires_gpu": False,
        "max_new_tokens": 512,
        "template": "phi2",
        "domain": "general"
    },
    "mistral": {
        "name": "mistralai/Mistral-7B-Instruct-v0.2",
        "description": "Best quality 7B model, needs GPU",
        "requires_gpu": True,
        "max_new_tokens": 1024,
        "template": "mistral",
        "domain": "general"
    },
    "flan-t5": {
        "name": "google/flan-t5-base",
        "description": "Efficient encoder-decoder model, great for instructions",
        "requires_gpu": False,
        "max_new_tokens": 256,
        "template": "flan",
        "domain": "general"
    }
}

# Default model - TinyLlama for best conversational experience
# Use biogpt for medical terminology understanding
DEFAULT_MODEL = os.environ.get("WELLNESS_LLM_MODEL", "tinyllama")
# Grounding model for state-of-union: produces evidence-based prompt from user stats (BioGPT)
GROUNDING_MODEL = os.environ.get("WELLNESS_GROUNDING_MODEL", "biogpt")


class LocalLLMService:
    """
    Local Language Model Service for Health Recommendations
    
    This class demonstrates:
    - HuggingFace Transformers integration
    - Efficient model loading with quantization
    - Prompt engineering for health domain
    - Context-aware text generation
    """
    
    def __init__(self, model_key: str = DEFAULT_MODEL):
        self.model_key = model_key
        self.model_config = MODEL_CONFIGS.get(model_key, MODEL_CONFIGS["tinyllama"])
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.is_loaded = False
        self.torch_available = TORCH_AVAILABLE
        
        # Determine device
        if TORCH_AVAILABLE and torch is not None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = "cpu"
        
        # Model cache directory
        self.cache_dir = Path(__file__).parent.parent / "models"
        self.cache_dir.mkdir(exist_ok=True)
        
        print(f"[LocalLLM] Initialized with model: {self.model_config['name']}")
        print(f"[LocalLLM] Device: {self.device}")
        print(f"[LocalLLM] PyTorch available: {TORCH_AVAILABLE}")
        print(f"[LocalLLM] Cache directory: {self.cache_dir}")
    
    def load_model(self) -> bool:
        """
        Load the language model with optimizations
        
        Uses:
        - 8-bit quantization for memory efficiency
        - Device mapping for optimal performance
        - Caching for faster subsequent loads
        """
        if self.is_loaded:
            return True
        
        if not TORCH_AVAILABLE:
            print("[LocalLLM] Cannot load model - PyTorch not installed")
            print("[LocalLLM] Run: pip install torch transformers accelerate")
            return False
        
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            from transformers import BitsAndBytesConfig
            
            model_name = self.model_config["name"]
            print(f"[LocalLLM] Loading model: {model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=str(self.cache_dir),
                trust_remote_code=True
            )
            
            # Set padding token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # GPU loading with fp16 for best performance on RTX GPUs
            if self.device == "cuda" and torch.cuda.is_available():
                print(f"[LocalLLM] Loading on GPU with fp16 for fast inference...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    cache_dir=str(self.cache_dir),
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
            else:
                # CPU loading with torch.float32
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    cache_dir=str(self.cache_dir),
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
            
            # Create text generation pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=self.model_config["max_new_tokens"],
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1
            )
            
            self.is_loaded = True
            print(f"[LocalLLM] Model loaded successfully!")
            return True
            
        except ImportError as e:
            print(f"[LocalLLM] Missing dependencies: {e}")
            print("[LocalLLM] Run: pip install transformers torch accelerate bitsandbytes")
            return False
        except Exception as e:
            print(f"[LocalLLM] Error loading model: {e}")
            return False
    
    def _build_prompt(
        self, 
        user_message: str, 
        health_context: str, 
        recommendations: str
    ) -> str:
        """
        Build an optimized prompt for the language model
        
        Prompt Engineering Techniques:
        1. Clear system instructions
        2. Structured context injection
        3. Role-based formatting
        4. Output format guidance
        5. Domain-specific framing for medical models
        """
        template = self.model_config["template"]
        domain = self.model_config.get("domain", "general")
        
        # Use medical-focused system prompt for health domain models
        if domain == "medical":
            system_prompt = """You are a clinical health advisor providing evidence-based recommendations.
Analyze the patient's biometric data and provide personalized health and nutrition guidance.
Use medical knowledge to explain correlations between metrics and recommendations.
Be clear, actionable, and cite relevant health factors."""
        else:
            system_prompt = """You are WellnessAI, a helpful health and nutrition assistant. 
You provide personalized advice based on biometric data from fitness wearables.
Be concise, practical, and supportive. Focus on actionable recommendations."""
        
        # BioGPT - Medical domain prompt (optimized for PubMed-trained model)
        # BioGPT works best with scientific/medical text completion style
        if template == "biogpt":
            prompt = f"""Background: A patient presents with the following health metrics from wearable device monitoring:
{health_context}

The patient asks: "{user_message}"

Based on current nutritional science and clinical evidence, the recommended interventions include dietary modifications focusing on"""
        
        # BioMedLM - Stanford biomedical model format
        elif template == "biomedlm":
            prompt = f"""Clinical Case Summary:

Patient Biometrics:
{health_context}

Evidence-Based Recommendations:
{recommendations}

Clinical Question: {user_message}

Assessment: Analyzing the patient's health data, the following evidence-based recommendations are provided:

"""
        
        elif template == "tinyllama":
            # TinyLlama chat format
            prompt = f"""<|system|>
{system_prompt}</s>
<|user|>
## User's Health Data:
{health_context}

## Recommended Actions (from analysis):
{recommendations}

## User's Question:
{user_message}

Please provide a helpful, personalized response based on the health data and recommendations above.</s>
<|assistant|>
"""
        elif template == "phi2":
            # Phi-2 instruction format
            prompt = f"""Instruct: {system_prompt}

Context:
{health_context}

Analysis Recommendations:
{recommendations}

User Question: {user_message}

Output: Let me provide personalized health advice based on your data:
"""
        elif template == "mistral":
            # Mistral instruction format
            prompt = f"""[INST] {system_prompt}

## Health Context:
{health_context}

## Analysis Recommendations:
{recommendations}

User asks: {user_message}
[/INST]

"""
        else:
            # Default format
            prompt = f"""Health Assistant Response:

User Health Data:
{health_context}

Recommended Actions:
{recommendations}

Question: {user_message}

Personalized Response:
"""
        
        return prompt
    
    def generate_response(
        self,
        user_message: str,
        health_context: str,
        recommendations: str,
        max_length: int = 500
    ) -> str:
        """
        Generate a response using the local LLM
        
        Args:
            user_message: User's question
            health_context: Formatted health data context
            recommendations: Pre-computed recommendations from engine
            max_length: Maximum response length
            
        Returns:
            Generated response text
        """
        if not self.is_loaded:
            if not self.load_model():
                return self._fallback_response(user_message, recommendations)
        
        try:
            # Build the prompt
            prompt = self._build_prompt(user_message, health_context, recommendations)
            
            # Generate response
            outputs = self.pipeline(
                prompt,
                max_new_tokens=min(max_length, self.model_config["max_new_tokens"]),
                pad_token_id=self.tokenizer.eos_token_id,
                return_full_text=False
            )
            
            response = outputs[0]["generated_text"].strip()
            
            # Clean up response
            response = self._clean_response(response)
            
            return response
            
        except Exception as e:
            print(f"[LocalLLM] Generation error: {e}")
            return self._fallback_response(user_message, recommendations)
    
    def _clean_response(self, response: str) -> str:
        """Clean and format the generated response"""
        # Remove any trailing special tokens
        stop_tokens = ["</s>", "[/INST]", "<|", "|>", "User:", "Question:"]
        for token in stop_tokens:
            if token in response:
                response = response.split(token)[0]
        
        # Remove excessive whitespace
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        response = '\n'.join(lines)
        
        return response.strip()
    
    def _build_grounding_prompt(self, health_context: str, user_message: str) -> str:
        """Build prompt for grounding model (BioGPT): evidence-based summary from stats + query."""
        template = self.model_config.get("template", "biogpt")
        if template == "biogpt":
            return f"""Based on the following patient biometrics and their question, provide a brief evidence-based clinical summary (2-4 sentences). Summarize the patient's status and what the evidence suggests—do not give direct advice to the patient. A health advisor will use this summary to give a personalized response.

Patient data:
{health_context[:2500]}

Patient question: {user_message}

Evidence-based summary:"""
        # Generic medical-style prompt for biomedlm etc.
        return f"""Clinical summary task. Given the patient data and question below, output a short evidence-based summary (2-4 sentences) for use by a health advisor. Do not address the patient directly.

Patient data:
{health_context[:2500]}

Question: {user_message}

Summary:"""

    def generate_grounding(self, health_context: str, user_message: str, max_tokens: int = 200) -> str:
        """
        Generate an evidence-based grounding prompt from user stats and query.
        Used in state-of-union: BioGPT produces this; TinyLlama uses it as context for the final response.
        """
        if not self.is_loaded:
            if not self.load_model():
                return ""
        try:
            prompt = self._build_grounding_prompt(health_context, user_message)
            outputs = self.pipeline(
                prompt,
                max_new_tokens=min(max_tokens, self.model_config.get("max_new_tokens", 256)),
                pad_token_id=self.tokenizer.eos_token_id,
                return_full_text=False,
            )
            text = (outputs[0].get("generated_text") or "").strip()
            for stop in ["Patient question:", "Summary:", "\n\n\n"]:
                if stop in text:
                    text = text.split(stop)[0].strip()
            return text[:1200]
        except Exception as e:
            print(f"[LocalLLM] Grounding generation error: {e}")
            return ""

    def _fallback_response(self, user_message: str, recommendations: str) -> str:
        """Fallback when model isn't available"""
        return f"""## WellnessAI Analysis

Based on your health data, here are my recommendations:

{recommendations}

I'm currently running in offline mode. For more detailed, conversational responses, 
please ensure the local language model is properly loaded.

**Quick Tips:**
- Stay hydrated throughout the day
- Prioritize quality sleep (7-9 hours)
- Balance your activity with recovery
- Focus on whole, nutritious foods

Is there anything specific you'd like me to elaborate on?
"""
    
    def get_model_info(self) -> Dict:
        """Get information about the current model"""
        return {
            "model_key": self.model_key,
            "model_name": self.model_config["name"],
            "description": self.model_config["description"],
            "device": self.device,
            "is_loaded": self.is_loaded,
            "requires_gpu": self.model_config["requires_gpu"],
            "cache_directory": str(self.cache_dir)
        }


# Singleton instances: primary (response) and grounding (evidence-based prompt)
_local_llm_service = None
_grounding_llm_service = None

def get_local_llm_service(model_key: str = None) -> LocalLLMService:
    """Get or create primary LocalLLMService (e.g. TinyLlama for final response)."""
    global _local_llm_service
    if model_key is None:
        model_key = DEFAULT_MODEL
    if _local_llm_service is None:
        _local_llm_service = LocalLLMService(model_key)
    return _local_llm_service


def get_grounding_llm_service(model_key: str = None) -> LocalLLMService:
    """Get or create grounding LocalLLMService (e.g. BioGPT for evidence-based prompt from stats)."""
    global _grounding_llm_service
    if model_key is None:
        model_key = GROUNDING_MODEL
    if _grounding_llm_service is None:
        _grounding_llm_service = LocalLLMService(model_key)
    return _grounding_llm_service


def preload_model(model_key: str = DEFAULT_MODEL) -> bool:
    """Preload the model (call during startup)"""
    service = get_local_llm_service(model_key)
    return service.load_model()
