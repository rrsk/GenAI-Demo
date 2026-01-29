#!/usr/bin/env python3
"""
WellnessAI - Run Script
Starts the FastAPI server for the health assistant
"""
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uvicorn
from backend.config import HOST, PORT

if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║   🌿 WellnessAI - Your Personal Health Assistant              ║
    ║                                                               ║
    ║   Starting server at http://localhost:8000                    ║
    ║                                                               ║
    ║   Features:                                                   ║
    ║   • Personalized meal plans based on Whoop data               ║
    ║   • Weather-adapted health recommendations                    ║
    ║   • AI-powered health insights                                ║
    ║   • Sleep and recovery optimization                           ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(
        "backend.main:app",
        host=HOST,
        port=PORT,
        reload=True,
        log_level="info"
    )
