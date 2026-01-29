"""
Web Search Service for WellnessAI
Allows the AI to search the web for current health and nutrition information

Uses DuckDuckGo Instant Answer API (no API key required)
Includes fallback health knowledge base for offline operation
"""

import httpx
from typing import Dict, List, Optional
import re
from urllib.parse import quote_plus


# Built-in health knowledge base for common queries
HEALTH_KNOWLEDGE_BASE = {
    "magnesium": {
        "title": "Magnesium Benefits",
        "content": "Magnesium is essential for sleep quality, muscle relaxation, and stress reduction. Good sources include dark leafy greens, nuts, seeds, and whole grains. It helps regulate melatonin and calms the nervous system.",
        "foods": ["spinach", "almonds", "pumpkin seeds", "dark chocolate", "avocado"]
    },
    "sleep": {
        "title": "Sleep & Nutrition",
        "content": "Foods that promote sleep include those rich in tryptophan, magnesium, and melatonin. Avoid caffeine and heavy meals before bed. Cherries, bananas, and warm milk can help improve sleep quality.",
        "foods": ["cherries", "bananas", "warm milk", "chamomile tea", "fatty fish"]
    },
    "headache": {
        "title": "Headache Relief",
        "content": "Dehydration is a common cause of headaches. Magnesium-rich foods, ginger, and omega-3 fatty acids can help. Avoid trigger foods like aged cheese, processed meats, and alcohol.",
        "foods": ["water", "ginger", "leafy greens", "fatty fish", "peppermint tea"]
    },
    "energy": {
        "title": "Energy Boosting Foods",
        "content": "Complex carbohydrates, lean proteins, and iron-rich foods provide sustained energy. B vitamins are crucial for energy metabolism. Avoid sugar crashes by choosing whole foods.",
        "foods": ["oatmeal", "eggs", "nuts", "bananas", "lean meats", "quinoa"]
    },
    "stress": {
        "title": "Stress-Reducing Nutrition",
        "content": "Foods rich in vitamin C, B vitamins, and omega-3s help manage cortisol levels. Dark chocolate, green tea, and fermented foods support stress response.",
        "foods": ["oranges", "salmon", "dark chocolate", "green tea", "yogurt"]
    },
    "hrv": {
        "title": "Heart Rate Variability",
        "content": "HRV can be improved through omega-3 fatty acids, antioxidants, and proper hydration. Reducing alcohol and processed foods helps. Regular exercise and sleep optimization are key.",
        "foods": ["fatty fish", "berries", "leafy greens", "nuts", "olive oil"]
    },
    "recovery": {
        "title": "Recovery Nutrition",
        "content": "Post-exercise recovery requires protein for muscle repair, carbohydrates for glycogen replenishment, and antioxidants to reduce inflammation. Hydration is critical.",
        "foods": ["chicken", "eggs", "sweet potato", "berries", "tart cherry juice"]
    }
}


class WebSearchService:
    """
    Web Search Service using DuckDuckGo with offline fallback
    
    This demonstrates:
    - Retrieval-Augmented Generation (RAG) concept
    - External knowledge integration
    - Real-time information fetching
    - Fallback to local knowledge base
    """
    
    def __init__(self):
        self.base_url = "https://api.duckduckgo.com/"
        self.timeout = 10.0
        
    async def search(self, query: str, max_results: int = 5) -> Dict:
        """
        Search the web for information
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            Dict with search results and summary
        """
        # First, check local knowledge base for health-related queries
        local_results = self._search_local_knowledge(query)
        if local_results.get("results"):
            return local_results
        
        # Try DuckDuckGo API
        try:
            params = {
                "q": query,
                "format": "json",
                "no_html": 1,
                "skip_disambig": 1
            }
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(self.base_url, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    parsed = self._parse_results(data, query)
                    if parsed.get("results"):
                        return parsed
                        
        except Exception as e:
            print(f"[WebSearch] API error: {e}")
        
        # Fallback to local knowledge base
        return local_results if local_results.get("results") else self._generate_fallback(query)
    
    def _search_local_knowledge(self, query: str) -> Dict:
        """Search the local health knowledge base"""
        query_lower = query.lower()
        results = []
        
        for keyword, data in HEALTH_KNOWLEDGE_BASE.items():
            if keyword in query_lower:
                results.append({
                    "title": data["title"],
                    "snippet": data["content"],
                    "source": "WellnessAI Knowledge Base",
                    "url": "",
                    "foods": data.get("foods", [])
                })
        
        if results:
            return {
                "success": True,
                "query": query,
                "results": results[:5],
                "summary": results[0]["snippet"] if results else "",
                "source": "WellnessAI Knowledge Base",
                "result_count": len(results)
            }
        
        return {"success": False, "results": [], "result_count": 0}
    
    def _generate_fallback(self, query: str) -> Dict:
        """Generate a helpful fallback response"""
        return {
            "success": True,
            "query": query,
            "results": [{
                "title": "Health Information",
                "snippet": f"For specific information about '{query}', I recommend consulting with a healthcare professional or registered dietitian. General nutrition advice: stay hydrated, eat whole foods, get adequate sleep, and maintain regular physical activity.",
                "source": "WellnessAI",
                "url": ""
            }],
            "summary": "General health advice provided. For specific queries, consult a healthcare professional.",
            "source": "WellnessAI",
            "result_count": 1
        }
    
    def _parse_results(self, data: Dict, query: str) -> Dict:
        """Parse DuckDuckGo API response"""
        results = []
        
        # Abstract (main summary)
        if data.get("Abstract"):
            results.append({
                "title": data.get("Heading", "Summary"),
                "snippet": data["Abstract"],
                "source": data.get("AbstractSource", "DuckDuckGo"),
                "url": data.get("AbstractURL", "")
            })
        
        # Related topics
        for topic in data.get("RelatedTopics", [])[:5]:
            if isinstance(topic, dict) and topic.get("Text"):
                # Extract first sentence for cleaner display
                text = topic["Text"]
                first_sentence = text.split(". ")[0] + "." if ". " in text else text
                results.append({
                    "title": topic.get("FirstURL", "").split("/")[-1].replace("_", " "),
                    "snippet": first_sentence,
                    "source": "DuckDuckGo",
                    "url": topic.get("FirstURL", "")
                })
        
        # Infobox (structured data)
        infobox_data = {}
        if data.get("Infobox"):
            for item in data["Infobox"].get("content", []):
                if item.get("label") and item.get("value"):
                    infobox_data[item["label"]] = item["value"]
        
        return {
            "success": True,
            "query": query,
            "results": results[:5],
            "summary": data.get("Abstract", ""),
            "source": data.get("AbstractSource", ""),
            "infobox": infobox_data,
            "result_count": len(results)
        }
    
    def _error_response(self, message: str) -> Dict:
        """Generate error response"""
        return {
            "success": False,
            "query": "",
            "results": [],
            "summary": "",
            "error": message,
            "result_count": 0
        }
    
    async def search_health_topic(self, topic: str) -> Dict:
        """
        Search for health-related information
        Adds health context to improve results
        """
        # Enhance query for health context
        health_query = f"{topic} health nutrition benefits"
        return await self.search(health_query)
    
    async def search_nutrition_info(self, food: str) -> Dict:
        """
        Search for nutrition information about a food
        """
        nutrition_query = f"{food} nutrition facts health benefits"
        return await self.search(nutrition_query)
    
    async def search_symptom(self, symptom: str) -> Dict:
        """
        Search for information about a health symptom
        """
        symptom_query = f"{symptom} causes remedies natural treatment"
        return await self.search(symptom_query)
    
    def format_for_llm(self, search_results: Dict) -> str:
        """
        Format search results for LLM context
        """
        if not search_results.get("success"):
            return "Web search unavailable."
        
        if not search_results.get("results"):
            return "No relevant web results found."
        
        formatted = "**Web Search Results:**\n\n"
        
        if search_results.get("summary"):
            formatted += f"**Summary:** {search_results['summary']}\n\n"
        
        for i, result in enumerate(search_results["results"][:3], 1):
            formatted += f"{i}. **{result['title']}**: {result['snippet']}\n"
        
        if search_results.get("source"):
            formatted += f"\n*Source: {search_results['source']}*"
        
        return formatted


# Singleton instance
_web_search_service = None

def get_web_search_service() -> WebSearchService:
    """Get or create WebSearchService singleton"""
    global _web_search_service
    if _web_search_service is None:
        _web_search_service = WebSearchService()
    return _web_search_service
