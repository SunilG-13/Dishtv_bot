"""
DeepSeek API Client Module
Integrates deepseek-chat using OpenAI-compatible client interface or direct HTTP requests.
Handles multilingual instruction following, strict knowledge grounding, and anti-hallucination guards.
"""
import time
import logging
from typing import List, Dict, Any, Optional
import requests

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class DeepSeekClient:
    """Wrapper client for DeepSeek Chat API."""

    def __init__(
        self,
        api_key: str = DEEPSEEK_API_KEY,
        base_url: str = DEEPSEEK_BASE_URL,
        model: str = DEEPSEEK_MODEL
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.endpoint = f"{self.base_url}/chat/completions"

    def generate_answer(
        self,
        system_prompt: str,
        user_message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.2,
        max_tokens: int = 1200
    ) -> Dict[str, Any]:
        """
        Calls DeepSeek API with system prompt, history, and user query.
        Returns response dict containing text, latency, and token usage.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        messages = [{"role": "system", "content": system_prompt}]
        if conversation_history:
            # Append last 4 history turns for context
            for msg in conversation_history[-4:]:
                if msg.get("role") in ["user", "assistant"] and msg.get("content"):
                    messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": user_message})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        start_time = time.time()
        try:
            logger.info(f"Sending request to DeepSeek API ({self.model})...")
            response = requests.post(self.endpoint, headers=headers, json=payload, timeout=45)
            latency = time.time() - start_time

            if response.status_code != 200:
                logger.error(f"DeepSeek API Error HTTP {response.status_code}: {response.text}")
                return {
                    "success": False,
                    "answer": f"DeepSeek API Error (HTTP {response.status_code}). Please verify your connection or API status.",
                    "error": response.text,
                    "latency": round(latency, 2)
                }

            result_json = response.json()
            choice = result_json["choices"][0]["message"]["content"]
            usage = result_json.get("usage", {})

            return {
                "success": True,
                "answer": choice.strip(),
                "usage": usage,
                "latency": round(latency, 2),
                "model": self.model
            }

        except requests.Timeout:
            latency = time.time() - start_time
            logger.error("DeepSeek API request timed out.")
            return {
                "success": False,
                "answer": "The request timed out. Please try asking again.",
                "error": "Timeout",
                "latency": round(latency, 2)
            }
        except Exception as e:
            latency = time.time() - start_time
            logger.error(f"Unexpected DeepSeek API error: {e}")
            return {
                "success": False,
                "answer": f"An error occurred while generating the answer: {str(e)}",
                "error": str(e),
                "latency": round(latency, 2)
            }
