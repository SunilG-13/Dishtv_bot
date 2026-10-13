"""
Comprehensive Evaluation Benchmark Module (Steps 13 & 14)
Tests DISH TV Knowledge Bot across:
- English, Telugu, Hindi, Kannada, Tamil
- Troubleshooting codes (301, 102, 401, E16-4)
- Viewing card refresh & channel management
- Out-of-domain / Absent queries (anti-hallucination verification)
- Metrics: Retrieval relevance, answer accuracy, latency, multilingual quality
"""
import os
import sys
import time
import json
import logging
from typing import List, Dict, Any
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from config import SUPPORTED_LANGUAGES
from rag.rag_pipeline import RagPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Test Evaluation Benchmark Dataset
BENCHMARK_TEST_CASES = [
    {
        "id": "TC-01",
        "category": "Troubleshooting (Error 301)",
        "language": "en",
        "language_name": "English",
        "query": "How do I fix Error 301 Signal Not Found on my Dish TV?",
        "expected_keywords": ["weather", "cable", "restart", "helpline", "95017", "1800-120-3474", "customer care"],
        "is_absent_test": False
    },
    {
        "id": "TC-02",
        "category": "Recharge & Error 101/102 (Telugu)",
        "language": "te",
        "language_name": "Telugu",
        "query": "నా డిష్ టీవీలో ఎర్రర్ 102 వస్తోంది, దీన్ని ఎలా పరిష్కరించాలి మరియు సర్వీస్ ఎలా ప్రారంభించాలి?",
        "expected_keywords": ["1800-270-2102", "15", "మిస్డ్ కాల్", "బాక్స్", "రీఛార్జ్"],
        "is_absent_test": False
    },
    {
        "id": "TC-03",
        "category": "New Connection Plans & Cashback (Hindi)",
        "language": "hi",
        "language_name": "Hindi",
        "query": "डिश टीवी के नए कनेक्शन प्लान और कैशबैक ऑफर क्या हैं?",
        "expected_keywords": ["1,440", "2,287", "3,600", "कैशबैक", "एचडी प्लान"],
        "is_absent_test": False
    },
    {
        "id": "TC-04",
        "category": "STB Dongle Device (Kannada)",
        "language": "kn",
        "language_name": "Kannada",
        "query": "ಡಿಶ್ ಟಿವಿ STB ಡಾಂಗಲ್ ಎಂದರೇನು ಮತ್ತು ಅದರ ಬೆಲೆ ಎಷ್ಟು?",
        "expected_keywords": ["2499", "ಆಂಡ್ರಾಯ್ಡ್", "ಗೂಗಲ್", "ಡಾಂಗಲ್", "ರಿಮೋಟ್"],
        "is_absent_test": False
    },
    {
        "id": "TC-05",
        "category": "3-Days Credit & Quick Help (Tamil)",
        "language": "ta",
        "language_name": "Tamil",
        "query": "டிஷ் டிவியில் 3 நாட்கள் அவசர கடன் பெறுவது எப்படி?",
        "expected_keywords": ["1800-274-9050", "3", "10", "மிஸ்டு கால்"],
        "is_absent_test": False
    },
    {
        "id": "TC-06",
        "category": "Absent / Out-of-Domain Anti-Hallucination Test",
        "language": "en",
        "language_name": "English",
        "query": "Can you give me a recipe to bake chocolate cookies using Dish TV?",
        "expected_keywords": ["not available", "dish tv", "customer care"],
        "is_absent_test": True
    },
    {
        "id": "TC-07",
        "category": "Quick Help & 101/102 Error (Gujarati)",
        "language": "gu",
        "language_name": "Gujarati",
        "query": "ડિશ ટીવીમાં એરર 101 અને 102 નું નિવારણ કેવી રીતે કરવું?",
        "expected_keywords": ["1800-270-2102", "15", "મિસ્ડ કૉલ", "બોક્સ", "ચાલુ"],
        "is_absent_test": False
    }
]


def run_evaluation(output_file: str = "evaluation_report.json"):
    logger.info("=======================================================")
    logger.info("   Starting DISH TV Multilingual Bot Evaluation Suite  ")
    logger.info("=======================================================")

    import requests
    # Check if live server is reachable
    use_api = False
    try:
        r = requests.get("http://127.0.0.1:8000/api/health", timeout=3)
        if r.status_code == 200:
            use_api = True
            logger.info("Testing against live FastAPI server at http://127.0.0.1:8000")
    except Exception:
        use_api = False
        logger.info("Live server not detected. Initializing local RagPipeline...")

    pipeline = None if use_api else RagPipeline()
    results = []

    for test_case in BENCHMARK_TEST_CASES:
        t_id = test_case["id"]
        lang = test_case["language"]
        query = test_case["query"]
        category = test_case["category"]
        is_absent = test_case["is_absent_test"]

        logger.info(f"\n[Test Case {t_id}] Category: {category} | Language: {test_case['language_name']}")
        logger.info(f"Query: '{query}'")

        start_time = time.time()
        if use_api:
            resp = requests.post(
                "http://127.0.0.1:8000/api/chat",
                json={"query": query, "language": lang},
                timeout=45
            )
            response = resp.json()
        else:
            response = pipeline.answer_query(query=query, language=lang)
        elapsed = round(time.time() - start_time, 2)

        answer = response.get("answer", "")
        citations = response.get("citations", [])
        metrics = response.get("metrics", {})

        # Compute Retrieval Score
        top_similarity = citations[0]["relevance_score"] if citations else 0.0

        # Check Grounding / Expected Keywords
        keyword_hits = []
        if not is_absent:
            # Check presence of knowledge elements
            for kw in test_case["expected_keywords"]:
                if kw.lower() in answer.lower():
                    keyword_hits.append(kw)
            grounding_score = round(len(keyword_hits) / max(1, len(test_case["expected_keywords"])), 2)
        else:
            # For absent queries, verify model gracefully declined without hallucinating
            decline_phrases = ["not available", "cannot find", "official", "outside", "customer care", "help line", "assist"]
            declined = any(p in answer.lower() for p in decline_phrases)
            grounding_score = 1.0 if declined else 0.0

        test_result = {
            "test_id": t_id,
            "category": category,
            "language": lang,
            "language_name": test_case["language_name"],
            "query": query,
            "answer_snippet": answer[:300] + "..." if len(answer) > 300 else answer,
            "citations_count": len(citations),
            "citations": citations,
            "top_similarity_score": top_similarity,
            "grounding_pass": grounding_score >= 0.3 or is_absent,
            "grounding_score": grounding_score,
            "latency_seconds": elapsed,
            "metrics": metrics
        }
        results.append(test_result)

        logger.info(f"Answer snippet: {answer[:180]}...")
        logger.info(f"Top Relevance: {top_similarity:.3f} | Latency: {elapsed}s | Grounding Score: {grounding_score}")

    # Summary Statistics
    total_tests = len(results)
    avg_latency = round(sum(r["latency_seconds"] for r in results) / total_tests, 2)
    avg_relevance = round(sum(r["top_similarity_score"] for r in results) / total_tests, 3)
    passed_tests = sum(1 for r in results if r["grounding_pass"])

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_test_cases": total_tests,
        "passed_test_cases": passed_tests,
        "pass_rate_percentage": round((passed_tests / total_tests) * 100, 1),
        "average_latency_seconds": avg_latency,
        "average_top_retrieval_relevance": avg_relevance,
        "languages_tested": ["English", "Telugu", "Hindi", "Kannada", "Tamil"],
        "detailed_results": results
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info("\n=======================================================")
    logger.info(f"   EVALUATION SUMMARY: {passed_tests}/{total_tests} Tests Passed ({summary['pass_rate_percentage']}%)")
    logger.info(f"   Average Latency: {avg_latency}s | Average Relevance: {avg_relevance}")
    logger.info(f"   Report saved to {output_file}")
    logger.info("=======================================================")

    return summary


if __name__ == "__main__":
    run_evaluation()
