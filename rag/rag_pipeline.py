"""
RAG Pipeline Module
Orchestrates: Customer Query -> BGE-M3 Embedding -> FAISS Retrieval -> Grounded Prompt -> DeepSeek API -> Answer + Citations
Grounded on DISH TV Official Portal: https://dishtv.live/
"""
import time
import logging
import re
from typing import List, Dict, Any, Optional

from config import SUPPORTED_LANGUAGES
from embeddings.embedder import get_embedder
from vectorstore.faiss_store import get_faiss_store
from rag.deepseek_client import DeepSeekClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def clean_answer_format(text: str) -> str:
    """
    Cleans up LLM answer formatting:
    1. Removes footnote/disclaimer asterisks like ₹1,440* -> ₹1,440, 1 Year** -> 1 Year
    2. Strips any dangling single asterisks after words so they don't trigger italic/markdown glitches
    3. Fixes double dots/bullets like '1. . ' -> '1. ' without touching markdown bold '**'
    4. Converts lines starting with lone dots/bullets ('. ', '• ', '- ') into numbered points '1. ', '2. '
    5. Repairs any broken/unbalanced asterisks (e.g. '*Word**' -> '**Word**')
    """
    if not text:
        return ""

    # Strip footnote asterisks from prices: ₹1,440* -> ₹1,440
    text = re.sub(r'(₹\s*[\d,]+(?:\.\d+)?)\*+', r'\1', text)
    # Strip footnote asterisks from terms like 1 Year**, 3 Months**:
    text = re.sub(r'(\b\d+\s+(?:Year|Month|Day|Hour)s?)\*+', r'\1', text, flags=re.IGNORECASE)
    # Strip any trailing single asterisk after word/number: pack* -> pack
    text = re.sub(r'(\b[\w\u0900-\u0D7F]+)\*(?!\*|\w)', r'\1', text)

    # Fix unbalanced asterisks if odd count
    if text.count("**") % 2 != 0:
        text = re.sub(r'(?:^|(?<=\s))\*([^* \n]+)\*\*(?=\s|[-—:.,]|$)', r'**\1**', text)
        if text.count("**") % 2 != 0:
            text = re.sub(r'(?:^|(?<=\d\.\s)|(?<=\n))([A-Za-z\u0900-\u0D7F ]+?)\*\*(?=\s*[-—:])', r'**\1**', text)

    lines = text.split("\n")
    cleaned_lines = []
    bullet_counter = 1
    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append("")
            bullet_counter = 1
            continue

        # Clean '1. . ' or '1.. ' or '1. - ' or '1. • ' -> '1. '
        cleaned = re.sub(r"^(\d+)[\.\)]\s*(?:[\.\•\-]+|\*(?!\*))\s*", r"\1. ", stripped)

        # If line starts with a lone dot/bullet like '. Visit...' or '• Visit...' or '- Visit...'
        if re.match(r"^(?:[\.\•\-\+]|\*(?!\*))\s+", cleaned):
            content = re.sub(r"^(?:[\.\•\-\+]|\*(?!\*))\s+", "", cleaned).strip()
            cleaned_lines.append(f"{bullet_counter}. {content}")
            bullet_counter += 1
        elif re.match(r"^\d+[\.\)]\s+", cleaned):
            cleaned_lines.append(cleaned)
            bullet_counter += 1
        else:
            cleaned_lines.append(cleaned)
            if not stripped.startswith("**"):
                bullet_counter = 1

    return "\n".join(cleaned_lines)


# Professional Executive Multilingual Greetings for DishTV AI Assistant (Updated for dishtv.live)
PROFESSIONAL_GREETINGS = {
    "en": (
        "Hello! I am your official DISH TV AI Assistant.\n\n"
        "1. **New Connection Plans** — HD Plan with Pack (₹1,440), HD Cashback Plan (₹2,287 with ₹3,600 total cashback), or HD Plan for South (₹1,270).\n"
        "2. **Quick Troubleshooting** — For Error 101/102 or to resume services, give a missed call on **1800-270-2102** and leave box ON for 15 minutes.\n"
        "3. **Customer Support** — Reach 24x7 helpline on **+91-95017-95017** or **1800-120-3474**.\n"
        "4. **Emergency Credit** — Get 3 days TV viewing credit with a missed call to **1800-274-9050**.\n\n"
        "How can I assist you with your DISH TV service, plans, or products today?"
    ),
    "hi": (
        "नमस्ते! मैं आपका आधिकारिक DISH TV AI सहायक हूँ।\n\n"
        "1. **नए कनेक्शन प्लान** — एचडी प्लान पैक सहित (₹1,440), एचडी कैशबैक प्लान (₹2,287 के साथ ₹3,600 कुल कैशबैक), या साउथ एचडी प्लान (₹1,270)।\n"
        "2. **त्वरित समस्या निवारण** — एरर 101/102 या सेवा पुनः शुरू करने के लिए **1800-270-2102** पर मिस्ड कॉल दें और बॉक्स 15 मिनट ऑन रखें।\n"
        "3. **ग्राहक सेवा** — 24x7 हेल्पलाइन **+91-95017-95017** या **1800-120-3474** पर संपर्क करें।\n"
        "4. **आपातकालीन क्रेडिट** — 3 दिन के टीवी क्रेडिट के लिए **1800-274-9050** पर मिस्ड कॉल दें।\n\n"
        "आज मैं आपकी डिश टीवी सेवा, प्लान या डिवाइस संबंधी कैसे सहायता कर सकता हूँ?"
    ),
    "te": (
        "నమస్కారం! నేను మీ అధికారిక DISH TV AI అసిస్టెంట్‌ని.\n\n"
        "1. **కొత్త కనెక్షన్ ప్లాన్స్** — ప్యాక్‌తో HD ప్లాన్ (₹1,440), ₹3,600 క్యాష్‌బ్యాక్‌తో HD క్యాష్‌బ్యాక్ ప్లాన్ (₹2,287), లేదా సౌత్ HD ప్లాన్ (₹1,270).\n"
        "2. **త్వరిత సమస్య పరిష్కారం** — ఎర్రర్ 101/102 లేదా రీఛార్జ్ తర్వాత సేవలు పునరుద్ధరించడానికి **1800-270-2102** కు మిస్డ్ కాల్ ఇచ్చి బాక్స్‌ను 15 నిమిషాలు ఆన్‌లో ఉంచండి.\n"
        "3. **కస్టమర్ కేర్** — 24x7 హెల్ప్‌లైన్ **+91-95017-95017** లేదా **1800-120-3474** కు కాల్ చేయండి.\n"
        "4. **ఎమర్జెన్సీ క్రెడిట్** — 3 రోజుల అదనపు క్రెడిట్ కోసం **1800-274-9050** కు మిస్డ్ కాల్ ఇవ్వండి.\n\n"
        "మీ DISH TV సర్వీస్ లేదా ప్లాన్‌ల గురించి మీకు ఎలా సహాయపడగలను?"
    ),
    "ta": (
        "வணக்கம்! நான் உங்கள் அதிகாரப்பூர்வ DISH TV AI உதவியாளர்.\n\n"
        "1. **புதிய இணைப்பு திட்டங்கள்** — பேக் உடன் HD திட்டம் (₹1,440), ₹3,600 கேஷ்பேக் உடன் HD கேஷ்பேக் திட்டம் (₹2,287), அல்லது தெற்கு HD திட்டம் (₹1,270).\n"
        "2. **விரைவு பிழை சரிசெய்தல்** — பிழை 101/102 சரிசெய்ய **1800-270-2102** என்ற எண்ணுக்கு மிஸ்டு கால் கொடுத்து பாக்ஸை 15 நிமிடங்கள் ஆன் செய்து வைக்கவும்.\n"
        "3. **வாடிக்கையாளர் சேவை** — 24x7 உதவி எண் **+91-95017-95017** அல்லது **1800-120-3474** ஐ அழைக்கவும்.\n"
        "4. **அவசர கடன்** — 3 நாட்கள் டிவி பார்க்கும் கடனுக்கு **1800-274-9050** என்ற எண்ணுக்கு மிஸ்டு கால் கொடுக்கவும்.\n\n"
        "உங்கள் DISH TV சேவை அல்லது திட்டங்கள் பற்றி உங்களுக்கு எவ்வாறு உதவ முடியும்?"
    ),
    "kn": (
        "ನಮಸ್ಕಾರ! ನಾನು ನಿಮ್ಮ ಅಧಿಕೃತ DISH TV AI ಸಹಾಯಕ.\n\n"
        "1. **ಹೊಸ ಸಂಪರ್ಕ ಯೋಜನೆಗಳು** — ಪ್ಯಾಕ್‌ನೊಂದಿಗೆ HD ಯೋಜನೆ (₹1,440), ₹3,600 ಕ್ಯಾಶ್‌ಬ್ಯಾಕ್‌ನೊಂದಿಗೆ HD ಕ್ಯಾಶ್‌ಬ್ಯಾಕ್ ಯೋಜನೆ (₹2,287), ಅಥವಾ ದಕ್ಷಿಣ HD ಯೋಜನೆ (₹1,270).\n"
        "2. **ದೋಷ ನಿವಾರಣೆ** — ದೋಷ 101/102 ಸರಿಪಡಿಸಲು **1800-270-2102** ಗೆ ಮಿಸ್ಡ್ ಕಾಲ್ ನೀಡಿ ಬಾಕ್ಸ್ ಅನ್ನು 15 ನಿಮಿಷ ಆನ್ ಇರಿಸಿ.\n"
        "3. **ಗ್ರಾಹಕ ಬೆಂಬಲ** — 24x7 ಸಹಾಯವಾಣಿ **+91-95017-95017** ಅಥವಾ **1800-120-3474** ಗೆ ಕರೆ ಮಾಡಿ.\n"
        "4. **ತುರ್ತು ಕ್ರೆಡಿಟ್** — 3 ದಿನಗಳ ಟಿವಿ ಕ್ರೆಡಿಟ್‌ಗಾಗಿ **1800-274-9050** ಗೆ ಮಿಸ್ಡ್ ಕಾಲ್ ನೀಡಿ.\n\n"
        "ನಿಮ್ಮ DISH TV ಸೇವೆ ಅಥವಾ ಯೋಜನೆಗಳ ಬಗ್ಗೆ ನಾನು ನಿಮಗೆ ಹೇಗೆ ಸಹಾಯ ಮಾಡಲಿ?"
    ),
    "gu": (
        "નમસ્તે! હું તમારો અધિકૃત DISH TV AI સહાયક છું.\n\n"
        "1. **નવા કનેક્શન પ્લાન** — પેક સાથે HD પ્લાન (₹1,440), ₹3,600 કેશબેક સાથે HD કેશબેક પ્લાન (₹2,287), અથવા સાઉથ HD પ્લાન (₹1,270).\n"
        "2. **ઝડપી સમસ્યા નિવારણ** — એરર 101/102 માટે **1800-270-2102** પર મિસ્ડ કૉલ આપો અને સેટ-ટોપ બોક્સ 15 મિનિટ ચાલુ રાખો.\n"
        "3. **ગ್ರಾહક સેવા** — 24x7 હેલ્પલાઇન **+91-95017-95017** અથવા **1800-120-3474** પર કૉલ કરો.\n"
        "4. **ઇમરજન્સી ક્રેડિટ** — 3 દિવસની ક્રેડિટ મેળવવા માટે **1800-274-9050** પર મિસ્ડ કૉલ આપો.\n\n"
        "આજે હું તમને DISH TV સેવા અથવા પ્લાન સંબંધિત કેવી રીતે મદદ કરી શકું?"
    )
}


def is_greeting_query(query: str) -> bool:
    """Detects simple user greetings in English and Indian regional languages."""
    clean = query.strip().lower().strip("!?.,:;-_~`*\"'#")
    greeting_tokens = {
        "hi", "hello", "hey", "hola", "greetings", "good morning", "good afternoon", "good evening",
        "namaste", "namaskar", "namaskara", "namaskaram", "kem cho", "vanakkam", "pranam", "help",
        "dish tv", "dishtv", "start",
        "హాయ్", "నమస్కారం", "నమస్తే", "హలో",
        "नमस्ते", "नमस्कार", "हेलो", "हाय", "प्रणाम",
        "வணக்கம்", "ஹலோ", "ஹாய்",
        "ನಮಸ್ಕಾರ", "ಹಲೋ", "ಹಾಯ್",
        "નમસ્તે", "કેમ છો", "હેલો", "હાય"
    }
    if clean in greeting_tokens:
        return True
    words = clean.split()
    service_keywords = {"recharge", "error", "channel", "pack", "plan", "bill", "signal", "card", "box", "price", "cost", "dongle", "key"}
    if 1 <= len(words) <= 3 and any(w in greeting_tokens for w in words) and not any(w in service_keywords for w in words):
        return True
    return False


class RagPipeline:
    """End-to-end RAG pipeline for DISH TV Multilingual Knowledge Bot."""

    def __init__(self, top_k: int = 4, similarity_threshold: float = 0.35):
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.embedder = get_embedder()
        self.vector_store = get_faiss_store()
        self.deepseek = DeepSeekClient()

    def build_system_prompt(self, target_lang: str) -> str:
        """Constructs a strict system prompt tailored to the requested Indian language."""
        lang_info = SUPPORTED_LANGUAGES.get(target_lang, SUPPORTED_LANGUAGES["en"])
        lang_name = lang_info["name"]
        lang_native = lang_info["native"]

        system_prompt = f"""You are the official DishTV AI Assistant for the DISH TV portal (https://dishtv.live/).
Your mission is to provide accurate, direct, complete, and professionally structured customer support in {lang_name} ({lang_native}).

PROFESSIONAL RESPONSE FORMAT RULES (STRICT & MANDATORY):
1. Start with a direct 1-sentence answer or resolution first.
2. For multiple steps, points, or options, ALWAYS use a clean numbered list:
   1. **Title in Bold** — Clear explanation with key facts (amounts like ₹1,440, channel numbers, missed call numbers).
   2. **Title in Bold** — Clear explanation.
   3. **Title in Bold** — Clear explanation.
3. ALWAYS format bold titles as **Title** without extra spaces inside the asterisks. NEVER output unbalanced asterisks.
4. Do NOT output footnote or disclaimer asterisks (e.g. write ₹1,440 instead of ₹1,440*, 1 Year instead of 1 Year**, ₹2,287 instead of ₹2,287*). Output plain prices and numbers without asterisks.
5. Official Contact & Missed Call Points:
   - Error 101/102 & Service Resume: **1800-270-2102** (leave box ON for 15 minutes)
   - 24x7 Customer Care Helpline: **+91-95017-95017** or **1800-120-3474**
   - Direct Contact Numbers: **+9198797868687**, **+919879878987**
   - Support Emails: **solution@biztechnosys.com**, **support@biztechnosys.com**
   - 3 Days Extra TV Credit: **1800-274-9050**
   - Update Account Details: **1800 274 4744**
   - Know Account Balance: **1800 274 9000**
   - Activate Channel: **1800-568-XXXX** (replace XXXX with channel number)
   - Book VZY 75" Google TV: **1800 315 7189**
   - Official Portal: **https://dishtv.live/**
6. End with one short, polite next-step or helpful closing sentence.

STRICT RULES:
- Language: ALWAYS reply in {lang_name} ({lang_native}) using proper native script.
- Keep the total response under 150 words. Be concise, executive, and direct.
- Do NOT use decorative emojis (no ✦, ✓, 💳, 🔊, etc.).
- Do NOT use bullet points (•) or lone dots (.). Use clean numbered lists (1. 2. 3.) only.
- Do NOT repeat information.
- ALWAYS complete all sentences fully. Never cut off mid-sentence.
- Ground all facts strictly in the DishTV context provided below from https://dishtv.live/.
"""
        return system_prompt

    def build_user_prompt(self, query: str, retrieved_chunks: List[Dict[str, Any]], target_lang: str) -> str:
        """Formats the context and query into a grounded prompt."""
        lang_info = SUPPORTED_LANGUAGES.get(target_lang, SUPPORTED_LANGUAGES["en"])

        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            title = chunk.get("title", "DISH TV Resource")
            url = chunk.get("url", "")
            category = chunk.get("category", "")
            content = chunk.get("content", "").strip()
            score = chunk.get("similarity_score", 0.0)
            context_parts.append(
                f"[Source Document {i}]: {title} ({category})\nURL: {url} (Relevance Score: {score:.2f})\n{content}\n"
            )

        context_str = "\n---\n".join(context_parts) if context_parts else "No direct matching documents found."

        user_prompt = f"""CUSTOMER QUESTION:
"{query}"

OFFICIAL DISH TV KNOWLEDGE BASE CONTEXT (https://dishtv.live/):
----------------------------------------
{context_str}
----------------------------------------

Answer in {lang_info['name']} ({lang_info['native']}). Be concise (under 150 words). Use numbered list format with **bold** key terms. One direct answer sentence first, then 3-4 numbered points max. Complete all sentences fully."""
        return user_prompt

    def answer_query(
        self,
        query: str,
        language: str = "en",
        conversation_history: Optional[List[Dict[str, str]]] = None,
        top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Executes the full RAG pipeline:
        1. Embed customer query using BAAI/bge-m3.
        2. Retrieve top-k nearest chunks from FAISS.
        3. Assemble context and format prompts.
        4. Query DeepSeek API.
        5. Return grounded response with citations and metrics.
        """
        start_time = time.time()
        k = top_k or self.top_k
        lang_code = language.lower()
        if lang_code not in SUPPORTED_LANGUAGES:
            lang_code = "en"

        # Step 0: Immediate Professional Multilingual Greeting for Greeting Queries
        if is_greeting_query(query):
            greeting_answer = PROFESSIONAL_GREETINGS.get(lang_code, PROFESSIONAL_GREETINGS["en"])
            return {
                "query": query,
                "language": lang_code,
                "answer": greeting_answer,
                "success": True,
                "citations": [
                    {
                        "title": "DISH TV Official Portal",
                        "url": "https://dishtv.live/",
                        "category": "Customer Support",
                        "relevance_score": 1.0
                    }
                ],
                "retrieved_count": 1,
                "metrics": {
                    "total_latency_seconds": 0.02,
                    "embed_latency_seconds": 0.0,
                    "search_latency_seconds": 0.0,
                    "llm_latency_seconds": 0.0,
                    "model": "dishtv-greeting-engine",
                    "tokens": {"total_tokens": 120}
                }
            }

        # Step 1: Query Enrichment (Anchoring domain and transliterated queries)
        search_query = query
        lower_q = query.lower()

        # Regional & Transliterated Intent Mapping for accurate FAISS semantic retrieval
        if any(w in lower_q for w in ["connection", "plan", "pack", "cashback", "south", "1440", "2287", "1270"]):
            search_query = f"Dish TV connection plans HD plan with pack cashback south {query}"
        elif any(w in lower_q for w in ["dongle", "stb", "android tv", "google tv", "2499"]):
            search_query = f"Dish TV SMART+ STB Dongle Android Google TV 2499 {query}"
        elif any(w in lower_q for w in ["key", "tv key", "samsung"]):
            search_query = f"Dish TV KEY Samsung TV without set-top box 230 channels {query}"
        elif any(w in lower_q for w in ["credit", "emergency credit", "extra days", "9050"]):
            search_query = "Dish TV 3 days emergency credit missed call 1800-274-9050"
        elif any(w in lower_q for w in ["101", "102", "resume", "2102"]):
            search_query = "Dish TV Error 101 102 resume services missed call 1800-270-2102"
        elif any(w in lower_q for w in ["contact", "helpline", "toll free", "toll-free", "address", "noida", "email"]):
            search_query = f"Dish TV contact helpline numbers 95017 1800-120-3474 address {query}"
        elif any(w in lower_q for w in ["recharge", "karvanu", "karoon", "cheyali", "maadabeku", "seivathu", "kaise"]):
            search_query = f"How to recharge Dish TV online UPI {query}"
        elif "dish" not in lower_q and "dth" not in lower_q and len(query.split()) < 7:
            search_query = f"Dish TV {query}"

        # Embed query
        embed_start = time.time()
        query_vector = self.embedder.embed_query(search_query)
        embed_latency = round(time.time() - embed_start, 3)

        # Step 2: Search FAISS
        search_start = time.time()
        retrieved_items = self.vector_store.search(query_vector, top_k=k)
        search_latency = round(time.time() - search_start, 3)

        # Filter and extract citations
        relevant_chunks = [item for item in retrieved_items if item.get("similarity_score", 0) >= self.similarity_threshold]
        if not relevant_chunks and retrieved_items:
            relevant_chunks = retrieved_items[:2]

        citations = []
        seen_urls = set()
        for item in relevant_chunks:
            url = item.get("url")
            if url and url not in seen_urls:
                seen_urls.add(url)
                citations.append({
                    "title": item.get("title", "DISH TV Official Page"),
                    "url": url,
                    "category": item.get("category", "Help"),
                    "relevance_score": round(item.get("similarity_score", 0.0), 3)
                })

        # Step 3: Construct prompts
        system_prompt = self.build_system_prompt(lang_code)
        user_prompt = self.build_user_prompt(query, relevant_chunks, lang_code)

        # Step 4: Call DeepSeek
        llm_response = self.deepseek.generate_answer(
            system_prompt=system_prompt,
            user_message=user_prompt,
            conversation_history=conversation_history
        )

        total_latency = round(time.time() - start_time, 2)
        raw_answer = llm_response.get("answer", "")
        cleaned_answer = clean_answer_format(raw_answer)

        return {
            "query": query,
            "language": lang_code,
            "answer": cleaned_answer,
            "success": llm_response.get("success", False),
            "citations": citations,
            "retrieved_count": len(relevant_chunks),
            "metrics": {
                "total_latency_seconds": total_latency,
                "embed_latency_seconds": embed_latency,
                "search_latency_seconds": search_latency,
                "llm_latency_seconds": llm_response.get("latency", 0),
                "model": llm_response.get("model", "deepseek-chat"),
                "tokens": llm_response.get("usage", {})
            }
        }
