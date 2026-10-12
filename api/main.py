"""
FastAPI Backend Application
Exposes endpoints for chat conversations, health checks, voice synthesis,
semantic search, and knowledge base indexing. Serves static UI frontend.
"""
import sys
if sys.version_info >= (3, 14):
    import typing
    _original_eval_type = getattr(typing, '_eval_type', None)
    if _original_eval_type:
        def _patched_eval_type(*args, **kwargs):
            kwargs.pop('prefer_fwd_module', None)
            return _original_eval_type(*args, **kwargs)
        typing._eval_type = _patched_eval_type

import logging
import os
import urllib.parse
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger("dishtv_api")

from config import (
    BASE_DIR,
    SUPPORTED_LANGUAGES,
    DEEPSEEK_MODEL,
    EMBEDDING_MODEL_NAME,
    FAISS_DIR
)
from rag.rag_pipeline import RagPipeline
from voice.voice_service import get_voice_service
from vectorstore.faiss_store import get_faiss_store

app = FastAPI(
    title="DishTV Multilingual AI Assistant",
    description="Production Multilingual Knowledge Bot powered by DeepSeek API, BAAI/bge-m3, FAISS, and Voice Service",
    version="1.0.0"
)

# Enable CORS for cross-origin frontend interactions
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request & Response Models
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    query: Optional[str] = Field(default=None, description="Customer question")
    message: Optional[str] = Field(default=None, description="Alternative key for customer question")
    language: str = Field(default="auto", description="Target language code (en, te, hi, kn, ta, etc.)")
    history: Optional[List[ChatMessage]] = Field(default=[], description="Recent conversation turns")

    @property
    def effective_query(self) -> str:
        return (self.query or self.message or "").strip()

class ChatResponse(BaseModel):
    query: str
    language: str
    answer: str
    citations: List[Dict[str, Any]]
    audio_url: Optional[str] = None
    metrics: Dict[str, Any]

# Lazy-loaded pipeline components
_pipeline = None

def get_pipeline() -> RagPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RagPipeline()
    return _pipeline

# Routes
@app.get("/api/health")
def health_check():
    """Health check endpoint reporting system readiness."""
    store = get_faiss_store()
    is_indexed = store.is_indexed()
    vector_count = store.index.ntotal if store.index else (len(store.metadata) if store.load() else 0)

    return {
        "status": "healthy" if is_indexed else "indexing_required",
        "faiss_indexed": is_indexed,
        "total_knowledge_chunks": vector_count,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "llm_model": DEEPSEEK_MODEL,
        "supported_languages": list(SUPPORTED_LANGUAGES.keys())
    }

@app.get("/api/languages")
def get_languages():
    """Returns available Indian languages with native labels and voice codes."""
    return {
        "languages": [
            {
                "code": code,
                "name": info["name"],
                "native": info["native"],
                "voice_code": info["voice_code"],
                "flag": info["flag"],
                "greeting": info["greeting"]
            }
            for code, info in SUPPORTED_LANGUAGES.items()
        ]
    }

def detect_script_language(text: str) -> Optional[str]:
    """Detects Indian regional language by counting matching Unicode script characters."""
    import re
    counts = {
        'kn': len(re.findall(r'[\u0C80-\u0CFF]', text)),  # Kannada
        'te': len(re.findall(r'[\u0C00-\u0C7F]', text)),  # Telugu
        'ta': len(re.findall(r'[\u0B80-\u0BFF]', text)),  # Tamil
        'gu': len(re.findall(r'[\u0A80-\u0AFF]', text)),  # Gujarati
        'hi': len(re.findall(r'[\u0900-\u097F]', text)),  # Hindi (Devanagari)
    }
    max_lang = max(counts, key=counts.get)
    if counts[max_lang] > 0:
        return max_lang
    return None

def detect_user_language(query: str, fallback_lang: str = "en") -> str:
    """
    Automatically detects user language without requiring any manual user selection.
    Supports English, Hindi, Kannada, Telugu, Tamil, and Gujarati with rich transliteration scoring.
    """
    import re
    # 1. Native Indic Unicode Scripts
    script_lang = detect_script_language(query)
    if script_lang:
        return script_lang

    # 2. Tokenize Romanized input
    lower = query.lower()
    words = set(re.findall(r'\b[a-z]+\b', lower))

    scores = {
        'gu': 0,
        'te': 0,
        'kn': 0,
        'ta': 0,
        'hi': 0
    }

    gu_vocab = {
        "kem", "cho", "kemcho", "shu", "karvanu", "joiye", "maru", "aapo", "madad",
        "hu", "navu", "navi", "kevi", "rite", "karu", "mane", "mahino", "mahina",
        "chale", "evo", "evi", "batavo", "batao", "sasto", "sasta", "kayo", "chhe",
        "che", "mare", "tamare", "tamari", "tamaro", "aapvo", "karo", "ketla",
        "khabar", "nathi", "aave", "hove", "su", "karvu", "joie", "paisa", "rupiya",
        "nava", "jovu", "male"
    }

    te_vocab = {
        "namaskaram", "ela", "cheyali", "cheyyali", "enti", "kavali", "naa", "naaku",
        "cheyyandi", "meeku", "sahayam", "nenu", "kothaga", "kotha", "chesukovali",
        "chesi", "yentha", "enta", "undhi", "undi", "manchi", "eppudu", "ekkada",
        "chudali", "pettali", "teliyali", "chesuko", "chestaru"
    }

    kn_vocab = {
        "namaskara", "namaskar", "hege", "madodu", "maadabeku", "madabeku", "yenu",
        "beku", "nanna", "nanage", "illi", "yavudu", "yava", "nimage", "sahaya",
        "naanu", "hosa", "estu", "yeshtu", "hegidira", "agabeku", "nodabekagide", "kanisutta",
        "maadi", "maadodu"
    }

    ta_vocab = {
        "vanakkam", "eppadi", "seivathu", "seivadhu", "panrathu", "enna", "vendum",
        "venum", "enathu", "enakku", "udhavi", "unga", "ungalukku", "naan", "pudhiya",
        "pudhu", "evalavu", "ethana", "teriyum", "parka", "kedaikkum"
    }

    hi_vocab = {
        "namaste", "kaise", "kare", "karna", "karoon", "karein", "chahiye", "kya",
        "bataiye", "batao", "paisa", "paise", "rupaye", "mujhe", "mera", "meri", "mere",
        "nahi", "raha", "rahe", "hai", "hain", "kaunsa", "kaunsi", "kitna", "kitne",
        "karo", "apna", "apni", "shukriya", "dhanyawad", "chalta", "chalti"
    }

    scores['gu'] = len(words.intersection(gu_vocab))
    scores['te'] = len(words.intersection(te_vocab))
    scores['kn'] = len(words.intersection(kn_vocab))
    scores['ta'] = len(words.intersection(ta_vocab))
    scores['hi'] = len(words.intersection(hi_vocab))

    # Distinctive keyword multipliers
    if any(w in words for w in ["chhe", "kayo", "sasto", "mahino", "kevi", "navu", "mane", "karvanu"]):
        scores['gu'] += 4
    if any(w in words for w in ["kothaga", "cheyali", "cheyyali", "nenu", "meeku", "kavali"]):
        scores['te'] += 4
    if any(w in words for w in ["maadabeku", "madabeku", "naanu", "hosa", "hege"]):
        scores['kn'] += 4
    if any(w in words for w in ["eppadi", "seivathu", "pudhiya", "ungalukku"]):
        scores['ta'] += 4
    if any(w in words for w in ["karoon", "karein", "mujhe", "kaise"]):
        scores['hi'] += 4

    best_lang, max_score = max(scores.items(), key=lambda item: item[1])
    if max_score > 0:
        return best_lang

    return fallback_lang or 'en'

@app.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    """
    Main RAG conversational endpoint:
    Automatically detects user input language -> Retrieves DishTV context -> Generates grounded response in user's language.
    """
    pipeline = get_pipeline()
    history_dicts = [{"role": m.role, "content": m.content} for m in request.history] if request.history else []
    
    user_q = request.effective_query
    if not user_q:
        raise HTTPException(status_code=400, detail="Query or message cannot be empty")
        
    # Auto-detect language from user input (zero user selection needed)
    target_language = detect_user_language(user_q, fallback_lang=request.language or "en")

    result = pipeline.answer_query(
        query=user_q,
        language=target_language,
        conversation_history=history_dicts
    )

    # If the generated answer is in an Indic script, synchronize response language
    answer_script = detect_script_language(result["answer"])
    if answer_script:
        target_language = answer_script

    # Prepare synthesized speech audio link in the matching language
    voice_service = get_voice_service()
    audio_path = voice_service.synthesize_speech_file(text=result["answer"], lang=target_language)
    if audio_path and audio_path.exists():
        audio_url = f"/api/voice/audio/{audio_path.name}"
    else:
        encoded_text = urllib.parse.quote(result["answer"][:150])
        audio_url = f"/api/voice/synthesize?text={encoded_text}&lang={target_language}"

    return ChatResponse(
        query=result["query"],
        language=target_language,
        answer=result["answer"],
        citations=result["citations"],
        audio_url=audio_url,
        metrics=result["metrics"]
    )

class VoiceRequest(BaseModel):
    text: str = Field(..., min_length=1)
    lang: str = Field(default="en")

@app.get("/api/voice/audio/{filename}")
def stream_cached_audio(filename: str):
    """Streams cached MP3 audio file directly to browser."""
    voice_service = get_voice_service()
    audio_path = voice_service.get_audio_path_by_filename(filename)
    if not audio_path or not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(
        path=str(audio_path),
        media_type="audio/mpeg",
        filename=filename
    )

@app.get("/api/voice/synthesize")
def synthesize_voice(text: str = Query(..., min_length=1), lang: str = Query(default="en")):
    """Streams synthesized MP3 voice output for customer queries and responses."""
    voice_service = get_voice_service()
    audio_path = voice_service.synthesize_speech_file(text=text, lang=lang)
    if not audio_path or not audio_path.exists():
        raise HTTPException(status_code=500, detail="Voice synthesis failed")
    
    return FileResponse(
        path=str(audio_path),
        media_type="audio/mpeg",
        filename=audio_path.name
    )

@app.post("/api/voice/synthesize")
def synthesize_voice_post(request: VoiceRequest):
    """Accepts JSON payload to synthesize longer answers without URL length restrictions."""
    voice_service = get_voice_service()
    audio_path = voice_service.synthesize_speech_file(text=request.text, lang=request.lang)
    if not audio_path or not audio_path.exists():
        raise HTTPException(status_code=500, detail="Voice synthesis failed")
    return FileResponse(
        path=str(audio_path),
        media_type="audio/mpeg",
        filename=audio_path.name
    )

@app.post("/api/voice/transcribe")
async def transcribe_voice(
    file: UploadFile = File(...),
    lang: Optional[str] = Form(None),
    language: Optional[str] = Query(None)
):
    """
    Transcribes recorded audio (WebM, WAV, MP3) using OpenAI Whisper model.
    Enables accurate multilingual speech recognition across English and 5 Indian languages.
    """
    import uuid
    voice_service = get_voice_service()
    target_lang = lang or language
    ext = Path(file.filename or "recording.webm").suffix or ".webm"
    temp_path = voice_service.cache_dir / f"rec_{uuid.uuid4().hex[:12]}{ext}"
    try:
        content = await file.read()
        with open(temp_path, "wb") as f:
            f.write(content)
        result = voice_service.transcribe_audio_file(temp_path, language=target_lang)
        return JSONResponse(content={
            "success": result.get("success", False),
            "text": result.get("text", ""),
            "language": result.get("language", "en"),
            "error": result.get("error", "")
        })
    except Exception as e:
        logger.error(f"Whisper transcription error: {e}")
        return JSONResponse(status_code=200, content={"success": False, "error": str(e), "text": ""})
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass

@app.get("/api/search")
def search_knowledge_base(q: str = Query(..., min_length=1), k: int = Query(default=4, ge=1, le=10)):
    """Diagnostic semantic search endpoint directly querying FAISS."""
    pipeline = get_pipeline()
    query_vector = pipeline.embedder.embed_query(q)
    results = pipeline.vector_store.search(query_vector, top_k=k)
    return {
        "query": q,
        "count": len(results),
        "results": [
            {
                "title": r.get("title"),
                "url": r.get("url"),
                "category": r.get("category"),
                "similarity_score": round(r.get("similarity_score", 0), 4),
                "snippet": r.get("content", "")[:250] + "..."
            }
            for r in results
        ]
    }

# Mount static web UI files
static_dir = BASE_DIR / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/")
def serve_home():
    """Serves the main DISH TV Chatbot web interface."""
    index_file = static_dir / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return JSONResponse(
        content={"message": "DISH TV Bot API is active. UI is initializing."},
        status_code=200
    )
