# DISH TV Multilingual AI Assistant — Production RAG System

An enterprise-ready AI Customer Support Knowledge Assistant for **DISH TV**, capable of answering customer questions in 6 regional Indian languages (**English, Hindi, Kannada, Telugu, Tamil, and Gujarati**) using real-time RAG (Retrieval-Augmented Generation) grounded strictly on official DishTV portal content: [https://dishtv.biztechnosys.com/](https://dishtv.biztechnosys.com/).

---

## 🏗️ Architecture & Technology Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **LLM** | **DeepSeek API (`deepseek-chat`)** | Generates grounded, executive, and direct responses in the user's language without hallucinations |
| **Embedding Model** | **BAAI/bge-m3** | 1024-dimensional dense multilingual embeddings for cross-lingual semantic search |
| **Vector Database** | **FAISS (`IndexFlatIP`)** | Sub-millisecond cosine similarity retrieval over chunked knowledge vectors |
| **Backend API** | **FastAPI & Uvicorn** | High-performance asynchronous REST API for chat, search, voice transcription, and streaming |
| **Web Crawler** | **Requests & BeautifulSoup4** | Crawls official DishTV portal pages, FAQs, plans, and missed-call quick-help triggers |
| **Speech-to-Text** | **OpenAI Whisper (`small`)** | High-accuracy multilingual speech transcription with acoustic filtering for Indian languages |
| **Text-to-Speech** | **Google Text-to-Speech (gTTS)** | Natural speech synthesis with audio text-cleansing and persistent disk caching |
| **Frontend UI** | **Vanilla HTML5, CSS3 & JavaScript** | Responsive DishTV portal interface with dark/light mode, real-time voice recording, and audio playback |

---

## 📁 Repository Structure

```
Dishtv_bot/
├── config.py                  # API keys, model parameters, directories, language configurations
├── run.py                     # Main server entrypoint (FastAPI / Uvicorn runner)
├── build_index.py             # Knowledge ingestion pipeline (crawl -> clean -> embed -> FAISS index)
├── requirements.txt           # Production Python dependencies
├── .env                       # Environment variables (API keys & configuration)
├── .env.example               # Template environment configuration file
├── .gitignore                 # Excludes bytecode, venv, secrets, and cache
├── README.md                  # System architecture and setup documentation
├── test_questions_all_languages.md # Multilingual verification questions across 6 languages
├── evaluation_report.json     # Automated benchmark report
├── api/
│   ├── __init__.py
│   └── main.py                # FastAPI endpoints (/api/chat, /api/health, /api/voice/..., etc.)
├── crawler/
│   ├── __init__.py
│   └── dishtv_crawler.py      # Web crawler for https://dishtv.biztechnosys.com/
├── processor/
│   ├── __init__.py
│   └── content_cleaner.py     # HTML cleaner, FAQ extractor, and semantic text chunker
├── embeddings/
│   ├── __init__.py
│   └── embedder.py            # BAAI/bge-m3 vector embedder and normalizer
├── vectorstore/
│   ├── __init__.py
│   └── faiss_store.py         # FAISS IndexFlatIP vector database & metadata repository
├── rag/
│   ├── __init__.py
│   ├── deepseek_client.py     # DeepSeek API client with retry and error handling
│   └── rag_pipeline.py        # End-to-end RAG orchestrator with grounding guards
├── voice/
│   ├── __init__.py
│   └── voice_service.py       # OpenAI Whisper STT & gTTS audio caching service
├── evaluation/
│   ├── __init__.py
│   └── evaluate.py            # Automated benchmark evaluation suite
├── scripts/
│   ├── __init__.py
│   └── wipe_database.py       # Safe utility to reset/wipe vector index and cached pages
├── static/
│   ├── index.html             # Customer Chatbot Web Interface & Portal
│   ├── css/
│   │   └── style.css          # Modular styling sheet
│   ├── js/
│   │   └── app.js             # Modular frontend application logic
│   └── images/
│       ├── dishtv-logonew.png # Official dark logo asset
│       └── dishtv-logo-white-trans.png # Official white logo asset
└── data/
    ├── raw_pages/             # Crawled portal pages cached as JSON
    ├── chunks.json            # Semantic chunks with category and source metadata
    ├── faiss_index/           # FAISS binary vector index (index.faiss) & metadata.json
    └── audio_cache/           # Synthesized MP3 audio cache
```

---

## 🚀 Quick Start Guide

### 1. Set Up Virtual Environment

#### On Windows (PowerShell / Command Prompt):
```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate virtual environment
.\venv\Scripts\activate

# 3. Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Configure Environment (`.env`)
Create or verify your `.env` file based on `.env.example`:
```ini
DEEPSEEK_API_KEY=your_deepseek_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
EMBEDDING_MODEL=BAAI/bge-m3
HOST=127.0.0.1
PORT=8000
```

### 3. Build Knowledge Base Index
If you want to re-crawl and re-index the knowledge base:
```bash
python build_index.py
```
*Crawls official portal pages from `https://dishtv.biztechnosys.com/`, chunks text, generates BAAI/bge-m3 dense vectors, and saves `data/faiss_index/index.faiss`.*

### 4. Start the Application Server
```bash
python run.py
```
- **Web Interface:** [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- **Swagger Documentation:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Health Check Endpoint:** [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)

### 5. Reset / Wipe Database Utility (Optional)
To safely clear vectors and cached content:
```bash
# Interactive confirmation prompt:
python scripts/wipe_database.py

# Skip prompt and wipe immediately:
python scripts/wipe_database.py -y

# Wipe and immediately crawl & rebuild fresh index:
python scripts/wipe_database.py -y --rebuild
```

---

## 🌐 Supported Regional Languages

The assistant supports native script recognition, automatic Romanized transliteration detection, Speech-to-Text, and Text-to-Speech across 6 languages:

| Language | Native Script | Voice Code (STT/TTS) | Sample Query |
| :--- | :--- | :--- | :--- |
| **English** | English | `en-IN` / `en` | *What are the new connection plans available on DishTV?* |
| **Hindi** | हिन्दी | `hi-IN` / `hi` | *DishTV के नए कनेक्शन प्लान कौन-कौन से हैं?* |
| **Telugu** | తెలుగు | `te-IN` / `te` | *DishTV కొత్త కనెక్షన్ ప్లాన్లు ఏమిటి?* |
| **Kannada** | ಕನ್ನಡ | `kn-IN` / `kn` | *DishTV ಹೊಸ ಸಂಪರ್ಕ ಪ್ಲಾನ್‌ಗಳು ಯಾವುವು?* |
| **Tamil** | தமிழ் | `ta-IN` / `ta` | *DishTV புதிய இணைப்பு திட்டங்கள் என்னென்ன உள்ளன?* |
| **Gujarati** | ગુજરાતી | `gu-IN` / `gu` | *DishTV ના નવા કનેક્શન પ્લાન કયા કયા છે?* |

---

## 🧪 Automated Evaluation Benchmark (`evaluation/evaluate.py`)

Run the automated test suite across all categories:
```bash
python -m evaluation.evaluate
```

Results are saved to `evaluation_report.json` with retrieval relevance scores, end-to-end latency, and anti-hallucination checks.
