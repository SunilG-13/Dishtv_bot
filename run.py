"""
Main Application Runner
Launches the DISH TV Multilingual Knowledge Bot server with FastAPI and Uvicorn.
Ensures knowledge base index is ready and displays active endpoints.
"""
# Python 3.14 beta compatibility patch for pydantic
# Fixes: TypeError: _eval_type() got an unexpected keyword argument 'prefer_fwd_module'
import sys
if sys.version_info >= (3, 14):
    import typing
    _original_eval_type = typing._eval_type
    def _patched_eval_type(*args, **kwargs):
        kwargs.pop('prefer_fwd_module', None)
        return _original_eval_type(*args, **kwargs)
    typing._eval_type = _patched_eval_type

import uvicorn
from config import HOST, PORT
from vectorstore.faiss_store import get_faiss_store
from build_index import run_pipeline

def main():
    print("=" * 60)
    print("        DISH TV MULTILINGUAL AI ASSISTANT        ")
    print("      Powered by DeepSeek API & BAAI/bge-m3 + FAISS ")
    print("=" * 60)

    # Check vector database status
    store = get_faiss_store()
    if not store.is_indexed():
        print("[!] FAISS index not found. Building Knowledge Base from DISH TV...")
        run_pipeline()
    else:
        print("[OK] FAISS Vector Database is ready.")

    print(f"\n[+] Starting DISH TV AI Assistant Server:")
    print(f"     -> Web Interface: http://{HOST}:{PORT}/")
    print(f"     -> API Docs:      http://{HOST}:{PORT}/docs")
    print(f"     -> Health Check:  http://{HOST}:{PORT}/api/health")
    print(f"     -> Languages:     http://{HOST}:{PORT}/api/languages\n")

    uvicorn.run("api.main:app", host=HOST, port=PORT, reload=False, log_level="info")

if __name__ == "__main__":
    main()
