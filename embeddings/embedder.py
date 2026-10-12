"""
Multilingual Embeddings Module using BAAI/bge-m3
Converts text chunks and customer queries into high-dimensional dense vectors
supporting cross-lingual semantic search across Indian languages.
"""
import os
import logging
from typing import List, Union
import numpy as np

from config import EMBEDDING_MODEL_NAME

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class MultilingualEmbedder:
    """Embedder powered by BAAI/bge-m3."""

    def __init__(self, model_name_or_path: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name_or_path
        self.model = None
        self._load_model()

    def _load_model(self):
        """Loads the SentenceTransformer model from cache or HuggingFace."""
        from sentence_transformers import SentenceTransformer
        
        # Check if local snapshot is present to guarantee offline/cached loading
        cache_snapshot = os.path.expanduser(
            "~/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181"
        )
        target = cache_snapshot if os.path.exists(cache_snapshot) else self.model_name

        logger.info(f"Loading embedding model from: {target}...")
        try:
            import torch
            torch.set_num_threads(os.cpu_count() or 4)
            self.model = SentenceTransformer(target)
            self.model.max_seq_length = 512
            self.dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"Loaded BAAI/bge-m3 successfully. Embedding dimension: {self.dimension}")
        except Exception as e:
            logger.error(f"Failed to load {target}: {e}")
            logger.info("Attempting fallback to standard HuggingFace load...")
            self.model = SentenceTransformer("BAAI/bge-m3")
            self.model.max_seq_length = 512
            self.dimension = self.model.get_sentence_embedding_dimension()

    def embed_documents(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        """Embeds a list of text chunks and returns L2-normalized numpy array."""
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        logger.info(f"Embedding {len(texts)} chunks with BAAI/bge-m3 (batch_size={batch_size})...")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embeds a single customer query and returns normalized 1D float32 array."""
        embedding = self.model.encode(
            query,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        return embedding.astype(np.float32)


# Singleton instance for app reuse
_embedder_instance = None

def get_embedder() -> MultilingualEmbedder:
    """Returns singleton instance of MultilingualEmbedder."""
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = MultilingualEmbedder()
    return _embedder_instance
