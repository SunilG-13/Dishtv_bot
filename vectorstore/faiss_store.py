"""
Vector Database Module using FAISS
Stores chunk embeddings and page metadata for low-latency cosine similarity search.
"""
import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import faiss
import numpy as np

from config import FAISS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class FaissVectorStore:
    """FAISS-based vector database with metadata store."""

    def __init__(self, index_dir: Path = FAISS_DIR):
        self.index_dir = Path(index_dir)
        self.index_path = self.index_dir / "index.faiss"
        self.metadata_path = self.index_dir / "metadata.json"
        self.index: Optional[faiss.Index] = None
        self.metadata: List[Dict[str, Any]] = []

    def is_indexed(self) -> bool:
        """Returns True if the index and metadata files exist."""
        return self.index_path.exists() and self.metadata_path.exists()

    def build_index(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray) -> None:
        """Builds and stores a new FAISS index from vectors and chunk metadata."""
        if len(chunks) != len(embeddings):
            raise ValueError(f"Chunk count ({len(chunks)}) does not match embeddings count ({len(embeddings)})")

        dimension = embeddings.shape[1]
        logger.info(f"Building FAISS IndexFlatIP with dimension {dimension} for {len(chunks)} chunks...")
        
        # Inner Product with normalized vectors corresponds to Cosine Similarity
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        self.metadata = chunks

        logger.info(f"FAISS index built successfully. Total vectors: {self.index.ntotal}")

    def save(self) -> None:
        """Saves the FAISS index and metadata to disk."""
        if self.index is None or not self.metadata:
            raise ValueError("No index or metadata to save.")

        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

        logger.info(f"FAISS index saved to {self.index_path} and metadata to {self.metadata_path}")

    def load(self) -> bool:
        """Loads the FAISS index and metadata from disk."""
        if not self.is_indexed():
            logger.warning(f"Index or metadata not found in {self.index_dir}")
            return False

        try:
            logger.info(f"Loading FAISS index from {self.index_path}...")
            self.index = faiss.read_index(str(self.index_path))
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
            logger.info(f"Loaded {self.index.ntotal} vectors and {len(self.metadata)} metadata records.")
            return True
        except Exception as e:
            logger.error(f"Error loading FAISS store: {e}")
            return False

    def search(self, query_embedding: np.ndarray, top_k: int = 4) -> List[Dict[str, Any]]:
        """
        Searches the index for the most similar chunks.
        Returns list of chunk metadata dictionaries enriched with 'similarity_score'.
        """
        if self.index is None:
            if not self.load():
                return []

        # Ensure 2D shape (1, dim)
        if len(query_embedding.shape) == 1:
            query_embedding = np.expand_dims(query_embedding, axis=0)

        query_embedding = query_embedding.astype(np.float32)
        top_k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0 or idx >= len(self.metadata):
                continue
            item = dict(self.metadata[idx])
            item["similarity_score"] = float(score)
            item["rank"] = rank + 1
            results.append(item)

        return results


# Global singleton instance
_faiss_store_instance = None

def get_faiss_store() -> FaissVectorStore:
    """Returns singleton instance of FaissVectorStore."""
    global _faiss_store_instance
    if _faiss_store_instance is None:
        _faiss_store_instance = FaissVectorStore()
    return _faiss_store_instance
