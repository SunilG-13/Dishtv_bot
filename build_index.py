"""
Knowledge Base Ingestion & Indexing Pipeline
Crawls DISH TV pages, processes & chunks content, generates BGE-M3 embeddings,
and constructs the FAISS vector database.
"""
import time
import logging
from crawler.dishtv_crawler import DishTVCrawler
from processor.content_cleaner import ContentCleaner
from embeddings.embedder import MultilingualEmbedder
from vectorstore.faiss_store import FaissVectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_pipeline(force_recrawl: bool = False):
    logger.info("==================================================")
    logger.info("   Starting DISH TV Knowledge Base Ingestion      ")
    logger.info("==================================================")
    start_total = time.time()

    # Step 1: Crawl DISH TV Pages
    logger.info("Step 1: Crawling DISH TV Website Content...")
    crawler = DishTVCrawler()
    pages = crawler.crawl_all(force_refresh=force_recrawl)
    logger.info(f"Crawled {len(pages)} pages successfully.")

    # Step 2: Clean and Chunk Content
    logger.info("Step 2: Cleaning & Chunking Content...")
    cleaner = ContentCleaner(chunk_size=750, chunk_overlap=120)
    chunks = cleaner.process_all_pages(pages)
    logger.info(f"Generated {len(chunks)} knowledge chunks.")

    # Step 3: Multilingual Embedding Generation (BAAI/bge-m3)
    logger.info("Step 3: Generating Multilingual Embeddings with BAAI/bge-m3...")
    embedder = MultilingualEmbedder()
    texts = [c["content"] for c in chunks]
    embeddings = embedder.embed_documents(texts, batch_size=32)
    logger.info(f"Embeddings shape: {embeddings.shape}")

    # Step 4: Build FAISS Vector Database
    logger.info("Step 4: Building & Saving FAISS Index...")
    vector_store = FaissVectorStore()
    vector_store.build_index(chunks, embeddings)
    vector_store.save()
    logger.info("FAISS Index and metadata successfully saved!")

    # Step 5: Smoke Test Retrieval
    logger.info("Step 5: Testing Vector Retrieval...")
    test_queries = [
        "What are the DishTV HD connection plans and cashback offers?",
        "How to fix Error 101/102 and resume services?",
        "What is DishTV SMART+ STB Dongle and its price?"
    ]
    for q in test_queries:
        q_vec = embedder.embed_query(q)
        results = vector_store.search(q_vec, top_k=2)
        logger.info(f"Test Query: '{q}'")
        for r in results:
            logger.info(f"  -> [{r['similarity_score']:.3f}] {r['title']} ({r['url']})")

    elapsed = round(time.time() - start_total, 2)
    logger.info(f"Knowledge Base Build Complete in {elapsed} seconds!")
    return len(chunks)


if __name__ == "__main__":
    run_pipeline(force_recrawl=True)
