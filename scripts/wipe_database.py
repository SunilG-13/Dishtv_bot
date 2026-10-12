"""
Script to wipe all existing data, chunks, and FAISS vector index from the DishTV Bot database.
Preserves the directory structure and application configuration intact.

Usage:
    python scripts/wipe_database.py
    python scripts/wipe_database.py -y              # Skip confirmation prompt
    python scripts/wipe_database.py -y --rebuild    # Wipe and immediately crawl & rebuild
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

# Add project root to python path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config import DATA_DIR, RAW_DATA_DIR, FAISS_DIR, CHUNKS_FILE, AUDIO_CACHE_DIR

TARGETS = [
    ("Raw Scraped Pages", RAW_DATA_DIR, True),
    ("Semantic Chunks File", CHUNKS_FILE, False),
    ("FAISS Vector Index", FAISS_DIR, True),
    ("Cached TTS Audio", AUDIO_CACHE_DIR, True),
]


def wipe_all_data():
    """Wipes all data files, vector indices, and cached pages."""
    print("=" * 60)
    print("      DISH TV BOT — DATABASE WIPE & RESET UTILITY       ")
    print("=" * 60)
    print(f"Data root: {DATA_DIR}\n")

    cleared_items = 0

    for name, target_path, is_dir in TARGETS:
        target = Path(target_path)
        if is_dir:
            if target.exists():
                count = len(list(target.glob("*")))
                shutil.rmtree(target)
                target.mkdir(parents=True, exist_ok=True)
                print(f"  [CLEARED] {name:22} -> Emptied {count} file(s) ({target.relative_to(ROOT_DIR)})")
                cleared_items += 1
            else:
                target.mkdir(parents=True, exist_ok=True)
                print(f"  [READY]   {name:22} -> Created clean empty directory ({target.relative_to(ROOT_DIR)})")
        else:
            if target.exists():
                target.unlink()
                print(f"  [DELETED] {name:22} -> Removed file ({target.relative_to(ROOT_DIR)})")
                cleared_items += 1
            else:
                print(f"  [SKIPPED] {name:22} -> Already clean / not found")

    print("\n" + "=" * 60)
    print("SUCCESS: Database and vector store have been completely wiped!")
    print("=" * 60)
    print("\nNext steps to populate with fresh website data:")
    print("  1. Crawl website:    python crawler/dishtv_crawler.py")
    print("  2. Build vector DB:  python build_index.py")
    print("  3. Start server:     python run.py\n")


def rebuild_database():
    """Triggers crawler and build_index to ingest fresh website data."""
    print("\n[+] Triggering full rebuild pipeline...")
    try:
        from crawler.dishtv_crawler import DishTVCrawler
        from build_index import run_pipeline

        print("--> Step 1/2: Crawling live site (https://dishtv.biztechnosys.com/)...")
        crawler = DishTVCrawler()
        crawler.crawl_all(force_refresh=True)

        print("--> Step 2/2: Building FAISS dense vector embeddings...")
        run_pipeline(force_recrawl=False)

        print("\n[OK] Rebuild complete! Database is freshly populated and ready.")
    except Exception as e:
        print(f"\n[ERROR] Rebuild encountered an error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wipe DishTV bot database and vector store.")
    parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompt")
    parser.add_argument("--rebuild", action="store_true", help="Automatically crawl and rebuild fresh index after wiping")
    args = parser.parse_args()

    if args.yes:
        wipe_all_data()
        if args.rebuild:
            rebuild_database()
    else:
        print("WARNING: This will permanently delete all indexed vectors, scraped pages, and chunks!")
        confirm = input("Are you sure you want to wipe ALL database data? (yes/no): ").strip().lower()
        if confirm in ("yes", "y"):
            wipe_all_data()
            if args.rebuild:
                rebuild_database()
        else:
            print("Operation cancelled. No data was deleted.")
