"""
Content Processing and Chunking Module
Cleans crawled data from https://dishtv.biztechnosys.com/, parses structured FAQs,
connection plans, quick help actions, and splits text into clean semantic chunks
with rich metadata for vector retrieval.
"""
import re
import json
import logging
from typing import List, Dict, Any

from config import CHUNKS_FILE

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def clean_html_text(text: str) -> str:
    """Clean raw text extracted from HTML."""
    if not text:
        return ""
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    return text.strip()


def determine_category(url: str, title: str) -> str:
    """Classifies a page into a primary functional category."""
    lower_u = url.lower()
    lower_t = title.lower()

    if any(k in lower_u or k in lower_t for k in ["help", "troubleshoot", "101", "102", "error", "credit"]):
        return "Troubleshooting & Quick Help"
    elif any(k in lower_u or k in lower_t for k in ["contact", "care", "helpline", "support", "office"]):
        return "Customer Care & Corporate Contacts"
    elif any(k in lower_u or k in lower_t for k in ["connection", "pack", "plan", "cashback"]):
        return "Connection Plans & Booking"
    elif any(k in lower_u or k in lower_t for k in ["product", "dongle", "key", "stb", "vzy", "dish-hd"]):
        return "Products & Hardware"
    return "General Information & Offerings"


class ContentCleaner:
    """Processes crawled pages into chunked documents."""

    def __init__(self, chunk_size: int = 700, chunk_overlap: int = 120):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str, prefix: str = "") -> List[str]:
        """Splits long text into overlapping chunks, maintaining sentence boundaries."""
        if len(text) <= self.chunk_size:
            return [f"{prefix}\n{text}".strip()]

        sentences = re.split(r'(?<=[.!?\n])\s+', text)
        chunks = []
        current_chunk = []
        current_len = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            sentence_len = len(sentence)

            if current_len + sentence_len > self.chunk_size and current_chunk:
                chunk_str = " ".join(current_chunk)
                chunks.append(f"{prefix}\n{chunk_str}".strip())
                overlap_tokens = []
                overlap_len = 0
                for s in reversed(current_chunk):
                    if overlap_len + len(s) < self.chunk_overlap:
                        overlap_tokens.insert(0, s)
                        overlap_len += len(s)
                    else:
                        break
                current_chunk = overlap_tokens
                current_len = overlap_len

            current_chunk.append(sentence)
            current_len += sentence_len

        if current_chunk:
            chunk_str = " ".join(current_chunk)
            chunks.append(f"{prefix}\n{chunk_str}".strip())

        return chunks

    def process_page(self, page_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Processes a single page from dishtv.biztechnosys.com into chunks."""
        url = page_data.get("url", "")
        title = page_data.get("title", "DISH TV Official")
        meta_desc = page_data.get("meta_description", "")
        cleaned_body = page_data.get("cleaned_body", "")
        structured_faqs = page_data.get("structured_faqs", [])
        structured_plans = page_data.get("structured_plans", [])
        structured_quick_help = page_data.get("structured_quick_help", [])
        category = determine_category(url, title)

        chunks_list = []
        chunk_idx = 0

        # 1. Process Structured FAQs (Highest QA Precision)
        for faq in structured_faqs:
            q = faq.get("question", "").strip()
            a = faq.get("answer", "").strip()
            if q and a:
                faq_content = (
                    f"Official Question: {q}\n"
                    f"Official Answer: {a}\n"
                    f"Portal: DISH TV Official Portal ({url})\n"
                    f"Category: {category}"
                )
                chunks_list.append({
                    "id": f"{url}#faq-{chunk_idx}",
                    "title": f"FAQ: {q}",
                    "url": url,
                    "category": category,
                    "meta_description": f"Frequently asked question on DISH TV: {q}",
                    "content": faq_content,
                    "is_faq": True
                })
                chunk_idx += 1

        # 2. Process Structured Connection Plans
        for plan in structured_plans:
            p_name = plan.get("plan_name", "")
            p_details = plan.get("raw_details", "")
            if p_name and p_details:
                plan_content = (
                    f"DISH TV New Connection Plan: {p_name}\n"
                    f"Category: Connection Plans & Booking\n"
                    f"Source: {url}\n"
                    f"Details:\n{p_details}"
                )
                chunks_list.append({
                    "id": f"{url}#plan-{chunk_idx}",
                    "title": f"Plan: {p_name}",
                    "url": url,
                    "category": "Connection Plans & Booking",
                    "meta_description": f"DISH TV connection plan details for {p_name}",
                    "content": plan_content,
                    "is_plan": True
                })
                chunk_idx += 1

        # 3. Process Quick Help Cards
        for qh in structured_quick_help:
            topic = qh.get("topic", "")
            details = qh.get("details", "")
            if topic and details:
                qh_content = (
                    f"DISH TV Quick Help & Self-Care Action: {topic}\n"
                    f"Category: Troubleshooting & Quick Help\n"
                    f"Source: {url}\n"
                    f"Instructions: {details}"
                )
                chunks_list.append({
                    "id": f"{url}#quickhelp-{chunk_idx}",
                    "title": f"Quick Help: {topic}",
                    "url": url,
                    "category": "Troubleshooting & Quick Help",
                    "meta_description": f"Self-care instruction for {topic}",
                    "content": qh_content,
                    "is_quick_help": True
                })
                chunk_idx += 1

        # 4. Process Cleaned Body Text
        if cleaned_body:
            header = f"Page: {title}\nCategory: {category}\nSource: {url}"
            if meta_desc:
                header += f"\nSummary: {meta_desc}"
            body_chunks = self.chunk_text(cleaned_body, prefix=header)
            for b_chunk in body_chunks:
                chunks_list.append({
                    "id": f"{url}#body-{chunk_idx}",
                    "title": title,
                    "url": url,
                    "category": category,
                    "meta_description": meta_desc,
                    "content": b_chunk,
                    "is_body": True
                })
                chunk_idx += 1

        return chunks_list

    def process_all_pages(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Processes all crawled pages into a unified list of chunks."""
        all_chunks = []
        for page in pages:
            chunks = self.process_page(page)
            all_chunks.extend(chunks)

        logger.info(f"Processed {len(pages)} pages into {len(all_chunks)} semantic chunks purely from crawled website data.")

        # Save to disk
        with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=2)

        return all_chunks
