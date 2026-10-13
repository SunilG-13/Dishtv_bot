"""
DISH TV Website Crawler Module
Extracts products, plans, troubleshooting, FAQs, and service information
from the official portal: https://dishtv.live/
"""
import os
import re
import json
import logging
import hashlib
from typing import List, Dict, Any, Optional, Set
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup

from config import RAW_DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_SITE_URL = "https://dishtv.live/"

# Key curated seed URLs on dishtv.live
DISH_TV_TARGET_URLS = [
    "https://dishtv.live/",
    "https://dishtv.live/Get-Help",
    "https://dishtv.live/ContactUs",
    "https://dishtv.live/Get-a-Connection",
    "https://dishtv.live/Products",
    "https://dishtv.live/Products/Dish-HD",
    "https://dishtv.live/Products/DishTV-STB-Dongle",
    "https://dishtv.live/Products/DishTV-KEY"
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://dishtv.live/"
}


class DishTVCrawler:
    """Crawler to fetch DISH TV public web content from https://dishtv.live/ and extract structured data."""

    def __init__(self, cache_dir: str = str(RAW_DATA_DIR), base_url: str = BASE_SITE_URL):
        self.cache_dir = cache_dir
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    def _url_to_cache_path(self, url: str) -> str:
        url_hash = hashlib.md5(url.encode("utf-8")).hexdigest()
        clean_url = url.split("://")[-1]
        filename = re.sub(r'[^a-zA-Z0-9_-]', '_', clean_url)[:50] + f"_{url_hash}.json"
        return os.path.join(self.cache_dir, filename)

    def discover_internal_urls(self, seed_urls: List[str]) -> List[str]:
        """Discovers all internal URLs on the target domain starting from seeds."""
        discovered: Set[str] = set()
        queue: List[str] = list(seed_urls)
        base_domain = urlparse(self.base_url).netloc.lower()

        while queue:
            current_url = queue.pop(0).split("#")[0].rstrip("/")
            if not current_url or current_url in discovered:
                continue
            discovered.add(current_url)

            try:
                response = self.session.get(current_url, timeout=12)
                if response.status_code != 200:
                    continue
                soup = BeautifulSoup(response.text, "html.parser")
                for link in soup.find_all("a", href=True):
                    href = link["href"].strip()
                    if not href or href.startswith(("javascript:", "tel:", "mailto:", "#")):
                        continue
                    full_url = urljoin(current_url, href).split("#")[0].rstrip("/")
                    parsed = urlparse(full_url)
                    if parsed.netloc.lower() == base_domain:
                        if full_url not in discovered and full_url not in queue:
                            queue.append(full_url)
            except Exception as e:
                logger.warning(f"Could not discover links from {current_url}: {e}")

        ordered_urls = []
        for s in seed_urls:
            s_norm = s.rstrip("/")
            if s_norm in discovered and s_norm not in ordered_urls:
                ordered_urls.append(s_norm)
        for u in sorted(discovered):
            if u not in ordered_urls:
                ordered_urls.append(u)
        return ordered_urls

    def extract_faqs(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        """Extracts question-and-answer pairs from FAQ accordions."""
        faqs = []
        for faq_el in soup.find_all(class_="faq-item"):
            q_el = faq_el.find(class_="faq-question") or faq_el.find("button")
            ans_el = faq_el.find(id=lambda x: x and x.startswith("faq-answer")) or faq_el.find("p")
            if q_el and ans_el:
                q_text = q_el.get_text(strip=True)
                ans_text = ans_el.get_text(separator=" ", strip=True)
                if q_text and ans_text:
                    faqs.append({"question": q_text, "answer": ans_text})
        return faqs

    def extract_plans(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extracts connection plans with pricing and features."""
        plans = []
        # Look for plan cards in Get-a-Connection
        plan_titles = ["HD Plan With Pack", "HD Cashback Plan", "HD Plan For South"]
        for title in plan_titles:
            heading = soup.find(lambda el: el.name in ["h2", "h3", "h4", "p", "div"] and title.lower() in el.get_text().lower())
            if heading:
                parent = heading.find_parent(class_=lambda c: c and any(k in str(c) for k in ["card", "col", "grid", "plan"])) or heading.parent
                if parent:
                    text = parent.get_text(separator="\n", strip=True)
                    plans.append({"plan_name": title, "raw_details": text})
        return plans

    def extract_quick_help(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        """Extracts quick help cards and missed-call triggers."""
        quick_help_items = []
        keywords = ["Do it Yourself", "My Account", "Activate a Channel", "Get 3 days credit", "Great Offers", "101/102 Error"]
        for kw in keywords:
            el = soup.find(lambda e: e.name in ["h2", "h3", "h4", "h5", "div", "span"] and kw.lower() in e.get_text().lower())
            if el:
                card = el.find_parent(class_=lambda c: c and any(k in str(c) for k in ["card", "col", "item", "rounded"])) or el.parent
                if card:
                    card_text = card.get_text(separator=" ", strip=True)
                    quick_help_items.append({"topic": kw, "details": card_text})
        return quick_help_items

    def fetch_page(self, url: str, force_refresh: bool = False) -> Optional[Dict[str, Any]]:
        cache_path = self._url_to_cache_path(url)
        if not force_refresh and os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading cached page {cache_path}: {e}")

        logger.info(f"Crawling URL: {url}")
        try:
            response = self.session.get(url, timeout=12)
            if response.status_code != 200:
                logger.warning(f"Failed to fetch {url}: HTTP {response.status_code}")
                return None

            html = response.text
            soup = BeautifulSoup(html, "html.parser")

            # Extract Title
            title = ""
            if soup.title and soup.title.string:
                title = soup.title.string.strip()
            if not title or title.lower() in ["dishtv", "page", "live content"]:
                h1 = soup.find(["h1", "h2"])
                if h1:
                    title = f"DishTV - {h1.get_text(strip=True)}"
                else:
                    path_name = url.split("/")[-1].replace("-", " ").strip()
                    title = f"DishTV {path_name.capitalize()}" if path_name else "DishTV Official Portal"

            # Extract Meta Description
            meta_desc = ""
            desc_tag = soup.find("meta", attrs={"name": "description"}) or soup.find("meta", attrs={"property": "og:description"})
            if desc_tag and desc_tag.get("content"):
                meta_desc = desc_tag["content"].strip()
                if "Sitecore Next.js" in meta_desc:
                    meta_desc = "Official Dish TV portal: offerings, plans, customer care, and products."

            # Structured Extractions
            faqs = self.extract_faqs(soup)
            plans = self.extract_plans(soup)
            quick_help = self.extract_quick_help(soup)

            # Clean and extract main text content
            for remove_tag in soup(["script", "style", "noscript", "iframe", "svg"]):
                remove_tag.decompose()

            # Extract full text from main and footer sections to capture all cards, contact info, and specs
            parts = []
            main = soup.find("main")
            footer = soup.find("footer")
            if main:
                parts.append(main.get_text(separator="\n", strip=True))
            if footer:
                parts.append(footer.get_text(separator="\n", strip=True))
            if not parts:
                parts.append(soup.get_text(separator="\n", strip=True))

            full_text = "\n".join(parts)
            # Clean and deduplicate lines
            content_blocks = []
            prev_line = ""
            for line in full_text.splitlines():
                line = line.strip()
                if len(line) > 2 and line != prev_line and not any(skip in line.lower() for skip in ["login with otp", "forgot password", "enter the otp"]):
                    content_blocks.append(line)
                    prev_line = line

            cleaned_body = "\n".join(content_blocks)

            page_data = {
                "url": url,
                "title": title,
                "meta_description": meta_desc,
                "cleaned_body": cleaned_body,
                "structured_faqs": faqs,
                "structured_plans": plans,
                "structured_quick_help": quick_help,
                "raw_html_snippet": html[:1000]
            }

            # Save to cache
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(page_data, f, ensure_ascii=False, indent=2)

            return page_data

        except Exception as e:
            logger.error(f"Error crawling {url}: {e}")
            return None

    def crawl_all(self, urls: Optional[List[str]] = None, force_refresh: bool = False, auto_discover: bool = True) -> List[Dict[str, Any]]:
        target_urls = urls or DISH_TV_TARGET_URLS
        if auto_discover:
            logger.info("Discovering all internal URLs on https://dishtv.live/...")
            target_urls = self.discover_internal_urls(target_urls)

        logger.info(f"Starting crawl of {len(target_urls)} DISH TV URLs from {self.base_url}...")
        pages = []
        for url in target_urls:
            page_data = self.fetch_page(url, force_refresh=force_refresh)
            if page_data:
                pages.append(page_data)
        logger.info(f"Crawling completed. Successfully fetched {len(pages)} pages.")
        return pages
