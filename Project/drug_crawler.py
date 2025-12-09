"""
drug_crawler.py

Light polite crawler for PharmaIntel (Option A - ~100 pages per site)

- Starts from configured seed URLs
- Collects internal links up to depth=2
- Visits candidate "drug pages" filtered by heuristics
- Extracts structured fields using BeautifulSoup heuristics:
    generic_name, brand_names, indications, mechanism, dosage,
    side_effects, interactions, source_url, site
- Saves to `drug_dataset.jsonl` (one JSON object per line)
- Respects robots.txt via urllib.robotparser (basic)
- Uses small random delays between requests

Notes:
- This is a heuristic-based extractor. Web pages vary; extraction is best-effort.
- You can increase max_pages_per_site for more coverage (be polite).
- For production-grade scraping, build per-site parsers and add retry/backoff/captcha handling.
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import random
import json
import re
import os
import sys
import logging
from collections import deque
import urllib.robotparser

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# ----------------------------
# Configuration (adjustable)
# ----------------------------
SEEDS = {
    "drugs.com": "https://www.drugs.com",
    "medscape": "https://reference.medscape.com/drugs",
    "dailymed": "https://dailymed.nlm.nih.gov/dailymed"
}
OUTPUT_FILE = "drug_dataset.jsonl"
USER_AGENT = "PharmaIntelCrawler/1.0 (+https://example.com/) - student project - contact: you@example.com"
MAX_PAGES_PER_SITE = 100      # Option A: light crawling
REQUEST_TIMEOUT = 15
MIN_DELAY = 1.0
MAX_DELAY = 3.0

# Heuristics for candidate drug page URLs (simple substrings/patterns)
DRUG_URL_PATTERNS = [
    re.compile(r"/drug", re.I),
    re.compile(r"/drugs/", re.I),
    re.compile(r"drug-information", re.I),
    re.compile(r"drugmonograph", re.I),
    re.compile(r"drug\_info", re.I),
    re.compile(r"drugInfo", re.I),
    re.compile(r"setid=", re.I),              # dailymed setid param
    re.compile(r"/drug/"),                    # medscape/drugs
]

# Keywords for extracting sections (case-insensitive)
SECTION_KEYWORDS = {
    "indications": ["indications", "indication", "uses", "clinical uses"],
    "mechanism": ["mechanism", "mechanism of action", "pharmacology"],
    "dosage": ["dosage", "dosage and administration", "administration", "dosage & administration"],
    "side_effects": ["side effects", "adverse reactions", "adverse effects", "adverse events"],
    "interactions": ["interactions", "drug interactions", "drug–drug interactions", "drug–food interactions"],
    "brand_names": ["brand names", "brands", "trade names"]
}

HEADINGS = ["h1", "h2", "h3", "h4", "h5"]

# ----------------------------
# Utilities
# ----------------------------
session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT})

def is_allowed_by_robots(seed_url, path):
    """Check robots.txt for the given domain and path (basic)."""
    try:
        parsed = urlparse(seed_url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        rp = urllib.robotparser.RobotFileParser()
        rp.set_url(robots_url)
        rp.read()
        return rp.can_fetch(USER_AGENT, path)
    except Exception:
        # If robots.txt can't be fetched, default to conservative allow
        return True

def polite_get(url):
    """GET with timeout and polite delay + error handling."""
    try:
        time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))
        resp = session.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp
    except Exception as e:
        logging.warning(f"Request failed for {url}: {e}")
        return None

def is_internal_link(base_netloc, link):
    """Check if link is internal to base netloc."""
    try:
        p = urlparse(link)
        if not p.netloc:
            return True
        return p.netloc.endswith(base_netloc)
    except Exception:
        return False

def clean_text(txt):
    if not txt:
        return ""
    return re.sub(r"\s+", " ", txt).strip()

# ----------------------------
# Extraction heuristics
# ----------------------------
def extract_field_by_section(soup, keywords):
    """
    Find headings that match any keyword and return following text until next heading.
    Returns concatenated text (first match).
    """
    for h in soup.find_all(HEADINGS):
        h_text = clean_text(h.get_text())
        for kw in keywords:
            if kw.lower() in h_text.lower():
                # collect siblings until next heading of same level
                pieces = []
                for sib in h.next_siblings:
                    if getattr(sib, "name", None) and sib.name in HEADINGS:
                        break
                    # paragraphs, divs, lists
                    if getattr(sib, "get_text", None):
                        t = clean_text(sib.get_text())
                        if t:
                            pieces.append(t)
                if pieces:
                    return " ".join(pieces)
    # fallback: search for <p> containing keyword nearby
    for p in soup.find_all("p"):
        if any(kw.lower() in p.get_text().lower() for kw in keywords):
            return clean_text(p.get_text())
    return ""

def extract_brand_names(soup):
    # try specific patterns
    # look for text like "Brand names:" or small blocks
    text = extract_field_by_section(soup, SECTION_KEYWORDS["brand_names"])
    if text:
        return text
    # fallback: look for occurrences like "Brand names: X, Y"
    body_text = soup.get_text(separator="\n")
    m = re.search(r"Brand(?:\s|-)names?:\s*(.+)", body_text, re.I)
    if m:
        return clean_text(m.group(1).split("\n")[0])
    return ""

def extract_generic_name(soup):
    # Prefer the H1 title as generic name if it looks right
    h1 = soup.find("h1")
    if h1:
        name = clean_text(h1.get_text())
        # sometimes title includes extra suffix; trim at '-' or '–' if present
        name = re.split(r"[-–|:]", name)[0].strip()
        return name
    # fallback: meta og:title
    og = soup.find("meta", property="og:title") or soup.find("meta", attrs={"name": "title"})
    if og and og.get("content"):
        return clean_text(og.get("content"))
    return ""

def extract_structured_fields(url, html, site_key):
    """Heuristic extraction from BeautifulSoup object and return dict."""
    soup = BeautifulSoup(html, "html.parser")
    data = {
        "source_url": url,
        "site": site_key,
        "generic_name": None,
        "brand_names": None,
        "indications": None,
        "mechanism": None,
        "dosage": None,
        "side_effects": None,
        "interactions": None,
        "raw_excerpt": None
    }

    try:
        data["generic_name"] = extract_generic_name(soup)
        data["brand_names"] = extract_brand_names(soup)
        data["indications"] = extract_field_by_section(soup, SECTION_KEYWORDS["indications"])
        data["mechanism"] = extract_field_by_section(soup, SECTION_KEYWORDS["mechanism"])
        data["dosage"] = extract_field_by_section(soup, SECTION_KEYWORDS["dosage"])
        data["side_effects"] = extract_field_by_section(soup, SECTION_KEYWORDS["side_effects"])
        data["interactions"] = extract_field_by_section(soup, SECTION_KEYWORDS["interactions"])

        # raw excerpt: first 500 chars of visible text
        visible = soup.get_text(separator=" ")
        data["raw_excerpt"] = clean_text(visible)[:1000]
    except Exception as e:
        logging.warning(f"Extraction error for {url}: {e}")

    # final cleanup: ensure empty strings -> None
    for k, v in list(data.items()):
        if isinstance(v, str):
            v2 = v.strip()
            if v2 == "":
                data[k] = None
            else:
                data[k] = v2
    return data

# ----------------------------
# Crawler core
# ----------------------------
def gather_links_from_page(base_url, html, base_netloc):
    """Return set of absolute internal links found on page."""
    soup = BeautifulSoup(html, "html.parser")
    links = set()
    for a in soup.find_all("a", href=True):
        href = a['href'].split('#')[0].strip()
        if not href:
            continue
        absolute = urljoin(base_url, href)
        # filter mailto, javascript
        if absolute.startswith("mailto:") or absolute.startswith("javascript:"):
            continue
        # only internal domain
        if is_internal_link(base_netloc, absolute):
            # normalize: remove query strings for linking-level (but keep them for visiting)
            links.add(absolute)
    return links

def looks_like_drug_page(url):
    """Heuristic: return True if URL likely points to a drug page."""
    for pat in DRUG_URL_PATTERNS:
        if pat.search(url):
            return True
    # also heuristic: ends with .htm/.html and has path length > 1
    p = urlparse(url)
    if p.path and len(p.path.split("/")) >= 2 and p.path.endswith((".html", ".htm", "/")):
        return True
    return False

def crawl_site(seed_url, site_key, max_pages=100):
    logging.info(f"Starting crawl for {site_key} -> {seed_url}")
    parsed = urlparse(seed_url)
    base_netloc = parsed.netloc
    visited = set()
    to_visit = deque()
    to_visit.append((seed_url, 0))
    pages_visited = 0
    found_drug_pages = set()
    results = []

    while to_visit and pages_visited < max_pages:
        url, depth = to_visit.popleft()
        if url in visited:
            continue
        # check robots
        path_for_robots = urlparse(url).path or "/"
        if not is_allowed_by_robots(seed_url, path_for_robots):
            logging.info(f"Blocked by robots.txt: {url}")
            visited.add(url)
            continue
        resp = polite_get(url)
        visited.add(url)
        if resp is None:
            continue
        pages_visited += 1
        html = resp.text
        logging.info(f"[{site_key}] Visited ({pages_visited}/{max_pages}) depth={depth}: {url}")

        # collect links
        links = gather_links_from_page(url, html, base_netloc)

        # decide which links to add to queue (depth-limited)
        if depth < 1:  # allow following from seed page to its children (depth 0->1), but not deep
            for link in links:
                if link not in visited:
                    to_visit.append((link, depth + 1))

        # If the current page looks like a drug page, attempt extraction
        if looks_like_drug_page(url):
            logging.info(f"Candidate drug page found: {url}")
            data = extract_structured_fields(url, html, site_key)
            # Heuristic: accept if there's at least a generic_name or indications or raw text length
            if data.get("generic_name") or data.get("indications") or (data.get("raw_excerpt") and len(data["raw_excerpt"]) > 200):
                # avoid duplicates by URL
                if url not in found_drug_pages:
                    results.append(data)
                    found_drug_pages.add(url)
                    logging.info(f"Extracted {data.get('generic_name') or url} from {site_key}")
            else:
                logging.info(f"Rejected candidate (insufficient data): {url}")

    logging.info(f"Finished crawling {site_key}. Extracted {len(results)} candidate drug pages.")
    return results

# ----------------------------
# Main runner
# ----------------------------
def main(max_pages_per_site=MAX_PAGES_PER_SITE, output_file=OUTPUT_FILE):
    all_results = []
    for site_key, seed in SEEDS.items():
        try:
            res = crawl_site(seed, site_key, max_pages=max_pages_per_site)
            all_results.extend(res)
        except Exception as e:
            logging.error(f"Error crawling {site_key}: {e}")

    # Save to JSONL
    if not all_results:
        logging.warning("No results extracted — check seeds and network/robots.")
    else:
        with open(output_file, "w", encoding="utf-8") as fh:
            for r in all_results:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        logging.info(f"Saved {len(all_results)} records to {output_file}")

if __name__ == "__main__":
    # Allow optional CLI override for pages per site
    arg = None
    if len(sys.argv) > 1:
        try:
            arg = int(sys.argv[1])
        except Exception:
            arg = None
    if arg:
        main(max_pages_per_site=arg)
    else:
        main()
